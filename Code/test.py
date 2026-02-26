#test.py
import os
import torch
import numpy as np
from skimage.io import imsave
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import names from hybrid_model, supporting both old and new variants
try:
    from hybrid_model import HybridModel, LEFuse
except ImportError as e:
    logger.error(f"Failed to import HybridModel/LEFuse from hybrid_model: {e}")
    raise

# Prefer SRUnet if defined (older code), else fall back to EnhancedSRUnet (newer code)
UnetClass = None
try:
    from hybrid_model import SRUnet as UnetClass  # older projects
    logger.info("Using SRUnet from hybrid_model")
except Exception:
    try:
        from hybrid_model import EnhancedSRUnet as UnetClass  # newer project
        logger.info("Using EnhancedSRUnet from hybrid_model")
    except Exception as e:
        logger.error("Neither SRUnet nor EnhancedSRUnet is available in hybrid_model.")
        raise

from img_utils import image_read_cv2, YCbCr2RGB, RGB2YCrCb


def _ensure_dirs(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'debug'), exist_ok=True)
    os.makedirs(os.path.join(path, 'intermediate'), exist_ok=True)


def _build_models(device: str):
    """
    Create models in a way that works with BOTH hybrid_model variants:
    - Old: HybridModel(lefuse, unet)   + SRUnet
    - New: HybridModel() (no args)     + EnhancedSRUnet and optional self-ensemble
    """
    # First try the new no-arg constructor
    try:
        hybrid_model = HybridModel().to(device)
        lefuse = getattr(hybrid_model, "lefuse", None)
        unet = getattr(hybrid_model, "unet", None)
        if lefuse is None or unet is None:
            raise RuntimeError("HybridModel() did not expose 'lefuse'/'unet' attributes")
        logger.info("Initialized HybridModel() (no-arg) and found internal lefuse/unet")
        return hybrid_model, lefuse, unet, True
    except TypeError as e:
        # Old signature requires lefuse, unet
        logger.info(f"HybridModel() raised {e}. Falling back to HybridModel(lefuse, unet).")
        lefuse = LEFuse().to(device)
        # Use the detected UnetClass (SRUnet or EnhancedSRUnet)
        unet = UnetClass(in_channels=3, out_channels=3, scale=4).to(device)
        hybrid_model = HybridModel(lefuse, unet).to(device)
        logger.info("Initialized HybridModel(lefuse, unet)")
        return hybrid_model, lefuse, unet, False


def _load_lefuse_weights(lefuse, ckpt_lefuse, device):
    state_dict = torch.load(ckpt_lefuse, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        logger.info("LEFuse checkpoint: loaded from 'model' key")
        state_dict = state_dict["model"]
    else:
        logger.info("LEFuse checkpoint: using state_dict directly (no 'model' key)")

    lefuse_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('content.fuser.'):
            lefuse_state_dict[k.replace('content.fuser.', 'fuser.')] = v
        elif k.startswith('fuser.'):
            lefuse_state_dict[k] = v

    if not lefuse_state_dict:
        raise RuntimeError("No compatible keys found in LEFuse checkpoint. Expected 'fuser.' or 'content.fuser.' prefixes.")

    missing, unexpected = lefuse.load_state_dict(lefuse_state_dict, strict=False)
    logger.info(f"Loaded LEFuse weights. Missing keys: {missing}, Unexpected keys: {unexpected}")


def _load_unet_weights(unet, ckpt_unet, device):
    state_dict = torch.load(ckpt_unet, map_location=device)
    target_sd = unet.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in target_sd and v.shape == target_sd[k].shape}
    ignored = [k for k in state_dict.keys() if k not in compatible]
    target_sd.update(compatible)
    missing, unexpected = unet.load_state_dict(target_sd, strict=False)
    logger.info(f"Loaded UNet weights. Ignored: {ignored}. Missing: {missing}, Unexpected: {unexpected}")


def test(ckpt_lefuse, ckpt_unet, vi_path, ir_path, out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    _ensure_dirs(out_path)

    # Initialize models in a way that supports both APIs
    try:
        hybrid_model, lefuse, unet, new_api = _build_models(device)
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

    # Load weights
    try:
        _load_lefuse_weights(lefuse, ckpt_lefuse, device)
    except Exception as e:
        logger.error(f"Failed to load LEFuse weights: {e}")
        raise

    try:
        _load_unet_weights(unet, ckpt_unet, device)
    except Exception as e:
        logger.error(f"Failed to load UNet weights: {e}")
        raise

    hybrid_model.eval()

    # Verify image files
    exts = ('.png', '.jpg', '.jpeg')
    vi_files = sorted([f for f in os.listdir(vi_path) if f.lower().endswith(exts)])
    ir_files = sorted([f for f in os.listdir(ir_path) if f.lower().endswith(exts)])
    if vi_files != ir_files:
        logger.error("Mismatched filenames between vi and ir directories")
        raise ValueError("Mismatched filenames between vi and ir directories")

    with torch.no_grad():
        for img_name in vi_files:
            try:
                # Read visible and infrared images
                vi = image_read_cv2(os.path.join(vi_path, img_name), mode='RGB')[np.newaxis, ...] / 255.0
                vi = np.transpose(vi, (0, 3, 1, 2))  # [1, 3, H, W]
                vi = torch.from_numpy(vi).float().to(device)

                ir = image_read_cv2(os.path.join(ir_path, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
                ir = torch.from_numpy(ir).float().to(device)  # [1, 1, H, W]

                # Dimension check
                if vi.size(2) % 8 != 0 or vi.size(3) % 8 != 0:
                    logger.warning(f"Image {img_name} dimensions {tuple(vi.size()[2:])} not divisible by 8; may affect borders")

                # --- Prepare SR input (mimic HybridModel forward up to SR input) ---
                y, cr, cb = RGB2YCrCb(vi)                 # Y, Cr, Cb
                fused_y = lefuse(y, ir)                   # fused luminance
                fused_rgb_lr = YCbCr2RGB(fused_y, cb, cr) # SR input

                # Save SR input (debug)
                fused_rgb_lr_np_debug = fused_rgb_lr.detach().cpu().numpy().squeeze(0)
                fused_rgb_lr_np_debug = np.transpose(fused_rgb_lr_np_debug, (1, 2, 0))
                fused_rgb_lr_np_debug = (fused_rgb_lr_np_debug * 255.0).clip(0, 255).astype('uint8')
                debug_path = os.path.join(out_path, 'debug', f"{os.path.splitext(img_name)[0]}_sr_input.png")
                imsave(debug_path, fused_rgb_lr_np_debug)
                logger.info(f"Saved SR input: {debug_path} "
                            f"(min {fused_rgb_lr.min().item():.4f}, max {fused_rgb_lr.max().item():.4f})")

                # --- Super-resolution ---
                # Prefer self-ensemble if hybrid_model exposes it; else run the UNet directly.
                if hasattr(hybrid_model, "ensemble") and callable(getattr(hybrid_model, "ensemble")):
                    fused_rgb_hr = hybrid_model.ensemble(fused_rgb_lr)
                else:
                    fused_rgb_hr = unet(fused_rgb_lr)

                logger.info(f"SR output stats for {img_name} - min: {fused_rgb_hr.min().item():.4f}, "
                            f"max: {fused_rgb_hr.max().item():.4f}")

                # Save intermediate low-resolution fused image
                fused_rgb_lr_np = fused_rgb_lr.detach().cpu().numpy().squeeze(0)
                fused_rgb_lr_np = np.transpose(fused_rgb_lr_np, (1, 2, 0))
                fused_rgb_lr_np = (fused_rgb_lr_np * 255.0).clip(0, 255).astype('uint8')
                inter_path = os.path.join(out_path, 'intermediate', f"{os.path.splitext(img_name)[0]}_fused_lr.png")
                imsave(inter_path, fused_rgb_lr_np)

                # Save high-resolution output
                fused_hr_np = fused_rgb_hr.detach().cpu().numpy().squeeze(0)
                fused_hr_np = np.transpose(fused_hr_np, (1, 2, 0))
                fused_hr_np = (fused_hr_np * 255.0).clip(0, 255).astype('uint8')
                output_filename = os.path.join(out_path, f"{os.path.splitext(img_name)[0]}_fused_hr.png")
                imsave(output_filename, fused_hr_np)
                logger.info(f"Processed {img_name}, saved to {output_filename}")

            except Exception as e:
                logger.error(f"Failed to process {img_name}: {e}")
                continue


if __name__ == "__main__":
    # Paths to directories
    ckpt_lefuse = r"D:\paper_implimentation\project_root\Hybrid\weights\L2025.pth"
    ckpt_unet   = r"D:\paper_implimentation\project_root\Hybrid\weights\sr_unet_aug_enhanced.pth"
    vi_path     = r"D:\paper_implimentation\project_root\Hybrid\data\TEST_data\vi"
    ir_path     = r"D:\paper_implimentation\project_root\Hybrid\data\TEST_data\ir"
    out_path    = r"D:\paper_implimentation\project_root\Hybrid\output"

    test(ckpt_lefuse, ckpt_unet, vi_path, ir_path, out_path)
