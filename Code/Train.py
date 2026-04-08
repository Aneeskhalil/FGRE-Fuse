import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func
import warnings
warnings.filterwarnings('ignore')

# ==================== ENHANCED RESIDUAL DENSE BLOCK WITH MULTI-SCALE FEATURES ====================
class MultiScaleResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_channels=32):
        super().__init__()
        # Multi-scale feature extraction
        self.conv3x3 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, growth_channels, 5, padding=2)
        self.conv7x7 = nn.Conv2d(channels, growth_channels, 7, padding=3)
        
        # Feature fusion
        self.fusion = nn.Conv2d(growth_channels * 3, channels, 1, padding=0)
        
        # Dense connections within the block
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_channels, channels, 3, padding=1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # Multi-scale feature extraction (parallel)
        ms_feat1 = self.lrelu(self.conv3x3(x))
        ms_feat2 = self.lrelu(self.conv5x5(x))
        ms_feat3 = self.lrelu(self.conv7x7(x))
        
        # Fuse multi-scale features
        ms_fused = self.fusion(torch.cat([ms_feat1, ms_feat2, ms_feat3], dim=1))
        
        # Dense connections (sequential)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        
        # Combine multi-scale and dense features
        return x + x4 * 0.2 + ms_fused * 0.1

# ==================== ENHANCED CHANNEL ATTENTION ====================
class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        y = self.fc(torch.cat([avg_out, max_out], dim=1)).view(b, c, 1, 1)
        
        return x * (1 + y)

# ==================== SPATIAL ATTENTION MODULE ====================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        attention = self.sigmoid(y)
        
        return x * attention

# ==================== MULTI-SCALE ATTENTION AGGREGATION MODULE ====================
class MultiScaleAttentionAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2)
        self.conv7x7 = nn.Conv2d(channels, channels, 7, padding=3)
        
        self.channel_att = EnhancedChannelAttention(channels * 3)
        self.spatial_att = SpatialAttention()
        
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        
    def forward(self, x):
        # Extract multi-scale features
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        # Concatenate and apply attention
        feats = torch.cat([feat3, feat5, feat7], dim=1)
        feats = self.channel_att(feats)
        feats = self.spatial_att(feats)
        
        # Fuse features
        return self.fusion(feats)

# ==================== ENHANCED REVERSED U-BLOCK ====================
class EnhancedReversedUBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Multi-scale processing
        self.ms_attention = MultiScaleAttentionAggregation(channels)
        self.ms_rdb = MultiScaleResidualDenseBlock(channels)
        
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=scale_factor, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.skip = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        x_up = self.up(x)
        
        # Apply multi-scale attention and processing
        x_att = self.ms_attention(x_up)
        x_proc = self.ms_rdb(x_att)
        
        x_down = self.down(x_proc)
        
        return x_down + self.skip(x)

# ==================== ENHANCED ESR-U-NET WITH DRA-NET CONCEPTS ====================
class Enhanced_ESR_Unet_DRA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=4, num_rub=3):
        super().__init__()
        self.scale = scale
        
        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EnhancedChannelAttention(64)
        )
        
        # Progressive feature enhancement blocks
        self.rub_blocks = nn.ModuleList()
        for i in range(num_rub):
            self.rub_blocks.append(nn.Sequential(
                EnhancedReversedUBlock(64),
                EnhancedChannelAttention(64),
                MultiScaleResidualDenseBlock(64)
            ))
        
        # Multi-scale feature aggregation
        self.ms_aggregation = MultiScaleAttentionAggregation(64)
        
        # Progressive upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            MultiScaleResidualDenseBlock(64),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            MultiScaleResidualDenseBlock(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        # Store bicubic upsampled version
        bicubic_up = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        
        # Extract features
        features = self.head(x)
        
        # Process through progressive blocks
        block_features = [features]
        for rub in self.rub_blocks:
            features = rub(features)
            block_features.append(features)
        
        # Aggregate multi-scale features
        if len(block_features) > 1:
            # Use weighted combination of features from different blocks
            aggregated = block_features[0]
            for i, feat in enumerate(block_features[1:]):
                weight = 0.5 / (i + 1)  # Decreasing weight for later blocks
                aggregated = aggregated + feat * weight
            features = aggregated
        
        # Apply multi-scale attention aggregation
        features = self.ms_aggregation(features)
        
        # Upsample
        features = self.upsample(features)
        
        # Final reconstruction
        out = self.reconstruction(features)
        
        # Global residual connection
        out = out + bicubic_up
        
        return torch.clamp(out, 0, 1)

# ==================== DATASET CLASSES ====================
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        
        # Get matching filenames
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure matching
        assert len(self.lr_files) == len(self.hr_files), "Mismatched number of LR and HR images"
        assert all(l == h for l, h in zip(self.lr_files, self.hr_files)), "Filenames don't match"
        
        print(f"Loaded {len(self.lr_files)} paired images")
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.lr_files)
    
    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        
        # Load images
        lr_img = plt.imread(lr_path)
        hr_img = plt.imread(hr_path)
        
        # Convert to tensor and normalize to [0, 1]
        lr_tensor = self.to_tensor(lr_img).float()
        hr_tensor = self.to_tensor(hr_img).float()
        
        return lr_tensor, hr_tensor

# ==================== AUGMENTATION ====================
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, crop_size=128):
        self.dataset = dataset
        self.crop_size = crop_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        lr, hr = self.dataset[idx]
        
        # Random crop
        _, h, w = lr.shape
        if h > self.crop_size and w > self.crop_size:
            top = torch.randint(0, h - self.crop_size, (1,)).item()
            left = torch.randint(0, w - self.crop_size, (1,)).item()
            
            lr_crop = lr[:, top:top+self.crop_size, left:left+self.crop_size]
            hr_crop = hr[:, top*4:top*4+self.crop_size*4, left*4:left*4+self.crop_size*4]
        else:
            lr_crop = lr
            hr_crop = hr
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [2])
            hr_crop = torch.flip(hr_crop, [2])
        
        # Random rotation
        if torch.rand(1) > 0.5:
            lr_crop = torch.rot90(lr_crop, 1, [1, 2])
            hr_crop = torch.rot90(hr_crop, 1, [1, 2])
        
        return lr_crop, hr_crop

# ==================== ENHANCED LOSS FUNCTION ====================
class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg19
        
        # Load VGG19
        try:
            vgg = vgg19(pretrained=True).features[:16].eval()
        except:
            vgg = vgg19(weights=True).features[:16].eval()
        
        # Move VGG to specified device
        vgg = vgg.to(device)
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()
        
        # Store device for mean/std tensors
        self.device = device
        
        # Create mean and std tensors on the correct device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def forward(self, sr, hr):
        # Normalize for VGG
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        
        sr_features = self.vgg(sr_norm)
        hr_features = self.vgg(hr_norm)
        
        return self.l1_loss(sr_features, hr_features)

def rgb_to_ycbcr(x):
    """Convert RGB to YCbCr"""
    y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    cb = 0.5 - 0.168736 * x[:, 0] - 0.331264 * x[:, 1] + 0.5 * x[:, 2]
    cr = 0.5 + 0.5 * x[:, 0] - 0.418688 * x[:, 1] - 0.081312 * x[:, 2]
    return torch.stack([y, cb, cr], dim=1)

class EnhancedCombinedLoss(nn.Module):
    def __init__(self, lambda_p=0.1, lambda_c=0.05, lambda_g=0.05, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.lambda_g = lambda_g
        self.device = device
        
        # Sobel filters for gradient calculation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3).to(device)
    
    def gradient_loss(self, sr, hr):
        # Convert to grayscale for gradient calculation
        sr_gray = 0.299 * sr[:, 0] + 0.587 * sr[:, 1] + 0.114 * sr[:, 2]
        hr_gray = 0.299 * hr[:, 0] + 0.587 * hr[:, 1] + 0.114 * hr[:, 2]
        
        # Add channel dimension
        sr_gray = sr_gray.unsqueeze(1)
        hr_gray = hr_gray.unsqueeze(1)
        
        # Calculate gradients
        grad_sr_x = F.conv2d(sr_gray, self.sobel_x, padding=1)
        grad_sr_y = F.conv2d(sr_gray, self.sobel_y, padding=1)
        grad_sr = torch.sqrt(grad_sr_x**2 + grad_sr_y**2 + 1e-8)
        
        grad_hr_x = F.conv2d(hr_gray, self.sobel_x, padding=1)
        grad_hr_y = F.conv2d(hr_gray, self.sobel_y, padding=1)
        grad_hr = torch.sqrt(grad_hr_x**2 + grad_hr_y**2 + 1e-8)
        
        return self.l1_loss(grad_sr, grad_hr)
    
    def forward(self, sr, hr, vgg_loss=None):
        # Pixel loss
        l_pixel = self.l1_loss(sr, hr)
        
        # Gradient loss (for edge preservation)
        l_gradient = self.gradient_loss(sr, hr)
        
        # Color consistency loss
        sr_ycbcr = rgb_to_ycbcr(sr)
        hr_ycbcr = rgb_to_ycbcr(hr)
        l_color = self.l1_loss(sr_ycbcr[:, 1:], hr_ycbcr[:, 1:])
        
        # Perceptual loss if VGG is provided
        l_perceptual = 0
        if vgg_loss is not None:
            l_perceptual = vgg_loss(sr, hr)
        
        return (l_pixel + 
                self.lambda_p * l_perceptual + 
                self.lambda_c * l_color + 
                self.lambda_g * l_gradient)

# ==================== METRICS ====================
def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(sr, hr):
    sr_np = sr.detach().cpu().numpy().transpose(0, 2, 3, 1)
    hr_np = hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
    ssim_sum = 0
    for i in range(sr_np.shape[0]):
        ssim_sum += ssim_func(sr_np[i], hr_np[i], channel_axis=-1, data_range=1.0)
    return ssim_sum / sr_np.shape[0]

# ==================== COMPLETE TRAINING FUNCTION ====================
# ==================== COMPLETE TRAINING FUNCTION ====================
def train_enhanced_model_dra():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
    "lr_dir": "D:/paper_implimentation/project_root/train_data50/LR_fused",
    "hr_dir": "D:/paper_implimentation/project_root/train_data50/LR",
    "model_path": "D:/paper_implimentation/project_root/ESR_Unet_DRA.pth",
    "checkpoint_dir": "D:/paper_implimentation/project_root/checkpoints_dra",
    "output_dir": "D:/paper_implimentation/project_root/training_output_dra",
    "plot_dir": "D:/paper_implimentation/project_root/training_plots_dra",
    "epochs": 50,
    "batch_size": 1,  # Keep as is due to memory
    "learning_rate": 1e-4,
    "crop_size": 64,
    "val_split": 0.2,
    "patience": 10,
    "lambda_p": 0.05,
    "lambda_c": 0.1,
    "lambda_g": 0.02,
    "save_every": 5,
    "plot_every": 5,
    "use_mixed_precision": True,
    "gradient_clip": 0.5,
    "weight_decay": 0,
}
    
    # Set gradient accumulation steps based on batch size
    if config["batch_size"] == 1:
        config["gradient_accumulation_steps"] = 4
    else:
        config["gradient_accumulation_steps"] = 2
    
    # Create directories
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["plot_dir"], exist_ok=True)
    
    # Enable mixed precision if available
    if config["use_mixed_precision"] and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("✓ Mixed precision training enabled")
    else:
        scaler = None
        config["use_mixed_precision"] = False
        print("✗ Mixed precision not available")
    
    # Create dataset
    try:
        full_dataset = PairedDataset(config["lr_dir"], config["hr_dir"])
        
        # Split train/val
        val_size = int(config["val_split"] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Add augmentation with enhanced settings
        class EnhancedAugmentedDataset(AugmentedDataset):
            def __getitem__(self, idx):
                lr, hr = self.dataset[idx]
                
                # Random crop
                _, h, w = lr.shape
                if h > self.crop_size and w > self.crop_size:
                    top = torch.randint(0, h - self.crop_size, (1,)).item()
                    left = torch.randint(0, w - self.crop_size, (1,)).item()
                    
                    lr_crop = lr[:, top:top+self.crop_size, left:left+self.crop_size]
                    hr_crop = hr[:, top*4:top*4+self.crop_size*4, left*4:left*4+self.crop_size*4]
                else:
                    lr_crop = lr
                    hr_crop = hr
                
                # Random horizontal flip
                if torch.rand(1) > 0.5:
                    lr_crop = torch.flip(lr_crop, [2])
                    hr_crop = torch.flip(hr_crop, [2])
                
                # Random rotation (0, 90, 180, 270 degrees)
                rot = torch.randint(0, 4, (1,)).item()
                if rot > 0:
                    lr_crop = torch.rot90(lr_crop, rot, [1, 2])
                    hr_crop = torch.rot90(hr_crop, rot, [1, 2])
                
                # Random brightness adjustment
                if torch.rand(1) > 0.7:
                    brightness = 0.9 + torch.rand(1).item() * 0.2
                    lr_crop = lr_crop * brightness
                    hr_crop = hr_crop * brightness
                    lr_crop = torch.clamp(lr_crop, 0, 1)
                    hr_crop = torch.clamp(hr_crop, 0, 1)
                
                return lr_crop, hr_crop
        
        train_dataset = EnhancedAugmentedDataset(train_dataset, config["crop_size"])
        val_dataset = EnhancedAugmentedDataset(val_dataset, config["crop_size"])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True,
            pin_memory=True, num_workers=0, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False,
            pin_memory=True, num_workers=0
        )
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"Crop size: {config['crop_size']}, Batch size: {config['batch_size']}")
        print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Initialize enhanced model
    model = Enhanced_ESR_Unet_DRA().to(device)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load checkpoint if exists
    if os.path.exists(config["model_path"]):
        try:
            model.load_state_dict(torch.load(config["model_path"], map_location=device))
            print(f"Loaded model from {config['model_path']}")
        except:
            print("Could not load model, starting from scratch")
    
    # Initialize loss functions
    vgg_loss = VGGLoss(device=device)
    combined_loss = EnhancedCombinedLoss(
        lambda_p=config["lambda_p"],
        lambda_c=config["lambda_c"],
        lambda_g=config["lambda_g"],
        device=device
    )
    
    # Optimizer with weight decay - CONSTANT LEARNING RATE
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"],
                                  betas=(0.9, 0.999))
    
    # Remove warmup scheduler - Keep constant learning rate
    # No scheduler needed for constant LR
    
    # Training statistics
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    
    # Training loop with gradient accumulation
    best_val_loss = float('inf')
    best_val_psnr = 0
    no_improve = 0
    accumulation_steps = config["gradient_accumulation_steps"]
    
    print(f"\n{'='*60}")
    print(f"Starting Training with DRA-Enhanced ESR-U-Net")
    print(f"Batch size: {config['batch_size']}, Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {config['batch_size'] * accumulation_steps}")
    print(f"Learning rate: {config['learning_rate']} (CONSTANT)")
    print(f"{'='*60}\n")
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for batch_idx, (lr, hr) in enumerate(pbar):
            lr, hr = lr.to(device), hr.to(device)
            
            if config["use_mixed_precision"]:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    sr = model(lr)
                    loss = combined_loss(sr, hr, vgg_loss) / accumulation_steps
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training
                sr = model(lr)
                loss = combined_loss(sr, hr, vgg_loss) / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({"Loss": loss.item() * accumulation_steps})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for lr_val, hr_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=False):
                lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                
                if config["use_mixed_precision"]:
                    with torch.cuda.amp.autocast():
                        sr_val = model(lr_val)
                        val_loss += combined_loss(sr_val, hr_val, vgg_loss).item()
                else:
                    sr_val = model(lr_val)
                    val_loss += combined_loss(sr_val, hr_val, vgg_loss).item()
                
                val_psnr += calculate_psnr(sr_val, hr_val).item()
                val_ssim += calculate_ssim(sr_val, hr_val)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)
        avg_val_ssim = val_ssim / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        val_ssims.append(avg_val_ssim)
        
        # No scheduler step for constant LR
        
        # Print progress
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']} Results:")
        print(f"{'='*60}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Val PSNR:   {avg_val_psnr:.4f} dB")
        print(f"  Val SSIM:   {avg_val_ssim:.6f}")
        print(f"  Learning Rate: {config['learning_rate']:.2e} (CONSTANT)")
        print(f"{'='*60}")
        
        # Early stopping based on PSNR
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config["model_path"])
            print(f"✓ Saved BEST model (PSNR: {best_val_psnr:.4f} dB, Loss: {best_val_loss:.6f})")
            no_improve = 0
            
            # Also save checkpoint
            checkpoint_path = os.path.join(config["checkpoint_dir"], f"best_epoch_{epoch+1:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"⚠️ No improvement for {no_improve} epochs. Early stopping.")
                break
        
        # Save visualization every few epochs
        if (epoch + 1) % config["plot_every"] == 0 or epoch == 0:
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(val_loader))
                sample_lr = sample_lr[:1].to(device)
                sample_hr = sample_hr[:1].to(device)
                sample_sr = model(sample_lr)
                
                # Convert to numpy
                lr_np = sample_lr[0].cpu().permute(1, 2, 0).numpy()
                hr_np = sample_hr[0].cpu().permute(1, 2, 0).numpy()
                sr_np = sample_sr[0].cpu().permute(1, 2, 0).numpy()
                
                # Simple visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(np.clip(lr_np, 0, 1))
                axes[0].set_title('LR Input')
                axes[0].axis('off')
                
                axes[1].imshow(np.clip(hr_np, 0, 1))
                axes[1].set_title('HR Target')
                axes[1].axis('off')
                
                axes[2].imshow(np.clip(sr_np, 0, 1))
                axes[2].set_title(f'SR Output (PSNR: {calculate_psnr(sample_sr, sample_hr).item():.2f} dB)')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(config["output_dir"], f"epoch_{epoch+1:03d}.png"), 
                          dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved visualization for epoch {epoch+1}")
        
        # Save training plots
        if (epoch + 1) % config["plot_every"] == 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
            axes[0, 0].plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(range(1, len(val_psnrs) + 1), val_psnrs, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PSNR (dB)')
            axes[0, 1].set_title('PSNR Evolution')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(range(1, len(val_ssims) + 1), val_ssims, 'm-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('SSIM')
            axes[1, 0].set_title('SSIM Evolution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Combined metrics
            epochs_range = range(1, len(val_psnrs) + 1)
            ax2 = axes[1, 1].twinx()
            line1, = axes[1, 1].plot(epochs_range, val_psnrs, 'g-', label='PSNR')
            line2, = ax2.plot(epochs_range, val_ssims, 'm-', label='SSIM')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('PSNR (dB)', color='g')
            ax2.set_ylabel('SSIM', color='m')
            axes[1, 1].set_title('PSNR and SSIM')
            axes[1, 1].legend(handles=[line1, line2], loc='upper left')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Training Progress (Best PSNR: {best_val_psnr:.2f} dB)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(config["plot_dir"], f"progress_epoch_{epoch+1:03d}.png"), 
                      dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved training plots for epoch {epoch+1}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation PSNR: {best_val_psnr:.4f} dB")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Final Validation PSNR: {avg_val_psnr:.4f} dB")
    print(f"Final Validation SSIM: {avg_val_ssim:.6f}")
    print(f"Model saved to: {config['model_path']}")
    print(f"{'='*60}")
    
    # Save final training statistics
    stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnrs': val_psnrs,
        'val_ssims': val_ssims,
        'best_psnr': best_val_psnr,
        'best_loss': best_val_loss,
    }
    
    stats_path = os.path.join(config["plot_dir"], "training_statistics.npy")
    np.save(stats_path, stats)
    print(f"Training statistics saved to: {stats_path}")

# ==================== MAIN ====================
if __name__ == "__main__":
    # Set CUDA memory fraction if needed
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Train the enhanced DRA model
    train_enhanced_model_dra()