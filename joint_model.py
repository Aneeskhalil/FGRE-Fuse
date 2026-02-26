import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from einops import rearrange
import numbers
import warnings
import gc
import platform
warnings.filterwarnings('ignore')

# ==================== FUSION MODULE ====================
def clamp(value, min=0., max=1.0):
    return torch.clamp(value, min=min, max=max)

def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1, :, :]
    G = rgb_image[:, 1:2, :, :]
    B = rgb_image[:, 2:3, :, :]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cr, Cb

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cb, Cr], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim_in, dim_out, mlp_ratio=3):
        super().__init__()
        self.dwconv = ConvBN(dim_in, dim_in, 7, 1, (7 - 1) // 2, groups=dim_in, with_bn=True)
        self.f1 = ConvBN(dim_in, mlp_ratio * dim_in, 1, with_bn=False)
        self.f2 = ConvBN(dim_in, mlp_ratio * dim_in, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim_in, dim_out, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim_out, dim_out, 7, 1, (7 - 1) // 2, groups=dim_out, with_bn=False)
        self.act = nn.ReLU6()
        self.aux_g = ConvBN(dim_in, dim_out, 1, with_bn=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x) 
        x1, x2 = self.f1(x), self.f2(x)  
        x = self.act(x1) * x2  
        x = self.dwconv2(self.g(x)) 
        input = self.aux_g(input)
        x = input + x
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DualBranch_module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.num_heads = int(dim / 8)
        self.Global = TransformerBlock(dim=dim, num_heads=self.num_heads, ffn_expansion_factor=2, bias=False,
                                      LayerNorm_type='WithBias')
        self.Local = Block(dim, dim)
        self.se_conv1 = nn.Conv2d(int(dim * 2), int(dim * 2), kernel_size=3, stride=1, padding=1)
        self.se_conv2 = nn.Conv2d(int(dim * 2), int(dim * 2), kernel_size=1, stride=1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.se_conv3 = nn.Conv2d(int(dim * 2), dim, kernel_size=1, stride=1)
        self.se_conv4 = nn.Conv2d(int(dim * 2), dim, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        x_g = self.Global(x)
        x_l = self.Local(x)
        x = torch.cat([x_g, x_l], dim=1)
        x = self.se_conv1(x)
        x = self.se_conv2(x)
        x = self.avg(x)
        return res + x_l * self.sigmoid(self.se_conv4(x)) + x_g * self.sigmoid(self.se_conv3(x))

class up_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        
    def forward(self, x, tgt):
        x = self.up(x)
        if x.shape != tgt.shape:
            x = F.interpolate(x, size=(tgt.shape[2], tgt.shape[3]), mode='bilinear')
        return x

class Unet_fuser(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.emb1 = nn.Conv2d(2, dim, kernel_size=3, stride=1, padding=1)
        self.blk1 = DualBranch_module(dim)
        self.emb2 = nn.Conv2d(dim, int(dim * 2), kernel_size=3, stride=2, padding=1)
        self.blk2 = DualBranch_module(int(dim * 2))
        self.emb3 = nn.Conv2d(int(dim * 2), int(dim * 3), kernel_size=3, stride=2, padding=1)
        self.blk3 = DualBranch_module(int(dim * 3))
        self.emb4 = nn.Conv2d(int(dim * 3), int(dim * 4), kernel_size=3, stride=2, padding=1)
        self.blk4 = DualBranch_module(int(dim * 4))
        self.up4 = up_block(int(dim * 4), int(dim * 3))
        self.conv_3_1 = nn.Conv2d(int(dim * 6), int(dim * 3), kernel_size=3, stride=1, padding=1)
        self.upblk1 = DualBranch_module(int(dim * 3))
        self.up3 = up_block(int(dim * 3), int(dim * 2))
        self.conv_3_2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=3, stride=1, padding=1)
        self.upblk2 = DualBranch_module(int(dim * 2))
        self.up2 = up_block(int(dim * 2), dim)
        self.conv_3_3 = nn.Conv2d(int(dim * 2), dim, kernel_size=3, stride=1, padding=1)
        self.upblk3 = DualBranch_module(dim)
        self.upblk4 = nn.Sequential(
            DualBranch_module(dim),
            DualBranch_module(dim)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(dim, int(dim * 0.5), kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(int(dim * 0.5), 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, vi, ir):
        x = torch.cat([vi, ir], dim=1)
        x1 = self.blk1(self.emb1(x))
        x2 = self.blk2(self.emb2(x1))
        x3 = self.blk3(self.emb3(x2))
        x4 = self.blk4(self.emb4(x3))
        out = self.upblk1(self.conv_3_1(torch.cat([self.up4(x4, x3), x3], dim=1)))
        out = self.upblk2(self.conv_3_2(torch.cat([self.up3(out, x2), x2], dim=1)))
        out = self.upblk3(self.conv_3_3(torch.cat([self.up2(out, x1), x1], dim=1)))
        out = self.dec(self.upblk4(out))
        return out

class LEFuse(nn.Module):
    def __init__(self):
        super(LEFuse, self).__init__()
        self.fuser = Unet_fuser(dim=32)

    def forward(self, vi, ir):
        return self.fuser(vi, ir)

# ==================== ENHANCED ESR-U-NET  ====================
class MultiScaleResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_channels=32):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, growth_channels, 5, padding=2)
        self.conv7x7 = nn.Conv2d(channels, growth_channels, 7, padding=3)
        
        self.fusion = nn.Conv2d(growth_channels * 3, channels, 1, padding=0)
        
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_channels, channels, 3, padding=1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        ms_feat1 = self.lrelu(self.conv3x3(x))
        ms_feat2 = self.lrelu(self.conv5x5(x))
        ms_feat3 = self.lrelu(self.conv7x7(x))
        
        ms_fused = self.fusion(torch.cat([ms_feat1, ms_feat2, ms_feat3], dim=1))
        
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        
        return x + x4 * 0.2 + ms_fused * 0.1

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
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)
        
        feats = torch.cat([feat3, feat5, feat7], dim=1)
        feats = self.channel_att(feats)
        feats = self.spatial_att(feats)
        
        return self.fusion(feats)

class EnhancedReversedUBlock(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
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
        
        x_att = self.ms_attention(x_up)
        x_proc = self.ms_rdb(x_att)
        
        x_down = self.down(x_proc)
        
        return x_down + self.skip(x)

class Enhanced_ESR_Unet_DRA(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=4, num_rub=3):
        super().__init__()
        self.scale = scale
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EnhancedChannelAttention(64)
        )
        
        self.rub_blocks = nn.ModuleList()
        for i in range(num_rub):
            self.rub_blocks.append(nn.Sequential(
                EnhancedReversedUBlock(64),
                EnhancedChannelAttention(64),
                MultiScaleResidualDenseBlock(64)
            ))
        
        self.ms_aggregation = MultiScaleAttentionAggregation(64)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            MultiScaleResidualDenseBlock(64),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            MultiScaleResidualDenseBlock(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        bicubic_up = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        
        features = self.head(x)
        
        block_features = [features]
        for rub in self.rub_blocks:
            features = rub(features)
            block_features.append(features)
        
        if len(block_features) > 1:
            aggregated = block_features[0]
            for i, feat in enumerate(block_features[1:]):
                weight = 0.5 / (i + 1)
                aggregated = aggregated + feat * weight
            features = aggregated
        
        features = self.ms_aggregation(features)
        
        features = self.upsample(features)
        
        out = self.reconstruction(features)
        
        out = out + bicubic_up
        
        return torch.clamp(out, 0, 1)

# ==================== SIMPLIFIED MEMORY-EFFICIENT PIPELINE ====================
class SimpleFusionSRPipeline:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.config = config
        self.to_tensor = transforms.ToTensor()
        
        # Create output directories
        os.makedirs(config["fused_output_dir"], exist_ok=True)
        os.makedirs(config["sr_output_dir"], exist_ok=True)
        
        # Load models once
        self.load_models()
        
    def load_models(self):
        """Load both models at startup"""
        print("\n" + "="*60)
        print("Loading Models")
        print("="*60)
        
        # Load Fusion Model
        self.fusion_model = LEFuse().to(self.device)
        if os.path.exists(self.config["fusion_model_path"]):
            try:
                checkpoint = torch.load(self.config["fusion_model_path"], map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.fusion_model.load_state_dict(checkpoint['state_dict'])
                        print(f"✓ Loaded fusion model from {self.config['fusion_model_path']}")
                    elif 'model' in checkpoint:
                        self.fusion_model.load_state_dict(checkpoint['model'])
                        print(f"✓ Loaded fusion model from {self.config['fusion_model_path']}")
                    else:
                        self.fusion_model.load_state_dict(checkpoint)
                        print(f"✓ Loaded fusion model from {self.config['fusion_model_path']}")
                else:
                    self.fusion_model.load_state_dict(checkpoint)
                    print(f"✓ Loaded fusion model from {self.config['fusion_model_path']}")
            except Exception as e:
                print(f"✗ Error loading fusion model: {e}")
                print("Using randomly initialized fusion model")
        else:
            print(f"✗ Fusion model not found at {self.config['fusion_model_path']}")
            print("Using randomly initialized fusion model")
        
        # Load SR Model
        self.sr_model = Enhanced_ESR_Unet_DRA().to(self.device)
        if os.path.exists(self.config["sr_model_path"]):
            try:
                checkpoint = torch.load(self.config["sr_model_path"], map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        self.sr_model.load_state_dict(checkpoint['state_dict'])
                        print(f"✓ Loaded SR model from {self.config['sr_model_path']}")
                    elif 'model' in checkpoint:
                        self.sr_model.load_state_dict(checkpoint['model'])
                        print(f"✓ Loaded SR model from {self.config['sr_model_path']}")
                    else:
                        self.sr_model.load_state_dict(checkpoint)
                        print(f"✓ Loaded SR model from {self.config['sr_model_path']}")
                else:
                    self.sr_model.load_state_dict(checkpoint)
                    print(f"✓ Loaded SR model from {self.config['sr_model_path']}")
            except Exception as e:
                print(f"✗ Error loading SR model: {e}")
                print("Using randomly initialized SR model")
        else:
            print(f"✗ SR model not found at {self.config['sr_model_path']}")
            print("Using randomly initialized SR model")
        
        # Set models to eval mode
        self.fusion_model.eval()
        self.sr_model.eval()
        
        print("✓ Models loaded successfully")
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def load_image_pair(self, visible_path, infrared_path):
        """Load and preprocess a visible-infrared image pair"""
        # Load images
        visible_img = plt.imread(visible_path)
        infrared_img = plt.imread(infrared_path)
        
        # Handle infrared format
        if len(infrared_img.shape) == 3:
            infrared_img = infrared_img.mean(axis=2)  # Convert to grayscale
        
        # Add channel dimension if needed
        if len(infrared_img.shape) == 2:
            infrared_img = np.expand_dims(infrared_img, axis=2)
        
        # Normalize images to [0, 1]
        if visible_img.max() > 1.0:
            visible_img = visible_img.astype(np.float32) / 255.0
        
        if infrared_img.max() > 1.0:
            infrared_img = infrared_img.astype(np.float32) / 255.0
        
        # Convert to tensors
        visible_tensor = self.to_tensor(visible_img).float()
        infrared_tensor = torch.from_numpy(infrared_img.transpose(2, 0, 1)).float()
        
        # Ensure infrared is single channel
        if infrared_tensor.shape[0] != 1:
            infrared_tensor = infrared_tensor[:1, :, :]
        
        return visible_tensor, infrared_tensor
    
    def process_single_pair(self, visible_path, infrared_path, output_prefix, idx):
        """Process a single visible-infrared pair"""
        print(f"\n[{idx}] Processing: {os.path.basename(visible_path)}")
        
        try:
            # Step 1: Load images
            visible_tensor, infrared_tensor = self.load_image_pair(visible_path, infrared_path)
            
            # Move to device
            visible_tensor = visible_tensor.unsqueeze(0).to(self.device)
            infrared_tensor = infrared_tensor.unsqueeze(0).to(self.device)
            
            # Step 2: Fusion
            print("  Step 1: Fusing images...")
            with torch.no_grad():
                # Convert RGB to YCbCr
                y, cr, cb = RGB2YCrCb(visible_tensor)
                
                # Fuse Y channel with infrared
                fused_y = self.fusion_model(y, infrared_tensor)
                
                # Convert back to RGB
                fused_image = YCbCr2RGB(fused_y, cr, cb)
            
            # Save fused image
            fused_path = os.path.join(self.config["fused_output_dir"], f"{output_prefix}_fused.png")
            self.save_image(fused_image, fused_path)
            print(f"     Saved fused image to {fused_path}")
            
            # Clear memory after fusion
            del visible_tensor, infrared_tensor, y, cr, cb, fused_y
            self.clear_memory()
            
            # Step 3: Super-resolution
            print("  Step 2: Applying super-resolution...")
            with torch.no_grad():
                sr_image = self.sr_model(fused_image)
            
            # Save super-resolved image
            sr_path = os.path.join(self.config["sr_output_dir"], f"{output_prefix}_sr.png")
            self.save_image(sr_image, sr_path)
            print(f"     Saved super-resolved image to {sr_path}")
            
            # Clear memory after SR
            del fused_image, sr_image
            self.clear_memory()
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing image {idx}: {e}")
            # Clear memory on error
            self.clear_memory()
            return False
    
    def save_image(self, tensor, save_path):
        """Save tensor as image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy
        if tensor.shape[0] == 1:  # Grayscale
            img_np = tensor.cpu().float().numpy().squeeze()
            plt.imsave(save_path, np.clip(img_np, 0, 1), cmap='gray')
        else:  # RGB
            img_np = tensor.cpu().float().numpy().transpose(1, 2, 0)
            plt.imsave(save_path, np.clip(img_np, 0, 1))
    
    def run_batch_processing(self):
        """Run the complete pipeline on all images"""
        print("\n" + "="*60)
        print("Fusion-SR Pipeline")
        print("="*60)
        
        # Get list of image files
        visible_files = sorted([f for f in os.listdir(self.config["visible_dir"]) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        infrared_files = sorted([f for f in os.listdir(self.config["infrared_dir"]) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        
        # Match by name (assume same filenames)
        visible_files.sort()
        infrared_files.sort()
        
        # Create pairs
        pairs = []
        for vis_file in visible_files:
            # Find corresponding infrared file
            base_name = os.path.splitext(vis_file)[0]
            ir_file = f"{base_name}.jpg"  # Assuming same extension
            
            # Check if infrared file exists
            ir_path = os.path.join(self.config["infrared_dir"], ir_file)
            if os.path.exists(ir_path):
                pairs.append((vis_file, ir_file))
            else:
                print(f"Warning: No matching infrared file for {vis_file}")
        
        print(f"Found {len(pairs)} image pairs to process")
        
        # Process each pair
        successful = 0
        pbar = tqdm(enumerate(pairs), total=len(pairs), desc="Processing")
        
        for idx, (vis_file, ir_file) in pbar:
            vis_path = os.path.join(self.config["visible_dir"], vis_file)
            ir_path = os.path.join(self.config["infrared_dir"], ir_file)
            
            # Create output prefix
            base_name = os.path.splitext(vis_file)[0]
            output_prefix = f"{idx+1:03d}_{base_name}"
            
            # Process the pair
            success = self.process_single_pair(vis_path, ir_path, output_prefix, idx+1)
            if success:
                successful += 1
            
            pbar.set_postfix({"Success": successful, "Current": vis_file})
        
        # Generate summary
        self.generate_summary(successful, len(pairs))
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print(f" Successfully processed: {successful}/{len(pairs)} image pairs")
        print(f" Fused images saved to: {self.config['fused_output_dir']}")
        print(f" Super-resolved images saved to: {self.config['sr_output_dir']}")
        print("="*60)
    
    def generate_summary(self, successful, total):
        """Generate a summary of processed images"""
        summary_path = os.path.join(self.config["fused_output_dir"], "processing_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Fusion-SR Pipeline Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total images to process: {total}\n")
            f.write(f"Successfully processed: {successful}\n")
            f.write(f"Failed: {total - successful}\n\n")
            f.write(f"Fusion model: {self.config.get('fusion_model_path', 'Not specified')}\n")
            f.write(f"SR model: {self.config.get('sr_model_path', 'Not specified')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Operating System: {platform.system()}\n")
            f.write("\nOutput directories:\n")
            f.write(f"  Fused images: {self.config['fused_output_dir']}\n")
            f.write(f"  Super-resolved images: {self.config['sr_output_dir']}\n")
            f.write("\n" + "="*60 + "\n")
        
        print(f"✓ Summary saved to {summary_path}")

# ==================== MAIN FUNCTION ====================
def run_fusion_sr_pipeline():
    """Main function to run the fusion-super-resolution pipeline"""
    config = {
        # Input directories
        "visible_dir": "",
        "infrared_dir": "",
        
        # Model paths
        "fusion_model_path": "",
        "sr_model_path": "",
        
        # Output directories
        "fused_output_dir": "",
        "sr_output_dir": "",
        
        # Processing parameters
        "sr_scale": 4,
    }
    
    # Initialize and run pipeline
    pipeline = SimpleFusionSRPipeline(config)
    pipeline.run_batch_processing()

# ==================== SINGLE IMAGE TEST ====================
def test_single_image():
    """Test a single image pair"""
    config = {
        # Model paths
        "fusion_model_path": "",
        "sr_model_path": "",
        
        # Output directories
        "fused_output_dir": "",
        "sr_output_dir": "",
        
        # Processing parameters
        "sr_scale": 4,
    }
    
    # Create output directories
    os.makedirs(config["fused_output_dir"], exist_ok=True)
    os.makedirs(config["sr_output_dir"], exist_ok=True)
    
    # Initialize pipeline
    pipeline = SimpleFusionSRPipeline(config)
    
    # Test single pair
    visible_path = "190001.jpg"
    infrared_path = "190001.jpg"  # Assuming same filename in infrared directory
    
    if os.path.exists(visible_path) and os.path.exists(infrared_path):
        print("\n" + "="*60)
        print("Testing Single Image Pair")
        print("="*60)
        
        success = pipeline.process_single_pair(
            visible_path, infrared_path, 
            output_prefix="single_test",
            idx=1
        )
        
        if success:
            print("\n Single image test complete!")
            print(f"  Fused image saved to: {config['fused_output_dir']}")
            print(f"  SR image saved to: {config['sr_output_dir']}")
        else:
            print("\n✗ Failed to process single image pair")
    else:
        print("✗ Image files not found!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fusion-SR Pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'single'],
                       help='Processing mode: full (all images), single (one pair)')
    
    args = parser.parse_args()
    
    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if args.mode == 'full':
        run_fusion_sr_pipeline()
    elif args.mode == 'single':
        test_single_image()
    else:
        run_fusion_sr_pipeline()