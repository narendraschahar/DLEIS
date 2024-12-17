# src/models/eapt.py

import torch
import torch.nn as nn
import math
from typing import Tuple

class PyramidAttention(nn.Module):
    def __init__(self, channels: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        
        # Multi-scale processing
        self.downsample = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, stride=2**i, padding=1)
            for i in range(num_levels)
        ])
        
        # Attention at each level
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 8, 1),
                nn.Softmax(dim=1)
            )
            for _ in range(num_levels)
        ])
        
        # Feature fusion
        self.fusion = nn.Conv2d(channels * num_levels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        features = []
        
        # Process each pyramid level
        for i in range(self.num_levels):
            # Downsample
            level_feat = self.downsample[i](x)
            
            # Apply attention
            attn = self.attention[i](level_feat)
            level_feat = level_feat * attn
            
            # Upsample back to original size
            if i > 0:
                level_feat = F.interpolate(
                    level_feat,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            
            features.append(level_feat)
        
        # Fuse features
        return self.fusion(torch.cat(features, dim=1))

class EAPT(BaseSteganoModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "EAPT"
        self.embed_dim = config['embed_dim']
        self.num_pyramid_levels = config['num_pyramid_levels']
        
        # Initial feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, 3, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Pyramid attention blocks
        self.pyramid_blocks = nn.ModuleList([
            PyramidAttention(self.embed_dim, self.num_pyramid_levels)
            for _ in range(config['num_blocks'])
        ])
        
        # Message processing
        self.message_proc = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, 3, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            PyramidAttention(self.embed_dim, self.num_pyramid_levels)
        )
        
        # Output heads
        self.encode_head = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, self.embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(
        self, 
        cover_image: torch.Tensor, 
        secret_message: torch.Tensor
    ) -> torch.Tensor:
        # Extract initial features
        cover_feat = self.init_conv(cover_image)
        
        # Apply pyramid attention blocks
        for block in self.pyramid_blocks:
            cover_feat = cover_feat + block(cover_feat)
        
        # Process secret message
        msg_feat = self.message_proc(secret_message)
        
        # Combine features and generate stego image
        combined = torch.cat([cover_feat, msg_feat], dim=1)
        return self.encode_head(combined)

    def decode(self, stego_image: torch.Tensor) -> torch.Tensor:
        # Extract features
        feat = self.init_conv(stego_image)
        
        # Apply pyramid attention
        for block in self.pyramid_blocks:
            feat = feat + block(feat)
        
        # Generate recovered message
        return self.decode_head(feat)
