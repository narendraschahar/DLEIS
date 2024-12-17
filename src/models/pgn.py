# src/models/pgn.py

import torch
import torch.nn as nn
from .base_model import BaseSteganoModel

class ProgressiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class PGN(BaseSteganoModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "PGN"
        self.num_stages = config['num_stages']
        self.base_channels = config['base_channels']
        
        # Initial feature extraction
        self.init_conv = nn.Conv2d(3, self.base_channels, 3, padding=1)
        
        # Progressive stages
        self.stages = nn.ModuleList()
        self.refinement = nn.ModuleList()
        
        for _ in range(self.num_stages):
            stage = nn.Sequential(
                ProgressiveBlock(self.base_channels, self.base_channels),
                ProgressiveBlock(self.base_channels, self.base_channels)
            )
            self.stages.append(stage)
            
            refine = nn.Conv2d(self.base_channels, 3, 3, padding=1)
            self.refinement.append(refine)

        # Message encoding/decoding networks
        self.encode_net = nn.Sequential(
            nn.Conv2d(3 + self.base_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.decode_net = nn.Sequential(
            nn.Conv2d(3, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, cover_image, secret_message):
        # Initial features
        feat = self.init_conv(cover_image)
        
        # Progressive refinement
        for stage, refine in zip(self.stages, self.refinement):
            feat = stage(feat)
            refined = refine(feat)
            
            if self.training:
                self.intermediate_results.append(refined)
        
        # Combine features with secret message
        combined = torch.cat([feat, secret_message], dim=1)
        stego = self.encode_net(combined)
        
        return stego

    def decode(self, stego_image):
        return self.decode_net(stego_image)