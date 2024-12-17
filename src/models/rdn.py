# src/models/rdn.py

import torch
import torch.nn as nn
from .base_model import BaseSteganoModel

class DenseBlock(nn.Module):
    def __init__(self, channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels + growth_rate * i, growth_rate, 3, padding=1),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class RDN(BaseSteganoModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "RDN"
        
        # Network parameters
        self.num_features = config['num_features']
        self.growth_rate = config['growth_rate']
        self.num_blocks = config['num_blocks']
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(3, self.num_features, 3, padding=1)
        
        # RDB blocks
        self.rdb_blocks = nn.ModuleList([
            DenseBlock(
                self.num_features,
                self.growth_rate,
                config['layers_per_block']
            ) for _ in range(self.num_blocks)
        ])
        
        # Global feature fusion
        self.global_fusion = nn.Conv2d(
            self.num_features * self.num_blocks,
            self.num_features,
            1
        )
        
        # Message embedding network
        self.embed_net = nn.Sequential(
            nn.Conv2d(self.num_features + 3, self.num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_features, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Message extraction network
        self.extract_net = nn.Sequential(
            nn.Conv2d(3, self.num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_features, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, cover_image, secret_message):
        # Initial features
        feat = self.conv_input(cover_image)
        
        # Store RDB outputs
        rdb_outs = []
        for rdb in self.rdb_blocks:
            feat = rdb(feat)
            rdb_outs.append(feat)
        
        # Global feature fusion
        concat_feat = torch.cat(rdb_outs, dim=1)
        fused_feat = self.global_fusion(concat_feat)
        
        # Combine features with secret message
        combined = torch.cat([fused_feat, secret_message], dim=1)
        
        # Generate stego image
        stego = self.embed_net(combined)
        return stego

    def decode(self, stego_image):
        # Extract secret message
        return self.extract_net(stego_image)