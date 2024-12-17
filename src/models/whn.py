# src/models/whn.py

import torch
import torch.nn as nn
import pywt
import numpy as np

class DWT(nn.Module):
    def __init__(self, wave_type='haar'):
        super().__init__()
        self.wave_type = wave_type

    def forward(self, x):
        B, C, H, W = x.shape
        x_np = x.detach().cpu().numpy()
        coeffs = []
        
        for b in range(B):
            c = []
            for ch in range(C):
                # Apply 2D DWT
                coeff = pywt.dwt2(x_np[b, ch], self.wave_type)
                c.append(coeff)
            coeffs.append(c)
            
        # Convert back to tensors
        ll = torch.from_numpy(
            np.stack([[c[0][0] for c in b] for b in coeffs])
        ).float().to(x.device)
        lh = torch.from_numpy(
            np.stack([[c[1][0] for c in b] for b in coeffs])
        ).float().to(x.device)
        hl = torch.from_numpy(
            np.stack([[c[1][1] for c in b] for b in coeffs])
        ).float().to(x.device)
        hh = torch.from_numpy(
            np.stack([[c[1][2] for c in b] for b in coeffs])
        ).float().to(x.device)
        
        return ll, lh, hl, hh

class WHN(BaseSteganoModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "WHN"
        self.num_blocks = config['num_blocks']
        self.base_channels = config['base_channels']
        
        # DWT modules
        self.dwt = DWT(config['wavelet_type'])
        self.idwt = nn.ConvTranspose2d(
            self.base_channels * 4,
            self.base_channels,
            4,
            stride=2,
            padding=1
        )
        
        # Processing blocks
        self.blocks = nn.ModuleList([
            WaveletBlock(self.base_channels)
            for _ in range(self.num_blocks)
        ])
        
        # Message processing
        self.message_proc = nn.Sequential(
            nn.Conv2d(3, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1)
        )
        
        # Output heads
        self.encode_head = nn.Sequential(
            nn.Conv2d(self.base_channels * 2, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, cover_image, secret_message):
        # Apply DWT
        ll, lh, hl, hh = self.dwt(cover_image)
        coeffs = torch.cat([ll, lh, hl, hh], dim=1)
        
        # Process through wavelet blocks
        feat = coeffs
        for block in self.blocks:
            feat = block(feat)
            
        # Process message
        msg_feat = self.message_proc(secret_message)
        
        # Combine and generate stego image
        combined = torch.cat([feat, msg_feat], dim=1)
        stego = self.encode_head(combined)
        
        return stego

    def decode(self, stego_image):
        # Apply DWT to stego image
        ll, lh, hl, hh = self.dwt(stego_image)
        coeffs = torch.cat([ll, lh, hl, hh], dim=1)
        
        # Process through wavelet blocks
        feat = coeffs
        for block in self.blocks:
            feat = block(feat)
            
        # Extract message
        return self.decode_head(feat)