# src/models/dsa.py

class DualStreamBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.structure_stream = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
        self.texture_stream = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        self.fusion = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        struct = self.structure_stream(x)
        text = self.texture_stream(x)
        concat = torch.cat([struct, text], dim=1)
        out = self.fusion(concat)
        return out + x

class DSA(BaseSteganoModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "DSA"
        self.channels = config['channels']
        self.num_blocks = config['num_blocks']
        
        # Initial convolution
        self.init_conv = nn.Conv2d(3, self.channels, 3, padding=1)
        
        # Dual stream blocks
        self.blocks = nn.ModuleList([
            DualStreamBlock(self.channels) 
            for _ in range(self.num_blocks)
        ])
        
        # Message processing
        self.message_proc = nn.Sequential(
            nn.Conv2d(3, self.channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, 3, padding=1)
        )
        
        # Output heads
        self.encode_head = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, cover_image, secret_message):
        # Process cover image
        feat = self.init_conv(cover_image)
        for block in self.blocks:
            feat = block(feat)
            
        # Process message
        msg_feat = self.message_proc(secret_message)
        
        # Combine and generate stego image
        combined = torch.cat([feat, msg_feat], dim=1)
        stego = self.encode_head(combined)
        
        return stego

    def decode(self, stego_image):
        feat = self.init_conv(stego_image)
        for block in self.blocks:
            feat = block(feat)
        return self.decode_head(feat)