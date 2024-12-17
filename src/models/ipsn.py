# src/models/ipsn.py

class InterestPointDetector(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, 1, 1)
        
        # Harris corner response parameters
        self.k = 0.04
        self.sigma = 1.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute gradients
        dx = F.conv2d(x, self.sobel_x(x.device))
        dy = F.conv2d(x, self.sobel_y(x.device))
        
        # Compute products of gradients
        dx2 = dx * dx
        dy2 = dy * dy
        dxy = dx * dy
        
        # Gaussian smoothing
        dx2 = self.gaussian_smooth(dx2)
        dy2 = self.gaussian_smooth(dy2)
        dxy = self.gaussian_smooth(dxy)
        
        # Harris corner response
        det = dx2 * dy2 - dxy * dxy
        trace = dx2 + dy2
        response = det - self.k * trace * trace
        
        return response

    @staticmethod
    def sobel_x(device: torch.device) -> torch.Tensor:
        kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).to(device)
        return kernel.view(1, 1, 3, 3)

    @staticmethod
    def sobel_y(device: torch.device) -> torch.Tensor:
        kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).to(device)
        return kernel.view(1, 1, 3, 3)

class IPSN(BaseSteganoModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "IPSN"
        self.channels = config['channels']
        
        # Interest point detector
        self.detector = InterestPointDetector(self.channels)
        
        # Feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )
        
        # Message processing
        self.message_proc = nn.Sequential(
            nn.Conv2d(3, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
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

    def encode(
        self, 
        cover_image: torch.Tensor, 
        secret_message: torch.Tensor
    ) -> torch.Tensor:
        # Extract features
        cover_feat = self.feature_extract(cover_image)
        
        # Detect interest points
        interest_points = self.detector(cover_feat)
        
        # Weight features by interest points
        cover_feat = cover_feat * interest_points
        
        # Process secret message
        msg_feat = self.message_proc(secret_message)
        
        # Combine and generate stego image
        combined = torch.cat([cover_feat, msg_feat], dim=1)
        return self.encode_head(combined)

    def decode(self, stego_image: torch.Tensor) -> torch.Tensor:
        # Extract features
        feat = self.feature_extract(stego_image)
        
        # Detect interest points
        interest_points = self.detector(feat)
        
        # Weight features
        feat = feat * interest_points
        
        # Generate recovered message
        return self.decode_head(feat)