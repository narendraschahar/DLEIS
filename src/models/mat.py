# src/models/mat.py

class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.mutual_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        B, N, C = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(y).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(y).reshape(B, N, self.num_heads, self.head_dim)
        
        # Compute attention
        attn = torch.einsum('bnhd,bmhd->bnmh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        
        # Apply attention to values
        out = torch.einsum('bnmh,bmhd->bnhd', attn, v)
        out = out.reshape(B, N, C)
        
        # Mutual gating
        gate = self.mutual_gate(torch.cat([x, out], dim=-1))
        out = gate * out + (1 - gate) * x
        
        return self.out_proj(out)

class MAT(BaseSteganoModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "MAT"
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        
        # Image embedding
        self.image_embed = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, 3, padding=1),
            nn.LayerNorm([self.embed_dim, 256, 256]),
            nn.ReLU(inplace=True)
        )
        
        # Mutual attention layers
        self.attention_layers = nn.ModuleList([
            MutualAttention(self.embed_dim, self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        # Output heads
        self.encode_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
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

    def encode(self, cover_image, secret_message):
        # Embed images
        cover_feat = self.image_embed(cover_image)
        secret_feat = self.image_embed(secret_message)
        
        B, C, H, W = cover_feat.shape
        cover_feat = cover_feat.flatten(2).transpose(1, 2)
        secret_feat = secret_feat.flatten(2).transpose(1, 2)
        
        # Apply mutual attention
        for layer in self.attention_layers:
            cover_feat = layer(cover_feat, secret_feat)
            secret_feat = layer(secret_feat, cover_feat)
        
        # Generate stego image
        cover_feat = cover_feat.transpose(1, 2).reshape(B, C, H, W)
        return self.encode_head(cover_feat)

    def decode(self, stego_image):
        # Extract features
        feat = self.image_embed(stego_image)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        
        # Apply self-attention
        for layer in self.attention_layers:
            feat = layer(feat, feat)
        
        # Generate recovered message
        feat = feat.transpose(1, 2).reshape(B, C, H, W)
        return self.decode_head(feat)