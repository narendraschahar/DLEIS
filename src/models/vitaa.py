# src/models/vitaa.py

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ViTAA(BaseSteganoModel):
    def __init__(self, config):
        super().__init__(config)
        self.name = "ViT-AA"
        
        self.patch_size = config['patch_size']
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.depth = config['depth']
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim,
                self.num_heads
            ) for _ in range(self.depth)
        ])
        
        # Message embedding
        self.message_embed = nn.Sequential(
            nn.Linear(3 * self.patch_size ** 2, self.embed_dim),
            nn.GELU()
        )
        
        # Output heads
        self.encode_head = nn.Sequential(
            nn.Linear(self.embed_dim, 3 * self.patch_size ** 2),
            nn.Tanh()
        )
        
        self.decode_head = nn.Sequential(
            nn.Linear(self.embed_dim, 3 * self.patch_size ** 2),
            nn.Sigmoid()
        )

    def encode(self, cover_image, secret_message):
        # Extract patches
        patches = self.patch_embed(cover_image)
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        
        # Process through transformer
        for block in self.blocks:
            patches = block(patches)
            
        # Embed message
        message_feat = self.message_embed(
            secret_message.reshape(B, -1, 3 * self.patch_size ** 2)
        )
        
        # Combine and generate stego image
        combined = patches + message_feat
        stego = self.encode_head(combined)
        
        return stego.reshape(B, H, W, 3).permute(0, 3, 1, 2)

    def decode(self, stego_image):
        # Process stego image
        patches = self.patch_embed(stego_image).flatten(2).transpose(1, 2)
        
        # Extract message
        for block in self.blocks:
            patches = block(patches)
            
        message = self.decode_head(patches)
        B = message.shape[0]
        
        return message.reshape(B, 3, self.patch_size, self.patch_size)
