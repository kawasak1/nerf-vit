import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helper functions
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Positional embedding for transformers
def posemb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# Basic components
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context=None):
        x = self.norm(x)
        
        if context is None:
            context = x
        else:
            context = self.context_norm(context)
        
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
    
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.cross_attn = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
    
    def forward(self, x, context):
        x = self.cross_attn(x, context) + x
        x = self.ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Vision Transformer for feature extraction
class ViT(nn.Module):
    def __init__(
        self, 
        *, 
        image_size=224, 
        patch_size=16, 
        dim=768,
        depth=12, 
        heads=12, 
        mlp_dim=3072, 
        channels=3, 
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        
        # Print initialization parameters for debugging
        print(f"ViT init: image_size={image_size}, patch_size={patch_size}, dim={dim}, channels={channels}")
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        # Print calculated dimensions for debugging
        print(f"ViT patches: num_patches={num_patches}, patch_dim={patch_dim}")
        
        # Ensure the patch embedding uses the correct embedding dimension
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),  # Normalize with patch_dim
            nn.Linear(patch_dim, dim),  # Map to embedding dimension
            nn.LayerNorm(dim),  # Normalize with embedding dimension
        )
        
        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Output features directly without classification head
        self.feature_dim = dim
        
    def forward(self, img):
        device = img.device
        
        # Debug print for input shape
        print(f"ViT forward input shape: {img.shape}")
        
        # Patchify and embed
        x = self.to_patch_embedding(img)
        
        # Debug print for embedded shape
        print(f"ViT after patch embedding shape: {x.shape}")
        
        # Add positional embedding
        x += self.pos_embedding.to(device, dtype=x.dtype)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Debug print for output shape
        print(f"ViT transformer output shape: {x.shape}")
        
        # Return patch features
        return x  # Shape: [batch_size, num_patches, dim]
    
    def get_patch_features(self, img):
        """Returns patch-wise features from the input image"""
        return self.forward(img)
    
    def get_global_features(self, img):
        """Returns a global feature vector by mean-pooling patch features"""
        patch_features = self.forward(img)
        return patch_features.mean(dim=1)  # Shape: [batch_size, dim] 