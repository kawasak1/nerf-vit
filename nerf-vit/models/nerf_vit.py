import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import ViT, CrossTransformerBlock, CrossAttention

# Basic MLP block as used in original NeRF
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, width=256, depth=8, skips=[4]):
        super().__init__()
        self.skips = skips
        layers = []
        
        # First layer
        layers.append(nn.Linear(dim_in, width))
        
        # Hidden layers
        for i in range(depth-1):
            if i in skips:
                # If this is a skip connection, add the input dimensions
                layers.append(nn.Linear(width + dim_in, width))
            else:
                layers.append(nn.Linear(width, width))
                
        # Last layer
        layers.append(nn.Linear(width, dim_out))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, input_x=None):
        h = x
        for i, layer in enumerate(self.layers):
            # Apply ReLU activation except for the last layer
            if i < len(self.layers) - 1:
                # Handle skip connections
                if i in self.skips and input_x is not None:
                    h = torch.cat([h, input_x], dim=-1)
                h = layer(h)
                h = F.relu(h)
            else:
                h = layer(h)
        return h

# Base NeRF model
class BaseNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        Base NeRF model as in the original implementation
        Args:
            D: number of layers in the network
            W: width of each layer
            input_ch: number of input channels for points
            input_ch_views: number of input channels for view directions
            output_ch: number of output channels
            skips: list of layers with skip connections
            use_viewdirs: whether to use view directions as input
        """
        super(BaseNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # MLP to predict density features
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips 
             else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )
        
        # MLP to predict RGB
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Process input points
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        # Process view directions if used
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs

# Feature Conditioning NeRF with ViT
class FeatureConditioningNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], 
                 use_viewdirs=False, vit_dim=384, image_size=224):
        """
        NeRF model with feature conditioning from ViT
        Args:
            D: number of layers in the network
            W: width of each layer
            input_ch: number of input channels for points
            input_ch_views: number of input channels for view directions
            output_ch: number of output channels
            skips: list of layers with skip connections
            use_viewdirs: whether to use view directions as input
            vit_dim: dimension of the ViT features
            image_size: size of the input image for ViT
        """
        super(FeatureConditioningNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.vit_dim = vit_dim
        
        # Vision Transformer for feature extraction
        self.vit = ViT(
            image_size=image_size,
            patch_size=16,
            dim=vit_dim,
            depth=6,
            heads=8,
            mlp_dim=vit_dim * 2,
            dropout=0.1,
            emb_dropout=0.1,
            channels=4,
            dim_head=vit_dim // 8
        )
        
        # Project global ViT features to the conditioning dimension
        self.feature_projector = nn.Linear(vit_dim, W)
        
        # MLP to predict density features
        # Add feature conditioning to the input
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + W, W)] + 
            [nn.Linear(W, W) if i not in self.skips 
             else nn.Linear(W + input_ch + W, W) for i in range(D-1)]
        )
        
        # MLP to predict RGB
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
            
        # Store the image features
        self.image_features = None
    
    def extract_features(self, image):
        """Extract features from input image using ViT"""
        with torch.no_grad():
            # Extract global features - image should be [B, C, H, W]
            return self.vit.get_global_features(image)  # Returns [B, vit_dim]
    
    def set_image_features(self, image):
        """Set image features for conditioning"""
        features = self.extract_features(image)  # features: [B, vit_dim]
        # Take the first batch element as reference if batch size > 1
        if features.shape[0] > 1:
            features = features[0]  # Get first batch item
        # Detach features to avoid backward pass issues
        self.image_features = self.feature_projector(features).detach()
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Expand image features to match batch size
        batch_size = input_pts.shape[0]
        # Handle the features properly - they might be 1D or 2D depending on how they were set
        if len(self.image_features.shape) == 1:
            # Detach features to avoid backward pass issues
            expanded_features = self.image_features.unsqueeze(0).expand(batch_size, -1).detach()
        else:
            # Assuming image_features is [1, feature_dim]
            # Detach features to avoid backward pass issues
            expanded_features = self.image_features.expand(batch_size, -1).detach()
        
        # Condition the input with image features
        conditioned_input = torch.cat([input_pts, expanded_features], dim=-1)
        
        # Process conditioned input points
        h = conditioned_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([conditioned_input, h], -1)
        
        # Process view directions if used
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs

# Transformer-based NeRF with ViT
class TransformerNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], 
                 use_viewdirs=False, vit_dim=384, transformer_depth=2, num_heads=4, image_size=224):
        """
        Memory-efficient NeRF model with Transformer-inspired architecture
        Args:
            D: number of layers in the network
            W: width of each layer
            input_ch: number of input channels for points
            input_ch_views: number of input channels for view directions
            output_ch: number of output channels
            skips: list of layers with skip connections
            use_viewdirs: whether to use view directions as input
            vit_dim: dimension of the ViT features
            transformer_depth: number of transformer-inspired layers
            num_heads: not used in this implementation (kept for compatibility)
            image_size: size of the input image for ViT
        """
        super(TransformerNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.vit_dim = vit_dim
        
        # Vision Transformer for feature extraction
        self.vit = ViT(
            image_size=image_size,
            patch_size=16,
            dim=vit_dim,
            depth=6,
            heads=8,
            mlp_dim=vit_dim * 2,
            dropout=0.1,
            emb_dropout=0.1,
            channels=4,
            dim_head=vit_dim // 8
        )
        
        # Project global ViT features to the conditioning dimension
        self.feature_projector = nn.Linear(vit_dim, W)
        
        # MLP to predict density features (exactly like BaseNeRF but with feature conditioning)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + W, W)] + 
            [nn.Linear(W, W) if i not in self.skips 
             else nn.Linear(W + input_ch + W, W) for i in range(D-1)]
        )
        
        # Instead of transformer blocks, use simple feature gating
        self.gate_layers = nn.ModuleList(
            [nn.Linear(W, W) for _ in range(transformer_depth)]
        )
        
        # MLP to predict RGB
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        # Store the image features
        self.image_features = None
    
    def extract_features(self, image):
        """Extract global features from input image using ViT"""
        with torch.no_grad():
            # Extract global features - image should be [B, C, H, W]
            return self.vit.get_global_features(image)  # Returns [B, vit_dim]
    
    def set_image_features(self, image):
        """Set image features for conditioning"""
        features = self.extract_features(image)  # features: [B, vit_dim]
        # Take the first batch element as reference if batch size > 1
        if features.shape[0] > 1:
            features = features[0]  # Get first batch item
        # Detach features to avoid backward pass issues
        self.image_features = self.feature_projector(features).detach()
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Expand image features to match batch size
        batch_size = input_pts.shape[0]
        # Handle the features properly - they might be 1D or 2D depending on how they were set
        if len(self.image_features.shape) == 1:
            # Detach features to avoid backward pass issues
            expanded_features = self.image_features.unsqueeze(0).expand(batch_size, -1).detach()
        else:
            # Assuming image_features is [1, feature_dim]
            # Detach features to avoid backward pass issues
            expanded_features = self.image_features.expand(batch_size, -1).detach()
        
        # Condition the input with image features
        conditioned_input = torch.cat([input_pts, expanded_features], dim=-1)
        
        # Process conditioned input points (same as BaseNeRF but with conditioning)
        h = conditioned_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([conditioned_input, h], -1)
        
        # Apply simplified "transformer-inspired" gating at the end
        # This adds transformer-like modulation without memory-intensive operations
        for gate_layer in self.gate_layers:
            # Simple gating mechanism instead of full transformer attention
            gate = torch.sigmoid(gate_layer(h))
            h = h * gate
        
        # Process view directions if used
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs 

# Cross-Attention NeRF with ViT
class CrossAttentionNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], 
                 use_viewdirs=False, vit_dim=384, num_heads=4, image_size=224):
        """
        NeRF model with Cross-Attention mechanism to relate 3D points to image features
        Args:
            D: number of layers in the network
            W: width of each layer
            input_ch: number of input channels for points
            input_ch_views: number of input channels for view directions
            output_ch: number of output channels
            skips: list of layers with skip connections
            use_viewdirs: whether to use view directions as input
            vit_dim: dimension of the ViT features
            num_heads: number of attention heads
            image_size: size of the input image for ViT
        """
        super(CrossAttentionNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.vit_dim = vit_dim
        
        # Vision Transformer for feature extraction
        self.vit = ViT(
            image_size=image_size,
            patch_size=16,
            dim=vit_dim,
            depth=6,
            heads=8,
            mlp_dim=vit_dim * 2,
            dropout=0.1,
            emb_dropout=0.1,
            channels=4,
            dim_head=vit_dim // 8
        )
        
        # Input embedding
        self.input_embedding = nn.Linear(input_ch, W)
        
        # MLP to process input before cross-attention
        self.pre_mlp = MLP(W, W, width=W, depth=D//2, skips=[])
        
        # Cross-attention layer to relate points to image features
        self.cross_attention = CrossAttention(
            dim=W,
            context_dim=vit_dim,
            heads=num_heads,
            dim_head=W // num_heads,
            dropout=0.1
        )
        
        # MLP to process after cross-attention
        self.post_mlp = MLP(W, W, width=W, depth=D//2, skips=[])
        
        # Final layers
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        
        # Store the image features
        self.image_features = None
    
    def extract_features(self, image):
        """Extract features from input image using ViT"""
        with torch.no_grad():
            # Extract patch features - image should be [B, C, H, W]
            return self.vit.get_patch_features(image)  # Returns [B, num_patches, vit_dim]
    
    def set_image_features(self, image):
        """Set image features for conditioning"""
        features = self.extract_features(image)  # features: [B, num_patches, vit_dim]
        # Take the first batch element as reference
        if features.shape[0] > 1:
            features = features[0:1]  # Keep batch dimension but use only first item
        # Detach features to avoid backward pass issues
        self.image_features = features.detach()  # Store [1, num_patches, vit_dim]
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # Embed input points
        h = self.input_embedding(input_pts)
        
        # Process with first MLP
        h = self.pre_mlp(h)
        
        # Apply cross-attention with image features
        batch_size = h.shape[0]
        h_unsqueezed = h.unsqueeze(1)  # [batch_size, 1, W]
        
        # Ensure image features have the right batch dimension
        context = self.image_features  # [1, num_patches, vit_dim]
        if batch_size > context.shape[0]:
            # Expand context to match batch size
            # Detach to avoid backward pass issues
            context = context.expand(batch_size, -1, -1).detach()
            
        h_attended = self.cross_attention(h_unsqueezed, context)  # [batch_size, 1, W]
        
        # Process with second MLP
        h = self.post_mlp(h_attended.squeeze(1))
        
        # Process view directions if used
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs 