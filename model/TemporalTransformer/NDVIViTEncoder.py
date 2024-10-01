import torch
from torch import nn, einsum
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *
from torchvision.models import resnet18


class NDVIViTEncoder(nn.Module):
    def __init__(self, image_size=64,num_patches=25, patch_size=3, in_channel=1, dim=128, depth=2, heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size=image_size, patch_size=patch_size, in_chans=in_channel, embed_dim=dim)
        self.transformer = Transformer(dim=dim, depth=depth, num_heads=heads, mlp_ratio=mlp_ratio)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformerblock = Transformer(dim=dim, depth=depth, num_heads=heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        bat = x.shape[0]
        x = rearrange(x, 'b s p h w -> (b s) p h w')
        b, n, _, _ = x.shape
        x = self.patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformerblock(x)
        x = self.norm(x)
        x = rearrange(x, '(b s) n d -> b s n d', b=bat)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(dim * mlp_ratio), dim)
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + mlp(norm2(x))
        return x

class Sen12MSViTEncoder(nn.Module):
    def __init__(self, image_size=64, in_channels=1, dim=256, output_dim=256, depth=6, heads=8, mlp_ratio=4.):
        super().__init__()
        
        # Lightweight feature extractor (modified ResNet18)
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor.maxpool = nn.Identity()  # Remove maxpool to maintain spatial dimensions
        # Remove the final fully connected layer and avgpool
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        
        # Add adaptive pooling to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Adjust the feature dimension
        self.feature_projection = nn.Linear(512 * 8 * 8, dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)
        
        self.transformer = Transformer(dim=dim, depth=depth, num_heads=heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (batch_size, 25, 64, 64)
        b, s,  _, _ = x.shape
        # Add channel dimension and reshape to process all patches at once
        x = rearrange(x, 'b s h w -> (b s ) 1 h w')
        
        # Extract features using ResNet18
        features = self.feature_extractor(x)
        
        # Apply adaptive pooling
        features = self.adaptive_pool(features)
        
        # Flatten features and project to desired dimension
        features = rearrange(features, 'b c h w -> b (c h w)')
        features = self.feature_projection(features)
        
        # Reshape features to (batch_size, sequence_length, num_patches, dim)
        features = rearrange(features, '(b s p) d -> b s p d', b=x.shape[0] // 25, s=25)
        
        # Add cls tokens and positional embeddings
        cls_tokens = repeat(self.cls_token, '() n d -> b s n d', b=features.shape[0], s=features.shape[1])
        features = torch.cat((cls_tokens, features), dim=2)
        features += self.pos_embedding[:, :(features.shape[2])]
        features = self.dropout(features)
        
        # Reshape to (batch_size * sequence_length, num_patches + 1, dim) for transformer
        features = rearrange(features, 'b s n d -> (b s) n d')
        
        # Apply transformer
        x = self.transformer(features)
        x = self.norm(x)
        
        # Remove cls tokens
        # x = x[:, 1:]
        
        # Reshape back to (batch_size, sequence_length, num_patches, dim)
        x = rearrange(x, '(b s p) c h -> b s p (c h)', b=b, s=s )
        
        return x