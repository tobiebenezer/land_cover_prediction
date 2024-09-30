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
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
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

class Sen12MSViTEncoder(nn.Module):
    def __init__(self, image_size=64, patch_size=4, in_channels=18, dim=256, depth=6, heads=8, mlp_ratio=4.):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        # Lightweight feature extractor (modified ResNet18)
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor.maxpool = nn.Identity()  # Remove maxpool to maintain spatial dimensions
        
        # Remove the final fully connected layer and avgpool
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape

        # Extract features using modified ResNet18
        x = self.feature_extractor(x)

        # Create patch embeddings
        x = self.to_patch_embedding(x)
        
        # Add class token and positional embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(x.size(1))]
        x = self.dropout(x)

        # Apply transformer
        x = self.transformer(x)
        x = self.norm(x)

        return x