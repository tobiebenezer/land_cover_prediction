import torch
from torch import nn, einsum
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *
from model.base import MBase


class NDVIViTEncoder(MBase):
    def __init__(self, image_size=64,num_patches=20, patch_size=3, in_channel=1, dim=128, depth=2, heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size=image_size, patch_size=patch_size, in_chans=in_channel, embed_dim=dim)
        self.transformer = Transformer(dim=dim, depth=depth, num_heads=heads, mlp_ratio=mlp_ratio)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformerblock = Transformer(dim=dim, depth=depth, num_heads=heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, _, _ = x.shape
        x = self.patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape,n)
        print(self.pos_embedding[:, :].shape)
        x += self.pos_embedding[:, :(n + 1)]
    
        x = self.dropout(x)
        x = self.transformerblock(x)
        print(x.shape,)

        x = self.norm(x)
        print(x.shape)
        return x[:, 0]

    

