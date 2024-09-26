import torch
from torch import nn, einsum
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *
from model.TemporalTransformer.NDVIViTEncoder import *
from model.TemporalTransformer.tft import *
from model.base import MBase



class NDVIViTFT(MBase):
    def __init__(self,
        num_heads=8, 
        hidden_size=128,
        output_size=128, 
        image_size=64,
        dropout=0.2, 
        num_patches=25,
        patch_size=3, 
        in_channel=1,
        dim=128,
        depth=2,
        heads=8, 
        mlp_ratio=4.,
        num_layers=1,
        past_size=10):
        super().__init__()

        self.encoder = NDVIViTEncoder(image_size=image_size,num_patches=num_patches, patch_size=patch_size, 
                        in_channel=in_channel, dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio)

        input_size = dim * num_patches 
        self.tft = TemporalFusionTransformer(input_size=input_size, hidden_size=hidden_size, 
                        output_size=output_size, num_heads=num_heads, 
                        dropout=dropout, num_layers=num_layers, past_size=past_size)

    def forward(self,x,context):
        encoded_output = self.encoder(x)
        temporal_output = self.tft(encoded_output, context)
        return temporal_output

# class NDVIViTEncoder(nn.Module):
    # def __init__(self, image_size=64,passnum_patches=25, patch_size=3, in_channel=1, dim=128, depth=2, heads=8, mlp_ratio=4.):
# 
# class TemporalFusionTransformer(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size, context_size, num_heads, dropout, num_layers=1, past_size=10, future_size=10):