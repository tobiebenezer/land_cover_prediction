import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *


class NDVIViTDecoder(nn.Module):
    def __init__(self, input_dim=64, output_channels=1, output_size=64 ,num_patches=25):
        super(NDVIViTDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size

        # Calculate the number of upsampling steps needed
        self.num_upsample = 3  # This will upsample 8x (2^3 = 8)
        initial_size = output_size // (2**self.num_upsample)

        self.fc = nn.Linear(input_dim, 256 * initial_size * initial_size)

        self.deconv_layers = nn.ModuleList()
        current_channels = 256
        for i in range(self.num_upsample):
            out_channels = current_channels // 2 if i < self.num_upsample - 1 else output_channels
            self.deconv_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            current_channels = out_channels

        self.final_conv = nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        x = rearrange(x, 'b s p d -> (b s p) d')  
        
        # Reshape the input to prepare for deconvolution
        initial_size = self.output_size // (2**self.num_upsample)
        x = self.fc(x)
        x = x.view(batch_size, 256, initial_size, initial_size)

        # Apply deconvolution layers
        for i, deconv in enumerate(self.deconv_layers):
            x = deconv(x)
            if i < len(self.deconv_layers) - 1:
                x = F.relu(x)

        # Apply final convolution
        x = self.final_conv(x)

        x = torch.sigmoid(x)

        return x

