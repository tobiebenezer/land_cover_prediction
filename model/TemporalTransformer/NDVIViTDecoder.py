import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from model.TemporalTransformer.module import *

class NDVIViTDecoder(nn.Module):
    def __init__(self, input_dim=128, output_channels=1, output_size=64):
        super(NDVIViTDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size

        # Calculate the number of upsampling steps needed
        self.num_upsample = 3  # This will upsample 8x (2^3 = 8)
        initial_size = output_size // (2**self.num_upsample)

        self.fc = nn.Linear(input_dim, 256 * initial_size * initial_size)

        self.deconv_layers = nn.ModuleList()
        current_channels = 256
        for _ in range(self.num_upsample):
            self.deconv_layers.append(
                nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=4, stride=2, padding=1)
            )
            current_channels //= 2

        self.final_conv = nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x shape: (batch_size * sequence_length, input_dim)
        batch_size = x.shape[0]
        
        # Reshape the input to prepare for deconvolution
        initial_size = self.output_size // (2**self.num_upsample)
        h = self.fc(x)
        h = h.view(batch_size, 256, initial_size, initial_size)

        # Apply deconvolution layers
        for i, deconv in enumerate(self.deconv_layers):
            h = deconv(h)
            if i < len(self.deconv_layers) - 1:
                h = F.relu(h)

        # Apply final convolution
        h = self.final_conv(h)
        h = torch.sigmoid(h)

        return h

# class NDVIViTDecoder(nn.Module):
#     def __init__(self, input_dim=64, output_channels=1, output_size=64 ,num_patches=25):
#         super(NDVIViTDecoder, self).__init__()
#         self.input_dim = input_dim
#         self.output_size = output_size

#         # Calculate the number of upsampling steps needed
#         self.num_upsample = 3  # This will upsample 8x (2^3 = 8)
#         initial_size = output_size // (2**self.num_upsample)

#         self.fc = nn.Linear(input_dim, 256 * initial_size * initial_size)

#         self.deconv_layers = nn.ModuleList()
#         current_channels = 256
#         for i in range(self.num_upsample):
#             out_channels = current_channels // 2 if i < self.num_upsample - 1 else output_channels
#             self.deconv_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
#             current_channels = out_channels

#         self.final_conv = nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # x shape: (batch_size, input_dim)
#         x = rearrange(x, 'b s p d -> (b s p) d')
#         batch_size = x.size(0)

        
#         # Reshape the input to prepare for deconvolution
#         initial_size = self.output_size // (2**self.num_upsample)
#         x = self.fc(x)
#         print(x.shape,'x')  
#         # x = x.view(batch_size, 256, initial_size, initial_size)

#         # # Apply deconvolution layers
#         # for i, deconv in enumerate(self.deconv_layers):
#         #     x = deconv(x)
#         #     if i < len(self.deconv_layers) - 1:
#         #         x = F.relu(x)

#         # # Apply final convolution
#         # x = self.final_conv(x)

#         # x = torch.sigmoid(x)

#         return x


# class NDVIViTDecoder(nn.Module):
#     def __init__(self, input_dim=64, output_channels=1, output_size=64, num_patches=25):
#         super(NDVIViTDecoder, self).__init__()
#         self.input_dim = input_dim
#         self.output_size = output_size
#         self.num_patches = num_patches

#         # Calculate the number of upsampling steps needed
#         self.num_upsample = 3  # This will upsample 8x (2^3 = 8)
#         initial_size = output_size // (2**self.num_upsample)

#         self.fc = nn.Linear(input_dim, 256 * initial_size * initial_size)

#         self.deconv_layers = nn.ModuleList()
#         current_channels = 256
#         for i in range(self.num_upsample):
#             out_channels = current_channels // 2 if i < self.num_upsample - 1 else output_channels
#             self.deconv_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
#             current_channels = out_channels

#         self.final_conv = nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1)

    # def forward(self, x):
    #     # x shape: (batch_size, sequence_length, num_patches, input_dim)
    #     batch_size, seq_len, num_patches, _ = x.shape
        
    #     # Process each timestep separately
    #     outputs = []
    #     for t in range(seq_len):
    #         # Process each patch
    #         patch_outputs = []
    #         for p in range(num_patches):
    #             patch = x[:, t, p, :]  # (batch_size, input_dim)
                
    #             # Reshape the input to prepare for deconvolution
    #             initial_size = self.output_size // (2**self.num_upsample)
    #             h = self.fc(patch)
    #             h = h.view(batch_size, 256, initial_size, initial_size)

    #             # Apply deconvolution layers
    #             for i, deconv in enumerate(self.deconv_layers):
    #                 h = deconv(h)
    #                 if i < len(self.deconv_layers) - 1:
    #                     h = F.relu(h)

    #             # Apply final convolution
    #             h = self.final_conv(h)
    #             h = torch.sigmoid(h)
                
    #             patch_outputs.append(h)
            
    #         # Combine patch outputs
    #         timestep_output = torch.stack(patch_outputs, dim=1)  # (batch_size, num_patches, output_channels, output_size, output_size)
    #         timestep_output = rearrange(timestep_output, 'b p c h w -> b c p h w')
    #         outputs.append(timestep_output)
       
    #     # Combine all timesteps
    #     final_output = torch.stack(outputs, dim=1)  # (batch_size, sequence_length, output_channels, output_size, output_size)
    #     # final_output = rearrange(final_output, 'b s c (p h) w -> b s c p h w', p=self.num_patches)

    #     return final_output