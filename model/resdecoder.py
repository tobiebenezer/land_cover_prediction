import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18Decoder(nn.Module):
    def __init__(self, out_channels=1):
        super(ResNet18Decoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Layers for upsampling
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

        
        # Add an extra convolutional layer to further upsample
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x4):
        x = self.upsample(x4)  # Upsample from latent space (size h/32 -> h/16)
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.upsample(x)  # Upsample (h/16 -> h/8)
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.upsample(x)  # Upsample (h/8 -> h/4)
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.upsample(x)  # Upsample (h/4 -> h/2)
        x = self.conv4(x)

        x = self.upsample(x)  # Final upsample (h/2 -> h), output 64x64
        x = self.conv5(x)  # Extra convolution to refine output
        
        return x
