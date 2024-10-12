import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet18Encoder, self).__init__()
        
        # Load pretrained ResNet18 model
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        
        # Modify the first convolution layer to accept 1 input channel instead of 3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Keep other layers from ResNet
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.output_dim =[2048]

    def forward(self, x):
        x = rearrange(x, 'b h w -> b 1 h w')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # 64 channels
        x = self.layer2(x)  # 128 channels
        x = self.layer3(x)  # 256 channels
        x4 = self.layer4(x)  # Latent space with 512 channels (final feature map)
        self.output_dim = x4.shape
        x4 = x4.view(x.size(0), -1)

        return x4 , self.output_dim  # Return only the latent space

