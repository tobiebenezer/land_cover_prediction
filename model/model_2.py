from .base import MBase
import torch
import torch.nn as nn
import torch.nn.functional as F

class NDVIModel(MBase):
    def __init__(self, sequence_length, num_channels, img_size=(32, 32), lstm_hidden_size=64, lstm_num_layers=1):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.img_size = img_size
        
        # Convolution layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._get_conv_output_size((num_channels, img_size[0], img_size[1]))
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, 100 * 1024)  # Directly output the desired shape
        
    def _get_conv_output_size(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._conv_forward(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    def _conv_forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, _, _, _ = x.size()
        
        # Reshape for 2D convolution
        x = x.view(batch_size * seq_len, self.num_channels, *self.img_size)
        
        # Convolutional layers
        x = self._conv_forward(x)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM layer
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape to match the desired output shape (100, 1024)
        x = x.view(batch_size,100, 1024)
        
        return x
