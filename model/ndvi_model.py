from .base import MBase
import torch
import torch.nn as nn
import torch.nn.functional as F


class NDVIModel(MBase):
    def __init__(self, sequence_length, img_size=(32, 32), lstm_hidden_size=64, lstm_num_layer=2):
        super().__init__()
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Convolution layers
        self.conv1 = nn.Conv2d(sequence_length, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=32 * 32, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layer,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, img_size[0] * img_size[1])
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)

        # LSTM layer
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        print(x)
        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x