import torch
import torch.nn as nn

from model.base import MBase, accuracy
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class SRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ComplexSRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # initial hidden state
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = self.h0.repeat(1, batch_size, 1)
        
        out, _ = self.rnn(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

