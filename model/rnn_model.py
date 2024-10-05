import torch
import torch.nn as nn
from einops import rearrange

from model.base import MBase, accuracy
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class SRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size,pred_size=4,sequence_length=16,input_size = 128,  dropout=0.5):
        super(SRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_size = pred_size
        self.sequence_length = sequence_length
        
        self.rnn1 = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        
    def forward(self, x):
        b, _ = x.shape
        x = rearrange(x,'(b s) h -> b s  h', s = (self.sequence_length - self.pred_size))
        batch, seq,_ = x.shape
        
        h0 = self.h0.expand(-1, batch, -1).contiguous()
        out, h1 = self.rnn1(x, h0)
        out = self.fc1(out)
        out = self.relu(out)
        out,_ = self.rnn2(out,h1)
        out = self.fc2(out)
        return out

