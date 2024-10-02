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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ComplexLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Learned initial hidden and cell states
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = self.h0.repeat(1, batch_size, 1)
        c0 = self.c0.repeat(1, batch_size, 1)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# Example usage:
# input_size = patch_height * patch_width
# hidden_size = 128
# num_layers = 3
# output_size = patch_height * patch_width
# sequence_length = 10

# srnn_model = ComplexSRNN(input_size, hidden_size, num_layers, output_size)
# lstm_model = ComplexLSTM(input_size, hidden_size, num_layers, output_size)
# gru_model = ComplexGRU(input_size, hidden_size, num_layers, output_size)

# x = torch.randn(32, sequence_length - 1, input_size)  # (batch_size, sequence_length, input_size)
# y_pred_srnn = srnn_model(x)
# y_pred_lstm = lstm_model(x)
# y_pred_gru = gru_model(x)