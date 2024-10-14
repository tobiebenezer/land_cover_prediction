import torch
import torch.nn as nn
from model.base import MBase, accuracy
from einops import rearrange


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size,pred_size=4,sequence_length=16,input_size = 128,  dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_size = pred_size
        self.sequence_length = sequence_length

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Learned initial hidden and cell states
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))

    def forward(self, x):
      
        b, _, _ = x.shape
        # x = rearrange(x,'(b s) h -> b s  h', s = (self.sequence_length - self.pred_size))
    
        # batch, seq,_ = x.shape
       
        h0 = self.h0.expand(-1, batch, -1).contiguous()
        c0 = self.c0.expand(-1, batch, -1).contiguous()
    
        out, (hn, cn) = self.lstm1(x, (h0, c0))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out, (hn, cn) = self.lstm2(out, (hn, cn))
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Final output layer
       
        out = self.fc3(out)  
        # out = rearrange(out, "b s h -> (b s) h")

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