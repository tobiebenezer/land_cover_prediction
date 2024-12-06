import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction
from einops import rearrange


class PatchTST(torch.nn.Module):
    def __init__(self,config):
        super(PatchTST, self).__init__()
        self.config = PatchTSTConfig(**config)
        self.model = PatchTSTForPrediction(self.config)

    def forward(self, x,y=None):
        
        if not( y is None):
            y = rearrange(y, 's b h -> b s h')
            return self.model(
                past_values=x,
                future_values=y)
        else:
            return self.model(past_values=x)
