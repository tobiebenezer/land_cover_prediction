import torch
form transformers import PatchTSTConfig, PatchTSTForPrediction


class PatchTST(torch.nn.Module):
    def __init__(self,config):
        super(PatchTST, self).__init__()
        self.config = PatchTSTConfig(**config)
        self.model = PatchTSTForPrediction(self.config)

    def forward(self, x):
        return self.model(x)