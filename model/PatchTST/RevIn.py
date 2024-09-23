import torch
import torch.nn as nn
import einops

""" 
implementation source: https://github.dev/yuqinie98/PatchTST/tree/main/PatchTST_supervised
"""

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.subtract_last:
            self.last = einops.reduce(x[:, -1, :], 'b c -> b 1 c', 'mean')
        else:
            self.mean = einops.reduce(x, 'b t c -> b 1 c', 'mean')
        self.stdev = torch.sqrt(einops.reduce(x.var(dim=1, unbiased=False), 'b c -> b 1 c', 'mean') + self.eps)

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = torch.einsum('btc,c->btc', x, self.affine_weight) + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x