from .base import MBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import *


class TemporalFusionTransformer(nn.Module):
    def __init__(self,hidden_size, num_heads, dropout):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
