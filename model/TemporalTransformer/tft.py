from .base import MBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import *


class TemporalFusionTransformer(nn.Module):
    def __init__(self,hidden_size, context_size, num_heads, dropout,num_layers=1):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.context_size = context_size

        self.input_embedding = nn.Linner
