from .base import MBase
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalFusionTransformer(MBase):
    def __init__(self,parameters):
        super(TemporalFusionTransformer, self).__init__()

        #inputs
        self.col_to_idx = parameters["col_to_idx"]
        self.static_covariates = parameters["static_covariates"]
        self.time_dependent_categorical = parameters["time_dependent_categorical"]
        self.time_dependent_continuous = parameters["time_dependent_continuous"]
        self.category_counts = parameters["category_counts"]
        self.known_time_dependent = parameters["known_time_dependent"]
        self.observed_time_dependent = parameters["observed_time_dependent"]
        self.time_dependent = self.known_time_dependent + self.observed_time_dependent

        #Architecture
        self.batch_size = parameters["batch_size"]
        self.encoder_steps = parameters["encoder_steps"]
        self.hidden_size = parameters["hidden_size"]
        self.num_lstm_layers = parameters["num_lstm_layers"]
        self.dropout = parameters["dropout"]
        self.embedding_dim = parameters["embedding_dim"]
        self.num_heads = parameters["num_heads"]
        
        #Outputs
        self.quantiles = parameters['quantiles']
        self.device = parameters['device']