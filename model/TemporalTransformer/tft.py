import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import einops
from model.TemporalTransformer.module import *

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, dropout, num_layers=1, past_size=10, patch_size=25, sequence_length=12):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.past_size = past_size
        self.output_size = output_size
        self.patch_size = patch_size
        self.sequence_length = sequence_length

        self.input_embedding = nn.Linear(hidden_size, hidden_size)
        self.day_embedding = PositionalEncoder(hidden_size // 2, 1, dropout, max_len=366)
        self.month_embedding = PositionalEncoder(hidden_size // 2, 2, dropout, max_len=12)
        self.encoder_lstm = nn.LSTM(hidden_size * (patch_size + 1), hidden_size * (patch_size ), num_layers, dropout=dropout, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size * (patch_size + 1), hidden_size * (patch_size), num_layers, dropout=dropout, batch_first=True)
        self.temporal_grn = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout)
        self.final_grn = GatedResidualNetwork(hidden_size, hidden_size, output_size, dropout)

        self.gated_skip_connection = TemporalLayer(GLU(hidden_size))
        self.add_norm = TemporalLayer(nn.BatchNorm1d(hidden_size))
        self.position_wise_feed_forward = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        self.context_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size * (patch_size), dropout)
        self.static_context_state_h = GatedResidualNetwork(hidden_size, hidden_size, hidden_size * (patch_size), dropout, is_temporal=False)
        self.static_context_state_c = GatedResidualNetwork(hidden_size, hidden_size, hidden_size * (patch_size), dropout, is_temporal=False)
        
        self.static_enrichment = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout)
        self.attention_gated_skip_connection = TemporalLayer(GLU(hidden_size))
        self.attention_add_norm = TemporalLayer(nn.BatchNorm1d(hidden_size))
        self.output_gated_skip_connection = TemporalLayer(GLU(hidden_size))
        self.output_add_norm = TemporalLayer(nn.BatchNorm1d(hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    # @torch.jit.script
    def define_lstm_encoder(self, x, static_context_h, static_context_c):
        # static_context_h = rearrange(static_context_h, "(s p) n -> s (p n)" , p=self.sequence_length)
        # static_context_c = rearrange(static_context_c, "(s p) n -> s (p n)" , p=self.sequence_length)
        b, _, _ = x.shape
        static_context_h = static_context_h[:b,:]
        static_context_c = static_context_c[:b,:]

        output, (state_h, state_c) = self.encoder_lstm(x, (static_context_h.unsqueeze(0).repeat(self.num_layers,1,1),
                             static_context_c.unsqueeze(0).repeat(self.num_layers,1,1)))

        return output, state_h, state_c

    # @torch.jit.script
    def define_lstm_decoder(self, x, state_h, state_c):
        output, (_, _) = self.decoder_lstm(x, (state_h, state_c))

        return output

    def define_static_covariate_encoders(self, context):
        day_of_year = context[:, :,0]
        month = context[:, :,1]
        day_context = self.day_embedding(day_of_year)
        month_context = self.month_embedding(month - 1) 
        
        static_encoder = torch.cat([day_context, month_context], dim=-1)
       
        static_context_e = self.context_grn(static_encoder)
        static_context_h = self.static_context_state_h(static_encoder)
        static_context_c = self.static_context_state_c(static_encoder)

        return static_context_e, static_context_h, static_context_c


    def get_mask(self, tensor):
        return torch.triu(torch.ones(tensor.size(0), tensor.size(0)), diagonal=1).bool().to(tensor.device)

    def forward(self, x, context):
        print(x.shape,"x")
        x = self.input_embedding(x)
        b, s ,_ ,_ = x.shape 
        x = rearrange(x, "b s n h -> (b s) n h")
        future_size = x.shape[0] * 0.75
        future_size = int(future_size)
        
        print(context.shape)
        static_context_e, static_context_h, static_context_c = self.define_static_covariate_encoders(context)
        
        past_input = rearrange(x[:future_size, :, :], "(b s) n h -> b s (n h)", b=b)
        future_input = rearrange(x[future_size:, :, :], "(b s) n h -> b s (n h)", b=b)
     
        encoder_output, state_h, state_c = self.define_lstm_encoder(past_input, static_context_h, static_context_c)       
        decoder_output = self.define_lstm_decoder(future_input, state_h, state_c)
        print(encoder_output.shape,"encoder_output")
        print(decoder_output.shape,"decoder_output")
        lstm_outputs = torch.cat([encoder_output, decoder_output], dim=1)
        print(lstm_outputs.shape,"lstm_outputs")
         
        lstm_outputs = rearrange(lstm_outputs, "b s (n h) -> b (s n) h", h=self.hidden_size)
        print(lstm_outputs.shape,"lstm_outputs")
        gated_outputs = self.gated_skip_connection(lstm_outputs)
        gated_outputs = rearrange(gated_outputs, "b (s n) h -> (b s) n h", s=s)
        print(gated_outputs.shape,"gated_outputs")
        print(x.shape,"x")
        temporal_feature_outputs = self.add_norm(x[:,:-1,:] + gated_outputs)

        static_context_e_reshaped = rearrange(static_context_e, "b (s h) -> (b s) h", s=gated_outputs.shape[1] ) #.reshape(24, 25, 128)
        static_er = torch.cat([temporal_feature_outputs, static_context_e_reshaped.unsqueeze(1).repeat(1, temporal_feature_outputs.shape[1], 1)], dim=2)
        print(static_er.shape,"static_er")

        # stactic_er = torch.cat([temporal_feature_outputs, static_context_e.unsqueeze(1).expand(-1, temporal_feature_outputs.size(1), -1)])
        # print(stactic_er.shape,"stactic_er")

        static_enrichment_outputs = self.static_enrichment(torch.cat([temporal_feature_outputs, static_context_e.unsqueeze(1).expand(-1, temporal_feature_outputs.size(1), -1)], dim=-1))

        # mask = self.get_mask(static_enrichment_outputs)
        # multihead_outputs, multihead_attention = self.multihead_attn(static_enrichment_outputs, static_enrichment_outputs, static_enrichment_outputs, attn_mask=mask)
        
        # attention_gated_outputs = self.attention_gated_skip_connection(multihead_outputs)
        # attention_outputs = self.attention_add_norm(attention_gated_outputs + static_enrichment_outputs)

        # temporal_fusion_decoder_outputs = self.position_wise_feed_forward(attention_outputs)

        # gate_outputs = self.output_gated_skip_connection(temporal_fusion_decoder_outputs)
        # norm_outputs = self.output_add_norm(gate_outputs + temporal_feature_outputs)

        # output = self.output(norm_outputs[:, self.past_size:, :]).view(-1, self.output_size)
        
        # attention_weights = {
        #     'multihead_attention': multihead_attention,
        # }

        # return output, attention_weights
        return x