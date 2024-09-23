import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()  
        self.fnet = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fnet(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        internal_dim = dim_head * heads
        is_projected =  not (head == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** (-0.5) # sqrt of num of head

        self.to_qkv = nn.Linear(dim, internal_dim *3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(internal_dim, dim),
            nn.Dropout(dropout)
        ) if is_projected else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda tran: rearrange(tran, "b n (h d) -> b h n d", h=h), qkv)
        q_dot_k = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale   

        attention = q_dot_k.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Reattention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Reattention, self).__init__()
        internal_dim = dim_head * heads
        is_projected =  not (head == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5) # sqrt of num of head
        self.to_qkv = nn.Linear(dim, internal_dim *3, bias=False)
        self.reattn_weig = nn.Parameter(torch.randn(heads,head))
        self.reattn_norm = nn.Sequential(
            Rearrange("b h i j -> b i j h"),
            nn.LayerNorm(heads),
            Rearrange("b i j h -> b h i j")
        )

        self.to_out = nn.Sequential(
            nn.Linear(internal_dim, dim),
            nn.Dropout(dropout)
        ) if is_projected else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda tran: rearrange(tran, "b n (h d) -> b h n d", h=h), qkv)

        #attention 
        q_dot_k = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale   
        attention = q_dot_k.softmax(dim=-1)

        #reattention
        attention = einsum("b h i d, b h j d -> b h i j", attention, self.reattn_weig)
        attention = self.reattn_norm(attention)

        out = einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.proj = nn.Linear(dim, dim )

        #gate 
        self.sigmoid = nn.Sigmoid()
        self.proj_gate = nn.Linear(dim, dim)

    def forword(self,x):
        gate = self.sigmoid(self.proj_gate(x))
        x = self.proj(x)
        return torch.mul(gate,x)

class TemporalLayer(nn.Module):
    def __init__(self, module):
        super(TemporalLayer, self).__init__()

        self.module = module

    def forword(self, x):
        x = rearrange("t n h -> (t n) h",x)
        x = self.module(x)
        x = rearrange("(t n) h -> t n h", x)

        return x

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0., context_size=None, is_temporal=True):
        super(GateResidualNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size
        self.is_temporal = is_temporal

        if self.is_temporal:
            if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size, self.output_size))
            
            #context
            if self.context_size != None:
                self.context = TemporalLayer(nn.Linear(self.input_size, self.output_size))

            #Dense & ELU
            self.dense1 = TemporalLayer(nn.Linear(self.input_size, self.hidden_size))
            self.elu = nn.ELU()
            self.dense2 = TemporalLayer(nn.Linear(self.hidden_size, self.output_size))
            self.dropout = nn.Dropout(self.dropout)

            #Gate, Add and Norm
            self.gate = TemporalLayer(GLU(self.output_size))
            self.layer_norm = TemporalLayer(nn.BatchNorm1d(self.output_size))

        else:
            if self.input_size != self.output_size:
                self.skip_layer = nn.Linear(self.input_size, self.output_size)
            
            #context
            if self.context_size != None:
                self.context = nn.Linear(self.input_size, self.output_size)

            #Dense & ELU
            self.dense1 = nn.Linear(self.input_size, self.hidden_size)
            self.elu = nn.ELU()
            self.dense2 = nn.Linear(self.hidden_size, self.output_size)
            self.dropout = nn.Dropout(self.dropout)

            #Gate, Add and Norm
            self.gate = GLU(self.output_size)
            self.layer_norm = nn.BatchNorm1d(self.output_size)

    def forward(self,x, context = None):

        if self.input_size != self.output_size:
            a = self.skip_layer(x)
        else:
            a

        x = self.dense1(x)
        if context != None:
            context = self.context(context.unsqueeze(1))
            x += context

        eta_2 = self.elu(x)
        eta_1 = self.dropout(self.dense2(eta_2))

        gate = self.gate(eta_1) + a
        x = self.layer_norm(gate)

        return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, context_size=None, is_temporal=True):
        super(VariableSelectionNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_sze
        self.dropout = dropout
        self.context_size = context_size
        self.is_temp = is_temp

        self.flattened_inputs = GatedRedsidualNetwork(self.output_size*self.input_size, 
                                                     self.hidden_size, self.output_size, 
                                                     self.dropout, self.context_size, 
                                                     self.is_temp)

        self.transformer_inputs = nn.ModuleList([
            GatedResidualNetwork(
                self.input_size, self.hidden_size, self.dropout,
                self.context_size, self.is_temp) for _ in range(self.output_size)
        ])

        self.softMax = nn.Softmax(dim=-1)

    def forword(self, embedding, context=None):
        # Generation of variable selection weights
        sparse_weights = self.flattened_inputs(embedding, context)
        if self.is_temporal:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        else:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(1)

        # Additional non-linear processing for each feature vector
        transformed_embeddings = torch.stack(
            [self.transformed_inputs[i](embedding[
                Ellipsis, i*self.input_size:(i+1)*self.input_size]) for i in range(self.output_size)], axis=-1)

        # Processed features are weighted by their corresponding weights and combined
        combined = transformed_embeddings*sparse_weights
        combined = combined.sum(axis=-1)

        return combined, sparse_weights

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model

        # Vectorized positional encoding computation
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        # Compute sine and cosine
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  

        pe = rearrange(pe, 'l d_model -> 1 seq_len d_model')
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        
        # Scale the input and add positional encodings
        x = x * math.sqrt(self.d_model)
        pe_slice = self.pe[:, :seq_len, :]
        x = x + rearrange(pe_slice, '1 seq_len d_model -> seq_len 1 d_model')
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()

