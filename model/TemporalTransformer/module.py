import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import math


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
        is_projected =  not (heads == 1 and dim_head == dim)
        
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
        return self.to_out(out), attention


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

    def forward(self,x):
        gate = self.sigmoid(self.proj_gate(x))
        x = self.proj(x)
        return torch.mul(gate,x)

class TemporalLayer(nn.Module):
    def __init__(self, module):
        super(TemporalLayer, self).__init__()

        self.module = module

    def forward(self, x):
        b, s, _ = x.shape
        x = rearrange(x,"t n h -> (t n) h", t=b)
        x = self.module(x)
        x = rearrange(x,"(t n) h -> t n h", t=b, n=s)

        return x

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0., context_size=None, is_temporal=False):
        super(GatedResidualNetwork, self).__init__()

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

    def forward(self, x, context=None):
        if self.input_size != self.output_size:
            a = self.skip_layer(x)
        else:
            a = x

        x = self.dense1(x)
        if context is not None:
            context = self.context(context.unsqueeze(1))
            x += context

        eta_2 = self.elu(x)
        eta_1 = self.dropout(self.dense2(eta_2))

        gate = self.gate(eta_1) + a

        # Reshape if necessary
        if self.is_temporal:
            gate = gate.transpose(1, 2)  # Change to (batch_size, features, sequence_length)
        else:
            gate = gate.view(-1, self.output_size)

        x = self.layer_norm(gate)

        # Reshape back if necessary
        if self.is_temporal:
            x = x.transpose(1, 2)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model,pe, dropout=0.1,max_len=366):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        
        # Vectorized positional encoding computation
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        # Compute sine and cosine
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x.long().clamp(1, self.max_len)
        x = self.pe[x-1]
        
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self,dim,depth=1, num_heads=8, mlp_ratio=4.0, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=num_heads, dropout=drop)),
                PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout=drop)),
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = x + attn(x)[0]
            x = x + mlp(x)
        return self.norm(x)

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention using shared values for each head.
    Uses einsum and einops for efficient tensor manipulations.

    Args:
        num_attention_heads (int): Number of attention heads
        hidden_size (int): Hidden size of the model
        dropout (float): Fraction between 0 and 1 corresponding to the degree of dropout used
    """
    def __init__(self, num_attention_heads, hidden_size, dropout=0.0):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Linear layers for queries, keys, and shared values
        self.qs = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self.num_attention_heads)])
        self.ks = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(self.num_attention_heads)])
        self.vs = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # shared value layer

        self.attention = ScaledDotProductAttention()
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, query, key, value, mask=None):
        batch_size, tgt_len, embed_dim = query.shape
        head_dim = embed_dim // self.num_attention_heads

        # Project queries, keys, and values for each head using einsum for efficiency
        queries = [self.qs[i](query) for i in range(self.num_attention_heads)]
        keys = [self.ks[i](key) for i in range(self.num_attention_heads)]
        values = self.vs(value)  # shared value

        heads = []
        attentions = []

        # Process each head separately
        for q_i, k_i in zip(queries, keys):
            # Rearrange and apply attention
            q_i = rearrange(q_i, 'b t (h d) -> b h t d', h=self.num_attention_heads)
            k_i = rearrange(k_i, 'b t (h d) -> b h t d', h=self.num_attention_heads)
            v_i = rearrange(values, 'b t (h d) -> b h t d', h=self.num_attention_heads)

            head, attention = self.attention(q_i, k_i, v_i, mask)

            # Revert shape and apply dropout
            head = rearrange(head, 'b h t d -> b t (h d)')
            heads.append(self.dropout(head))
            attentions.append(attention)

        # Average over heads for interpretability
        heads = torch.stack(heads, dim=2) 
        outputs = torch.mean(heads, dim=2)  

        attentions = torch.stack(attentions, dim=2)
        attention = torch.mean(attentions, dim=2)  
        
        # Final linear transformation and dropout
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)

        return outputs, attention


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with einsum and einops for efficient tensor manipulation.
    
    Args:
        dropout (float): Fraction between 0 and 1 corresponding to the degree of dropout used
    """
    def __init__(self, dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, tgt_len, head_dim)
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, src_len, head_dim)
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, src_len, head_dim)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, tgt_len, src_len)

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, num_heads, tgt_len, head_dim)
            attention (torch.Tensor): Attention weights of shape (batch_size, num_heads, tgt_len, src_len)
        """
        # Scaled dot-product between query and key using einsum
        scaling_factor = torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float32))
        attention_scores = torch.einsum('bhtd,bhsd->bhts', query, key) / scaling_factor

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention = self.softmax(attention_scores)
        attention = self.dropout(attention)

        # Apply attention to value using einsum
        output = torch.einsum('bhts,bhsd->bhtd', attention, value)

        return output, attention

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, in_chans=1, embed_dim=128):
        super().__init__()
        self.image_size = img_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(64, 64, kernel_size=1),  
            nn.BatchNorm2d(64),  
            nn.ReLU() ,
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(128, 128, kernel_size=1),  
            nn.BatchNorm2d(128),  
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Conv2d(256, embed_dim, kernel_size=3),  
            nn.BatchNorm2d(embed_dim),  
            nn.ReLU()  
        )

        self.linear = nn.Linear(embed_dim * 4 * 4, embed_dim)

    def forward(self, x):
        b, _, _ = x.shape
        x = rearrange(x, 'b h w -> b 1 h w')
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (c h w)', b=b)
        x = self.linear(x)

        return x








