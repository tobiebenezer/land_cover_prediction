import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from typing import Optional

from model.PatchTST.PatchTST_layers import positional_encoding, get_activation_fn
from model.PatchTST.RevIN import RevIN

""" 
implementation source: https://github.dev/yuqinie98/PatchTST/tree/main/PatchTST_supervised
"""

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, 
                 max_seq_len: Optional[int] = 1024, n_layers: int = 3, d_model: int = 128, n_heads: int = 16, 
                 d_k: Optional[int] = None, d_v: Optional[int] = None, d_ff: int = 256, norm: str = 'BatchNorm', 
                 attn_dropout: float = 0., dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True, 
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True, 
                 fc_dropout: float = 0., head_dropout: float = 0, padding_patch = None, pretrain_head: bool = False, 
                 head_type: str = 'flatten', individual: bool = False, revin: bool = True, affine: bool = True, 
                 subtract_last: bool = False, verbose: bool = False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        
        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                    padding_var=padding_var, attn_mask=attn_mask, res_attention=res_attention,
                                    pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z):
        # z: [bs x nvars x seq_len]
        if self.revin:
            z = rearrange(z, 'b v s -> b s v')
            z = self.revin_layer(z, 'norm')
            z = rearrange(z, 'b s v -> b v s')
        
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = rearrange(z, 'b v (p l) -> b v p l', l=self.patch_len)
        z = rearrange(z, 'b v p l -> b v l p')
        
        z = self.backbone(z)
        z = self.head(z)
        
        if self.revin:
            z = rearrange(z, 'b v s -> b s v')
            z = self.revin_layer(z, 'denorm')
            z = rearrange(z, 'b s v -> b v s')
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(head_nf, vars, 1)
        )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList([
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(nf, target_window),
                    nn.Dropout(head_dropout)
                ) for _ in range(self.n_vars)
            ])
        else:
            self.linear = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(nf, target_window),
                nn.Dropout(head_dropout)
            )
            
    def forward(self, x):
        # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            return torch.stack([linear(x[:, i]) for i, linear in enumerate(self.linears)], dim=1)
        else:
            return self.linear(x)
        
        
class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = patch_num

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(patch_num, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout, pre_norm=pre_norm,
                                  activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x):
        # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        x = rearrange(x, 'b v p l -> (b v) p l')
        x = self.W_P(x)
        
        u = self.dropout(x + self.W_pos)
        z = self.encoder(u)
        z = rearrange(z, '(b v) p d -> b v d p', v=n_vars)
        return z
    
    
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                                     norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) for _ in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(nn.BatchNorm1d(d_model), nn.Identity())
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(nn.BatchNorm1d(d_model), nn.Identity())
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        if self.store_attn:
            self.attn = attn
        
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.res_attention = res_attention
        self.lsa = lsa

        self.to_qkv = nn.Linear(d_model, (d_k + d_v + d_v) * n_heads, bias=qkv_bias)
        
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q, k, v, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

        # Reattention parameters
        self.reattn_weights = nn.Parameter(torch.randn(n_heads, n_heads))
        self.reattn_norm = nn.Sequential(
            rearrange('b h i j -> b i j h'),
            nn.LayerNorm(n_heads),
            rearrange('b i j h -> b h i j')
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        # Scaled Dot-Product Attention
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if prev is not None and self.res_attention:
            attn_scores = attn_scores + prev

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Reattention mechanism
        attn_weights = torch.einsum('bhij,hk->bkij', attn_weights, self.reattn_weights)
        attn_weights = self.reattn_norm(attn_weights)

        output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class Reattention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.Sequential(
            rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention
        attn = torch.einsum('bhij,hk->bkij', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention = _MultiheadAttention(hidden_size, num_attention_heads)

    def forward(self, x):
        return self.attention(x)[0]  # Return only the output, not the attention weights