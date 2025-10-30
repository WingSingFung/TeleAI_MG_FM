import torch
from torch import nn, einsum
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
import math
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return F.silu(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
    

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.swish = Swish()
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * self.swish(gate)
    

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    r"""Modulate input with scale and shift.

    Args:
        x: (b, t, d)
        shift: (b, t, d)
        scale: (b, t, d)

    Outputs:
        out: (b, t, d)
    """
    return x * (1 + scale) + shift

class PreNormWithModulation(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.RMSNorm(dim)

    def forward(self, x, shift=0, scale=0, alpha=1, **kwargs):
        x = modulate(self.norm(x), shift, scale)
        return alpha * self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        # self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim = dim_head)

        self.dropout = dropout

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, torch.ones(*context.shape[:2], device = device))
            mask = (mask[:, None, :, None] * context_mask[:,None, None, :]).to(torch.bool)
        else:
            mask = None
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_inner
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner*2),
            SwiGLU(dim=-1),
            # nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            # RMSNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            SwiGLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_inner,
        dim_head = 64,
        heads = 8,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        t_cond = False, # set this true for decoder
        ca = False, # set this true for encoder
        use_conv = False
    ):
        super().__init__()

        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.ff_post = FeedForward(dim = dim, dim_inner=dim_inner, dropout = ff_dropout)
        self.attn = PreNormWithModulation(dim, self.attn)
        self.ff_post = PreNormWithModulation(dim, self.ff_post)

        self.use_conv = use_conv
        self.ca = ca
        self.t_cond = t_cond
        if use_conv:
            self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
            self.ff_pre = FeedForward(dim = dim, dim_inner=dim_inner, dropout = ff_dropout)
            self.conv = PreNormWithModulation(dim, self.conv)
            self.ff_pre = PreNormWithModulation(dim, self.ff_pre)
            self.post_norm = nn.RMSNorm(dim)
        if ca:
            self.cross_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            self.cross_attn = PreNormWithModulation(dim, self.cross_attn)
        if t_cond:
            self.t_local = nn.Embedding(1, dim*6)

    def forward(self, x, t = None, context = None, mask = None, context_mask = None):
        '''
        t: (b,1,6*dim)
        '''
        if self.t_cond:
            assert t is not None
            b = t.shape[0]
            t = (t + self.t_local.weight.unsqueeze(0).repeat(b, 1, 1)).chunk(6, dim=-1)
        else:
            t = [0,0,1,0,0,1]
        ff_scale = 0.5 if self.use_conv else 1
        x = ff_scale * self.ff_pre(x) + x if self.use_conv else x
        x = self.attn(x, t[0], t[1], t[2], mask = mask) + x
        x = self.cross_attn(x, context=context, mask=mask, context_mask=context_mask) + x if self.ca else x
        x = self.conv(x) + x if self.use_conv else x
        x = ff_scale * self.ff_post(x, t[3], t[4], t[5]) + x
        x = self.post_norm(x) if self.use_conv else x
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, latent_dim, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout),
        )
        self.t_local = nn.Embedding(1, hidden_dim*2)
        self.net = PreNormWithModulation(hidden_dim, self.net)

    def forward(self, x, t):
        b, d = t.shape[0], t.shape[2]//6
        t = (t[...,:d*2] + self.t_local.weight.unsqueeze(0).repeat(b, 1, 1)).chunk(2, dim=-1)
        x = self.net(x, t[0], t[1])
        return x


class TimestepEmbedder(nn.Module):
    r"""Time step embedder.
    
    References:
    [1] https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/unet/nn.py
    [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """
    def __init__(
        self, 
        dim: int, 
        freq_size: int = 256,
        scale: float = 1.  # Use 100. for flow matching
    ):
        super().__init__()

        self.freq_size = freq_size
        self.scale = scale

        self.mlp = nn.Sequential(
            nn.Linear(freq_size, dim, bias=True),
            Swish(),
            nn.Linear(dim, dim, bias=True),
        )

    def timestep_embedding(self, t: Tensor, max_period=10000) -> Tensor:
        r"""

        Args:
            t: (b,), between 0. and 1.

        Outputs:
            embedding: (b, d)
        """
        
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half).to(t.device)  # (b,)
        args = self.scale * t[:, None] * freqs[None, :]  # (b, dim/2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (b, dim)
        
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        r"""Calculate time embedding.

        Args:
            t: (b,), between 0. and 1.

        Outputs:
            out: (b, d)
        """

        t = self.timestep_embedding(t)
        t = self.mlp(t)
        
        return t


# Conformer

class DiCDecoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        latent_dim,
        cond_dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        final_dropout = 0.,
        conv_causal = False,
        model_type = 'vae'
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim=dim,
                dim_inner=dim_inner,
                dim_head=dim_head,
                heads=heads,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                conv_causal=conv_causal,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                conv_dropout=conv_dropout,
                t_cond=True,
                ca=True,
                use_conv=False
            ))
        self.t_embedder = TimestepEmbedder(dim, scale=100.)
        self.modulation = nn.Sequential(
            Swish(),
            nn.Linear(dim, 6 * dim)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)
        adapter_stride = 2 if model_type == 'mel' else 1
        self.input_adapter = nn.Sequential(
            nn.Conv1d(latent_dim, dim, kernel_size=3, padding=1),
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=adapter_stride, padding=1),
            Rearrange('b d n -> b n d')
        )
        self.cond_adapter = nn.Sequential(
            # nn.RMSNorm(dim),
            nn.Linear(cond_dim, dim),
            Swish(),
            nn.Linear(dim, dim),
            nn.RMSNorm(dim)
        )
        self.final_layer = FinalLayer(dim, latent_dim, final_dropout)

    def forward(self, x_t, t, cond, context_mask=None):
        '''
        x: (b, n, latent_dim)
        t: (b,)
        cond: (b, cond_dim, n)
        '''
        B, D, T = x_t.shape
        if t.dim() == 0:
            t = t.repeat(B)
        t_emb = self.t_embedder(t)
        t_emb = self.modulation(t_emb.unsqueeze(1))
        x_t = self.input_adapter(x_t)
        cond = self.cond_adapter(cond)

        for block in self.layers:
            x_t = block(x_t, t=t_emb, context=cond, context_mask=context_mask)
        
        v = self.final_layer(x_t, t_emb)

        v = rearrange(v, 'b n d -> b d n')

        return v
    

class DiCEncoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        mel_dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        model_type = '50hz',
        use_conv = False,
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim=dim,
                dim_inner=dim_inner,
                dim_head=dim_head,
                heads=heads,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                conv_causal=conv_causal,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                conv_dropout=conv_dropout,
                t_cond=False,
                ca=False,
                use_conv=use_conv
            ))
        if model_type == '50hz':
            adapter_stride_1 = adapter_stride_2 = 1
        elif model_type == '25hz':
            adapter_stride_1 = 1
            adapter_stride_2 = 2
        elif model_type == '12.5hz':
            adapter_stride_1 = adapter_stride_2 = 2
        self.input_adapter = nn.Sequential(
            nn.Conv1d(mel_dim, dim, kernel_size=3, stride=adapter_stride_1, padding=1),
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=adapter_stride_2, padding=1),
            Rearrange('b d n -> b n d')
        )
        self.post_norm = nn.RMSNorm(dim) if not self.layers[-1].use_conv else nn.Identity()

    def forward(self, mel, quantizer=None):
        '''
        input:
            x: (b, mel_dim, n)
            aux: (b, aux_dim, n), optional auxilary information from other modality
        return:
            x: (b, n, d)
        '''
        x = self.input_adapter(mel)
        for block in self.layers:
            x = block(x)
        x = self.post_norm(x)

        quant_loss = torch.tensor(0., device=x.device)
        quant_out = quantizer(x)
        x, code = quant_out[0:2]
        if len(quant_out) == 3:
            quant_loss += quant_out[2].mean()
        return x, code, quant_loss
    
    @torch.no_grad()
    def encode(self, mel, quantizer=None):
        x = self.input_adapter(mel)
        for block in self.layers:
            x = block(x)
        x = self.post_norm(x)
        code = quantizer(x)[1] if quantizer is not None else x
        return code
    
    @torch.no_grad()
    def decode(self, code, decode_fn=None):
        x = decode_fn(code) if decode_fn is not None else code
        return x


if __name__ == '__main__':
    model = DiCEncoder(dim=1280, mel_dim=128, depth=12, dim_head=80, heads=16)
    data = torch.randn((8, 128, 100))
    out = model(data)
    print(out.shape)