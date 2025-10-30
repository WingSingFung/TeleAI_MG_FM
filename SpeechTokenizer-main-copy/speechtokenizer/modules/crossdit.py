import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, LongTensor
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    r"""Modulate input with scale and shift.

    Args:
        x: (b, n, d)
        shift: (b, n, d)
        scale: (b, n, d)

    Outputs:
        out: (b, n, d)
    """
    return x * (1 + scale) + shift

class PreNormWithModulation(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, x, shift=0, scale=0, alpha=1, **kwargs):
        x = modulate(self.norm(x), shift, scale)
        return alpha * self.fn(x, **kwargs)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000):
        r"""Rotary position embedding.

        [1] Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary 
        position embedding." Neurocomputing, 2024

        h: head_dim
        l: seq_len
        """
        super().__init__()

        self.head_dim = head_dim

        # Calculate θ = 1 / 10000**(2i/h)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))  # (h/2,)

        # Matrix pθ
        pos_theta = torch.outer(torch.arange(max_len), theta).float()  # (l, h/2)

        # Rotation matrix
        w = torch.stack([torch.cos(pos_theta), torch.sin(pos_theta)], dim=-1)  # (l, h/2, 2)
        self.register_buffer(name="w", tensor=w)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply RoPE.

        b: batch_size
        l->n: seq_len
        n->h: heads_num
        h->d: head_dim

        Args:
            x: (b, n, h, d)

        Outputs:
            out: (b, n, h, d)
        """
        L = x.shape[1]
        x = rearrange(x, 'b n h (d c) -> b n h d c', c=2)  # (b, n, h, d/2, 2)
        w = self.w[0 : L][None, :, None, :, :]  # (1, n, 1, d/2, 2)
        x = self.rotate(x, w)  # (b, n, h, d/2, 2)
        x = rearrange(x, 'b n h d c -> b n h (d c)')  # (b, n, h, d)
        
        return x

    def rotate(self, x: Tensor, w: Tensor) -> Tensor:
        r"""Rotate x.

        x0 = cos(θp)·x0 - sin(θp)·x1
        x1 = sin(θp)·x0 + cos(θp)·x1

        b: batch_size
        n: seq_len
        h: heads_num
        d: head_dim

        Args:
            x: (b, n, h, d/2, 2)
            w: (1, n, 1, d/2, 2)

        Outputs:
            out: (b, n, h, d/2, 2)
        """

        out = torch.stack([
            w[..., 0] * x[..., 0] - w[..., 1] * x[..., 1],
            w[..., 0] * x[..., 1] + w[..., 1] * x[..., 0]
            ],
            dim=-1,
        )  # (b, n, h, d/2, 2)

        return out

    def apply_nd(self, x: Tensor, pos: LongTensor) -> Tensor:
        r"""Apply Nd RoPE with sparse positions.

        b: batch_size
        n: seq_len
        h: heads_num
        d: head_dim
        k: data dim

        Args:
            x: (b, n, h, d)
            pos: (n, k)
            n_dim: int

        Outputs:
            out: (b, n, h, d)
        """
        
        B, L, N, H = x.shape
        K = pos.shape[1]  # rope_dim
        assert H == K * self.head_dim

        x = rearrange(x, 'b n h (k d c) -> k b n h d c', k=K, c=2)  # (k, b, n, h, d/2, 2)
        x = x.contiguous()

        for i in range(K):
            p = pos[:, i]  # (l,)
            w = self.w[p][None, :, None, :, :]  # (1, n, 1, d/2, 2)
            x[i] = self.rotate(x[i], w).clone()  # x: (k, b, n, h, d/2, 2)

        out = rearrange(x, 'k b n h d c -> b n h (k d c)')  # (b, n, h, d)
        
        return out




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
        self.norm_q = nn.RMSNorm(inner_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(inner_dim, eps=1e-6)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        context: Tensor = None,
        rope: RoPE = None,
        mask: Tensor = None,
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q = self.norm_q(q)
        k = self.norm_k(k)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), (q, k, v))

        if exists(rope):
            q = rope(q)
            k = rope(k)
        
        # with torch.autocast('cuda', enabled=False):
        #     out = F.scaled_dot_product_attention(
        #         rearrange(q, 'b n h d -> b h n d').float(), 
        #         rearrange(k, 'b n h d -> b h n d').float(), 
        #         rearrange(v, 'b n h d -> b h n d').float(), 
        #         attn_mask=mask, 
        #         dropout_p=self.dropout if self.training else 0.0
        #     )
        
        out = F.scaled_dot_product_attention(
            rearrange(q, 'b n h d -> b h n d'), 
            rearrange(k, 'b n h d -> b h n d'), 
            rearrange(v, 'b n h d -> b h n d'), 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0.0
        )

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





class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_inner,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        t_cond = False, # set this true for decoder
        ca = False, # set this true for encoder
    ):
        super().__init__()

        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.ff_post = FeedForward(dim = dim, dim_inner=dim_inner, dropout = ff_dropout)
        self.attn = PreNormWithModulation(dim, self.attn)
        self.ff_post = PreNormWithModulation(dim, self.ff_post)
        self.ca = ca
        self.t_cond = t_cond
        if ca:
            self.cross_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            self.cross_attn = PreNormWithModulation(dim, self.cross_attn)
        if t_cond:
            self.t_modulation = nn.Sequential(
                Swish(),
                nn.Linear(dim, 6 * dim)
            )
        self._init_weights()
            
    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        if self.t_cond:
            nn.init.constant_(self.t_modulation[-1].weight, 0)
            nn.init.constant_(self.t_modulation[-1].bias, 0)


    def forward(self, x, t = None, context = None, rope=None, sa_mask=None, ca_mask=None):
        '''
        t: (b,dim)
        '''
        if self.t_cond:
            assert t is not None
            t = self.t_modulation(t).unsqueeze(1).chunk(6, dim=-1)
        else:
            t = [0,0,1,0,0,1]

        x = self.attn(x, t[0], t[1], t[2], rope=rope, mask=sa_mask) + x
        if self.ca:
            x = self.cross_attn(x, context=context, rope=rope, mask=ca_mask) + x
        x = self.ff_post(x, t[3], t[4], t[5]) + x
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
        self.t_modulation = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.net = PreNormWithModulation(hidden_dim, self.net)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.net.fn[0].weight, 0)
        nn.init.constant_(self.net.fn[0].bias, 0)
        nn.init.constant_(self.t_modulation[-1].weight, 0)
        nn.init.constant_(self.t_modulation[-1].bias, 0)

    def forward(self, x, t):
        t = self.t_modulation(t).unsqueeze(1).chunk(2, dim=-1)
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
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

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

class DiTDecoder(nn.Module):
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
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_dropout = 0.,
        model_type = 'vae'
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(DiTBlock(
                dim=dim,
                dim_inner=dim_inner,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                t_cond=True,
                ca=False,
            ))
        self.rope = RoPE(head_dim=dim_head)
        self.t_embedder = TimestepEmbedder(dim, scale=100.)

        adapter_stride = 2 if model_type == 'mel' else 1
        self.input_adapter = nn.Sequential(
            nn.Conv1d(latent_dim, dim, kernel_size=3, padding=1),
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=adapter_stride, padding=1),
            Rearrange('b d n -> b n d')
        )
        # cond_adapter 需要对 cond 进行下采样以匹配 VAE latent 的分辨率
        # 对于 VAE 模式: codec embedding (50 Hz) -> 下采样 2 倍 -> VAE latent space (25 Hz)
        self.cond_adapter = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Conv1d(cond_dim, dim, kernel_size=3, padding=1),
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1),  # 固定 stride=2 下采样
            Rearrange('b d n -> b n d')
        )
        self.final_layer = FinalLayer(dim, latent_dim, final_dropout)

    def forward(self, x_t, t, cond):
        '''
        x: (b, n, latent_dim)
        t: (b,)
        cond: (b, cond_dim, n)
        '''
        B, D, T = x_t.shape
        if t.dim() == 0:
            t = t.repeat(B)
        t_emb = self.t_embedder(t)
        # print(f"x_t shape: {x_t.shape}")
        # print(f"cond shape: {cond.shape}")
        x_t = self.input_adapter(x_t)
        # print(f"x_t shape after input_adapter: {x_t.shape}")
        # print(f"cond shape after cond_adapter: {self.cond_adapter(cond).shape}")
        x_t += self.cond_adapter(cond)
        

        for block in self.layers:
            x_t = block(x_t, t=t_emb, rope=self.rope)
        
        v = self.final_layer(x_t, t_emb)

        v = rearrange(v, 'b n d -> b d n')

        return v
    

class DiTEncoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        mel_dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        model_type = '50hz',
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(DiTBlock(
                dim=dim,
                dim_inner=dim_inner,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                t_cond=False,
                ca=False,
            ))
        self.rope = RoPE(head_dim=dim_head)
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
        self.post_norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, mel):
        '''
        input:
            x: (b, mel_dim, n)
            aux: (b, aux_dim, n), optional auxilary information from other modality
        return:
            x: (b, n, d)
        '''
        x = self.input_adapter(mel)
        for block in self.layers:
            x = block(x, rope=self.rope)
        x = self.post_norm(x)


        return x
    
    # @torch.no_grad()
    # def encode(self, mel, quantizer=None):
    #     x = self.input_adapter(mel)
    #     for block in self.layers:
    #         x = block(x, rope=self.rope)
    #     x = self.post_norm(x)
    #     code = quantizer(x)[1] if quantizer is not None else x
    #     return code
    
    # @torch.no_grad()
    # def decode(self, code, decode_fn=None):
    #     x = decode_fn(code) if decode_fn is not None else code
    #     return x


if __name__ == '__main__':
    model = DiTDecoder(dim=1024, dim_inner=3072, latent_dim=128, cond_dim=1024, depth=3, dim_head=64, heads=16)
    data = torch.randn((8, 128, 100))
    c = torch.randn((8, 100, 1024))
    t = torch.tensor([0,0,0,0,0,0,0,0])
    out = model(data, t, c)
    print(out.shape)