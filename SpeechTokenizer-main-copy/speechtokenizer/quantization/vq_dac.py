from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from dac.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, num_experts: int=1, router_hidden_dim: int=None):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size//num_experts, codebook_dim) for _ in range(num_experts)])
        self.router = nn.Sequential(
            nn.Linear(codebook_dim, router_hidden_dim or codebook_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim or codebook_dim, num_experts)
        )
    
    def _ste_routing(self, x, tau=1.0):
        """
        Straight-Through Estimator Routing
        Args:
            router_logits: [B, N, E]  每个 token 的专家选择 logits
            tau: 温度系数 (softmax平滑程度)
        Returns:
            router_gate: [B, N, E]  one-hot (forward)，softmax (backward)
            expert_idx: [B, N]  选中的专家索引
        """
        # softmax (有梯度)
        router_logits = self.router(x)           # [B, N, E]
        probs = F.softmax(router_logits / tau, dim=-1)  # [B, N, E]

        # 取 argmax 做 one-hot
        idx = torch.argmax(probs, dim=-1, keepdim=True)   # [B, N, 1]
        onehot = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        p = probs.mean(dim=(0,1))  # [E], 光滑平均
        f = onehot.float().mean(dim=(0,1))  # [E], top-k 负载分布

        # 2. 计算 Aux Loss
        loss = (f * p).sum()  # ∑ Fi Pi
        if torch.distributed.get_rank() == 0:
            # 1. 计算负载均衡损失
            print(f)


        # STE: forward 用 one-hot，backward 用 softmax
        # gate = onehot + (probs - probs.detach())
        gate = probs * onehot / torch.max(probs, dim=-1, keepdim=True)[0].detach()

        return gate, idx.squeeze(-1), loss

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z).transpose(-1, -2)  # z_e : (B x N x D)
        B, N, D = z_e.shape

        router_gate, expert_choice, aux_loss = self._ste_routing(z_e, tau=1.0)
        # router_gate: [B, N, E], one-hot forward, soft backward

        quantized_all = [] # receive quantized vectors with gradients flow from codebook for codebook loss
        quantized_all_ste = [] # receive quantized vectors with gradients flow from encoder for reconstruction loss
        embed_inds_all = []

        for eid, expert in enumerate(self.codebooks):
            # expert 对所有 token 做 quantization
            q_e, ind_e = self.decode_latents(z_e, eid)  # [B, N, D], [B, N]
            quantized_all.append(q_e)
            quantized_all_ste.append(z_e + (q_e - z_e).detach())
            global_embed_ind = ind_e + eid * expert.weight.shape[0]
            embed_inds_all.append(global_embed_ind)

        quantized_all = torch.stack(quantized_all, dim=-2)    # [B, N, E, D]
        quantized_all_ste = torch.stack(quantized_all_ste, dim=-2)    # [B, N, E, D]
        embed_inds_all = torch.stack(embed_inds_all, dim=-1)  # [B, N, E]

        # 根据 router_gate 选择专家 (hard forward, soft backward)
        z_q = torch.sum(router_gate.unsqueeze(-1).detach() * quantized_all, dim=-2)  # [B, N, D]
        z_q_ste = torch.sum(router_gate.unsqueeze(-1) * quantized_all_ste, dim=-2)  # [B, N, D]
        indices = torch.sum(router_gate * embed_inds_all.float(), dim=-1).long() # [B, N]


        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # z_q = (
        #     z_e + (z_q - z_e).detach()
        # )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q_ste.transpose(-1, -2))

        return z_q, commitment_loss, codebook_loss, aux_loss, indices, z_e

    def embed_code(self, embed_id, eid):
        return F.embedding(embed_id, self.codebooks[eid].weight)

    def decode_code(self, embed_id, eid):
        return self.embed_code(embed_id, eid)

    def decode_latents(self, latents, eid):
        encodings = rearrange(latents, "b n d -> (b n) d")
        codebook = self.codebooks[eid].weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b n) -> b n", b=latents.size(0))
        # indices = (-dist).max(1)[1]
        z_q = self.decode_code(indices, eid)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        vq_experts: int=8
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i], vq_experts)
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        aux_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, aux_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            aux_loss += aux_loss_i

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss, aux_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)
