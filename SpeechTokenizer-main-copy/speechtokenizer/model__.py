# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""


from .quantization  import ResidualVectorQuantizer
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
import numpy as np
from vector_quantize_pytorch import ResidualFSQ as RFSQ
from vector_quantize_pytorch import ResidualVQ, SimVQ
from dac.nn.quantize import ResidualVectorQuantize
# from .quantization.vq_dac import ResidualVectorQuantize
from .modules.conformer import DiCEncoder, DiCDecoder
import torchdiffeq


class UniTok(nn.Module):
    def __init__(self, config):
        '''
        
        Parameters
        ----------
        config : json
            Model Config.

        '''
        super().__init__()

        self.cfg_drop_rate = config.get('cfg_drop_rate', 0.)
        self.cfg_guidance_scale = config.get('cfg_guidance_scale', 1.)
        if self.cfg_drop_rate > 0:
            self.null_cond = nn.Embedding(1, config['decoder_cfg']['cond_dim'])

        self.decoder = DiCDecoder(
            **config.get('decoder_cfg')
        )
   
        

    
    def forward(self, 
                mel: torch.tensor, 
                x_t: torch.tensor,
                u_t: torch.tensor,
                t: torch.tensor,
                aux: torch.tensor=None,
                context_mask: torch.tensor=None,
                global_step: float=np.inf,
                layers: list=[0]):
        '''
        
        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        v : torch.tensor
            Input semantic vectors. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is the first layer.

        Returns
        -------
        o : torch.tensor
            Output wavs. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from residual vector quantizers.
        feature : torch.tensor
            Output of RVQ's first layer. Shape: (batch, timesteps, dimension)

        '''
        e = aux
        if self.cfg_drop_rate > 0:
            # mask: (B, 1, 1)，保证在 batch 维度做 drop
            rand_vals = torch.rand(e.shape[0], device=e.device)
            mask = (rand_vals > self.cfg_drop_rate).float().view(-1, 1, 1)

            null_cond = F.normalize(self.null_cond.weight.unsqueeze(0), dim=2)  # mock rmsnorm

            # 替换
            e = e * mask + null_cond.expand(e.shape[0], e.shape[1], -1) * (1 - mask)
        v_t = self.decoder(x_t, t, e, context_mask)
        loss_fm = torch.mean((v_t - u_t) ** 2)
        quantizer_loss = torch.tensor(0.).type_as(loss_fm)
        return loss_fm, quantizer_loss
    
    
    
    def encode(self, 
               x_mel: torch.tensor
               ):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (n_q, batch, timesteps)

        '''
        e = self.encoder(x_mel)
        if self.vq_type == 'fsq':
            quantized, code = self.quantizer(e)
        elif self.vq_type == 'rvq_dac':
            quantized, code, _, _, _ = self.quantizer(e)
        elif self.vq_type == 'sim_vq':
            quantized, code, _ = self.quantizer(e)
        return code, e, quantized
    
    def decode(self, 
               codes: torch.tensor,
               continues_latent,
               context_mask,
               global_step,
               audio_dur=10.
               ):
        '''

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        '''
        # quantized, _, _ = self.quantizer.from_codes(codes)
        def cfg_decoder(x, t, cond, context_mask, decoder, guidance_scale):
            # 有条件预测
            f_cond = decoder(x, t, cond, context_mask)

            if guidance_scale == 1.0:
                return f_cond

            # 无条件预测：用 null_cond 扩展到 (B, dim, L)
            # null_cond = self.null_cond.weight.unsqueeze(0).transpose(1, 2)  # (1, dim, 1)
            # f_uc = decoder(x, t, null_cond.expand(cond.shape[0], -1, cond.shape[2]))
            null_cond = F.normalize(self.null_cond.weight.unsqueeze(0), dim=2)  # (1, 1, dim)
            f_uc = decoder(x, t, null_cond.expand(cond.shape[0], cond.shape[1], -1))

            # CFG 组合
            return f_uc + guidance_scale * (f_cond - f_uc)
        
        quantized = continues_latent
        noise = torch.randn(quantized.shape[0], 32, int(25*audio_dur)).type_as(quantized)
        traj = torchdiffeq.odeint(
                lambda t, x: cfg_decoder(x, t, quantized, context_mask, self.decoder, self.cfg_guidance_scale),
                y0=noise,
                t=torch.linspace(0, 1, 10, device=quantized.device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        return traj[-1]
    
        
