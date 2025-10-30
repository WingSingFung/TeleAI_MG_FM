# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""


from .quantization  import ResidualVectorQuantizer
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np
from vector_quantize_pytorch import ResidualFSQ as RFSQ
from vector_quantize_pytorch import ResidualVQ, SimVQ
from .quantization.simvq import SimVQ1D
from dac.nn.quantize import ResidualVectorQuantize
# from .quantization.vq_dac import ResidualVectorQuantize
from .modules.conformer import *
import torchdiffeq



class SemanticBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.get('semantic_branch')
        self.input_adapter = nn.Sequential(
            nn.Conv1d(h['mel_dim'], h['dim'], kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv1d(h['dim'], h['dim'], kernel_size=3, stride=2, padding=1),
            Rearrange('b d n -> b n d')
        )
        self.layers = nn.ModuleList()
        for _ in range(h['depth']):
            self.layers.append(ConformerBlock(
                dim=h['dim'],
                dim_head=h['dim_head'],
                heads=h['heads'],
                t_cond=False,
                ca=False
            ))
        
        self.lm_adapter = nn.Sequential(
            nn.Linear(h['dim'], h['lm_dim']*2),
            SwiGLU(dim=-1),
            nn.Linear(h['lm_dim'], h['lm_dim']),
        )


        self.quantizer = SimVQ(
            # codebook_transform=codebook_transform,
            **config.get('quantizer_cfg')
        )
    
    def forward(self, mel, text_in, text_out):
        x = self.input_adapter(mel)

        for i, block in enumerate(self.encoder.layers):
            x = block(x)
            if i == 11:
                x, _, commit_loss = self.quantizer(x)
        








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
            self.null_cond = nn.Parameter(torch.empty(1, config['decoder_cfg']['cond_dim']))
            nn.init.normal_(self.null_cond, mean=0.0, std=1.0)

        self.encoder = DiCEncoder(
            **config.get('encoder_cfg')
        )

        self.proj_before_quant = nn.Sequential(
            nn.Linear(self.encoder.dim, self.encoder.dim),
            nn.SiLU(),
            nn.Linear(self.encoder.dim, config['decoder_cfg']['cond_dim']),
            nn.LayerNorm(config['decoder_cfg']['cond_dim'], elementwise_affine=False)
        )

        self.decoder = DiCDecoder(
            **config.get('decoder_cfg')
        )

        self.quantize_start_step = config.get('quantize_start_step', 0)

        self.vq_type = config.get('vq_type', 'rvq_dac')
        if self.vq_type == 'rvq_dac':
            self.quantizer = ResidualVectorQuantize(
                **config.get('quantizer_cfg')
            )
        elif self.vq_type == 'fsq':
            self.quantizer = RFSQ(
                **config.get('quantizer_cfg')
            )
        elif self.vq_type == 'sim_vq':
            # codebook_transform = nn.Sequential(
            #     nn.Linear(config.get('quantizer_cfg')['frozen_codebook_dim'], config.get('quantizer_cfg')['frozen_codebook_dim']),
            #     nn.ReLU(),
            #     nn.Linear(config.get('quantizer_cfg')['frozen_codebook_dim'], config.get('quantizer_cfg')['dim'])
            # )
            self.quantizer = SimVQ(
                # codebook_transform=codebook_transform,
                **config.get('quantizer_cfg')
            )
        else:
            raise ValueError(f"Unknown vq type: {config.get('vq_type')}")
   
        

    
    def forward(self, 
                mel: torch.tensor, 
                x_t: torch.tensor,
                u_t: torch.tensor,
                t: torch.tensor,
                global_step: float=np.inf
                ):
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
        e = self.encoder(mel)
        e = self.proj_before_quant(e)

        quantizer_loss = torch.tensor(0.).to(e.device)
        if global_step >= self.quantize_start_step:
            if self.vq_type == 'rvq_dac':
                e, _, _, commitment_loss, codebook_loss = self.quantizer(e)
                quantizer_loss += (codebook_loss + 0.25*commitment_loss)
            elif self.vq_type == 'fsq':
                e, _, = self.quantizer(e)
            elif self.vq_type == 'sim_vq':
                e, _, commit_loss = self.quantizer(e)
                quantizer_loss += commit_loss

        if self.cfg_drop_rate > 0:
            # mask: (B, 1, 1)，保证在 batch 维度做 drop
            rand_vals = torch.rand(e.shape[0], device=e.device)
            mask = (rand_vals > self.cfg_drop_rate).float().view(-1, 1, 1)

            null_cond = self.null_cond.unsqueeze(0)

            # 替换
            e = e * mask + null_cond.expand(e.shape[0], e.shape[1], -1) * (1 - mask)
        v_t = self.decoder(x_t, t, e)
        loss_fm = torch.mean((v_t - u_t) ** 2)
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
        e = self.proj_before_quant(e)
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
        def cfg_decoder(x, t, cond, decoder, guidance_scale):
            # 有条件预测
            f_cond = decoder(x, t, cond)

            if guidance_scale == 1.0:
                return f_cond

            # 无条件预测：用 null_cond 扩展到 (B, dim, L)
            null_cond = self.null_cond.unsqueeze(0)  # (1, 1, dim)
            f_uc = decoder(x, t, null_cond.expand(cond.shape[0], cond.shape[1], -1))

            # CFG 组合
            return f_uc + guidance_scale * (f_cond - f_uc)
        if global_step >= self.quantize_start_step:
            if self.vq_type == 'fsq':
                quantized = self.quantizer.get_output_from_indices(codes)
            elif self.vq_type == 'rvq_dac':
                quantized = self.quantizer.from_codes(codes)
            elif self.vq_type == 'sim_vq':
                quantized = self.quantizer.indices_to_codes(codes)
        else:
            quantized = continues_latent
        quantized = quantized.float()
        noise = torch.randn(quantized.shape[0], self.decoder.latent_dim, int(25*audio_dur)).type_as(quantized)
        traj = torchdiffeq.odeint(
                lambda t, x: cfg_decoder(x, t, quantized, self.decoder, self.cfg_guidance_scale),
                y0=noise,
                t=torch.linspace(0, 1, 10, device=quantized.device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
        return traj[-1]
    
        
