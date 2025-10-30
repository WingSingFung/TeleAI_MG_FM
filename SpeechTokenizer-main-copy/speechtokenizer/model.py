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
from vector_quantize_pytorch import FSQ
# from vector_quantize_pytorch import ResidualVQ, SimVQ
from .quantization.simvq import SimVQ1D as SimVQ
from .quantization.core_vq_ddp import VectorQuantization as VQEMA
from dac.nn.quantize import ResidualVectorQuantize
# from .quantization.vq_dac import ResidualVectorQuantize
from .modules.crossdit import *
from .modules.whisper_encoder import WhisperEncoder
import torchdiffeq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import whisper
from torch.nn.utils.parametrizations import weight_norm



def get_prefix_mask(valid_second, seq_len, frame_rate, offset=0):
    valid_frames = (valid_second * frame_rate).long() + offset
    mask = (torch.arange(seq_len)[None, :].to(valid_frames.device) < valid_frames[:, None]).float()
    return mask
    




class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.lm = AutoModelForCausalLM.from_pretrained(config.get('lm_name'))
        self.lm = get_peft_model(self.lm, lora_config)
        embedding_dim = self.lm.get_input_embeddings().embedding_dim
        # for p in self.lm.parameters():
        #     p.requires_grad = False
        self.lm_adapter = nn.Sequential(
            nn.Linear(config.get('semantic_quantizer_cfg')['dim'], embedding_dim),
        )
        self.sep_token = nn.Embedding(1, embedding_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('lm_name'))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def forward(self, x, valid_len, text_in, labels, mask_text):
        lm_in = []
        lm_prefix = self.lm_adapter(x)
        device = x.device
        for i, lm_prefix_i in enumerate(lm_prefix):
            prefix_len_i = (valid_len[i] * 12.5).long() + 1
            lm_prefix_i = lm_prefix_i[:prefix_len_i,:]
            text_in_i = text_in[i,:mask_text[i].sum()]
            text_emb_i = self.lm.get_input_embeddings()(text_in_i)
            lm_in_i = torch.concat([lm_prefix_i, self.sep_token.weight.unsqueeze(0), text_emb_i], dim=0)
            prefix_ignore_i = torch.full((prefix_len_i + 1), -100, dtype=labels.dtype, device=device)
            label_i = torch.concat([prefix_ignore_i, labels[i,:mask_text[i].sum()]], dim=0)



        # 6. 调用 LM
        outputs = self.lm(inputs_embeds=lm_in, labels=labels)
        return outputs['loss']
    
    @torch.no_grad()
    def decode_from_latents(self, x, valid_len, max_gen_len=128, num_beams=4, do_sample=False, **gen_kwargs):
        """
        使用 HuggingFace generate 直接生成
        """
        self.eval()

        # latent -> prefix embedding
        lm_prefix = self.lm_adapter(x)  # [B, T_prefix, D]
        lm_prefix = torch.cat(
            [lm_prefix, self.sep_token.weight.unsqueeze(0).repeat(lm_prefix.shape[0], 1, 1)],
            dim=1
        )
        B, T_prefix, D = lm_prefix.shape
        mask = get_prefix_mask(valid_len, T_prefix, 12.5)
        mask[:, -1] = 1


        # 调用 generate（直接用 inputs_embeds）
        generated_ids = self.lm.generate(
            inputs_embeds=lm_prefix,
            attention_mask=mask,
            max_new_tokens=max_gen_len,   # 比 max_length 更直观
            num_beams=num_beams,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs
        )

        # 解码
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)



class SemanticBranch(nn.Module):
    def __init__(self, config):
        super().__init__()
        whisper_model = whisper.load_model('/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/large-v3.pt', device='cpu')
        self.encoder = WhisperEncoder(n_mels=whisper_model.dims.n_mels, 
                                      n_ctx=whisper_model.dims.n_audio_ctx, 
                                      n_head=whisper_model.dims.n_audio_head, 
                                      n_layer=whisper_model.dims.n_audio_layer, 
                                      n_state=whisper_model.dims.n_audio_state
                                    )
        self.encoder.load_state_dict(whisper_model.encoder.state_dict())

        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        self.quant_bottleneck = nn.Sequential(
            Rearrange('b (t r) d -> b t (r d)', r=4),
            nn.Linear(whisper_model.dims.n_audio_state*4, config.get('semantic_quantizer_cfg')['dim']),
            nn.RMSNorm(config.get('semantic_quantizer_cfg')['dim'], eps=1e-6)
        )

        codebook_transform = nn.Sequential(
            nn.Linear(config.get('semantic_quantizer_cfg')['frozen_codebook_dim'],  config.get('semantic_quantizer_cfg')['dim']*2),
            nn.ReLU(),
            nn.Linear(config.get('semantic_quantizer_cfg')['dim']*2,  config.get('semantic_quantizer_cfg')['dim']),
        )

        self.quantizer = SimVQ(
            codebook_transform=codebook_transform,
            **config.get('semantic_quantizer_cfg')
        )

        self.quantize_warmup_step = config.get('quantize_warmup_step', 0)

    def forward_encoder(self, x, valid_len, global_step, valid):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        mask_enc = get_prefix_mask(valid_len, (x.shape[2] - 1) // 2 + 1, 50)
        x = self.encoder(x, mask_enc)

        x = self.quant_bottleneck(x)
        # x_quant, code, quant_loss = self.quantizer(x)
        quant_out = self.quantizer(x)
        x_quant, code = quant_out[0:2]
        quant_loss = quant_out[2] if len(quant_out) > 2 else torch.tensor(0.0, device=x.device)
        
        # quant_loss = (quant_loss.mean(-1) * mask_loss).sum() / mask_loss.sum()

        # if valid:
        #     return x_quant, code, quant_loss.mean()

        # p = self._get_p(global_step)
        p = 0

        B, T, _ = x.shape

        mask_fea = (torch.rand((B, T), device=x.device) < p).float().view(B, T, 1)

        # 混合特征
        x = mask_fea * x_quant + (1 - mask_fea) * x
        mask_loss = get_prefix_mask(valid_len, quant_loss.shape[1], 12.5) * mask_fea.squeeze(-1)

        # # 根据mask缩放quant_loss（只对被选中量化的样本生效）
        if mask_loss.sum() != 0:
            quant_loss = (quant_loss.mean(-1) * mask_loss.squeeze(-1)).sum() / mask_loss.sum()
        else:
            quant_loss = (quant_loss.mean(-1) * mask_loss.squeeze(-1)).sum()

        return x, code, quant_loss
    
    def forward(self, mel, valid_len, global_step=np.inf, valid=False):
        x_out, code, quant_loss = self.forward_encoder(mel, valid_len, global_step, valid)

        return x_out, code, quant_loss
    
    def _get_p(self, step):
        # 如果 step 未达到 warmup 步数，则线性上升
        if step < self.quantize_warmup_step:
            p = 1 * (step / self.quantize_warmup_step)
        else:
            p = 1
        return p



class MergeMdoule(nn.Module):
    def __init__(self, in_dim_semantic, in_dim_acoustic, out_dim, acoustic_expand_factor=2):
        super().__init__()
        self.semantic_adapter = nn.Sequential(
            nn.Linear(in_dim_semantic, in_dim_acoustic*acoustic_expand_factor),
            Rearrange('b t (r d) -> b (t r) d', r=acoustic_expand_factor),
            nn.RMSNorm(in_dim_acoustic, eps=1e-6)
        )
        assert out_dim == in_dim_acoustic
        self.post_norm = nn.RMSNorm(out_dim, eps=1e-6)
    
    def forward(self, x_sem, x_acoust):
        x_sem = self.semantic_adapter(x_sem)
        merged = self.post_norm(x_acoust + x_sem)
        return merged





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
            self.null_cond = nn.Parameter(torch.randn(1, config['decoder_cfg']['cond_dim']) / config['decoder_cfg']['cond_dim'] ** 0.5)

        # semantic_states = torch.load(config.get('semantic_ckpt'), map_location='cpu')['state_dict']
        # semantic_states_ = dict()
        # for k, v in semantic_states.items():
        #     if k.startswith('generator'):
        #         semantic_states_[k.replace('generator.', '')] = v
        # self.semantic_branch = SemanticBranch(config)
        # for name, param in self.semantic_branch.named_parameters():
        #     if not param.requires_grad:
        #         semantic_states_[name] = param

        # self.semantic_branch.load_state_dict(semantic_states_)
        # for p in self.semantic_branch.parameters():
        #     p.requires_grad = False


        self.acoustic_branch = DiTEncoder(
            **config.get('acoustic_cfg')
        )

        self.quant_bottleneck = nn.Sequential(
            weight_norm(nn.Linear(config.get('acoustic_cfg')['dim'], config.get('quantizer_cfg')['e_dim'])),
        ) # necessary for simvq training

        self.quantizer = SimVQ(
            **config.get('quantizer_cfg')
        )

        self.decoder = DiTDecoder(
            **config.get('decoder_cfg')
        )

        # self.merge_adapter = MergeMdoule(
        #     in_dim_semantic=config.get('semantic_quantizer_cfg')['dim'],
        #     in_dim_acoustic=config.get('quantizer_cfg')['dim'],
        #     out_dim=config.get('decoder_cfg')['cond_dim']
        # )

        

    
    def forward(self, 
                mel: torch.tensor, 
                mel_sem: torch.tensor, 
                x_t: torch.tensor,
                u_t: torch.tensor,
                t: torch.tensor,
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
        # e, quantizer_loss = self.encoder(mel, quantizer=self.quantizer, quant_layer=9)
        # with torch.no_grad():
        #     e_sem, _, quantizer_loss = self.semantic_branch(mel_sem, valid=True)
        x = self.acoustic_branch(mel)
        quant_out = self.quantizer(self.quant_bottleneck(x))
        quant_loss = torch.tensor(0., device=x.device)
        e, code = quant_out[0:2]
        if len(quant_out) == 3:
            quant_loss += quant_out[2].mean()


        if self.cfg_drop_rate > 0:
            # mask: (B, 1, 1)，保证在 batch 维度做 drop
            rand_vals = torch.rand(e.shape[0], device=e.device)
            mask = (rand_vals > self.cfg_drop_rate).float().view(-1, 1, 1)

            null_cond = self.null_cond.unsqueeze(0)

            # 替换
            e = e * mask + null_cond.expand(e.shape[0], e.shape[1], -1) * (1 - mask)
        v_t = self.decoder(x_t, t, e)
        loss_fm = torch.mean((v_t - u_t) ** 2)
        return loss_fm, quant_loss, code
    
    
    
    def encode(self, 
               x_mel: torch.tensor,
               x_mel_sem: torch.tensor, 
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
        
        # e_sem, sem_code, _ = self.semantic_branch(x_mel_sem, valid=True)
        x = self.acoustic_branch(x_mel)
        quant_out = self.quantizer(self.quant_bottleneck(x))
        e, code = quant_out[0:2]
        return e, code
    
    def decode(self, 
               quantized: torch.tensor,
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
    
        
class DiTFM(nn.Module):
    """
    要求：
    1. 编码和量化部分使用别的codec获得code，再使用DiTDecoder进行解码
    2. DiTDecoder部分要和UniTok的Decoder部分一致，loss部分的计算去除量化损失quant_loss
    3. codec从外部传入，提取code部分参考codes, scale = self.codec_model.encode(wav)
    """
    def __init__(self, config):
        '''
        Parameters
        ----------
        config : dict
            Model Config. 需要包含:
            - codec_model_path: str, codec模型的路径
            - decoder_cfg: dict, DiTDecoder的配置
            - cfg_drop_rate: float, classifier-free guidance的drop rate
            - cfg_guidance_scale: float, classifier-free guidance的scale
        '''
        super().__init__()
        
        self.cfg_drop_rate = config.get('cfg_drop_rate', 0.)
        self.cfg_guidance_scale = config.get('cfg_guidance_scale', 1.)
        
        # 加载外部codec模型（使用audiocraft的加载方式）
        from pathlib import Path
        import sys
        sys.path.append('/gemini/platform/public/aigc/fys/separation/mg')
        # from audiocraft.models.loaders import load_compression_model
        from audiocraft.solvers.compression import CompressionSolver
        codec_path = config.get('codec_model_path')
        device = config.get('device', 'cpu')
        sample_rate = config.get('sample_rate')
        self.num_codebooks = config.get('n_q')
        # self.codec_model = load_compression_model(codec_path, device=device)
        self.codec_model = CompressionSolver.model_from_checkpoint(
            codec_path, device=device)
        self.codec_model.set_num_codebooks(self.num_codebooks)
        assert self.codec_model.sample_rate == sample_rate, (
            f"Codec model sample rate is {self.codec_model.sample_rate} but "
            f"Solver sample rate is {sample_rate}."
            )
        assert self.codec_model.sample_rate == sample_rate, \
            f"Sample rate of solver {sample_rate} and codec {self.codec_model.sample_rate} " \
            "don't match."
        
        # 冻结codec模型参数
        for p in self.codec_model.parameters():
            p.requires_grad = False
        self.codec_model.eval()
        
        # 获取codec的输出维度
        # encodec的codes shape: (B, K, T), K是quantizer数量
        # 我们需要将codes转换为embedding
        # self.codec_dim = self.codec_model.model.quantizer.dimension
        # self.codec_dim = 256
        # 创建一个dummy codes来推断维度
        dummy_codes = torch.zeros(1, self.num_codebooks, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            latent = self.codec_model.decode_latent(dummy_codes)  # (B, D, T)
            self.codec_dim = latent.shape[1]
            print(f"Codec dimension: {self.codec_dim}")
        
        
        # 将codec的量化输出映射到decoder的条件维度
        self.codec_adapter = nn.Sequential(
            nn.Linear(self.codec_dim, config.get('decoder_cfg')['cond_dim']),
            nn.RMSNorm(config.get('decoder_cfg')['cond_dim'], eps=1e-6)
        )
        
        # Decoder部分（和UniTok一致）
        self.decoder = DiTDecoder(
            **config.get('decoder_cfg')
        )
        
        # Classifier-free guidance的null condition
        if self.cfg_drop_rate > 0:
            self.null_cond = nn.Parameter(
                torch.randn(1, config['decoder_cfg']['cond_dim']) / config['decoder_cfg']['cond_dim'] ** 0.5
            )
    
    def forward(self, 
                wav: torch.tensor,
                x_t: torch.tensor,
                u_t: torch.tensor,
                t: torch.tensor,
                ):
        '''
        Parameters
        ----------
        wav : torch.tensor
            Input waveforms. Shape: (batch, channels, timesteps).
        x_t : torch.tensor
            Noisy latent at time t. Shape: (batch, latent_dim, T).
        u_t : torch.tensor
            Target velocity. Shape: (batch, latent_dim, T).
        t : torch.tensor
            Time step. Shape: (batch, 1).

        Returns
        -------
        loss_fm : torch.tensor
            Flow matching loss.
        codes : torch.tensor
            Quantized codes from codec model.
        '''
        # 使用外部codec进行编码
        with torch.no_grad():
            codes, scale = self.codec_model.encode(wav)
            # codes shape: (B, K, T), K是quantizer数量
            # 使用codec的dequantize获取embedding
            # e = self.codec_model.quantizer.decode(codes)  # (B, D, T)
            e = self.codec_model.decode_latent(codes)
            e = e.transpose(1, 2)  # (B, T, D)
        
        # 映射到decoder的条件空间
        e = self.codec_adapter(e)  # (B, T, cond_dim)
        
        # Classifier-free guidance: 随机drop条件
        if self.cfg_drop_rate > 0 and self.training:
            rand_vals = torch.rand(e.shape[0], device=e.device)
            mask = (rand_vals > self.cfg_drop_rate).float().view(-1, 1, 1)
            null_cond = self.null_cond.unsqueeze(0)
            e = e * mask + null_cond.expand(e.shape[0], e.shape[1], -1) * (1 - mask)
        
        # Decoder预测velocity
        v_t = self.decoder(x_t, t, e)
        
        # Flow matching loss
        loss_fm = torch.mean((v_t - u_t) ** 2)
        
        return loss_fm, codes
    
    def encode(self, wav: torch.tensor):
        '''
        使用外部codec编码音频
        
        Parameters
        ----------
        wav : torch.tensor
            Input waveforms. Shape: (batch, channels, timesteps).

        Returns
        -------
        quantized : torch.tensor
            Quantized embedding. Shape: (batch, T, cond_dim)
        codes : torch.tensor
            Quantized codes. Shape: (batch, K, T)
        '''
        with torch.no_grad():
            codes, scale = self.codec_model.encode(wav)
            # e = self.codec_model.quantizer.decode(codes)  # (B, D, T)
            e = self.codec_model.decode_latent(codes)
            e = e.transpose(1, 2)  # (B, T, D)
        
        # 映射到decoder的条件空间
        quantized = self.codec_adapter(e)
        
        return quantized, codes
    
    def decode(self, quantized: torch.tensor, audio_dur=10.):
        '''
        从量化的embedding解码音频latent
        
        Parameters
        ----------
        quantized : torch.tensor
            Quantized embedding. Shape: (batch, T, cond_dim)
        audio_dur : float
            音频时长（秒）

        Returns
        -------
        traj : torch.tensor
            Reconstructed latent. Shape: (batch, latent_dim, T_vae)
        '''
        def cfg_decoder(x, t, cond, decoder, guidance_scale):
            # 有条件预测
            f_cond = decoder(x, t, cond)
            
            if guidance_scale == 1.0:
                return f_cond
            
            # 无条件预测：用 null_cond
            null_cond = self.null_cond.unsqueeze(0)  # (1, 1, dim)
            f_uc = decoder(x, t, null_cond.expand(cond.shape[0], cond.shape[1], -1))
            
            # CFG 组合
            return f_uc + guidance_scale * (f_cond - f_uc)
        
        # VAE latent 的 frame_rate 是 25 Hz (mel 的 50 Hz 下采样 2 倍)
        # 根据音频时长计算 VAE latent 的序列长度
        vae_latent_length = int(25 * audio_dur)
        noise = torch.randn(quantized.shape[0], self.decoder.latent_dim, vae_latent_length).type_as(quantized)
        traj = torchdiffeq.odeint(
            lambda t, x: cfg_decoder(x, t, quantized, self.decoder, self.cfg_guidance_scale),
            y0=noise,
            t=torch.linspace(0, 1, 10, device=quantized.device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        return traj[-1]