
from typing import Dict, Any, Optional, List

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
from .optimizer import get_optimizer
from .loss import *
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import copy
from torchmetrics.text import WordErrorRate
from whisper_normalizer.english import EnglishTextNormalizer
from jiwer import wer
from speechtokenizer.modules.moun import MuonWithAuxAdam
import whisper
import os
from torchaudio.functional import resample
from pytorch_lightning.utilities import grad_norm

class UniTokWrapper(pl.LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        vae: nn.Module,
        vocoder: nn.Module,
        lm_head: nn.Module,
        fn_STFT: nn.Module,
        cfg: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(cfg, ignore=['generator'])
        self.fm = ConditionalFlowMatcher(sigma=0.)
        # Models
        self.generator = generator
        # Config parameters
        self.cfg = cfg
        self.learning_rate = cfg.get("learning_rate")
        self.initial_lr = cfg.get("initial_learning_rate", self.learning_rate)
        self.epochs = cfg.get("epochs")
        self.num_warmup_steps = cfg.get("num_warmup_steps", 0)
        self.batch_size = cfg.get("batch_size")
        self.sample_rate = cfg.get('sample_rate')
        self.showpiece_num = cfg.get('showpiece_num', 16)
        self.wd = cfg.get("wd", 0.0)
        self.betas = cfg.get("betas", (0.9, 0.999))
        self.eps = cfg.get("eps", 1e-8)

        self.loss_scale_quantizer = cfg.get("loss_scale_quantizer", 1000.)
        self.training_stage = cfg.get("training_stage", 1)
        if self.training_stage == 0:
            self.lm_head = lm_head
            # self.ctc_head = ctc_head
        else:
            self.fn_STFT = fn_STFT
            self.vae = vae
            self.vocoder = vocoder
            for p in self.fn_STFT.parameters():
                p.requires_grad = False
            for p in self.vocoder.parameters():
                p.requires_grad = False
            for p in self.vae.parameters():
                p.requires_grad = False
            self.fn_STFT.eval()
            self.vae.eval()
            self.vocoder.eval()
        self.normalizer = EnglishTextNormalizer()


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=1e-4, eps=1e-3)
        base_lr = self.learning_rate
        init_lr = self.initial_lr  # 可以指定 init_lr
        optimizer = get_optimizer(
            self.parameters(),
            lr=base_lr,
            wd=self.wd,
            betas=self.betas,
            eps=self.eps
        )

        warmup_steps = self.num_warmup_steps
        total_steps = self.trainer.estimated_stepping_batches
        decay_start = int(total_steps * 0.9)

        init_coef = init_lr / base_lr

        def lr_lambda(step):
            # 1. warmup: init_lr -> base_lr
            if step < warmup_steps:
                return init_coef + (1.0 - init_coef) * (step / float(max(1, warmup_steps)))

            # 2. 恒定阶段: base_lr
            if step < decay_start:
                return 1.0

            # 3. 最后10%: 直接降到 0.1*base_lr
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        frozen_keys = {name for name, p in self.named_parameters() if not p.requires_grad}
        for k in frozen_keys:
            if k in state:
                state.pop(k)
        return state

    def load_state_dict(self, state_dict, strict=True):
        # 保留当前冻结参数
        for name, param in self.named_parameters():
            if not param.requires_grad:
                state_dict[name] = param.data
        return super().load_state_dict(state_dict, strict=strict)
    
    @torch.no_grad()
    def get_mel_from_wav(self, waveform):
        self.fn_STFT.eval()
        with torch.amp.autocast('cuda', enabled=False):
            waveform = torch.clip(waveform.float(), -1, 1)
            melspec, magnitudes, phases, energy = self.fn_STFT.mel_spectrogram(waveform)
        return melspec, magnitudes, energy
    
    @torch.no_grad()
    def sample_vae_latent(self, mel):
        self.vae.eval()
        # with torch.amp.autocast('cuda', enabled=False):
        posterior = self.vae.encode(mel)
        return posterior.mode()
    
    @torch.no_grad()
    def reconstruct_wav_from_latent(self, latent):
        self.vae.eval()
        self.vocoder.eval()
        mel = self.vae.decode(latent)
        wav = self.vocoder(mel)
        return mel, wav

    def training_step_codec(self, batch, batch_idx):
        x = batch['waveform']
        
        # 判断是DiTFM还是UniTok
        is_ditfm = hasattr(self.generator, 'codec_model')
        
        if is_ditfm:
            # DiTFM: 直接使用wav作为输入
            x_1 = self.sample_vae_latent(self.get_mel_from_wav(x.squeeze(1))[0][...,:-1])
            noise = torch.randn_like(x_1)
            t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0=noise, x1=x_1)
            
            # DiTFM forward: wav -> codes -> loss_fm
            loss_fm, codes = self.generator(x, x_t, u_t, t)
            loss_q = torch.tensor(0.0, device=x.device)  # DiTFM没有量化损失
            indices = codes[:, 0, :]  # 取第一个codebook的codes用于统计
        else:
            # UniTok: 使用mel作为输入
            x_mel = self.get_mel_from_wav(x.squeeze(1))[0][...,:-1]
            with torch.amp.autocast('cuda', enabled=False):
                x_16 = resample(x.squeeze(1).float(), self.sample_rate, 16000)
                x_mel_sem = whisper.audio.log_mel_spectrogram(x_16, n_mels=128)
            x_1 = self.sample_vae_latent(x_mel)
            noise = torch.randn_like(x_1)
            t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0=noise, x1=x_1)
            
            loss_fm, loss_q, indices = self.generator(x_mel, x_mel_sem, x_t, u_t, t)
        
        if indices is not None:
            for ind in indices.unique():
                self.codebook_count[ind] = 1
        
        train_code_util = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))

        self.log_dict({
            "train/codebook_util": train_code_util.item(),
            "train/flowmatching_loss": loss_fm.item(),
            "train/quantizer_loss": loss_q.item(),
            "train/current_lr": self.optimizers().param_groups[0]['lr']
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)
        
        return loss_fm + loss_q * self.loss_scale_quantizer
    


    def training_step_semantic(self, batch, batch_idx):
        x = batch['waveform']
        valid_len = batch['valid_len']
        with torch.amp.autocast('cuda', enabled=False):
            x_mel = whisper.audio.log_mel_spectrogram(x.squeeze(1).float(), n_mels=128)

        input_ids = batch['input_ids']
        labels = batch['labels']
        mask_text = batch['attention_mask']
        feas, indices, loss_q = self.generator(x_mel, valid_len, self.global_step)
        if indices is not None:
            for ind in indices.unique():
                self.codebook_count[ind] = 1

        loss_lm = self.lm_head(feas, valid_len, input_ids, labels, mask_text)

        train_code_util = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
        self.log_dict({
            "train/lm_loss": loss_lm.item(),

            "train/codebook_util": train_code_util.item(),
            "train/quantizer_loss": loss_q.item(),
            "train/current_lr": self.optimizers().param_groups[0]['lr']
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)
        return loss_lm + loss_q * self.loss_scale_quantizer
    
        
    def training_step_ditfm(self, batch, batch_idx):
        """DiTFM专用的训练步骤"""
        x = batch['waveform']
        
        # DiTFM: 直接使用wav作为输入
        x_1 = self.sample_vae_latent(self.get_mel_from_wav(x.squeeze(1))[0][...,:-1])
        noise = torch.randn_like(x_1)
        t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0=noise, x1=x_1)
        
        # DiTFM forward: wav -> codes -> loss_fm
        loss_fm, codes = self.generator(x, x_t, u_t, t)
        loss_q = torch.tensor(0.0, device=x.device)  # DiTFM没有量化损失，因为使用外部codec
        
        # 统计codebook使用情况 - codes shape: (B, K, T)
        indices = codes[:, 0, :]  # 取第一个codebook的codes用于统计
        
        if indices is not None:
            for ind in indices.unique():
                self.codebook_count[ind] = 1
        
        train_code_util = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))

        self.log_dict({
            "train/codebook_util": train_code_util.item(),
            "train/flowmatching_loss": loss_fm.item(),
            "train/quantizer_loss": loss_q.item(),
            "train/current_lr": self.optimizers().param_groups[0]['lr']
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)
        
        return loss_fm
        
    def training_step(self, batch, batch_idx):
        if self.training_stage == 0:
            return self.training_step_semantic(batch, batch_idx)
        elif self.training_stage == 1:
            return self.training_step_codec(batch, batch_idx)
        elif self.training_stage == 2:
            return self.training_step_ditfm(batch, batch_idx)
        
    
    def validation_step_semantic(self, batch, batch_idx):
        x = batch['waveform']
        with torch.amp.autocast('cuda', enabled=False):
            x_mel = whisper.audio.log_mel_spectrogram(x.squeeze(1).float(), n_mels=128)
        valid_len = batch['valid_len']
        mask_text = batch['attention_mask']
        feas, code, _ = self.generator(x_mel, valid_len, self.global_step, valid=True)
        feas_continue, _, _ = self.generator(x_mel, valid_len, 0)
        input_ids = batch['input_ids']
        labels = batch['labels']
        loss_lm = self.lm_head(feas, valid_len, input_ids, labels, mask_text)
        decoded_texts = self.lm_head.decode_from_latents(feas, valid_len)
        decoded_texts_continue = self.lm_head.decode_from_latents(feas_continue, valid_len)
        gt_texts = batch['texts']
        # 收集 code
        if code is not None:
            self.all_codes_epoch.append(code.detach().cpu())
        else:
            self.all_codes_epoch.append(torch.zeros(x_mel.shape[0:2]).long())

        self.decoded_epoch.extend([self.normalizer(t) for t in decoded_texts])
        self.decoded_continue_epoch.extend([self.normalizer(t) for t in decoded_texts_continue])
        self.gt_epoch.extend([self.normalizer(t) for t in gt_texts])
        if batch_idx < self.showpiece_num:

            self.logger.experiment.add_audio(
                f'groundtruth/x_{batch_idx}', 
                x[0].float().cpu().detach(), 
                global_step=self.global_step, 
                sample_rate=self.sample_rate
            )
            # log text
            self.logger.experiment.add_text(
                f'generated_text/x_{batch_idx}',
                decoded_texts[0],
                global_step=self.global_step
            )
        self.log_dict({
            "val/loss": loss_lm.item(),
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)

    def on_train_epoch_start(self):
        # 判断模型类型，获取codebook_size
        if self.training_stage == 2 or hasattr(self.generator, 'codec_model'):
            # DiTFM (training_stage=2) 或带有codec_model的模型
            # 注意：这里需要的是每个codebook的大小(cardinality)，不是codebook数量(num_codebooks)
            codebook_size = self.generator.codec_model.cardinality
        else:
            # UniTok 或 SemanticBranch: 从config获取
            codebook_size = self.cfg.get('codebook_size')
        self.codebook_count = [0] * codebook_size

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.generator, norm_type=2)
    #     self.log_dict(norms)


    def validation_step_codec(self, batch, batch_idx):
        x = batch['waveform']

        # 判断是DiTFM还是UniTok
        is_ditfm = hasattr(self.generator, 'codec_model')
        
        if is_ditfm:
            # DiTFM: 直接使用wav作为输入
            quantized, code = self.generator.encode(x)
            vae_latent = self.generator.decode(quantized)
            
            # 用于对比的mel
            x_mel = self.get_mel_from_wav(x.squeeze(1))[0][...,:-1]
        else:
            # UniTok: 使用mel作为输入
            x_mel = self.get_mel_from_wav(x.squeeze(1))[0][...,:-1]
            with torch.amp.autocast('cuda', enabled=False):
                x_16 = resample(x.squeeze(1).float(), self.sample_rate, 16000)
                x_mel_sem = whisper.audio.log_mel_spectrogram(x_16, n_mels=128)
            quantized, code = self.generator.encode(x_mel, x_mel_sem)
            vae_latent = self.generator.decode(quantized)

        x_mel_hat, x_hat = self.reconstruct_wav_from_latent(vae_latent)

        # Calculate validation metrics
        mel_error = torch.nn.functional.l1_loss(x_mel, x_mel_hat)

        # 对于DiTFM，code是(B, K, T)的形状，我们取第一个codebook
        if is_ditfm:
            code_to_save = code[:, 0, :]  # (B, T)
            # code_to_save = code
        else:
            code_to_save = code
        
        self.all_codes_epoch.append(code_to_save.detach().cpu())


        # Log audio samples for the first few batches
        if batch_idx < self.showpiece_num:
            # if not self.plot_gt_once:
                # Log ground truth once
            self.logger.experiment.add_audio(
                f'groundtruth/x_{batch_idx}', 
                x[0].float().cpu().detach(), 
                global_step=self.global_step, 
                sample_rate=self.sample_rate
            )
            self.logger.experiment.add_figure(
                f'groundtruth/x_spec_{batch_idx}', 
                plot_spectrogram(x_mel[0].float().cpu().numpy()), 
                global_step=self.global_step
            )
            
            # Log generated audio
            self.logger.experiment.add_audio(
                f'generate/x_hat_{batch_idx}', 
                x_hat[0].float().cpu().detach(), 
                global_step=self.global_step, 
                sample_rate=self.sample_rate
            )
            self.logger.experiment.add_figure(
                f'generate/x_hat_spec_{batch_idx}', 
                plot_spectrogram(x_mel_hat[0].float().cpu().numpy()), 
                global_step=self.global_step
            )
        
        self.log_dict({
            "val/loss": mel_error.item(),
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)
    

    def validation_step_ditfm(self, batch, batch_idx):
        """DiTFM专用的验证步骤"""
        x = batch['waveform']

        # DiTFM: 直接使用wav作为输入
        quantized, code = self.generator.encode(x)
        vae_latent = self.generator.decode(quantized)
        
        # 用于对比的mel
        x_mel = self.get_mel_from_wav(x.squeeze(1))[0][...,:-1]
        x_mel_hat, x_hat = self.reconstruct_wav_from_latent(vae_latent)

        # Calculate validation metrics
        mel_error = torch.nn.functional.l1_loss(x_mel, x_mel_hat)

        # DiTFM的code是(B, K, T)的形状，我们取第一个codebook用于统计
        code_to_save = code[:, 0, :]  # (B, T)
        
        self.all_codes_epoch.append(code_to_save.detach().cpu())

        # Log audio samples for the first few batches
        if batch_idx < self.showpiece_num:
            # Log ground truth
            self.logger.experiment.add_audio(
                f'groundtruth/x_{batch_idx}', 
                x[0].float().cpu().detach(), 
                global_step=self.global_step, 
                sample_rate=self.sample_rate
            )
            self.logger.experiment.add_figure(
                f'groundtruth/x_spec_{batch_idx}', 
                plot_spectrogram(x_mel[0].float().cpu().numpy()), 
                global_step=self.global_step
            )
            
            # Log generated audio
            self.logger.experiment.add_audio(
                f'generate/x_hat_{batch_idx}', 
                x_hat[0].float().cpu().detach(), 
                global_step=self.global_step, 
                sample_rate=self.sample_rate
            )
            self.logger.experiment.add_figure(
                f'generate/x_hat_spec_{batch_idx}', 
                plot_spectrogram(x_mel_hat[0].float().cpu().numpy()), 
                global_step=self.global_step
            )
        
        self.log_dict({
            "val/loss": mel_error.item(),
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)

    def on_validation_epoch_start(self):
        # 清空收集容器
        self.all_codes_epoch = []
        if self.training_stage == 0:
            self.decoded_epoch = []
            self.decoded_continue_epoch = []
            self.gt_epoch = []


    def on_validation_epoch_end(self):
        # 本地收集
        local_codes = torch.cat(self.all_codes_epoch, dim=0).reshape(-1).to(self.device)

        # 分布式收集 (gather 到所有 rank)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            tensor_list = [torch.zeros_like(local_codes) for _ in range(world_size)]
            dist.all_gather(tensor_list, local_codes)
            all_codes = torch.cat(tensor_list, dim=0)
            if self.training_stage == 0:
                # all_gather_object 能直接收集任意 Python 对象（如 list[str]）
                decoded_lists = [None for _ in range(world_size)]
                decoded_continue_lists = [None for _ in range(world_size)]
                gt_lists = [None for _ in range(world_size)]
                dist.all_gather_object(decoded_lists, self.decoded_epoch)
                dist.all_gather_object(decoded_continue_lists, self.decoded_continue_epoch)
                dist.all_gather_object(gt_lists, self.gt_epoch)

                # 展平成一个大 list
                all_decoded = sum(decoded_lists, []) if any(decoded_lists) else []
                all_decoded_continue = sum(decoded_continue_lists, []) if any(decoded_continue_lists) else []
                all_gt = sum(gt_lists, []) if any(gt_lists) else []
        else:
            all_codes = local_codes
            if self.training_stage == 0:
                all_decoded = self.decoded_epoch
                all_decoded_continue = self.decoded_continue_epoch
                all_gt = self.gt_epoch

        # 统计频率
        all_codes = all_codes.cpu().numpy()
        
        # 判断模型类型，获取codebook_size
        if self.training_stage == 2 or hasattr(self.generator, 'codec_model'):
            # DiTFM (training_stage=2) 或带有codec_model的模型
            codebook_size = self.generator.codec_model.num_codebooks
        else:
            # UniTok 或 SemanticBranch: 从config获取
            codebook_size = self.cfg.get('codebook_size')
        
        freq = np.bincount(all_codes, minlength=codebook_size)

        # 计算利用率
        utilization = (freq > 0).sum() / codebook_size

        if self.global_rank == 0:
            if self.training_stage == 0:
                # print(f'test sample: {all_decoded[0]}, {len(all_decoded)}')
                wer_score = wer(all_gt, all_decoded)
                wer_score_continue = wer(all_gt, all_decoded_continue)
                self.logger.log_metrics({"val/wer": wer_score}, step=self.global_step)
                self.logger.log_metrics({"val/wer_continue": wer_score_continue}, step=self.global_step)


            self.logger.log_metrics({"val/utilization": utilization}, step=self.global_step)

            # 绘制直方图
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(np.arange(len(freq)), freq)
            ax.set_ylim(0, freq.max()*1.1)
            ax.set_title(f"Codebook Frequency Histogram")
            ax.set_xlabel("Code Index")
            ax.set_ylabel("Frequency")
            self.logger.experiment.add_figure(
                "val/codebook_histogram", fig, global_step=self.global_step
            )
            plt.close(fig)

    def validation_step(self, batch, batch_idx):
        if self.training_stage == 0:
            return self.validation_step_semantic(batch, batch_idx)
        elif self.training_stage == 1:
            return self.validation_step_codec(batch, batch_idx)
        elif self.training_stage == 2:
            return self.validation_step_ditfm(batch, batch_idx)








