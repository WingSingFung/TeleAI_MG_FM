#!/usr/bin/env python3
"""
Example training script using PyTorch Lightning for SpeechTokenizer
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from functools import partial
import argparse
import sys
import torch
from speechtokenizer import UniTok, SemanticBranch, LMHead, DiTFM
from speechtokenizer.utils import AttrDict, load_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from speechtokenizer.trainer.dataset import get_dataloader, audioDataset
from speechtokenizer.trainer.trainer_lightning import UniTokWrapper
from third_parties import AutoencoderKL
from speechtokenizer.modules.bigvgan import BigVGAN
from speechtokenizer.modules.stft import TacotronSTFT

torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.fp32_precision = "tf32"


def load_vae_vocoder(cfg):


    vocoder_cfg = cfg.get('vocoder_cfg')
    vocoder_cfg = AttrDict(vocoder_cfg)

    # loading vocoder
    bigvgan = BigVGAN(vocoder_cfg)
    bigvgan.remove_weight_norm()
    bigvgan.load_state_dict(torch.load(cfg.get("vocoder_ckpt"), map_location='cpu', weights_only=True))
    # loading vae

    cfg_vae = cfg.get('vae_cfg')

    vae = AutoencoderKL(
        embed_dim=cfg_vae['embed_dim'],
        ddconfig=cfg_vae['ddconfig'],
    )
    vae.load_state_dict(torch.load(cfg.get('vae_ckpt'), map_location="cpu", weights_only=True))

    return bigvgan, vae
    
def create_trainer_and_callbacks(cfg: Dict[str, Any], results_folder: Path):
    """Create PyTorch Lightning trainer and callbacks"""
    
    # Create results folder
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(results_folder / 'config.json', 'w+') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=results_folder,
        name="logs",
        version=None  # Automatically create unique version directories
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        # No dirpath: use default logger path: save_dir/name/version/checkpoints
        filename='SpeechTokenizer_{epoch:02d}_{step:08d}',
        save_top_k=cfg.get("num_ckpt_keep", 3),
        monitor='val/loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,  # 添加每个 epoch 保存的功能
        verbose=True       # 添加详细日志
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.get("epochs", -1),
        accelerator=cfg.get("accelerator", "auto"),
        devices=cfg.get("devices", "auto"),
        strategy=cfg.get("strategy", 'auto'),
        precision=cfg.get("precision", 32),
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        # val_check_interval=cfg.get("val_check_interval", 0.01),
        check_val_every_n_epoch=1,
        # detect_anomaly=True,
        # gradient_clip_algorithm='norm',
        # gradient_clip_val=0.1
    )
    
    return trainer

def create_data_loaders(cfg: Dict[str, Any]):
    """Setup datasets"""
    print('creating data loaders!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Setup datasets
    segment_size = cfg.get("segment_size")
    train_files = cfg.get("train_files")
    valid_files = cfg.get("valid_files")
    sample_rate = cfg.get("sample_rate")
    batch_size = cfg.get("batch_size")
    padding_mode = cfg.get("padding_mode", "zero")
    chunk_mode = cfg.get("chunk_mode", "constant")
    
    if valid_files is None:
        full_dataset = audioDataset(
            file_list=train_files,
            segment_size=segment_size,
            sample_rate=sample_rate,
            padding_mode=padding_mode,
            chunk_mode=chunk_mode
        )
        
        val_split = cfg.get("val_split", 0.001)
        total_files = len(full_dataset.audio_files)
        val_size = int(val_split * total_files)
        
        import random
        files = full_dataset.audio_files.copy()
        random.seed(cfg.get("seed", 42))
        random.shuffle(files)
        
        train_files_split = files[val_size:]
        val_files_split = files[:val_size]
        
        # 创建训练和验证数据集
        train_ds = audioDataset(
            file_list=train_files_split,
            segment_size=segment_size,
            sample_rate=sample_rate,
            valid=False,
            padding_mode=padding_mode,
            chunk_mode=chunk_mode
        )
        
        valid_ds = audioDataset(
            file_list=val_files_split,
            segment_size=max(sample_rate * 15., segment_size),
            sample_rate=sample_rate,
            valid=True,
            padding_mode=padding_mode,
            chunk_mode=chunk_mode
        )
            
        print(f"Auto-split: {len(train_ds)} train, {len(valid_ds)} validation")
        
    else:
        # 独立验证文件
        train_ds = audioDataset(
            file_list=train_files,
            segment_size=segment_size,
            sample_rate=sample_rate,
            padding_mode=padding_mode,
            chunk_mode=chunk_mode
        )
        valid_ds = audioDataset(
            file_list=valid_files,
            segment_size=max(sample_rate * 5., segment_size),
            sample_rate=sample_rate,
            valid=True,
            padding_mode=padding_mode,
            chunk_mode=chunk_mode
        )
    drop_last = cfg.get("drop_last", True)
    num_workers = cfg.get("num_workers", 72)
    train_loader = get_dataloader(
        train_ds, 
        tokenizer_name=cfg.get('lm_name'),
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    val_loader = get_dataloader(
        valid_ds,
        tokenizer_name=cfg.get('lm_name'),
        batch_size=16, 
        shuffle=False, 
        drop_last=False, 
        num_workers=8
    )
    
    print(f'Training with dataset of {len(train_ds)} samples and validating with {len(valid_ds)} samples')
    return train_loader, val_loader



def main():
    parser = argparse.ArgumentParser(description='Train SpeechTokenizer with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='/gemini/platform/public/aigc/fys/separation/TeleAI_MG_FM/SpeechTokenizer-main-copy/config/unitok_semantic.json', help='Path to config JSON file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--results_folder', type=str, default=None, help='Override results folder from config')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override results folder if provided
    if args.results_folder:
        cfg['results_folder'] = args.results_folder
    
    # Set random seed
    torch.manual_seed(cfg.get('seed', 42))
    
    # Create generator model
    if cfg.get('training_stage', 1) == 0:
        generator = SemanticBranch(cfg)
        vocoder = None
        vae = None
        lm_head = LMHead(cfg)
        fn_STFT = None
        # ctc_head = CTCHead(**cfg.get('ctc_cfg'))

    elif cfg.get('training_stage', 1) == 1:
        generator = UniTok(cfg)
        vocoder, vae = load_vae_vocoder(cfg)
        lm_head = None
        stft_config = cfg.get('stft_cfg')
        fn_STFT = TacotronSTFT(
            filter_length=stft_config['filter_length'],
            hop_length=stft_config['hop_length'],
            win_length=stft_config['win_length'],
            n_mel_channels=stft_config['n_mel_channels'],
            sampling_rate=stft_config['sampling_rate'],
            mel_fmin=stft_config['mel_fmin'],
            mel_fmax=stft_config['mel_fmax'],
        )
        # ctc_head = None
    elif cfg.get('training_stage', 1) == 2:
        generator = DiTFM(cfg)
        vocoder, vae = load_vae_vocoder(cfg)
        lm_head = None
        stft_config = cfg.get('stft_cfg')
        fn_STFT = TacotronSTFT(
            filter_length=stft_config['filter_length'],
            hop_length=stft_config['hop_length'],
            win_length=stft_config['win_length'],
            n_mel_channels=stft_config['n_mel_channels'],
            sampling_rate=stft_config['sampling_rate'],
            mel_fmin=stft_config['mel_fmin'],
            mel_fmax=stft_config['mel_fmax'],
        )

 
    model_ckpt = dict()
    if cfg.get('resume_generator', None) is not None:
        print(f'loading pre-trained generator from: {cfg.get("resume_generator")}')
        lightning_ckpt = torch.load(cfg.get('resume_generator'), map_location='cpu')
        for key in lightning_ckpt['state_dict'].keys():
            if key.startswith('generator'):
                model_ckpt[key.replace('generator.', '')] = lightning_ckpt['state_dict'][key]
        for k, v in generator.named_parameters():
            if not v.requires_grad:
                model_ckpt[k] = v

        generator.load_state_dict(model_ckpt, strict=False) # ignore quantizer
    
    results_folder = Path(cfg.get('results_folder'))
    
    # Create model
    model = UniTokWrapper(generator, vae, vocoder, lm_head, fn_STFT, cfg)
    
    # Create trainer and callbacks
    trainer = create_trainer_and_callbacks(cfg, results_folder)
    train_loader, val_loader = create_data_loaders(cfg)
    
    # Start training
    trainer.fit(model, ckpt_path=args.resume, train_dataloaders=train_loader, val_dataloaders=val_loader)
    



if __name__ == '__main__':
    main()
