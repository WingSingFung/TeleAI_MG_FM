from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from functools import partial
import argparse
import sys
import torch
sys.path.append('/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main')
from speechtokenizer.trainer.dataset_ import get_dataloader, audioDataset
from speechtokenizer.utils import AttrDict, load_config
from speechtokenizer.modules.stft import TacotronSTFT
import os
import numpy as np
from tqdm import tqdm
from third_parties import AutoencoderKL

def load_vae_vocoder(cfg):


    cfg_vae = cfg.get('vae_cfg')

    vae = AutoencoderKL(
        embed_dim=cfg_vae['embed_dim'],
        ddconfig=cfg_vae['ddconfig'],
    )
    vae.load_state_dict(torch.load(cfg.get('vae_ckpt'), map_location="cpu"))

    return vae

@torch.inference_mode()
def sample_vae_latent(vae, mel):
    vae.eval()
    posterior = vae.encode(mel)
    return posterior.mode()

@torch.inference_mode()
def get_mel_from_wav(fn_STFT, waveform):
    with torch.amp.autocast('cuda', enabled=False):
        waveform = torch.clip(waveform.float(), -1, 1)
        melspec, magnitudes, phases, energy = fn_STFT.mel_spectrogram(waveform)
    return melspec, magnitudes, energy

if __name__ == '__main__':
    train_files = "/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/data_processing/audioset_train.txt"
    dataset = audioDataset(
        file_list=train_files,
        segment_size=320000,
        sample_rate=32000
    )
    dataloader = get_dataloader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        drop_last=False, 
        num_workers=64
    )
    cfg = load_config('/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/config/unitok.json')
    vae = load_vae_vocoder(cfg)
    vae.eval()
    vae.cuda()
    stft_config = cfg.get('stft_config')
    fn_STFT = TacotronSTFT(
        filter_length=stft_config['filter_length'],
        hop_length=stft_config['hop_length'],
        win_length=stft_config['win_length'],
        n_mel_channels=stft_config['n_mel_channels'],
        sampling_rate=stft_config['sampling_rate'],
        mel_fmin=stft_config['mel_fmin'],
        mel_fmax=stft_config['mel_fmax'],
    )
    fn_STFT.cuda()

    save_dir = "/gemini/platform/public/aigc/mh-data/audioset_vae/train_latents_64"
    os.makedirs(save_dir, exist_ok=True)

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            wav = batch['waveform'].cuda()
            wav_paths = batch['file_path']
            
            # mel
            x_mel = get_mel_from_wav(fn_STFT, wav.squeeze(1))[0][...,:-1]
            # latent
            vae_latents = sample_vae_latent(vae, x_mel)  # (B,...)

            vae_latents = vae_latents.cpu().to(torch.float).numpy()
            for latent, path in zip(vae_latents, wav_paths):
                fname = os.path.splitext(os.path.basename(path))[0] + ".npy"
                fpath = os.path.join(save_dir, fname)
                np.save(fpath, latent)