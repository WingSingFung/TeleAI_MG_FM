import argparse
import torchaudio
import torch
from speechtokenizer import SpeechTokenizer
from scipy.io.wavfile import write
import numpy as np
import json
import os
import shutil

# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")


# Set up argument parser
parser = argparse.ArgumentParser(
    description="Load SpeechTokenizer model and process audio file."
)
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the model configuration file.",
    default="/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/exps/dac_4vq0826/config.json",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="Path to the model checkpoint file.",
    default="/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/exps/dac_4vq0826/logs/version_6/checkpoints/last.ckpt",
)
parser.add_argument(
    "--speech_file",
    type=str,
    default='/gemini/platform/public/aigc/mh-data/audioset/AudioSet/data/audio/eval/0izHOfrwPn4.flac',
    help="Path to the speech file to be processed.",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="Path to save the output audio file.",
    default="my_dac_fine_highfreq.wav",
)

args = parser.parse_args()

# Load model from the specified checkpoint
with open(args.config_path) as f:
    cfg = json.load(f)
model = SpeechTokenizer(cfg)
lightning_ckpt = torch.load(args.ckpt_path, map_location='cpu')
model_ckpt = dict()
for key in lightning_ckpt['state_dict'].keys():
    if key.startswith('generator'):
        model_ckpt[key.replace('generator.', '')] = lightning_ckpt['state_dict'][key]

model.load_state_dict(model_ckpt, strict=True)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

with open('/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/data_processing/audioset_eval_data_subset.json') as f:
    all_items = json.load(f)

source_root = "/gemini/platform/public/aigc/mh-data/audioset/AudioSet/data/audio/eval"
tgt_root = '/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/eval_audioset_4'
oracle_root = '/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/oracle_audioset'

all_items = all_items['data']
for item in all_items:
    filename = os.path.basename(item['wav'])
    filename = filename[1:].replace('.wav', '.flac')
    input_path = os.path.join(source_root, filename)
    try:
        output_path = os.path.join(tgt_root, filename)
        oracle_path = os.path.join(oracle_root, filename)
        shutil.copy(input_path, oracle_path)

        model.eval()

        # Determine the model's expected sample rate
        model_sample_rate = model.sample_rate

        # Load and preprocess speech waveform with the model's sample rate
        wav, sr = torchaudio.load(input_path)

        if sr != model_sample_rate:
            resample_transform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=model_sample_rate
            )
            wav = resample_transform(wav)

        # Ensure the waveform is monophonic
        if wav.shape[0] > 1:
            wav = wav[:1, :]

        wav = wav.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        max_val = float(wav.abs().max())
        wav *= (0.5/max_val)


        # Extract discrete codes from SpeechTokenizer
        with torch.inference_mode():
            codes = model.encode(wav)  # codes: (B, n_q, T)

            RVQ_1 = codes[:2, :, :]  # Contain content info, can be considered as semantic tokens
            RVQ_supplement = codes[
                1:, :, :
            ]  # Contain timbre info, complete info lost by the first quantizer

            # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
            wav_out = model.decode(torch.cat([RVQ_1], axis=1))
            wav_out = torchaudio.transforms.Resample(
                    orig_freq=model_sample_rate, new_freq=sr
                )(wav_out.detach().cpu())*(max_val/0.5)

        # Decoding from RVQ-i:j tokens from the ith quantizers to the jth quantizers
        # Example: decoding from quantizer 0 to quantizer 2
        wav_out = wav_out.numpy()
        write(output_path, sr, wav_out.astype(np.float32))
    except Exception as e:
        print(e)
        continue
