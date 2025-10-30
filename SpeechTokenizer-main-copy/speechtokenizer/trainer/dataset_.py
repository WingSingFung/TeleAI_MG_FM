from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer
import csv

def get_dataloader(ds, tokenizer_name, **kwargs):
    return DataLoader(ds, collate_fn=SemanticCollator(tokenizer_name=tokenizer_name), **kwargs)

class audioDataset(Dataset):
    
    def __init__(self,
                 file_list,
                 segment_size,
                 sample_rate,
                 padding_mode='repeat',
                 valid=False):
        super().__init__()
        if isinstance(file_list, str):
            with open(file_list) as f:
                self.audio_files = f.read().splitlines()
        else:
            self.audio_files = file_list
        self.captions = {}
        with open('/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/data_processing/AudioSetCaps_caption.csv', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.captions[row['id']] = row['caption']
        self.sampling_rate = sample_rate
        self.segment_size = segment_size
        self.valid = valid
        self.padding_mode = padding_mode

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            file_path = self.audio_files[idx]
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)

            # Check for silent/empty audio
            if waveform.numel() == 0:
                raise RuntimeError("Loaded audio is empty.")
            elif waveform.abs().max() < 1e-6:
                raise RuntimeError("Loaded audio is silent.")

            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

            waveform *= 0.5 / waveform.abs().max()
            # clip_fea = np.load(file_path.replace('.flac', '.npy').replace('vggsound_a_48k', 'vggsound_v_clip'))
            # clip_fea, start = self.pad_clip_fea(torch.tensor(clip_fea, dtype=torch.float32).transpose(-1,-2))
            waveform = self.pad_waveform(waveform)
            caption = self.captions[os.path.splitext(os.path.basename(file_path))[0]]
            # latent_path = "/gemini/platform/public/aigc/mh-data/audioset_vae/train_latents_64/" + os.path.splitext(os.path.basename(file_path))[0] + ".npy"
            # vae_latent = torch.tensor(np.load(latent_path), dtype=torch.float32)
            # t5_path = "/gemini/platform/public/aigc/mh-data/audioset/t5_embeddings/" + os.path.splitext(os.path.basename(file_path))[0][1:] + ".npz"
            # t5_emb = torch.tensor(np.load(t5_path)['caption_feature'], dtype=torch.float32)
            # context_mask = torch.tensor(np.load(t5_path)['attention_mask'], dtype=torch.float32)
            
            return {
                "waveform": waveform,
                'file_path': file_path,
                "caption": caption
                # 'vae_latent': vae_latent,
                # "t5_emb": t5_emb,
                # "context_mask": context_mask
                # 'clip_fea': clip_fea
            }
        except Exception as e:
            print(f"Warning: Failed to load file at index {idx} ({self.audio_files[idx]}). Error: {e}. Grabbing a random new sample.")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
    
    
    def pad_waveform(self, waveform):
        """Pad the waveform to a fixed length."""
        target_length = self.segment_size
        n_samples = waveform.shape[1]
        if n_samples < target_length:
            if self.padding_mode == 'repeat':
                n_repeats = (target_length + n_samples - 1) // n_samples
                waveform = waveform.repeat(1, n_repeats)
                waveform = waveform[:, :target_length]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, target_length - n_samples), mode='constant', value=0)
        elif n_samples > target_length:
            start = random.randint(0, n_samples - target_length) if not self.valid else 0
            waveform = waveform[:, start:start + target_length]
        return waveform



# def collate_fn(batch):
#     waveforms, vae_latents, file_paths = [], [], []
#     for item in batch:
#         waveforms.append(item['waveform'])
#         # vae_latents.append(item['vae_latent'])
#         # t5_embs.append(item['t5_emb'])
#         # context_masks.append(item['context_mask'])
#         file_paths.append(item['file_path'])
#     return {
#         'waveform': torch.stack(waveforms, dim=0),
#         # 'vae_latent': torch.stack(vae_latents, dim=0),
#         # "t5_emb": torch.concat(t5_embs, dim=0),
#         # "context_mask": torch.concat(context_masks, dim=0),
#         'file_path': file_paths,
#     }


class SemanticCollator:
    def __init__(self, tokenizer_name="Qwen/Qwen2.5-0.5B", max_length=128):
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = None
        self.max_length = max_length

    def __call__(self, batch):
        waveforms, captions, file_paths = [], [], [] 
        for item in batch:
            waveforms.append(item['waveform'])
            captions.append(item['caption'])
            file_paths.append(item['file_path'])

        # stack waveforms
        waveforms = torch.stack(waveforms, dim=0)

        # tokenizer 编码 captions
        if self.tokenizer is not None:
            captions = [c + self.tokenizer.eos_token for c in captions]
            enc = self.tokenizer(
                captions,
                padding=True if self.max_length is None else "max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']

            # 构造 labels
            labels = input_ids.clone()
            pad_mask = labels == self.tokenizer.pad_token_id

            # 找到每条序列第一个 eos
            eos_id = self.tokenizer.eos_token_id
            first_eos_idx = (labels == eos_id).int().cumsum(dim=1) == 1  # 第一个 eos 的位置

            # pad_mask 去掉第一个 eos
            final_mask = pad_mask & (~first_eos_idx)

            # 只对 final_mask 的位置置 -100
            labels[final_mask] = -100
        else:
            input_ids = attention_mask = labels = None

        return {
            'waveform': waveforms,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'file_path': file_paths
        }



if __name__ == "__main__":
    # 数据集配置
    audio_files_list = "/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/data_processing/audioset_train.txt"  # 替换为你的文件列表路径
    segment_size = 320000  # 5秒
    sample_rate = 32000

    # 初始化 dataset 和 dataloader
    ds = audioDataset(file_list=audio_files_list, segment_size=segment_size, sample_rate=sample_rate)
    collator = SemanticCollator(tokenizer_name="Qwen/Qwen2.5-0.5B")
    dataloader = DataLoader(ds, batch_size=4, collate_fn=collator, shuffle=True)

    # 输出目录
    save_dir = "./audio_samples"
    os.makedirs(save_dir, exist_ok=True)

    # 读取一批数据
    for batch in dataloader:
        waveforms = batch['waveform']  # [B, 1, T]
        captions = [collator.tokenizer.decode(ids, skip_special_tokens=True) 
                    for ids in batch['input_ids']]  # 解码 captions

        for j in range(len(waveforms)):
            waveform = waveforms[j].squeeze(0)  # [T]

            # 构造安全文件名
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in captions[j])
            if len(safe_name) > 100:
                safe_name = safe_name[:100]

            out_path = os.path.join(save_dir, f"{safe_name}.wav")
            torchaudio.save(out_path, waveform.unsqueeze(0), sample_rate)
            print(f"Saved audio to {out_path}")

        break  # 只保存一批做测试