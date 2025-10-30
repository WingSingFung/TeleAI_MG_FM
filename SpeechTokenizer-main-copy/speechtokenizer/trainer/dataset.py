from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np
import os
from transformers import AutoTokenizer
import csv
import json

def get_dataloader(ds, tokenizer_name, **kwargs):
    return DataLoader(ds, collate_fn=SemanticCollator(tokenizer_name=tokenizer_name), **kwargs)

class audioDataset(Dataset):
    
    def __init__(self,
                 file_list,
                 segment_size,
                 sample_rate,
                 padding_mode='zero',
                 chunk_mode='constant',
                 valid=False):
        super().__init__()

        self.captions = {}
        with open(file_list, "r") as f:
            data = json.load(f)

        if valid:
            # 如果存在 validation，则取 dev
            self.metadata = [item for item in data if "dev-clean" in item["split"]]
        else:
            # 否则取 train
            self.metadata = [item for item in data if "train" in item["split"]]
        self.sampling_rate = sample_rate
        self.segment_size = segment_size
        self.valid = valid
        self.padding_mode = padding_mode
        self.chunk_mode = chunk_mode

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            item = self.metadata[idx]
            file_path = item['wav_path']
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)

            # Check for silent/empty audio
            if waveform.numel() == 0:
                raise RuntimeError("Loaded audio is empty.")

            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            if self.padding_mode == 'zero':
                valid_len = min(waveform.shape[1] / self.sampling_rate, self.segment_size / self.sampling_rate)
            else:
                valid_len = self.segment_size / self.sampling_rate
            waveform = self.pad_waveform(waveform, target_length = self.segment_size)


            if waveform.abs().max() < 1e-3:
                raise RuntimeError("Loaded audio is silent.")
            waveform *= 0.5 / waveform.abs().max()

            caption = item['text'].strip().lower()

            
            return {
                "waveform": waveform,

                'file_path': file_path,
                "caption": caption,
                "valid_len": valid_len,
                # 'vae_latent': vae_latent,
                # "t5_emb": t5_emb,
                # "context_mask": context_mask
                # 'clip_fea': clip_fea
            }
        except Exception as e:
            print(f"Warning: Failed to load file at index {idx} ({self.metadata[idx]}). Error: {e}. Grabbing a random new sample.")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
    
    
    def pad_waveform(self, waveform, target_length):
        """Pad the waveform to a fixed length."""
        n_samples = waveform.shape[1]
        if n_samples < target_length:
            if self.padding_mode == 'repeat':
                n_repeats = (target_length + n_samples - 1) // n_samples
                waveform = waveform.repeat(1, n_repeats)
                waveform = waveform[:, :target_length]
            elif self.padding_mode == 'zero':
                waveform = torch.nn.functional.pad(waveform, (0, target_length - n_samples), mode='constant', value=0)
        elif n_samples > target_length:
            # start = random.randint(0, n_samples - target_length) if not self.valid else 0
            if self.chunk_mode == 'constant' or self.valid:
                start = 0
            elif self.chunk_mode == 'random':
                start = random.randint(0, n_samples - target_length)
            waveform = waveform[:, start:start + target_length]
        return waveform




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
        waveforms, captions, file_paths, valid_len = [], [], [], []
        for item in batch:
            waveforms.append(item['waveform'])
            valid_len.append(item['valid_len'])
            captions.append(item['caption'])
            file_paths.append(item['file_path'])

        # stack waveforms
        waveforms = torch.stack(waveforms, dim=0)
        valid_len = torch.tensor(valid_len)


        # tokenizer 编码 captions
        if self.tokenizer is not None:
            enc = self.tokenizer(
                [c + self.tokenizer.eos_token for c in captions],
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
            'valid_len': valid_len,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'texts': captions,
            'file_path': file_paths
        }



if __name__ == "__main__":
    # 数据集配置
    print(1)
    audio_files_list = "/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/data_processing/libritts_raw_data.json"
    segment_size = 480000
    sample_rate = 16000
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from whisper_normalizer.english import EnglishTextNormalizer
    from jiwer import wer
    from tqdm import tqdm
    import soundfile as sf
    # 初始化 dataset 和 dataloader
    ds = audioDataset(file_list=audio_files_list,
                      segment_size=segment_size,
                      sample_rate=sample_rate,
                      padding_mode='zero',
                      valid=True)
    collator = SemanticCollator(tokenizer_name="openai/whisper-small.en")
    dataloader = DataLoader(ds, batch_size=32, collate_fn=collator, shuffle=False)
    print(2)
    # 输出目录
    save_dir = "./audio_samples"
    os.makedirs(save_dir, exist_ok=True)

    # 加载 Whisper 模型 (比如 medium.en)
    # model_name = "/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/checkpoints/openai/whisper-small.en"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # processor = WhisperProcessor.from_pretrained(model_name)
    # model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    tn = EnglishTextNormalizer()
    all_refs, all_preds = [], []
    print(3)
    wer_score = []
    # 遍历数据集并解码
    for batch in tqdm(dataloader):
        waveforms = batch['waveform'].squeeze(1)  # [B, T]
        texts = batch['texts']                     # list[str]
        # print(texts)
        # for i in range(waveforms.shape[0]):
        #     sf.write('test.wav', waveforms[i,...], samplerate=16000)
        


        # # Whisper 预处理
        # inputs = processor(waveforms.numpy(),
        #                    sampling_rate=sample_rate,
        #                    return_tensors="pt",
        #                    padding=True).to(device)

        # with torch.no_grad():
        #     predicted_ids = model.generate(**inputs, num_beams=1)

        # # 解码预测文本
        # predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # all_refs.extend([tn(t) for t in texts])
        # all_preds.extend(tn(t) for t in predictions)
        # # 临时打印前几个
        # for ref, pred in zip(texts, predictions):
        #     print(f"REF: {tn(ref)}")
        #     print(f"HYP: {tn(pred)}")
        #     print("----")
        #     # 计算整体 WER
        #     print(f"Final WER: {wer(all_refs, all_preds):.4f}")