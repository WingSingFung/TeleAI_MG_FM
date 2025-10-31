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
            # self.metadata = [item for item in data if "dev-clean" in item["split"]]
            self.metadata = [item for item in data if "test" in item["split"]]
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
            
            # 1. 首先获取音频元信息（不加载完整音频）
            try:
                audio_info = torchaudio.info(file_path)
                sr = audio_info.sample_rate
                n_frames = audio_info.num_frames
                n_channels = audio_info.num_channels
            except Exception as e:
                print(f"Error reading audio info from {file_path}: {e}")
                raise RuntimeError(f"Failed to read audio info: {e}")
            
            # 检查音频是否为空
            if n_frames == 0:
                raise RuntimeError("Audio file is empty.")
            
            # 尝试选择非静音片段，最多重试10次
            max_retries = 10
            waveform = None
            successfully_loaded = False
            
            for retry in range(max_retries):
                try:
                    # 2. 计算需要加载的片段范围（在原始采样率下）
                    target_frames_in_sr = int(self.segment_size * sr / self.sampling_rate)
                    
                    if n_frames > target_frames_in_sr:
                        # 音频长度足够，根据重试次数选择不同的起始位置策略
                        if retry == 0:
                            # 第一次：根据chunk_mode选择
                            if self.chunk_mode == 'random' and not self.valid:
                                start_frame = random.randint(0, n_frames - target_frames_in_sr)
                            else:
                                start_frame = 0
                        elif retry == 1:
                            # 第二次：从音频中间选择
                            start_frame = (n_frames - target_frames_in_sr) // 2
                        elif retry == 2:
                            # 第三次：从前半部分的中间选择 (1/4位置)
                            start_frame = (n_frames - target_frames_in_sr) // 4
                        elif retry == 3:
                            # 第四次：从后半部分的中间选择 (3/4位置)
                            start_frame = 3 * (n_frames - target_frames_in_sr) // 4
                        else:
                            # 后续重试：在关键点附近随机选择
                            key_points = [
                                0,
                                (n_frames - target_frames_in_sr) // 4,
                                (n_frames - target_frames_in_sr) // 2,
                                3 * (n_frames - target_frames_in_sr) // 4,
                                n_frames - target_frames_in_sr
                            ]
                            base_point = key_points[retry % len(key_points)]
                            # 在关键点附近随机偏移
                            offset_range = max(1, (n_frames - target_frames_in_sr) // 8)
                            offset = random.randint(-offset_range, offset_range)
                            start_frame = max(0, min(n_frames - target_frames_in_sr, base_point + offset))
                        
                        # 加载指定片段
                        waveform, _ = torchaudio.load(file_path, frame_offset=start_frame, num_frames=target_frames_in_sr)
                    else:
                        # 音频长度不足，加载全部音频
                        waveform, _ = torchaudio.load(file_path)
                    
                    # 检查加载的音频是否为空
                    if waveform.numel() == 0:
                        raise RuntimeError("Loaded audio segment is empty.")
                    
                    # 3. 先检查是否为静音（在进行其他处理之前）
                    # 注意：这里检查的是多通道或原始采样率的音频
                    if waveform.abs().max() >= 1e-3:
                        # 非静音，跳出重试循环
                        successfully_loaded = True
                        break
                    
                    # 静音处理
                    if retry < max_retries - 1:
                        if self.chunk_mode != 'random':
                            print(f"Warning: Loaded audio segment is silent. Retry {retry + 1} of {max_retries} for {file_path}.")
                    
                except Exception as e:
                    # 加载过程中出错，记录并继续重试
                    print(f"Error loading audio chunk (retry {retry + 1}/{max_retries}) from {file_path}: {e}")
                    if retry == max_retries - 1:
                        # 最后一次重试失败，使用全0 tensor
                        print(f"Failed to load any valid chunk from {file_path} after {max_retries} retries. Using zero tensor.")
                        waveform = None
                        break
            
            # 如果所有重试都失败或都是静音，使用全0 tensor
            if waveform is None or not successfully_loaded:
                if waveform is not None and waveform.abs().max() < 1e-3:
                    print(f"Warning: All loaded segments are silent for {file_path}. Using zero tensor.")
                waveform = torch.zeros((1, self.segment_size), dtype=torch.float32)
                valid_len = self.segment_size / self.sampling_rate
                
                return {
                    "waveform": waveform,
                    'file_path': file_path,
                    "valid_len": valid_len,
                }
            
            # 4. 对加载的片段进行后处理：mono转换
            try:
                if waveform.shape[0] > 1:
                    # Convert to mono by averaging channels
                    waveform = waveform.mean(dim=0, keepdim=True)
            except Exception as e:
                print(f"Error converting to mono for {file_path}: {e}. Using zero tensor.")
                waveform = torch.zeros((1, self.segment_size), dtype=torch.float32)
                valid_len = self.segment_size / self.sampling_rate
                return {
                    "waveform": waveform,
                    'file_path': file_path,
                    "valid_len": valid_len,
                }
            
            # 5. resample到目标采样率
            try:
                if sr != self.sampling_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
            except Exception as e:
                print(f"Error resampling audio for {file_path}: {e}. Using zero tensor.")
                waveform = torch.zeros((1, self.segment_size), dtype=torch.float32)
                valid_len = self.segment_size / self.sampling_rate
                return {
                    "waveform": waveform,
                    'file_path': file_path,
                    "valid_len": valid_len,
                }
            
            # 6. 如果音频长度不足，进行padding
            try:
                current_samples = waveform.shape[1]
                if current_samples < self.segment_size:
                    if self.padding_mode == 'repeat':
                        n_repeats = (self.segment_size + current_samples - 1) // current_samples
                        waveform = waveform.repeat(1, n_repeats)
                        waveform = waveform[:, :self.segment_size]
                    elif self.padding_mode == 'zero':
                        waveform = torch.nn.functional.pad(
                            waveform, (0, self.segment_size - current_samples), 
                            mode='constant', value=0
                        )
                elif current_samples > self.segment_size:
                    # 由于resample可能导致轻微的长度变化，进行微调
                    waveform = waveform[:, :self.segment_size]
            except Exception as e:
                print(f"Error padding audio for {file_path}: {e}. Using zero tensor.")
                waveform = torch.zeros((1, self.segment_size), dtype=torch.float32)
                valid_len = self.segment_size / self.sampling_rate
                return {
                    "waveform": waveform,
                    'file_path': file_path,
                    "valid_len": valid_len,
                }
            
            # 计算有效长度
            try:
                if self.padding_mode == 'zero':
                    # 原始音频长度（秒）和segment_size长度（秒）的最小值
                    original_duration = n_frames / sr
                    segment_duration = self.segment_size / self.sampling_rate
                    valid_len = min(original_duration, segment_duration)
                else:
                    valid_len = self.segment_size / self.sampling_rate
            except Exception as e:
                print(f"Error calculating valid length for {file_path}: {e}")
                valid_len = self.segment_size / self.sampling_rate
            
            # 7. 归一化音频
            try:
                max_val = waveform.abs().max()
                if max_val > 1e-3:
                    waveform *= 0.5 / max_val
            except Exception as e:
                print(f"Error normalizing audio for {file_path}: {e}")
                # 如果归一化失败，继续使用未归一化的waveform

            return {
                "waveform": waveform,
                'file_path': file_path,
                "valid_len": valid_len,
            }
        except Exception as e:
            print(f"Warning: Failed to load file at index {idx} ({self.metadata[idx]}). Error: {e}. Grabbing a random new sample.")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)




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
        # waveforms, captions, file_paths, valid_len = [], [], [], []
        waveforms, file_paths, valid_len = [], [], []
        for item in batch:
            waveforms.append(item['waveform'])
            valid_len.append(item['valid_len'])
            # captions.append(item['caption'])
            file_paths.append(item['file_path'])

        # stack waveforms
        waveforms = torch.stack(waveforms, dim=0)
        valid_len = torch.tensor(valid_len)


        # tokenizer 编码 captions
        if self.tokenizer is not None:
            # enc = self.tokenizer(
            #     [c + self.tokenizer.eos_token for c in captions],
            #     padding=True if self.max_length is None else "max_length",
            #     truncation=True,
            #     max_length=self.max_length,
            #     return_tensors="pt"
            # )

            # input_ids = enc['input_ids']
            # attention_mask = enc['attention_mask']

            # 构造 labels
            # labels = input_ids.clone()
            # pad_mask = labels == self.tokenizer.pad_token_id

            # # 找到每条序列第一个 eos
            # eos_id = self.tokenizer.eos_token_id
            # first_eos_idx = (labels == eos_id).int().cumsum(dim=1) == 1  # 第一个 eos 的位置

            # # pad_mask 去掉第一个 eos
            # final_mask = pad_mask & (~first_eos_idx)

            # # 只对 final_mask 的位置置 -100
            # labels[final_mask] = -100
            input_ids = attention_mask = labels = None
        else:
            input_ids = attention_mask = labels = None

        return {
            'waveform': waveforms,
            'valid_len': valid_len,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            # 'texts': captions,
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