import dac
import torchaudio
import soundfile as sf
import torch
import json
import os

# Download a model
# model = dac.DAC.load('/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/weights_44khz_8kbps_0.0.1.pth')
# model.cuda()
model_sr = 44100
source_root = "/gemini/platform/public/aigc/mh-data/audioset/AudioSet/data/audio/eval"
tgt_root = '/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/1'
with open('/gemini/platform/public/aigc/mah_1/mah/SpeechTokenizer-main/data_processing/audioset_eval_data_subset.json') as f:
    all_items = json.load(f)
wav_files = []
all_items = all_items['data']
for item in all_items:

    filename = os.path.basename(item['wav'])
    filename = filename[1:].replace('.wav', '.flac')
    input_path = os.path.join(source_root, filename)
    try:
        output_path = os.path.join(tgt_root, filename)
        wav, sr = torchaudio.load(input_path)
        wav_files.append(input_path)


        # if wav.shape[0] != 1:
        #     wav = wav[:1,...]

        # if sr != model_sr:
        #     wav = torchaudio.transforms.Resample(sr, model_sr)(wav).cuda()

        # with torch.inference_mode():
        #     wav = model.decode(model.encode(wav.unsqueeze(1))[0]).detach().cpu().squeeze(1)

        # if sr != model_sr:
        #     wav = torchaudio.transforms.Resample(model_sr, sr)(wav)

        # sf.write(output_path, wav.squeeze().numpy(), samplerate=sr)
    except:
        continue
with open("./audioset_eval_sub.txt", "w") as f:
    f.write("\n".join(wav_files))

print(f"共收集到 {len(wav_files)} 个 wav 文件，已写入 ./audioset_eval_sub.txt")


# from transformers import EncodecModel, AutoProcessor
# model = EncodecModel.from_pretrained("./encodec_32khz")

# print(model)

# from speechtokenizer.modules.seanet import SEANetEncoder

# model = SEANetEncoder(n_filters=64, 
#                      dimension=128,
#                      ratios=[8, 5, 4, 4],
#                      lstm=2,
#                      bidirectional=False,
#                      dilation_base=2,
#                      residual_kernel_size=3,
#                      n_residual_layers=1,
#                      activation='ELU')
# print(model)