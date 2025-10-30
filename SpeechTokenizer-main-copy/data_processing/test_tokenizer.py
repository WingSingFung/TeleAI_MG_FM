import torch

state_dict = torch.load('/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/exps_semantic/12_5hz_asr_1010_tinyllama_whisper/logs/version_30/checkpoints/last.ckpt', map_location='cpu')

for k, v in state_dict['state_dict'].items():
    print(k)