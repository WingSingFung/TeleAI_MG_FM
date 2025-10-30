import kaldiio
import soundfile as sf

mat = kaldiio.load_mat('/gemini/platform/public/aigc/crj/xql/asr2_crj/dump/audio_raw/org/train/data/format.1/data_wav.ark:6717100')
sf.write('test.wav', mat[1], 16000)


print(mat.shape)