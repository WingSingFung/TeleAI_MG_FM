import os
from tqdm import tqdm

# 所有包含 wav 的目录
train_files = (
    [f"/gemini/platform/public/aigc/mh-data/unbalanced_train_segments_part{i}" for i in range(10, 41)]
    + [f"/gemini/platform/public/aigc/mh-data/audios/unbalanced_train_segments/unbalanced_train_segments_part0{i}" for i in range(0, 10)]
)

output_txt = "./audioset_train.txt"

wav_files = []
for d in tqdm(train_files, desc="Scanning directories"):
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".wav"):
                    wav_files.append(os.path.join(root, f))

# 写入 txt，一行一个路径
with open(output_txt, "w") as f:
    f.write("\n".join(wav_files))

print(f"共收集到 {len(wav_files)} 个 wav 文件，已写入 {output_txt}")
