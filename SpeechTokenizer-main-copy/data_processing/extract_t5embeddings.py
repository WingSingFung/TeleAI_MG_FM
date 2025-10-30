import sys
sys.path.append('/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main')
from speechtokenizer.modules.t5 import T5Embedder
from pathlib import Path
import threading
from queue import Queue
import torch
import os
import numpy as np
import csv
from tqdm import tqdm


def extract_caption_t5_do(q):
    while not q.empty():
        item = q.get()
        extract_caption_t5_job(item)
        q.task_done()


def extract_caption_t5_job(item):
    global mutex
    global t5
    global t5_save_dir

    with torch.no_grad():
        caption = item['caption'].strip()
        if isinstance(caption, str):
            caption = [caption]

        save_path = os.path.join(t5_save_dir, item['id'][1:])
        if os.path.exists(f"{save_path}.npz"):
            return
        try:
            mutex.acquire()
            caption_emb, emb_mask = t5.get_text_embeddings(caption)
            mutex.release()
            emb_dict = {
                'caption_feature': caption_emb.float().cpu().data.numpy(),
                'attention_mask': emb_mask.cpu().data.numpy(),
            }
            np.savez_compressed(save_path, **emb_dict)
        except Exception as e:
            print(e)


def extract_caption_t5():
    global t5
    global t5_save_dir
    # global images_extension
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir='/root/.cache/IF_/t5-v1_1-xxl/AI-ModelScope', model_max_length=120)
    caption_emb, emb_mask = t5.get_text_embeddings([''])
    print(caption_emb)
    print(emb_mask)
    # t5_save_dir = '/gemini/platform/public/aigc/mah_1/mah/unitok/SpeechTokenizer-main/data_processing/t5_embeddings'
    # os.makedirs(t5_save_dir, exist_ok=True)

    # global mutex
    # mutex = threading.Lock()
    # jobs = Queue()
    # with open('./AudioSetCaps_caption.csv', newline='', encoding='utf-8') as f:
    #     reader = csv.DictReader(f)
    #     for row in tqdm(reader):
    #         jobs.put(row)

    # for _ in range(20):
    #     worker = threading.Thread(target=extract_caption_t5_do, args=(jobs,))
    #     worker.start()

    # jobs.join()


if __name__ == '__main__':
    extract_caption_t5()