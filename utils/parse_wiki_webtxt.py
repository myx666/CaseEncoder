import json
import os
from tqdm import tqdm
import kara_storage
from nltk.tokenize import sent_tokenize

import kara_storage
import os
from tqdm import tqdm
import json
import random

storage = kara_storage.KaraStorage("file:///data/home/scv0540/xcj/datamux/data/wiki-webtext")
dataset = storage.open("pretrain", "corpus", "w", version="1st", serialization=kara_storage.serialization.JSONSerializer())

valid_text = []
for fn in tqdm(os.listdir("/data/home/scv0540/xcj/ReadOnce/data/wikipedia/passages-512")):
    fin = open("/data/home/scv0540/xcj/ReadOnce/data/wikipedia/passages-512/%s" % fn, "r")
    for line in fin.readlines():
        if random.random() < 0.0005 and len(valid_text) < 3000:
            valid_text.append(json.loads(line))
        else:
            dataset.write(json.loads(line))

fin = open("/data/home/scv0540/zzy/openwebtxt/all_txt", "r")
now_txt, now_len = [], 0
for line in fin:
    line = line.strip()
    if len(line) == 0:
        continue
    now_txt.append(line)
    now_len += len(line.split())
    if now_len > 460:
        if random.random() < 0.0005 and len(valid_text) < 12000:
            valid_text.append({"text": "\n".join(now_txt)})
        else:
            dataset.write({"text": "\n".join(now_txt)})
        now_txt, now_len = [], 0
dataset.close()

fout = open("/data/home/scv0540/xcj/datamux/data/valid_text.jsonl", "w")
random.shuffle(valid_text)
for line in valid_text:
    print(json.dumps(line, ensure_ascii=False), file=fout)
fout.close()
