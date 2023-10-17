import json
import os
from tqdm import tqdm
import kara_storage
from nltk.tokenize import sent_tokenize

out_path = "/data/disk1/private/xcj/ReadOnce/wiki_storage"
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)

storage = kara_storage.KaraStorage("file://%s" % out_path)
dataset = storage.open("wiki", "para-512", "w", version="lastest")

path = "/data2/private/xcj/BMSearch/wikiextractor-master/pages"
for fn in tqdm(os.listdir(path)):
    fin = open(os.path.join(path, fn), "r")
    data = json.load(fin)
    for doc in tqdm(data):
        doc = doc[doc.find("\n") : -8].strip()
        doc = doc[doc.find("\n") + 1: ]
        sents = sent_tokenize(doc)
        if len(sents) <= 1:
            continue
        paras = []
        nowsents, nowlen = [], 0
        for s in sents:
            slen = len(s.split())
            nowsents.append(s)
            nowlen += slen
            if nowlen > 460:
                paras.append(" ".join(nowsents))
                nowsents, nowlen = [], 0
        if nowlen > 0:
            paras.append(" ".join(nowsents))
        for p in paras:
            dataset.write(p)
dataset.flush()
