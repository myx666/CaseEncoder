import json
from operator import mod
import os
import random

path = "/data_new/private/xiaochaojun/DocAsPara/data/RACE/RACE"
for mode in ["train", "dev", "test"]:
    data = []
    for dif in ["high", "middle"]:
        folder = os.path.join(path, mode, dif)
        for f in os.listdir(folder):
            doc = json.load(open(os.path.join(folder, f), "r"))
            ques, ops, ans = doc["questions"], doc["options"], doc["answers"]
            questions = []
            for q, o, a in zip(ques, ops, ans):
                questions.append({
                    "question": q,
                    "options": o,
                    "answer": a
                })
            data.append({
                "document": doc["article"],
                "questions": questions
            })
    random.shuffle(data)
    fout = open("/data_new/private/xiaochaojun/DocAsPara/data/RACE/%s.json" % mode, "w")
    print(json.dumps(data, ensure_ascii=False, indent=2), file=fout)
    fout.close()
            