import json
from click import option
import torch
import os
import numpy as np

import random
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForLanguageModeling
from tools import print_rank

class WikiMLMFormatter:
    def __init__(self, config, mode, *args, **params):
        self.doc_len = config.getint("train", "doc_len")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.reuse_num = config.getint("train", "reuse_num")
        self.mode = mode
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "pretrained_model"))
        self.masker = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=self.mlm_prob, return_tensors="pt")

    def process(self, data):
        if self.mode == "train":
            selected = random.sample(list(range(len(data))), min(len(data), self.reuse_num))
            randorder = list(range(len(data))) + selected
            random.shuffle(randorder)
            ctxs = [data[i]["text"] for i in randorder]
            left = [randorder.index(selected[i]) for i in range(len(selected))]
            right = [randorder.index(selected[i], left[i] + 1) for i in range(len(selected))]
        # if type(data[0]) is dict and "text" in data[0]:
        else:
            ctxs = [d["text"] for d in data]
        # else:
        #     ctxs = data #[d["doc"] for d in data]

        ctx_tokens = self.tokenizer(ctxs, max_length=self.doc_len, padding="max_length", truncation=True, return_tensors="pt")

        qinp, qlabels = self.masker.torch_mask_tokens(ctx_tokens["input_ids"])

        ret = {
            "input_ids": qinp,
            "attention_mask": ctx_tokens["attention_mask"],
            "labels": qlabels
        }
        if self.mode == "train":
            ret["left"] = torch.LongTensor(left)
            ret["right"] = torch.LongTensor(right)
            ret["input_ids"][ret["right"]] = ret["input_ids"][ret["left"]]
            ret["attention_mask"][ret["right"]] = ret["attention_mask"][ret["left"]]
            ret["labels"][ret["right"]] = ret["labels"][ret["left"]]
            # print("left: %s \t right: %s" % (left, right))
        return ret