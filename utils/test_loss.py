import json
import argparse
import random
import numpy as np
from statistics import mode
from torch import nn
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling,BertTokenizer,BertConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import math
from nltk.tokenize import sent_tokenize

local_rank = -1
global_size = -1

model_path = "huawei-noah/TinyBERT_General_6L_768D"
model_path = "/home/yedeming/TinyBERT/models/bert_3layer_O1"
class Formatter:
    def __init__(self, max_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.masker = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        self.max_len = max_len

    def collect(self, data):
        ctxs = [d["text"] for d in data]
        ctx_tokens = self.tokenizer(ctxs, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        qinp, qlabels = self.masker.torch_mask_tokens(ctx_tokens["input_ids"])

        return {
            "input_ids": qinp,
            "attention_mask": ctx_tokens["attention_mask"],
            "labels": qlabels
        }


def myeval(model, dataloader):
    total_loss = 0
    step = 0
    for data in tqdm(dataloader):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()
        out = model(data["input_ids"], attention_mask=data["attention_mask"], labels=data["labels"])
        total_loss += out["loss"].item()
        step += 1
    mytensor = torch.FloatTensor([total_loss / step]).cuda()
    mylist = [torch.FloatTensor(1).cuda() for i in range(global_size)]
    torch.distributed.all_gather(mylist, mytensor)
    print("avg loss:", sum(mylist) / global_size)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(12315)
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    datasets = [json.loads(line) for line in open("/data_new/private/xiaochaojun/datamux/data/valid_text.jsonl", "r")]
    print("read done")

    form = Formatter()
    sampler = DistributedSampler(datasets)
    dataloader = DataLoader(dataset=datasets,
                            batch_size=16,
                            #shuffle=shuffle,
                            num_workers=2,
                            collate_fn=form.collect,
                            drop_last=False,
                            sampler=sampler)
    print("init model")
    model = BertForMaskedLM.from_pretrained(model_path)
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank])

    local_rank = args.local_rank
    global_size = torch.distributed.get_world_size()
    print("local_rank: ", local_rank, "global_size: ", global_size)

    model.eval()
    with torch.inference_mode():
        myeval(model, dataloader)

