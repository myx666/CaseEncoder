import kara_storage
import torch
import pickle
import argparse
import os
from tqdm import tqdm
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--in_path', type=str, help="input path")
    parser.add_argument('--json_path', type=str, help="json path")
    args = parser.parse_args()
    print("reading from", args.in_path, "\nsave at", args.save_path)
    storage = kara_storage.KaraStorage("file://%s" % args.save_path)
    print("init storage done")
    dataset = storage.open("deltas", "race-bs4-st150-lr2e3", "w", version="2nd", serialization=kara_storage.serialization.PickleSerializer())
    print("open dataset done")

    json_data = json.load(open(args.json_path, "r"))

    all_data = []
    for f in os.listdir(args.in_path):
        print("loading", f)
        try:
            data = torch.load(os.path.join(args.in_path, f))
        except:
            print("fail loading", f)
            continue
        for did, ka, kb, va, vb in zip(data["docid"], data["kA"], data["kB"], data["vA"], data["vB"]):
            all_data.append({"json_data": json.dumps(json_data[did]), "docid": did, "kA": ka.detach().numpy(), "kB": kb.detach().numpy(), "vA": va.detach().numpy(), "vB": vb.detach().numpy()})
    all_data.sort(key=lambda x:x["docid"])
    print("the number of data", len(all_data))
    docid = 0
    for a in tqdm(all_data):
        dataset.write(a)
    dataset.close()
