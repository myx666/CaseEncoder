import kara_storage
import torch
import pickle
import argparse
import os
from tqdm import tqdm
# class MySerializer(kara_storage.serialization.Serializer):
#     def serialize(self, x): # 序列化x，将x转换为bytes
#         return pickle.dumps(x)
    
#     def deserialize(self, x): # 反序列化x，将x从bytes重新转换回对象
#         return pickle.loads(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--in_path', type=str, help="input path")
    args = parser.parse_args()
    print("reading from", args.in_path, "\nsave at", args.save_path)
    storage = kara_storage.KaraStorage("file://%s" % args.save_path)
    print("init storage done")
    dataset = storage.open("deltas", "race-bs4-st80-lr2e3", "w", version="1st", serialization=kara_storage.serialization.PickleSerializer())
    print("open dataset done")

    all_data = []
    for f in os.listdir(args.in_path):
        print("loading", f)
        data = torch.load(os.path.join(args.in_path, f))
        for did, ka, kb, va, vb in zip(data["docid"], data["kA"], data["kB"], data["vA"], data["vB"]):
            all_data.append({"docid": did, "kA": ka.detach().numpy(), "kB": kb.detach().numpy(), "vA": va.detach().numpy(), "vB": vb.detach().numpy()})
    all_data.sort(key=lambda x:x["docid"])
    print("the number of data", len(all_data))
    docid = 0
    for a in tqdm(all_data):
        if docid > a["docid"]:
            print("repeat", docid, "-->", a["docid"])
            continue
        if a["docid"] != docid:
            while docid < a["docid"]:
                dataset.write({"docid": docid, "gg": True})
                print("missing", docid, a["docid"])

                docid += 1
        else:
            # from IPython import embed; embed()
            dataset.write(a)
            docid += 1
    dataset.close()
