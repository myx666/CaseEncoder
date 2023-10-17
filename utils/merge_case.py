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
    args = parser.parse_args()
    print("reading from", args.in_path, "\nsave at", args.save_path)
    storage = kara_storage.KaraStorage("file://%s" % args.save_path)
    print("init storage done")
    dataset = storage.open_dataset("crime_data", "cail-lcr22", "w", version="1st", serialization=kara_storage.serialization.JSONSerializer())
    print("open dataset done")

    # write lecard
    all_data = []
    querys = open(os.path.join(args.in_path, 'query22.json'), 'r').readlines()
    labels = json.load(open(os.path.join(args.in_path, 'label_top30_dict.json'), 'r'))
    for query_line in tqdm(querys):
        dic = eval(query_line)
        qid = str(dic['ridx'])
        if qid not in labels: continue
        # tmp_list = [{'ridx': dic['ridx'], 'fact': dic['q'], 'label': -1}]
        # tmp_list = []
        all_data.append({'ridx': qid + '_' + qid, 'fact': dic['q'], 'label': -1})
        subdir = os.path.join(args.in_path, 'candidates', qid)
        candidates = os.listdir(subdir)
        for c in candidates:
            cdic = json.load(open(os.path.join(subdir, c), 'r'))
            cid = c.split('.')[0]
            ridx = qid + '_' + cid
            label = labels[qid][cid] if cid in labels[qid] else 0
            all_data.append({'ridx': ridx, 'fact': cdic['ajjbqk'], 'label': label})
        # all_data.append(tmp_list)

    # for f in tqdm(os.listdir(args.in_path)[:1]):
    #     if int(f.split('_')[-1].split('.')[0]) % 10 in [8,9]: continue
    #     # print("loading", f)
    #     # data = torch.load(os.path.join(args.in_path, f))
    #     raw_lines = open(os.path.join(args.in_path, f), 'r').readlines()[:12]
    #     lines = [eval(line) for line in raw_lines]
       
    #     for line in lines:
    #         # print(line)
    #         all_data.append({"ridx":line['ridx'], "fact": line['fact'], "xf_article": line['xf_article'], "pos_case":line['pos_case']})
    # # all_data.sort(key=lambda x:x["docid"])

    print("the number of data", len(all_data))
    for a in tqdm(all_data):
        dataset.write(a)

    dataset.close()
