import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import string
from collections import Counter
import math

def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value

def softmax_acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def squad_em(predict, answers):
    em = 0
    for pre, ans in zip(predict, answers):
        if pre in ans:
            em += 1
    return em

def squad_f1(predict, answers):
    ret = 0
    for pred, ans in zip(predict, answers):
        # if pred == "no answer":
        #     continue
        prediction_tokens = pred.split()
        cpred_token = Counter(prediction_tokens)
        curf1 = []
        for a in ans:
            ground_truth_tokens = a.split()
            common = cpred_token & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                curf1.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                curf1.append(f1)
        ret += max(curf1)
    return ret

def squad_NAF1(predict, answers, acc_result):
    for p, ans in zip(predict, answers):
        if p == "no answer":
            if "no answer" in ans:
                acc_result["NA_tp"] += 1
            else:
                acc_result["NA_fp"] += 1
        else:
            if "no answer" in ans:
                acc_result["NA_tn"] += 1
            else:
                acc_result["NA_fn"] += 1
    return acc_result

def squad_metric(predict, answers, acc_result, tokenizer):
    if acc_result is None:
        acc_result = {"train": False, "total": 0, "em_sum": 0, "f1_sum": 0., "NA_tp": 0, "NA_fp": 0, "NA_tn": 0, "NA_fn": 0}
    pred = []
    for p in predict:
        tmp = []
        for n in p:
            if n == 1:
                break
            tmp.append(int(n))
        pred.append(normalize_answer(tokenizer.decode(tmp, skip_special_tokens=True)))
    # pred = [normalize_answer([int(n) for n in p if n == 1 break], skip_special_tokens=True)) for p in predict]
    
    ground = [{normalize_answer(a) for a in ans} for ans in answers]

    # print(pred)
    # print(ground)
    # print("==" * 10)
    acc_result["em_sum"] += squad_em(pred, ground)
    acc_result["f1_sum"] += squad_f1(pred, ground)
    acc_result["total"] += len(pred)
    acc_result = squad_NAF1(pred, ground, acc_result)
    # print(acc_result)
    return acc_result

def squad_train_metric(predict, labels, acc_result):
    # predict: batch, len
    # labels: batch, len
    if acc_result is None:
        acc_result = {"train": True, "total": 0, "right": 0}
    acc_result["right"] += int((predict[labels > 0] == labels[labels > 0]).sum())
    acc_result["total"] += int((labels > 0).sum())
    return acc_result
