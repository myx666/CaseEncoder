import imp
import logging
import os
import json
import torch
from torch.autograd import Variable
from timeit import default_timer as timer
import math
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


from tools.eval_tool import gen_time_str, output_value

logger = logging.getLogger(__name__)

def eval_result(config, result):
    def ndcg(ranks, gt_ranks, K):
        dcg_value = 0.
        idcg_value = 0.
        # log_ki = []

        sranks = sorted(gt_ranks, reverse=True)

        for i in range(0,K):
            logi = math.log(i+2,2)
            dcg_value += ranks[i] / logi
            idcg_value += sranks[i] / logi
        
        return dcg_value/idcg_value if idcg_value!=0.0 else 0.0
    
   
    # res_dic = {}
    sndcg = [0.0] * 3
    ndcg_scores = [[], [], []]
    K = [10, 20, 30]
    for qid in result:
        q_embd = result[qid][qid][0]
        tmp_result = sorted([(cosine_similarity([q_embd], [result[qid][cid][0]])[0], result[qid][cid][1]) for cid in result[qid] if cid != qid], key = lambda x:x[0], reverse=True)
        ranks = [r[1] for r in tmp_result]
        gt_ranks = sorted([r[1] for r in tmp_result], reverse=True)
        for i, k in enumerate(K):
            score = ndcg(ranks, gt_ranks, k)
            ndcg_scores[i].append(score)
            sndcg[i] += score
    # print(ndcg_scores)
    
    return [s/len(list(result.keys())) for s in sndcg]

def test(parameters, config, gpu_list):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = "testing"

    output_time = config.getint("output", "output_time")
    step = -1
    # result = []
    result = {}

    for step, data in enumerate(dataset):
        # if cnt == 2: break
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = model(data, config, gpu_list, acc_result, "test")
        # result.append(results["output"])
        for qid in results["output"]:
            if qid in result: result[qid].update(results["output"][qid])
            else: result[qid] = results["output"][qid]
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = "testing"
    output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    
    ndcg10, ndcg20, ndcg30 = eval_result(config, result)
    print('final ndcg scores are: %.3f, %.3f, %.3f'%(ndcg10, ndcg20, ndcg30))
    # return result

    return 0
