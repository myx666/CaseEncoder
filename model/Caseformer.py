from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM, AutoModel
import torch
from torch import avg_pool1d, nn
import math
import numpy as np
import torch.nn.functional as F
from .loss import convert_label_to_similarity, CircleLoss, BiasedCircleLoss

# from pytorch_pretrained_bert import BertModel

class CosineSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Caseformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Caseformer, self).__init__()
        
        self.model = AutoModelForMaskedLM.from_pretrained('hfl/chinese-roberta-wwm-ext')

        self.CL_weight = config.getfloat("train", "CL_weight")
        if config.getboolean("distributed", "use"): self.gpu_num = config.getint("distributed", "gpu_num")
        self.cos = CosineSimilarity(temp = config.getfloat("train", "temp"))
        self.circle_loss = CircleLoss(m=0.25, gamma=16)
        self.biased_circle_loss = BiasedCircleLoss(m=0.25, gamma=16)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    def init_multi_gpu(self, device, config, *args, **params):
        self.model = nn.DataParallel(self.model, device_ids=device)
    
    def get_circle_loss(self, contra_hidden, contra_weights, UsingBiased):
        batch_size = contra_hidden.shape[0] // 2
        if UsingBiased:
            K = 0.25
            pos_weights = []
            labels = torch.Tensor([-1] * (2*batch_size))
            cls = 0
            for i in range(batch_size):
                if labels[i] == -1:
                    labels[i] = cls
                    cls += 1
                for j in range(i+1, 2*batch_size):
                    if labels[j] == -1 and contra_weights[i][j] >= K: 
                        labels[j] = labels[i]
                        
            for i in range(2*batch_size):
                for j in range(i+1, 2*batch_size):
                    if labels[i] == labels[j]: pos_weights.append(contra_weights[i][j])
        
            pos_weights = torch.Tensor(pos_weights).cuda().detach()

            inp_sp, inp_sn = convert_label_to_similarity(contra_hidden, labels)
            loss = self.biased_circle_loss(inp_sp, inp_sn, pos_weights)
        else:
            labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)])
            inp_sp, inp_sn = convert_label_to_similarity(contra_hidden, labels)
            loss = self.circle_loss(inp_sp, inp_sn)
        return loss
    
    def info_nce_loss(self, hidden, pos_hidden, contra_weights):
        batch_size = hidden.shape[0]
        sim_mat = self.cos(hidden.unsqueeze(1), pos_hidden.unsqueeze(0))
        # lossfunc = nn.NLLLoss()
        lossfunc = nn.CrossEntropyLoss()
        labels = torch.arange(batch_size).long().cuda()
        loss = lossfunc(sim_mat, labels)
        return loss

    def forward(self, data, config, gpu_list, acc_result, mode):
        if mode in ['train', 'valid' ]:
            mlm_input_ids, mlm_mask, mlm_labels, contra_input_ids, contra_attention_mask, contra_weights = data['mlm_input_ids'], data['mlm_mask'], data['mlm_labels'], data['contra_input_ids'], data['contra_attention_mask'], data['contra_weights']
            batch_size = mlm_input_ids.shape[0]
            ret = self.model(mlm_input_ids, attention_mask=mlm_mask, labels=mlm_labels, output_hidden_states=True)
            
            mlm_loss, logits, mlm_hidden = ret.loss, ret.logits, ret.hidden_states[-1]

            contra_hidden = self.model(contra_input_ids, attention_mask=contra_attention_mask, output_hidden_states=True).hidden_states[-1]
            # print(contra_hidden.shape)
            # contra_hidden = torch.mean(mlm_hidden, dim=1, keepdim=True)
            # contra_hidden = mlm_hidden

            # hidden, pos_hidden = contra_hidden[:batch_size, 0, :], contra_hidden[batch_size:, 0, :]
            # mean_contra_hidden = torch.mean(contra_hidden, dim=1, keepdim=True)
            # hidden, pos_hidden = mean_contra_hidden[:batch_size], mean_contra_hidden[batch_size:] 
            # contra_loss = self.CL_weight * self.info_nce_loss(hidden, pos_hidden, contra_weights)
            contra_loss = 3e-6 * self.get_circle_loss(contra_hidden[:, 0, :], contra_weights, True)

            loss = mlm_loss + contra_loss

            # gather for tensorboard
            loss_gather, mlm_loss_gather, contra_loss_gather = [ [ torch.zeros(1).cuda() for __ in range(self.gpu_num) ] for _ in range(3)]
            torch.distributed.all_gather(loss_gather, loss)
            torch.distributed.all_gather(mlm_loss_gather, mlm_loss)
            torch.distributed.all_gather(contra_loss_gather, contra_loss)
            avg_loss_gather = torch.mean(torch.tensor(loss_gather)).item()
            avg_mlm_loss_gather = torch.mean(torch.tensor(mlm_loss_gather)).item()
            avg_contra_loss_gather = torch.mean(torch.tensor(contra_loss_gather)).item()
            
            return {"loss": loss, "acc_result":{}, "mlm_loss": mlm_loss, "contra_loss":contra_loss, "avg_loss_gather": avg_loss_gather, "avg_mlm_loss_gather": avg_mlm_loss_gather, "avg_contra_loss_gather": avg_contra_loss_gather}
        elif mode == 'test':
            input_ids, attention_mask, labels, ridxs = data['input_ids'], data['attention_mask'], data['labels'], data['ridx'] 
            hidden = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:, 0, :]

            hidden = hidden.detach().cpu().numpy().tolist()

            tmp_dic = {}
            for id, ridx in enumerate(ridxs):
                qid, cid = ridx.split('_')
                if qid in tmp_dic: tmp_dic[qid][cid] = (hidden[id], labels[id])
                else: tmp_dic[qid] = {cid: (hidden[id], labels[id])}
            return {"output": tmp_dic}
            # return {"output": [0]}

if __name__ == '__main__':
    import torch as t
    from torch import avg_pool1d, nn
    cos =nn.CosineSimilarity(dim=-1)
    a = t.tensor([1,2,3]).float()
    b = t.tensor([4,5,6]).float()
    c = t.rand(5,1,3)
    d = t.rand(1,5,3)
    print(cos(c,d).shape)
    # print(c)