import json
import torch
import os
import numpy as np
from scipy.spatial import distance

import random
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, LongformerTokenizer

class CaseformerFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.mlm_prob = config.getfloat("train", "mlm_prob")
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)

    def get_article_weight(self, data, i, j, UsePosi=False, UsePosj=False):
        a_i = [a for a in data[i]['xf_article'] if int(a) >= 102] if not UsePosi else [a for a in data[i]['pos_article'] if int(a) >= 102]
        a_j = [a for a in data[j]['xf_article'] if int(a) >= 102] if not UsePosj else [a for a in data[j]['pos_article'] if int(a) >= 102]
        if len(a_i) == 0: weight = 0.0 
        else: 
            # 最简单的实现方式：只看法条交集
            weight = len([a for a in a_j if a in a_i])/len(a_i)  
            if weight == 0.0: return weight

            # CaseEncoder: 
            fg_dic_i = data[i]['fg_article_vector'] if not UsePosi else data[i]['pos_fg_article_vector']
            fg_dic_j = data[j]['fg_article_vector'] if not UsePosj else data[j]['pos_fg_article_vector']
            valid_a_i = fg_dic_i.keys()
            valid_a_j = fg_dic_j.keys()
            common_a = list(set(valid_a_i) & set(valid_a_j))
            if common_a == []: return weight
            
            sim_score = 0.0
            for article in common_a:
                # print('vector: ', fg_dic_i[article], fg_dic_j[article])
                vector_dim = len(fg_dic_i[article])
                min_idx_i = [i for i in range(vector_dim) if fg_dic_i[article][i]==min(fg_dic_i[article])]
                min_idx_j = [i for i in range(vector_dim) if fg_dic_j[article][i]==min(fg_dic_j[article])]
                if list(set(min_idx_i) & set(min_idx_j)) != []: 
                    # print('weight: ', weight)
                    return weight
                else:
                    sim_score = max(sim_score, 1 - distance.cosine([i**(-2) for i in fg_dic_i[article]], [i**(-2) for i in fg_dic_j[article]]))
                    # print('sim_score: ', sim_score)
            
                    
            weight = sim_score * weight
            # print('final weight: ', weight)
            
        return weight
        
    def process(self, data):
        input_ids = []
        attention_mask = []
        if self.mode in ['train', 'valid' ]:
            max_len = min(self.max_len, max([ max(len(inp['fact']), len(inp['pos_case'])) for inp in data]))
            # pos_max_len =  min(self.max_len, max([len(inp['pos_case']) for inp in data]))
            
            # inputx = []
            # mask = np.zeros((len(data), max_len))

            pos_input_ids = []
            pos_attention_mask = []

            data_len = len(data)
            # contra_weights_ori = [[self.get_article_weight(data, i, j, False) for j in range(data_len) if j!=i] for i in range(data_len)]
            contra_weights_oriori = [[self.get_article_weight(data, i, j, False, False) for j in range(data_len) ] for i in range(data_len)]
            contra_weights_oripos = [[self.get_article_weight(data, i, j, False, True) for j in range(data_len)] for i in range(data_len)]
            contra_weights_pospos = [[self.get_article_weight(data, i, j, True, True) for j in range(data_len)] for i in range(data_len)]
            contra_weights = np.concatenate( \
                (np.concatenate((contra_weights_oriori, contra_weights_oripos), axis=1),
                np.concatenate((np.array(contra_weights_oripos).T , contra_weights_pospos), axis=1)), axis=0
            )
            # contra_weights = []

            for line in data:
                tokens = self.tokenizer(line['fact'], padding='max_length', truncation=True, max_length=max_len)
                input_ids.append(tokens['input_ids'])
                attention_mask.append(tokens['attention_mask'])

                pos_tokens = self.tokenizer(line['pos_case'], padding='max_length', truncation=True, max_length=max_len)
                pos_input_ids.append(pos_tokens['input_ids'])
                pos_attention_mask.append(pos_tokens['attention_mask'])

            mlm_inp = [torch.LongTensor(np.array(inp, dtype=np.int16)) for inp in input_ids]
            mlm_ret = self.data_collator(mlm_inp)
            mlm_ret['mask'] = torch.LongTensor(np.array(attention_mask))

            input_ids.extend(pos_input_ids)
            attention_mask.extend(pos_attention_mask)

            contra_input_ids = torch.LongTensor(np.array(input_ids, dtype=np.int16))
            contra_attention_mask = torch.LongTensor(np.array(attention_mask))

            return {'mlm_input_ids': mlm_ret['input_ids'], 'mlm_mask':mlm_ret['mask'], 'mlm_labels':mlm_ret['labels'], 'contra_input_ids': contra_input_ids, 'contra_attention_mask': contra_attention_mask, 'contra_weights': contra_weights}
        elif self.mode in ['test']:
            # max_len = min(self.max_len, max([ len(inp['fact']) for inp in data]))
            max_len = self.max_len
            labels = []
            ridxs = []
            # SEQLEN = 1
            for line in data:
                tokens = self.tokenizer(line['fact'], padding='max_length', truncation=True, max_length=max_len)
                input_ids.append(tokens['input_ids'])
                attention_mask.append(tokens['attention_mask'])
                ridxs.append(line['ridx'])
                labels.append(line['label'])

            input_ids = torch.LongTensor(np.array(input_ids, dtype=np.int16))
            attention_mask = torch.LongTensor(np.array(attention_mask))
            return {'ridx': ridxs, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


        