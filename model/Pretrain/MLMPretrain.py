from transformers import BertTokenizer,BertConfig,BertForMaskedLM
import torch
from torch import nn
from model.metric import softmax_acc

class MLMPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(MLMPretrain, self).__init__()
        self.plm = config.get("model", "pretrained_model")
        self.mux_num = config.getint("train", "mux_num")

        self.plm_config = BertConfig.from_pretrained(self.plm)
        self.model = BertForMaskedLM.from_pretrained(self.plm)
        # self.model = BertForMaskedLM(self.plm_config)
        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

        self.mapper = nn.Parameter(torch.randn(self.mux_num, self.hidden_size, self.hidden_size))
        self.mapper2 = nn.Parameter(torch.randn(self.mux_num, self.hidden_size, self.hidden_size))
        self.demapper = nn.Parameter(torch.randn(self.mux_num, self.hidden_size, self.hidden_size))
        self.demapper2 = nn.Parameter(torch.randn(self.mux_num, self.hidden_size, self.hidden_size))
        nn.init.normal_(self.mapper, mean=0, std=0.02)
        nn.init.normal_(self.demapper, mean=0, std=0.02)
        nn.init.normal_(self.mapper2, mean=0, std=0.02)
        nn.init.normal_(self.demapper2, mean=0, std=0.02)
        self.tanh = nn.Tanh()

        self.mlm_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def mux(self, hiddens):
        # hiddens: batch, mux_num, seq_len, hidden_size
        ret_mid = self.tanh(torch.matmul(self.mapper.unsqueeze(0), hiddens.transpose(2, 3)).transpose(2, 3).contiguous())
        ret = torch.matmul(self.mapper2.unsqueeze(0), ret_mid.transpose(2, 3)).transpose(2, 3).contiguous()
        return ret.mean(dim=1) # batch, seq_len, hidden_size

    def demux(self, hiddens, reshape=True):
        # hiddens: batch, seq_len, hidden_size
        batch, seq_len, hidden_size = hiddens.size()
        demux_hiddens_mid = self.tanh(torch.matmul(self.demapper.unsqueeze(0), hiddens.transpose(1, 2).unsqueeze(1)).transpose(2, 3).contiguous())
        demux_hiddens = torch.matmul(self.demapper2.unsqueeze(0), demux_hiddens_mid.transpose(2, 3)).transpose(2, 3).contiguous()
        if reshape:
            return demux_hiddens.view(batch * self.mux_num, seq_len, hidden_size).contiguous()
        else:
            return demux_hiddens

    def dup_forward(self, data):
        batch_size, ctx_len = data["input_ids"].size()
        input_emb = self.model.get_input_embeddings()(data["input_ids"]) # total_batch_size, ctx_len, hidden_size
        # input_emb_view = input_emb.view(batch_size, self.mux_num, ctx_len, self.hidden_size) # batch_size, mux_num, ctx_len, hidden_size
        input_emb_view = input_emb.unsqueeze(1).repeat(1, self.mux_num, 1, 1) # batch_size, mux_num, ctx_len, hidden_size

        mux_inp = self.mux(input_emb_view) # torch.matmul(self.mapper.unsqueeze(0), input_emb.unsqueeze(3)).squeeze(-1)
        out = self.model.bert(inputs_embeds=mux_inp)
        hiddens = out["last_hidden_state"] # batch_size, ctx_len, hidden_size

        real_hiddens = self.demux(hiddens).view(batch_size, self.mux_num, ctx_len, self.hidden_size).mean(dim=1) # batch, ctx_len, hidden_size

        prediction_scores = self.model.cls(real_hiddens)#.view(batch_size, self.mux_num, ctx_len, -1).mean(dim=1) # batch_size, mux_num, seq_len
        # prediction_scores = prediction_scores.mean(dim=1)
        mlm_loss = self.mlm_loss(prediction_scores.view(-1, self.plm_config.vocab_size), data["labels"].view(-1))
        # mse_loss = self.mse_loss(input_emb, self.demux(mux_inp))
        loss = mlm_loss
        return loss
        # return {"loss": loss, "acc_result": cal_loss(mlm_loss, mlm_loss, acc_result)}

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()
        batch_size = total_batch_size // self.mux_num
        input_emb = self.model.get_input_embeddings()(data["input_ids"]) # total_batch_size, ctx_len, hidden_size
        input_emb_view = input_emb.view(batch_size, self.mux_num, ctx_len, self.hidden_size) # batch_size, mux_num, ctx_len, hidden_size

        mux_inp = self.mux(input_emb_view) # torch.matmul(self.mapper.unsqueeze(0), input_emb.unsqueeze(3)).squeeze(-1)
        out = self.model.bert(inputs_embeds=mux_inp)
        hiddens = out["last_hidden_state"] # batch_size, ctx_len, hidden_size
        # demux_hiddens = torch.matmul(self.demapper.unsqueeze(0), hiddens.transpose(1, 2).unsqueeze(1)).transpose(2, 3).contiguous()
        # real_hiddens = demux_hiddens.view(total_batch_size, ctx_len, self.hidden_size)
        real_hiddens = self.demux(hiddens)

        prediction_scores = self.model.cls(real_hiddens)
        mlm_loss = self.mlm_loss(prediction_scores.view(-1, self.plm_config.vocab_size), data["labels"].view(-1))
        mse_loss = self.mse_loss(input_emb, self.demux(mux_inp))

        if mode == "train":
            left_hidden, right_hidden = real_hiddens[data["left"]].contiguous(), real_hiddens[data["right"]].contiguous()
            order_loss = self.mse_loss(left_hidden, right_hidden)
            loss = mlm_loss + mse_loss + order_loss
            return {"loss": loss, "acc_result": cal_loss(mlm_loss, mse_loss + order_loss, acc_result)}
        else:
            loss = mlm_loss
            ens_loss = self.dup_forward(data)
            return {"loss": loss, "acc_result": cal_loss(mlm_loss, ens_loss, acc_result)}

def cal_loss(mlm_loss, mse_loss, acc_result):
    if acc_result is None:
        acc_result = {"mlm": 0, "mse": 0, "step": 0}
    if acc_result["step"] > 2000:
        acc_result = {"mlm": 0, "mse": 0, "step": 0}
    acc_result["step"] += 1
    acc_result["mlm"] += mlm_loss.item()
    if mse_loss is not None:
        acc_result["mse"] += mse_loss.item()
    return acc_result 