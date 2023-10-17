from transformers import BertTokenizer,BertConfig,BertForMaskedLM
import torch
from torch import nn
from model.metric import softmax_acc

class VallinaPretrain(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(VallinaPretrain, self).__init__()
        self.plm = config.get("model", "pretrained_model")

        self.plm_config = BertConfig.from_pretrained(self.plm)
        self.model = BertForMaskedLM.from_pretrained(self.plm)
        # self.model = BertForMaskedLM(self.plm_config)
        self.hidden_size = self.model.config.hidden_size
        self.layer_num = self.model.config.num_hidden_layers

    def forward(self, data, config, gpu_list, acc_result, mode):
        total_batch_size, ctx_len = data["input_ids"].size()

        out = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"]
        )

        return {"loss": out["loss"], "acc_result": cal_loss(out["loss"], None, acc_result)}

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