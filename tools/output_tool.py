import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return str(data)


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def output_function1(data, config, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    if 'labelset' in data and 'doc_num' in data and data['doc_num'] != 0:
        metric['ave_len'] = data['labelset'] / data['doc_num']
    return json.dumps(metric)

def binary_output_function(data, config, *args, **params):
    if data['total'] == 0:
        metric = {'acc': 0}
    else:
        metric = {'acc': round(data['right'] / data['total'], 4)}
    return json.dumps(metric)


def squad_output_function(data, config, *args, **params):
    if data["train"]:
        acc = round(data["right"] / data["total"], 4)
        return json.dumps({"tok_acc": acc})
    else:
        if data['NA_tp'] != 0 or data['NA_fp'] != 0:
            pre = float(data['NA_tp']) / (data['NA_tp'] + data["NA_fp"])
            recall = float(data['NA_tp']) / (data['NA_tp'] + data["NA_fn"])
            if pre + recall == 0:
                naf1 = 0
            else:
                naf1 = 2 * pre * recall / (pre + recall)
        else:
            naf1 = 0

        return json.dumps({
            "EM": round(data["em_sum"] / data["total"], 4),
            "F1": round(data["f1_sum"] / data["total"], 4),
            "NA_F1": round(naf1, 4)
            }
        )


def avgloss_output_function(data, config, *args, **params):
    return json.dumps({key: data[key] / data["step"] for key in data if key != "step"})