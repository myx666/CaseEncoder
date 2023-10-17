import logging
# from math import dist
import os
from statistics import mode
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler as lrs
from timeit import default_timer as timer

from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast
from kara_storage.pytorch.base import KaraPytorchDatasetBase
from tools import output_log,print_rank

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step, lr_scheduler):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step,
        "lr_scheduler": lr_scheduler.state_dict(),
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))

def train(parameters, config, gpu_list, do_test=False, local_rank=-1):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        output_log(logger, "Output path exists, check whether need to change a name of model", logging.WARNING)
        # logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model, optimizer, dataset = parameters["model"], parameters["optimizer"], parameters["train_dataset"]
    global_step, output_function = parameters["global_step"], parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    grad_accumulate = config.getint("train", "grad_accumulate")

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.getint('train', 'warmup_steps'), num_training_steps=config.getint('train', 'training_steps'))

    fp16 = config.getboolean('train', 'fp16')
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
        dtype = torch.bfloat16 if config.getboolean("train", "bf16") else torch.float16
    max_grad_norm = config.getfloat('train', 'max_grad_norm')
    valid_mode = config.get('train', 'valid_mode')
    if valid_mode != 'step' and valid_mode != 'batch':
        raise ValueError('The value of valid_mode is invalid.')
    no_valid = config.getboolean("train", "no_valid")

    print_rank('valid_mode', valid_mode, "no_valid", no_valid)
    step_epoch = None
    if valid_mode == 'step':
        step_epoch = config.getint('train', 'step_epoch')
    print_rank('step_epoch', step_epoch)
    output_log(logger, "Start training", logging.INFO)
    

    # model.save_pretrained('/liuzyai04/thuir/myx/datamux-master/huggingface_model/%s'%(config.get('train', 'checkpoint_name')))
    # print(1/0)

    print_rank("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
    total_len = len(dataset)

    writer = SummaryWriter(log_dir="runs/%s" % (config.get("output", "model_name")))

    for epoch_num in range(trained_epoch, epoch):
        model.train()
        start_time = timer()
        current_epoch = epoch_num

        # exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        total_mlm_loss = 0
        total_contra_loss = 0

        output_info = ""
        step = -1
        if hasattr(dataset, "dataset") and isinstance(dataset.dataset, KaraPytorchDatasetBase): 
            dataset.dataset.set_epoch(epoch_num)
        
        for step, data in enumerate(dataset):
            # print('in step%d ...'% step)
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            if fp16:
                with autocast(dtype=torch.bfloat16):
                # with autocast():
                    results = model(data, config, gpu_list, acc_result, "train")
            else:
                results = model(data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]
            
            avg_loss_gather, avg_mlm_loss_gather, avg_contra_loss_gather = results["avg_loss_gather"], results["avg_mlm_loss_gather"], results["avg_contra_loss_gather"]
            total_loss += float(avg_loss_gather)
            total_mlm_loss += float(avg_mlm_loss_gather)
            total_contra_loss += float(avg_contra_loss_gather)

            if local_rank == 0:
                writer.add_scalars(config.get("output", "model_name"), {'loss' : total_loss/ (step + 1), 
                                  'total_mlm_loss' : total_mlm_loss/ (step + 1), 
                                  'total_contra_loss' : total_contra_loss/ (step + 1)}, (current_epoch-1)*total_len + step)

            loss = loss / grad_accumulate
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # if torch.distributed.get_rank() == 0:
            #     if step % 100 == 0:
            #         print("att_mat1", model.module.att_mat1)
            #         print("======")
            
            if (step + 1) % grad_accumulate == 0:
                if max_grad_norm is not None and max_grad_norm > 0:
                    if fp16:
                        scaler.unscale_(optimizer)
                    if hasattr(optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        optimizer.clip_grad_norm(max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_grad_norm
                        )

                optimizer_was_run = True
                if fp16:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    optimizer.step()
                if optimizer_was_run:
                    lr_scheduler.step()
                optimizer.zero_grad()
            
            if step % output_time == 0 and local_rank <= 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time
                # print(local_rank, "%.3lf" % (total_loss / (step + 1)))
                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))), "%.3lf %.3lf %.3lf " % (total_loss / (step + 1), total_mlm_loss / (step + 1), total_contra_loss / (step + 1)), output_info, None, config)
            
            global_step += 1
            if (step + 1) % grad_accumulate == 0 and valid_mode == 'step' and int((step + 1) / grad_accumulate) % step_epoch == 0:
                acc_result = None
                if local_rank <= 0:
                    print_rank()
                    checkpoint(os.path.join(output_path, "%d_%d.pkl" % (current_epoch, step + 1)), model, optimizer, current_epoch, config, global_step, lr_scheduler)
                #     # path = os.path.join(output_path, 'model_%d_%d' % (current_epoch, (step + 1) // grad_accumulate))
                #     # if local_rank < 0:
                #     #     model.save_pretrained(path)
                #     # else:
                #     #     model.module.save_pretrained(path)
                if not no_valid:
                    with torch.no_grad():
                        valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)


            # break
        if step == -1:
            output_log(logger, "No data in this epoch", logging.ERROR)
            # logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        print_rank(valid_mode != "batch", no_valid)
        if (valid_mode != "batch") or no_valid:
            print_rank("skip validation")
            continue
        print_rank("==" * 10, "begin saving model and validation", "==" * 10)
        if local_rank <= 0:
            output_info = output_function(acc_result, config)
            delta_t = timer() - start_time
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf %.3lf %.3lf " % (total_loss / (step + 1), total_mlm_loss / (step + 1), total_contra_loss / (step + 1)), output_info, None, config)

        if local_rank <= 0:
            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, lr_scheduler)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)
                if do_test:
                    valid(model, test_dataset, current_epoch, config, gpu_list, output_function, mode="test")
    
    writer.close()
