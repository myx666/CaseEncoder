import torch.distributed as dist
import logging

def output_log(logger: logging.Logger, info: str, level: int = logging.INFO, *args):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        logger._log(level, info, args)

def print_rank(*arg):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        print(*arg)