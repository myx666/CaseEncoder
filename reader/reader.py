from torch.utils.data import DataLoader
import logging

import formatter as form
from dataset import dataset_list
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from kara_storage.pytorch.base import KaraPytorchDatasetBase
logger = logging.getLogger(__name__)
# from model_center.dataset import DistributedDataLoader

collate_fn = {}
formatter = {}

def init_formatter(config, task_list, *args, **params):
    for task in task_list:
        formatter[task] = form.init_formatter(config, task, *args, **params)
        collate_fn[task] = formatter[task].process


def init_one_dataset(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_dataset_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_dataset_type has not been defined in config file, use [dataset] train_dataset_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_dataset_type" % temp_mode)

    if which in dataset_list:
        dataset = dataset_list[which](config, mode, *args, **params)
        batch_size = config.getint("train", "batch_size")
        shuffle = config.getboolean("train", "shuffle")
        reader_num = config.getint("train", "reader_num")
        drop_last = True
        if mode in ["valid", "test"]:
            if mode == "test":
                drop_last = False

            try:
                batch_size = config.getint("eval", "batch_size")
            except Exception as e:
                logger.warning("[eval] batch size has not been defined in config file, use [train] batch_size instead.")

            try:
                shuffle = config.getboolean("eval", "shuffle")
            except Exception as e:
                shuffle = False
                logger.warning("[eval] shuffle has not been defined in config file, use false as default.")
            try:
                reader_num = config.getint("eval", "reader_num")
            except Exception as e:
                logger.warning("[eval] reader num has not been defined in config file, use [train] reader num instead.")

        # if config.getboolean('distributed', 'use') and not isinstance(dataset, KaraPytorchDatasetBase):
        #     sampler = DistributedSampler(dataset)
        # else:
        #     sampler = None
        #     collate_fn[mode] = formatter[mode].process
        
        if isinstance(dataset, KaraPytorchDatasetBase):
            sampler = None
            # sampler = DistributedSampler(dataset) if config.getboolean('distributed', 'use') else None
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    # shuffle=shuffle,
                                    num_workers=reader_num, 
                                    collate_fn=collate_fn[mode],
                                    # drop_last=drop_last,
                                    sampler=sampler
                                    )
        else:
            raise NotImplementedError
        return dataloader
    else:
        logger.error("There is no dataset called %s, check your config." % which)
        raise NotImplementedError


def init_test_dataset(config, *args, **params):
    init_formatter(config, ["test"], *args, **params)
    test_dataset = init_one_dataset(config, "test", *args, **params)

    return test_dataset


def init_dataset(config, *args, **params):
    init_formatter(config, ["train", "valid"], *args, **params)
    train_dataset = init_one_dataset(config, "train", *args, **params)
    valid_dataset = init_one_dataset(config, "valid", *args, **params)

    return train_dataset, valid_dataset


if __name__ == "__main__":
    pass
