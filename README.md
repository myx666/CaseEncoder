# CaseEncoder 

## Accepted by EMNLP2023
CaseEncoder: A Knowledge-enhanced Pre-trained Model for Legal Case
Encoding

## Before Start

Please unzip all .blk.zip files in ``data``

A few parameters in ``config/Caseformer.config`` need to be specified:

- ``train_data_path``, ``valid_data_path`` in ``config/Caseformer.config`` is the path of the pre-training and validation dataset, which is not included in this repo due to the space limit. But we provide exactly the same checkpoint of CaseEncoder reported in our paper [here](https://drive.google.com/file/d/1KL_cKyiRsnz4FOiFMfGBQrPbEXyNpl5Q/view?usp=drive_link).
- ``test_kara_dataset`` is the test dataset you would like to use. We provide three choices: ``lecard``, ``cail-lcr21``, and ``cail-lcr22``.

## Model Training

Training from the start:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=YOUR_GPU_NUMBER train.py --config config/Caseformer.config --gpu YOUR_GPU_LIST 2>&1 | tee -a log/Caseformer.log 
```

Training from a checkpoint:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=YOUR_GPU_NUMBER train.py --checkpoint YOUR_CHECKPOINT_PATH --config config/Caseformer.config --gpu YOUR_GPU_LIST 2>&1 | tee -a log/Caseformer.log 
```

## Model Evaluation

To validate the checkpoint of CaseEncoder

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 test.py --checkpoint YOUR_CHECKPOINT_PATH --config config/Caseformer.config --gpu 0 --result YOUR_RESULT_STORAGE_PATH
```

- where ``YOUR_CHECKPOINT_PATH`` is the path of ``CaseEncoder`` checkpoint you download.
