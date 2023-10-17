#!/bin/sh
# python3 -m torch.distributed.launch 
# python3 train.py --config config/Caseformer.config --gpu 7 2>&1 | tee -a log/AIS/Caseformer_1.log
# -m torch.distributed.launch --nproc_per_node=4

# torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --config config/Caseformer.config --gpu 0,1,2,3,4,5,6,7 2>&1 | tee -a log/AIS/Caseformer_test.log 

# torchrun --standalone --nnodes=1 --nproc_per_node=1 test.py --checkpoint /liuzyai04/thuir/myx/datamux-master/checkpoint/Caseformer_fg_bs=8/2_5000.pkl --config config/Caseformer.config --gpu 1 --result /liuzyai04/thuir/myx/eval_results/lecard_test.json

torchrun --standalone --nnodes=1 --nproc_per_node=1 test.py --checkpoint 123 --config config/Caseformer.config --gpu 2 --result /liuzyai04/thuir/myx/eval_results/lecard_test.json