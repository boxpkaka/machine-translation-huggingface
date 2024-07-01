#!/bin/bash

data='dataset.json'
model_dir='/home/mingdongyu/model/huggingface/nllb-200-distilled-600M/'

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=0,1

# 运行torchrun命令
torchrun --nproc_per_node=2 --master_port=12345 \
    train.py \
    --data $data \
    --model_dir $model_dir


