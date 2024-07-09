#!/bin/bash

train_data='data/opus_zh_en.json'
val_data='data/flores101-dev.json'
model_dir='/srv/model/huggingface/opus-mt-zh-en'

# Define src_path if necessary or remove this line
# src_path='your/src/path/here'
# config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')

log_path=log/$(date "+%Y-%m-%d").log

# Specify the GPUs to use
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Run torchrun command
torchrun --nproc_per_node=4 --master_port=12355 \
    train.py \
    --train_data $train_data \
    --val_data $val_data \
    --model_dir $model_dir \
    --train_batch 128 \
    --eval_batch 16 \
    --logging_dir './tensorboard/opus-mt-zh-en-625500' \
    --output_dir "./results" \
    --checkpoint results/checkpoint-625500 2>&1 | tee $log_path
