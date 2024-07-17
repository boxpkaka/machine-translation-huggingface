#!/bin/bash

train_data='data/opus-zh-ja.json'
val_data='data/flores101-dev-zh-ja.json'
model_dir='/home/extrotec/workspace/machine-translation-huggingface/results/checkpoint-7440'
logging_dir='./tensorboard/opus-mt-zh-ja-mingdong'

train_batch=64
eval_batch=32
num_gpu=2
num_epoch=10
eval_steps=200
save_steps=500
checkpoint=
init_model=true
# Define src_path if necessary or remove this line
# src_path='your/src/path/here'
# config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')

log_path=log/$(date "+%Y-%m-%d").log

# Specify the GPUs to use
export CUDA_VISIBLE_DEVICES="0,1"

# Run torchrun command
torchrun --nproc_per_node=$num_gpu --master_port=12355 \
    train.py \
    --train_data $train_data \
    --val_data $val_data \
    --model_dir $model_dir \
    --train_batch $train_batch \
    --eval_batch $eval_batch \
    --num_gpu $num_gpu \
    --num_epoch $num_epoch \
    --logging_dir $logging_dir \
    --output_dir "./results" \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    ${checkpoint:+--checkpoint $checkpoint} \
    ${init_model:+--init_model} \
    #  2>&1 | tee $log_path
