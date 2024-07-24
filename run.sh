#!/bin/bash

train_data='data/39m-zh-fr.json'
val_data='data/flores101-dev-zh-fr.json'
model_dir='/srv/model/huggingface/opus-mt-zh-fr-mingdong/'
logging_dir='./tensorboard/opus-mt-zh-fr'

train_batch=64
eval_batch=32
num_gpu=4
num_epoch=10
buffer_size=10000
eval_steps=200
save_steps=500
checkpoint=
init_model=true
use_iterable_dataset=true
# Define src_path if necessary or remove this line
# src_path='your/src/path/here'
# config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')

log_path=log/$(date "+%Y-%m-%d").log

# Specify the GPUs to use
export CUDA_VISIBLE_DEVICES="0,1,2,3"


torchrun --nproc_per_node=$num_gpu --master_port=12355 \
    train.py \
    --train_data $train_data \
    --val_data $val_data \
    --model_dir $model_dir \
    --train_batch $train_batch \
    --eval_batch $eval_batch \
    --buffer_size $buffer_size \
    --num_gpu $num_gpu \
    --num_epoch $num_epoch \
    --logging_dir $logging_dir \
    --output_dir "./results" \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    ${checkpoint:+--checkpoint $checkpoint} \
    ${init_model:+--init_model} \
    ${use_iterable_dataset:+--use_iterable_dataset} \
    2>&1 | tee $log_path