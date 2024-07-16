#!/bin/bash

train_data='data/opus_en_ko.json'
val_data='data/flores101-dev-en-ko.json'
model_dir='/srv/model/huggingface/opus-mt-ko-en-finetuned-en-to-ko/'
logging_dir='./tensorboard/test'

train_batch=64
eval_batch=32
num_gpu=4
num_epoch=10
eval_steps=200
save_steps=500
checkpoint=
# Define src_path if necessary or remove this line
# src_path='your/src/path/here'
# config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')

log_path=log/$(date "+%Y-%m-%d").log

# Specify the GPUs to use
export CUDA_VISIBLE_DEVICES="0,1,2,3"

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
    #  2>&1 | tee $log_path
