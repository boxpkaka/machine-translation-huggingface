#!/bin/bash

data=$1
# model='/home/extrotec/workspace/machine-translation-huggingface/results/opus-mt-zh-en-ft-718000'
model=$2

save_dir='./eval/'$(basename "$model")
# save_dir='/home/extrotec/workspace/machine-translation-huggingface/eval/sirui'
config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')
log_path=log/$(date "+%Y-%m-%d").log


python \
    eval.py \
    --data $data \
    --model $model \
    --batch_size 32 \
    --num_workers 16 \
    --save_dir $save_dir \
    --device 'cuda:0'



