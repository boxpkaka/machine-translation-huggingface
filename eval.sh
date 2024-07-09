#!/bin/bash

data='/home/extrotec/workspace/machine-translation-huggingface/data/flores101-dev.json'
model='/home/extrotec/workspace/machine-translation-huggingface/results/opus-mt-zh-en-ft-155500'

save_dir='./eval/'$(basename "$model")
config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')
log_path=log/$(date "+%Y-%m-%d").log

python \
    eval.py \
    --data $data \
    --model $model \
    --batch_size 8 \
    --num_workers 4 \
    --save_dir $save_dir \
    --device 'cuda:0'


