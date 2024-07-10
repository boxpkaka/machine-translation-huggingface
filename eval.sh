#!/bin/bash

data='/home/extrotec/workspace/machine-translation-huggingface/data/flores101-dev.json'
# model='/srv/model/huggingface/zhenhui_model/zh-en'
model='/srv/model/huggingface/opus-mt-zh-en'

save_dir='./eval/'$(basename "$model")
# save_dir='/home/extrotec/workspace/machine-translation-huggingface/eval/sirui'
config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')
log_path=log/$(date "+%Y-%m-%d").log

python \
    eval.py \
    --data $data \
    --model $model \
    --batch_size 32 \
    --num_workers 4 \
    --save_dir $save_dir \
    --device 'cuda:0'


