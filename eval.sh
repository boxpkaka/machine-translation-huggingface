#!/bin/bash

data='data/flores101-dev.json'
# model='/srv/model/huggingface/zhenhui_model/zh-en'
model='/workspace/volume/data3-lianxiang/300-MT-Pro/machine-translation-huggingface/results/opus-mt-en-ja/opus-mt-en-ja-ft-5000'

save_dir='./eval/'$(basename "$model")
# save_dir='/home/extrotec/workspace/machine-translation-huggingface/eval/sirui'
config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')
log_path=log/$(date "+%Y-%m-%d").log

python \
    eval.py \
    --data $data \
    --model $model \
    --batch_size 32 \
    --num_workers 8 \
    --save_dir $save_dir \
    --device 'mlu:0'


