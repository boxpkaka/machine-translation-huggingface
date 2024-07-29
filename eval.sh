#!/bin/bash

data=$1
model=$2
deivce=$3
only_eval=$4

save_dir='./eval/'$(basename "$model")
# save_dir='/home/extrotec/workspace/machine-translation-huggingface/eval/sirui'
config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')
log_path=log/$(date "+%Y-%m-%d").log


python \
    eval.py \
    --data $data \
    --model $model \
    --batch_size 64 \
    --num_workers 16 \
    --save_dir $save_dir-$(basename "$data") \
    --device 'cuda:'$deivce \
    ${only_eval:+--only_eval} \



