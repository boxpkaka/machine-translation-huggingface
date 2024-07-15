#!/bin/bash
export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1
export CNCL_MLU_DIRECT_LEVEL=1
export CNCL_SLICE_SIZE=2097152
export CNCL_MEM_POOL_ENABLE=0
# Specify the GPUs to use
export MLU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
train_data='data/opus_en_ja_shard.json'
val_data='data/flores101-dev.json'
model_dir='/workspace/volume/data3-lianxiang/300-MT-Pro/model/opus-mt-en-ja'

train_batch=64
eval_batch=32
num_gpu=8
num_epoch=10
checkpoint=

# Define src_path if necessary or remove this line
# src_path='your/src/path/here'
# config_name=src_name=$(echo $src_path | awk -F'/' '{print $NF}')

log_path=log/$(date "+%Y-%m-%d").log

# Run torchrun command
nohup \
torchrun --nproc_per_node=$num_gpu --master_port=12355 \
    train.py \
    --train_data $train_data \
    --val_data $val_data \
    --model_dir $model_dir \
    --train_batch $train_batch \
    --eval_batch $eval_batch \
    --num_gpu $num_gpu \
    --num_epoch $num_epoch \
    --logging_dir './tensorboard/opus-mt-en-ja' \
    --output_dir "./results/opus-mt-en-ja"  \
    > $log_path 2>&1 &
