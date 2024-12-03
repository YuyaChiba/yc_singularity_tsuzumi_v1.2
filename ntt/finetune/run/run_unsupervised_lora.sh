#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES="0,1,2,3"

prog='../src/unsupervised_lora.py'

# INPUT-DATA PATH
training_data='/ntt/finetune/data/train_unsupervised.jsonl'
num_train_data=$(wc -l < $training_data)
cache_dir='../data/'

# INPUT-LLM PATH
base_modelname='/models/huggingface-base/'

# OUTPUT-MODEL PATH
save_dirname='/ntt/finetune/output/unsupervised_lora/'

# LoRA-MODEL PARAM
lora_r=8 #経験的にこのくらい．16でもいいかも？
lora_alpha=16 #lora_rの2倍がよいと言われている
target_modules="Wqkv" #FFN層を学習する流派もある

# TRAIN PARAM
epochs=2 #データ数が少ない時は長めに学習してからよいcheckpointを探す
per_device_batch_size=1 #GPUに乗る範囲で大きくするのがよい
total_batch_size=32 #実質的なバッチサイズ．経験的にこのくらい
lr=3e-4 #データ数と相談しながら調整
max_seq_len=1024 # 必要に応じて2048
save_steps=1000
n_worker=0
dtype=fp16 #A世代ならbf16
save_total_limit=10

# accelerateからの起動が高速だが，メモリ利用量が増える
#accelerate launch $prog \
python $prog \
	--train_data $training_data \
	--num_train_data $num_train_data \
	--cache_dir $cache_dir \
	--per_device_batch_size $per_device_batch_size \
	--total_batch_size $total_batch_size \
	--lora_r $lora_r \
	--lora_alpha $lora_alpha \
	--dataloader_num_workers $n_worker \
	--epochs $epochs \
	--target_modules $target_modules \
	--base_modelname $base_modelname \
	--save_dirname $save_dirname \
        --lr $lr \
	--max_seq_len $max_seq_len \
	--save_steps $save_steps \
        --model_dtype $dtype \
	--save_total_limit $save_total_limit
