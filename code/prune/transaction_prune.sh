#!/bin/bash

# Скрипт для обрезки транзакционных данных
# Использование: sh transaction_prune.sh <lamda> <k> <n_fewshot> <gpu_id>

lamda=${1:-0.5}
k=${2:-25}
n_fewshot=${3:-1024}
gpu_id=${4:-0}

echo "Starting transaction data pruning..."
echo "Parameters: lamda=$lamda, k=$k, n_fewshot=$n_fewshot, gpu_id=$gpu_id"

python transaction_prune.py \
    --train_parquet_dir ../../train_trx_file.parquet \
    --test_parquet_dir ../../test_trx_file.parquet \
    --lamda $lamda \
    --k $k \
    --n_fewshot $n_fewshot \
    --gpu_id $gpu_id \
    --max_seq_length 50 \
    --batch_size 500 \
    --epochs 200 \
    --lr 0.001

echo "Transaction data pruning completed!" 