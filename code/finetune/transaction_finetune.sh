#!/bin/bash

# Скрипт для дообучения LLM на транзакционных данных
# Использование: sh transaction_finetune.sh <base_model_path> <fewshot_size> <gpu_ids>

base_model_path=${1:-"/path/to/your/llm/model"}
fewshot_size=${2:-1024}
gpu_ids=${3:-"0,1,2,3"}

echo "Starting LLM fine-tuning for transactions..."
echo "Base model: $base_model_path"
echo "Few-shot size: $fewshot_size"
echo "GPU IDs: $gpu_ids"

# Настройка accelerate
accelerate config

# Дообучение LLM
echo "Fine-tuning LLM with few-shot samples..."

CUDA_VISIBLE_DEVICES=$gpu_ids accelerate launch transaction_finetune.py \
    --base_model $base_model_path \
    --train_data_path ./data/transactions/fewshot/train-$fewshot_size.json \
    --val_data_path ./data/transactions/fewshot/valid-$fewshot_size.json \
    --output_dir ./models/transactions/$fewshot_size \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --seed 2023

echo "LLM fine-tuning completed!"
echo "Model saved to: ./models/transactions/$fewshot_size" 