#!/bin/bash

# Скрипт для оценки качества эмбедингов
# Использование: sh evaluate_embeddings.sh <base_model_path> <lora_weights_path> <embeddings_path>

base_model_path=${1:-"/path/to/your/llm/model"}
lora_weights_path=${2:-"./models/transactions/1024"}
embeddings_path=${3:-"embeddings/train/transaction_embeddings.npy"}

echo "=== Embedding Quality Evaluation ==="
echo "Base model: $base_model_path"
echo "LoRA weights: $lora_weights_path"
echo "Embeddings path: $embeddings_path"
echo "==================================="

# Проверяем существование файлов
if [ ! -d "$base_model_path" ]; then
    echo "Error: Base model path does not exist: $base_model_path"
    exit 1
fi

if [ ! -d "$lora_weights_path" ]; then
    echo "Error: LoRA weights path does not exist: $lora_weights_path"
    exit 1
fi

if [ ! -f "$embeddings_path" ]; then
    echo "Error: Embeddings file does not exist: $embeddings_path"
    exit 1
fi

# Запускаем оценку
echo ""
echo "Evaluating embedding quality..."
python code/finetune/transaction_evaluate.py \
    --base_model $base_model_path \
    --lora_weights $lora_weights_path \
    --embeddings_path $embeddings_path

if [ $? -ne 0 ]; then
    echo "Error in embedding evaluation"
    exit 1
fi

echo ""
echo "=== Evaluation completed successfully! ==="
echo "Results saved in: ${embeddings_path%.npy}_evaluation.json"
echo "==========================================" 