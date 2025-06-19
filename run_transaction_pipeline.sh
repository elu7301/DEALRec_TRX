#!/bin/bash

# Основной скрипт для запуска пайплайна DEALRec для транзакционных данных
# Использование: sh run_transaction_pipeline.sh <base_model_path> <lamda> <k> <n_fewshot> <gpu_ids>

base_model_path=${1:-"/path/to/your/llm/model"}
lamda=${2:-0.5}
k=${3:-25}
n_fewshot=${4:-1024}
gpu_ids=${5:-"0,1,2,3"}

echo "=== DEALRec Transaction Pipeline ==="
echo "Base model: $base_model_path"
echo "Lambda: $lamda"
echo "K (groups): $k"
echo "Few-shot size: $n_fewshot"
echo "GPU IDs: $gpu_ids"
echo "=================================="

# Шаг 1: Обработка данных
echo ""
echo "Step 1: Processing transaction data..."
python code/transaction_data_processor.py
if [ $? -ne 0 ]; then
    echo "Error in data processing step"
    exit 1
fi

# Шаг 2: Обрезка данных (Data Pruning)
echo ""
echo "Step 2: Data pruning with DEALRec..."
cd code/prune/
sh transaction_prune.sh $lamda $k $n_fewshot ${gpu_ids%,*}  # Берем первый GPU для pruning
cd ../../
if [ $? -ne 0 ]; then
    echo "Error in data pruning step"
    exit 1
fi

# Шаг 3: Дообучение LLM
echo ""
echo "Step 3: Fine-tuning LLM with few-shot samples..."
cd code/finetune/
sh transaction_finetune.sh $base_model_path $n_fewshot $gpu_ids
cd ../../
if [ $? -ne 0 ]; then
    echo "Error in LLM fine-tuning step"
    exit 1
fi

# Шаг 4: Извлечение эмбедингов
echo ""
echo "Step 4: Extracting embeddings..."
python code/finetune/extract_embeddings.py \
    --base_model $base_model_path \
    --lora_weights ./models/transactions/$n_fewshot \
    --parquet_dir train_trx_file.parquet \
    --output_dir embeddings/train

python code/finetune/extract_embeddings.py \
    --base_model $base_model_path \
    --lora_weights ./models/transactions/$n_fewshot \
    --parquet_dir test_trx_file.parquet \
    --output_dir embeddings/test

if [ $? -ne 0 ]; then
    echo "Error in embedding extraction step"
    exit 1
fi

# Шаг 5: Оценка качества эмбедингов
echo ""
echo "Step 5: Evaluating embedding quality..."
python code/finetune/transaction_evaluate.py \
    --base_model $base_model_path \
    --lora_weights ./models/transactions/$n_fewshot \
    --embeddings_path embeddings/train/transaction_embeddings.npy

python code/finetune/transaction_evaluate.py \
    --base_model $base_model_path \
    --lora_weights ./models/transactions/$n_fewshot \
    --embeddings_path embeddings/test/transaction_embeddings.npy

if [ $? -ne 0 ]; then
    echo "Error in embedding evaluation step"
    exit 1
fi

echo ""
echo "=== Pipeline completed successfully! ==="
echo "Results saved in:"
echo "- Selected samples: code/prune/selected/transactions_$n_fewshot.pt"
echo "- LLM model: models/transactions/$n_fewshot/"
echo "- Train embeddings: embeddings/train/transaction_embeddings.npy"
echo "- Test embeddings: embeddings/test/transaction_embeddings.npy"
echo "- Train evaluation: embeddings/train/transaction_embeddings_evaluation.json"
echo "- Test evaluation: embeddings/test/transaction_embeddings_evaluation.json"
echo "========================================" 