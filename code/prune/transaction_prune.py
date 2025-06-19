import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transaction_data_processor import TransactionDataProcessor
from surrogate import train
from utils import get_args
from influence_score import get_influence_score
from effort_score import get_effort_score

import torch
import math
import random
import argparse

def get_transaction_args():
    """Аргументы для транзакционных данных"""
    parser = argparse.ArgumentParser()
    
    # Пути к данным
    parser.add_argument('--train_parquet_dir', default='train_trx_file.parquet', type=str)
    parser.add_argument('--test_parquet_dir', default='test_trx_file.parquet', type=str)
    parser.add_argument('--output_dir', default='./models/', type=str)
    parser.add_argument('--processed_data_dir', default='data/transactions', type=str)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--log_name', type=str, default='log')

    # Параметры последовательности
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--val_split', default=0.1, type=float)

    # Параметры суррогатной модели (SASRec)
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--initializer_range", type=float, default=0.02)

    # Параметры обучения
    parser.add_argument('--do_eval', action='store_true', help='testing mode')
    parser.add_argument("--batch_size", type=int, default=500, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--topN", default='[5, 10, 20, 50]', help="the recommended item num")
    
    # Параметры оптимизации
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    
    # Параметры LLM
    parser.add_argument("--base_model", type=str, default="/path/to/your/llm/model/", help="path to your LLM model")
    parser.add_argument("--resume_from_checkpoint", type=str, default="/path/to/alpaca/lora/adapter/", help="path of the alpaca lora adapter")
    parser.add_argument("--cutoff_len", default=512, type=int, help="cut off length for LLM input")
    parser.add_argument("--lora_r", default=8, type=int, help="lora r")
    parser.add_argument("--lora_alpha", default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", default=0.05, help="lora dropout")

    # Параметры DEALRec
    parser.add_argument('--n_fewshot', default=1024, type=int)
    parser.add_argument("--lamda", type=float, default=0.5, help="strength of gap regularization (effort score)")
    parser.add_argument('--k', default=25, type=int, help="number of groups")
    parser.add_argument("--hard_prune", default=0.1, type=float, help='percentage of hard samples to prune at first')
    parser.add_argument("--iteration", default=1, type=int, help="number of iteration to get the averaged HVP estimation")
    parser.add_argument("--recursion_depth", type=int, default=5000, help="number of recursion depth to get the h_estimation")
    
    args = parser.parse_args()
    return args

def process_transaction_data(args):
    """Обработка транзакционных данных"""
    print("Processing transaction data...")
    
    processor = TransactionDataProcessor(
        train_parquet_dir=args.train_parquet_dir,
        test_parquet_dir=args.test_parquet_dir,
        max_seq_length=args.max_seq_length,
        val_split=args.val_split
    )
    
    train_data, val_data, test_data = processor.process_all_data(args.processed_data_dir)
    
    # Обновляем пути для суррогатной модели
    args.data_dir = args.processed_data_dir + "/"
    args.data_name = "transactions"
    args.data_file = args.data_dir + "train.json"
    
    return processor, train_data, val_data, test_data

def adapt_surrogate_for_transactions(args, processor):
    """Адаптация суррогатной модели для транзакционных данных"""
    # Создаем адаптированные аргументы для SASRec
    args.item_size = len(processor.small_group_map) + 1  # +1 для padding
    args.max_seq_length = processor.max_seq_length
    
    # Создаем последовательности в формате, ожидаемом SASRec
    train_sequences = []
    for seq_data in processor.train_sequences:
        sequence = []
        for trans in seq_data['sequence']:
            # Используем small_group как item_id для SASRec
            item_id = processor.small_group_map[trans['small_group']]
            sequence.append(item_id)
        train_sequences.append(sequence)
    
    return train_sequences

def main():
    """Основная функция для обрезки транзакционных данных"""
    args = get_transaction_args()
    
    # Обработка данных
    processor, train_data, val_data, test_data = process_transaction_data(args)
    
    # Адаптация для суррогатной модели
    train_sequences = adapt_surrogate_for_transactions(args, processor)
    
    # Обучение суррогатной модели
    print("Training surrogate model (SASRec)...")
    trainer = train(args)
    
    # Расчет influence score
    print("Calculating influence scores...")
    influence = get_influence_score(args, trainer)
    
    # Расчет effort score
    print("Calculating effort scores...")
    effort = get_effort_score(args)
    
    # Нормализация скоров
    influence_norm = (influence - torch.min(influence)) / (torch.max(influence) - torch.min(influence))
    effort_norm = (effort - torch.min(effort)) / (torch.max(effort) - torch.min(effort))

    # Общий скор
    overall = influence_norm + args.lamda * effort_norm
    scores_sorted, indices = torch.sort(overall, descending=True)

    # Coverage-enhanced sample selection
    n_prune = math.floor(args.hard_prune * len(scores_sorted))
    scores_sorted = scores_sorted[n_prune:]
    indices = indices[n_prune:]
    print(f"** after hard prune with {args.hard_prune*100}% data:", len(scores_sorted))

    # Разделение скоров на k диапазонов
    s_max = torch.max(scores_sorted)
    s_min = torch.min(scores_sorted)
    print("== max score:", s_max)
    print("== min score:", s_min)
    interval = (s_max - s_min) / args.k

    s_split = [min(s_min + (interval * _), s_max) for _ in range(1, args.k+1)]

    score_split = [[] for _ in range(args.k)]
    for idxx, s in enumerate(scores_sorted):
        for idx, ref in enumerate(s_split):
            if s.item() <= ref:
                score_split[idx].append({indices[idxx].item(): s.item()})
                break
    
    coreset = []
    m = args.n_fewshot
    while len(score_split):
        # Выбираем группу с наименьшим количеством сэмплов
        group = sorted(score_split, key=lambda x: len(x))
        if len(group) > 3:
            print("** sorted group length:", len(group[0]), len(group[1]), len(group[2]), len(group[3]), "...")
        
        group = [strat for strat in group if len(strat)]
        if len(group) > 3:
            print("** sorted group length after removing empty ones:", len(group[0]), len(group[1]), len(group[2]), len(group[3]), "...")

        budget = min(len(group[0]), math.floor(m/len(group)))
        print("** budget for current fewest group:", budget)
        
        # Случайный выбор и добавление в список fewshot индексов
        fewest = group[0]
        selected_idx = random.sample([list(_.keys())[0] for _ in fewest], budget)
        coreset.extend(selected_idx)

        # Удаляем группу с наименьшим количеством
        score_split = group[1:]
        m = m - len(selected_idx)
        
    print(f"** finish selecting {len(coreset)} samples.")

    # Сохранение отобранных сэмплов
    os.makedirs("selected", exist_ok=True)
    torch.save(coreset, f"selected/transactions_{args.n_fewshot}.pt")
    
    # Сохранение отобранных данных для LLM
    selected_train_data = [train_data[idx] for idx in coreset]
    selected_val_data = val_data[:len(val_data)//2]  # Берем половину validation
    
    os.makedirs(f"{args.processed_data_dir}/fewshot", exist_ok=True)
    with open(f"{args.processed_data_dir}/fewshot/train-{args.n_fewshot}.json", 'w') as f:
        import json
        json.dump(selected_train_data, f, indent=2)
    
    with open(f"{args.processed_data_dir}/fewshot/valid-{args.n_fewshot}.json", 'w') as f:
        json.dump(selected_val_data, f, indent=2)
    
    print(f"Selected data saved to {args.processed_data_dir}/fewshot/")
    print("Transaction data pruning completed!")

if __name__ == '__main__':
    main() 