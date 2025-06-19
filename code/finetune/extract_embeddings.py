import os
import sys
import torch
import numpy as np
import json
from typing import List, Dict, Tuple
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

class EmbeddingExtractor:
    """
    Класс для извлечения эмбедингов из обученной LLM
    """
    
    def __init__(self, base_model_path: str, lora_weights_path: str, 
                 vocabulary_path: str = "data/transactions/vocabulary.json"):
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.vocabulary_path = vocabulary_path
        
        # Загружаем словарь
        self.load_vocabulary()
        
        # Загружаем модель
        self.load_model()
        
    def load_vocabulary(self):
        """Загрузка словаря"""
        with open(self.vocabulary_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.small_group_map = vocab_data['small_group_map']
        self.small_group_map_reverse = vocab_data['small_group_map_reverse']
        self.amount_scaler_mean = np.array(vocab_data['amount_scaler_mean'])
        self.amount_scaler_scale = np.array(vocab_data['amount_scaler_scale'])
        self.target_encoder_classes = vocab_data['target_encoder_classes']
        
        print(f"Vocabulary loaded: {len(self.small_group_map)} categories")
    
    def load_model(self):
        """Загрузка модели"""
        print(f"Loading base model from {self.base_model_path}")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model_path)
        
        # Настройка токенизатора
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        
        print(f"Loading LoRA weights from {self.lora_weights_path}")
        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Загружаем LoRA адаптер
        self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def normalize_amount(self, amounts: List[float]) -> List[float]:
        """Нормализация сумм транзакций"""
        amounts_array = np.array(amounts).reshape(-1, 1)
        normalized = (amounts_array - self.amount_scaler_mean) / self.amount_scaler_scale
        return normalized.flatten().tolist()
    
    def create_transaction_prompt(self, sequence: List[Dict]) -> str:
        """Создание промпта для последовательности транзакций"""
        sequence_text = []
        for i, trans in enumerate(sequence):
            # Нормализуем сумму
            normalized_amount = self.normalize_amount([trans['amount_rur']])[0]
            category_name = self.small_group_map_reverse.get(
                self.small_group_map.get(trans['small_group'], trans['small_group']), 
                trans['small_group']
            )
            
            sequence_text.append(
                f"Transaction {i+1}: time={trans['event_time']:.1f}, "
                f"amount={normalized_amount:.2f}, category={category_name}"
            )
        
        sequence_str = "; ".join(sequence_text)
        
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Given the sequence of financial transactions, predict the next transaction pattern.

### Input:
The client has the following transaction sequence: {sequence_str}

### Response:"""
        
        return prompt
    
    def extract_embeddings(self, sequences: List[List[Dict]], 
                          layer_idx: int = -1) -> np.ndarray:
        """
        Извлечение эмбедингов из указанного слоя модели
        
        Args:
            sequences: Список последовательностей транзакций
            layer_idx: Индекс слоя для извлечения эмбедингов (-1 для последнего)
        
        Returns:
            Массив эмбедингов shape (n_sequences, hidden_size)
        """
        embeddings = []
        
        with torch.no_grad():
            for sequence in tqdm(sequences, desc="Extracting embeddings"):
                # Создаем промпт
                prompt = self.create_transaction_prompt(sequence)
                
                # Токенизируем
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Перемещаем на GPU
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Получаем выходы модели
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Извлекаем эмбединги из указанного слоя
                if layer_idx == -1:
                    # Последний слой
                    hidden_states = outputs.hidden_states[-1]
                else:
                    # Конкретный слой
                    hidden_states = outputs.hidden_states[layer_idx]
                
                # Берем эмбединг последнего токена (или средний по последовательности)
                if hidden_states.shape[1] > 0:
                    # Используем последний токен
                    embedding = hidden_states[0, -1, :].cpu().numpy()
                else:
                    # Если последовательность пустая, используем нулевой эмбединг
                    embedding = np.zeros(hidden_states.shape[-1])
                
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def extract_multilayer_embeddings(self, sequences: List[List[Dict]], 
                                    layer_indices: List[int] = None) -> Dict[int, np.ndarray]:
        """
        Извлечение эмбедингов из нескольких слоев
        
        Args:
            sequences: Список последовательностей транзакций
            layer_indices: Список индексов слоев для извлечения
        
        Returns:
            Словарь с эмбедингами для каждого слоя
        """
        if layer_indices is None:
            # По умолчанию извлекаем из последних 4 слоев
            layer_indices = [-4, -3, -2, -1]
        
        embeddings_dict = {}
        
        with torch.no_grad():
            for sequence in tqdm(sequences, desc="Extracting multi-layer embeddings"):
                prompt = self.create_transaction_prompt(sequence)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                
                for layer_idx in layer_indices:
                    if layer_idx not in embeddings_dict:
                        embeddings_dict[layer_idx] = []
                    
                    hidden_states = outputs.hidden_states[layer_idx]
                    if hidden_states.shape[1] > 0:
                        embedding = hidden_states[0, -1, :].cpu().numpy()
                    else:
                        embedding = np.zeros(hidden_states.shape[-1])
                    
                    embeddings_dict[layer_idx].append(embedding)
        
        # Преобразуем списки в массивы
        for layer_idx in embeddings_dict:
            embeddings_dict[layer_idx] = np.array(embeddings_dict[layer_idx])
        
        return embeddings_dict
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str, 
                       metadata: Dict = None):
        """Сохранение эмбедингов"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем эмбединги
        np.save(output_path, embeddings)
        
        # Сохраняем метаданные
        if metadata:
            metadata_path = output_path.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Embeddings saved to {output_path}")
        print(f"Shape: {embeddings.shape}")
    
    def process_parquet_data(self, parquet_dir: str, output_dir: str = "embeddings"):
        """Обработка данных из Parquet файлов и извлечение эмбедингов"""
        import glob
        
        # Загружаем данные
        parquet_files = glob.glob(f"{parquet_dir}/*.parquet")
        all_sequences = []
        all_metadata = []
        
        for file in tqdm(parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(file)
            
            for _, row in df.iterrows():
                # Извлекаем последовательность
                event_times = np.array(row['event_time'])
                amounts = np.array(row['amount_rur'])
                small_groups = np.array(row['small_group'])
                
                # Создаем последовательность транзакций
                sequence = []
                for i in range(min(len(event_times), 50)):  # Ограничиваем длину
                    transaction = {
                        'event_time': float(event_times[i]),
                        'amount_rur': float(amounts[i]),
                        'small_group': int(small_groups[i])
                    }
                    sequence.append(transaction)
                
                all_sequences.append(sequence)
                all_metadata.append({
                    'client_id': row['client_id'],
                    'target': int(row['target']),
                    'trx_count': int(row['trx_count'])
                })
        
        print(f"Loaded {len(all_sequences)} sequences")
        
        # Извлекаем эмбединги
        print("Extracting embeddings...")
        embeddings = self.extract_embeddings(all_sequences)
        
        # Сохраняем результаты
        output_path = f"{output_dir}/transaction_embeddings.npy"
        self.save_embeddings(embeddings, output_path, {
            'n_sequences': len(all_sequences),
            'embedding_dim': embeddings.shape[1],
            'metadata': all_metadata
        })
        
        return embeddings, all_metadata


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', required=True, help='Path to base LLM model')
    parser.add_argument('--lora_weights', required=True, help='Path to LoRA weights')
    parser.add_argument('--parquet_dir', required=True, help='Path to parquet data directory')
    parser.add_argument('--output_dir', default='embeddings', help='Output directory for embeddings')
    parser.add_argument('--layer_idx', type=int, default=-1, help='Layer index for embedding extraction')
    
    args = parser.parse_args()
    
    # Создаем экстрактор
    extractor = EmbeddingExtractor(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights
    )
    
    # Извлекаем эмбединги
    embeddings, metadata = extractor.process_parquet_data(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir
    )
    
    print(f"Embedding extraction completed!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main() 