import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from typing import List, Dict, Tuple
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class TransactionDataProcessor:
    """
    Обработчик транзакционных данных для DEALRec
    """
    
    def __init__(self, train_parquet_dir: str, test_parquet_dir: str, 
                 max_seq_length: int = 50, val_split: float = 0.1):
        self.train_parquet_dir = train_parquet_dir
        self.test_parquet_dir = test_parquet_dir
        self.max_seq_length = max_seq_length
        self.val_split = val_split
        
        # Энкодеры для категориальных признаков
        self.amount_scaler = StandardScaler()
        self.small_group_encoder = LabelEncoder()
        self.target_encoder = LabelEncoder()
        
        # Словари для маппинга
        self.small_group_map = {}
        self.small_group_map_reverse = {}
        
    def load_parquet_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Загрузка всех Parquet файлов"""
        print("Loading train data...")
        train_files = glob.glob(f"{self.train_parquet_dir}/*.parquet")
        train_data = []
        for file in train_files:
            df = pd.read_parquet(file)
            train_data.append(df)
        train_df = pd.concat(train_data, ignore_index=True)
        
        print("Loading test data...")
        test_files = glob.glob(f"{self.test_parquet_dir}/*.parquet")
        test_data = []
        for file in test_files:
            df = pd.read_parquet(file)
            test_data.append(df)
        test_df = pd.concat(test_data, ignore_index=True)
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_sequences(self, df: pd.DataFrame) -> List[Dict]:
        """Предобработка последовательностей транзакций"""
        processed_sequences = []
        
        for _, row in df.iterrows():
            # Проверяем наличие NaN в target
            if pd.isna(row['target']):
                print(f"Warning: NaN target found for client {row['client_id']}, skipping...")
                continue
                
            # Извлекаем последовательности
            event_times = np.array(row['event_time'])
            amounts = np.array(row['amount_rur'])
            small_groups = np.array(row['small_group'])
            
            # Проверяем на NaN в последовательностях
            if np.isnan(event_times).any() or np.isnan(amounts).any() or np.isnan(small_groups).any():
                print(f"Warning: NaN values found in sequences for client {row['client_id']}, skipping...")
                continue
            
            # Ограничиваем длину последовательности
            seq_length = min(len(event_times), self.max_seq_length)
            
            # Создаем последовательность транзакций
            sequence = []
            for i in range(seq_length):
                transaction = {
                    'event_time': float(event_times[i]),
                    'amount_rur': float(amounts[i]),
                    'small_group': int(small_groups[i])
                }
                sequence.append(transaction)
            
            processed_sequences.append({
                'client_id': row['client_id'],
                'sequence': sequence,
                'target': int(row['target']),
                'trx_count': int(row['trx_count']) if not pd.isna(row['trx_count']) else 0
            })
        
        print(f"Processed {len(processed_sequences)} valid sequences out of {len(df)} total")
        return processed_sequences
    
    def create_vocabulary(self, sequences: List[Dict]):
        """Создание словаря для токенизации"""
        # Собираем все уникальные small_group
        all_small_groups = set()
        all_amounts = []
        
        for seq_data in sequences:
            for trans in seq_data['sequence']:
                all_small_groups.add(trans['small_group'])
                all_amounts.append(trans['amount_rur'])
        
        # Создаем маппинг для small_group
        unique_groups = sorted(list(all_small_groups))
        self.small_group_map = {group: idx + 1 for idx, group in enumerate(unique_groups)}  # +1 для padding
        self.small_group_map_reverse = {idx + 1: group for idx, group in enumerate(unique_groups)}
        
        # Нормализуем суммы
        all_amounts = np.array(all_amounts).reshape(-1, 1)
        self.amount_scaler.fit(all_amounts)
        
        # Энкодим таргеты
        targets = [seq['target'] for seq in sequences]
        self.target_encoder.fit(targets)
        
        print(f"Vocabulary size (small_groups): {len(self.small_group_map)}")
        print(f"Amount range: {np.min(all_amounts):.2f} - {np.max(all_amounts):.2f}")
        print(f"Amount mean: {self.amount_scaler.mean_[0]:.2f}, std: {self.amount_scaler.scale_[0]:.2f}")
        print(f"Target classes: {len(self.target_encoder.classes_)}")
    
    def tokenize_sequence(self, sequence: List[Dict]) -> Tuple[List[int], List[float], List[int]]:
        """Токенизация последовательности транзакций"""
        event_times = []
        amounts = []
        small_groups = []
        
        for trans in sequence:
            event_times.append(trans['event_time'])
            amounts.append(trans['amount_rur'])
            small_groups.append(self.small_group_map[trans['small_group']])
        
        # Нормализуем суммы
        amounts = self.amount_scaler.transform(np.array(amounts).reshape(-1, 1)).flatten()
        
        return event_times, amounts.tolist(), small_groups
    
    def prepare_llm_data(self, sequences: List[Dict]) -> List[Dict]:
        """Подготовка данных для LLM в формате инструкций"""
        llm_data = []
        
        for seq_data in sequences:
            event_times, amounts, small_groups = self.tokenize_sequence(seq_data['sequence'])
            
            # Создаем текстовое представление последовательности
            sequence_text = []
            for i, (time, amount, group) in enumerate(zip(event_times, amounts, small_groups)):
                group_name = self.small_group_map_reverse[group]
                sequence_text.append(f"Transaction {i+1}: time={time:.1f}, amount={amount:.2f}, category={group_name}")
            
            sequence_str = "; ".join(sequence_text)
            
            # Создаем инструкцию для LLM
            instruction = "Given the sequence of financial transactions, predict the next transaction pattern."
            input_text = f"The client has the following transaction sequence: {sequence_str}"
            
            # Создаем несколько вариантов ответов для разных аспектов
            target_class = self.target_encoder.inverse_transform([seq_data['target']])[0]
            
            # Предсказываем следующую категорию (если есть)
            if len(small_groups) > 1:
                next_category = self.small_group_map_reverse[small_groups[-1]]
                output = f"Next transaction category: {next_category}, Client risk class: {target_class}"
            else:
                output = f"Client risk class: {target_class}"
            
            llm_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "client_id": seq_data['client_id'],
                "target": seq_data['target'],
                "sequence_length": len(seq_data['sequence'])
            })
        
        return llm_data
    
    def split_data(self, sequences: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Разделение данных на train и validation"""
        train_sequences, val_sequences = train_test_split(
            sequences, test_size=self.val_split, random_state=42, 
            stratify=[seq['target'] for seq in sequences]
        )
        return train_sequences, val_sequences
    
    def save_processed_data(self, train_data: List[Dict], val_data: List[Dict], 
                          test_data: List[Dict], output_dir: str = "data/transactions"):
        """Сохранение обработанных данных"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем LLM данные
        with open(f"{output_dir}/train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(f"{output_dir}/valid.json", 'w') as f:
            json.dump(val_data, f, indent=2)
        
        with open(f"{output_dir}/test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Сохраняем словари
        vocab_data = {
            'small_group_map': self.small_group_map,
            'small_group_map_reverse': self.small_group_map_reverse,
            'amount_scaler_mean': self.amount_scaler.mean_.tolist(),
            'amount_scaler_scale': self.amount_scaler.scale_.tolist(),
            'target_encoder_classes': self.target_encoder.classes_.tolist()
        }
        
        with open(f"{output_dir}/vocabulary.json", 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Data saved to {output_dir}")
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
    
    def process_all_data(self, output_dir: str = "data/transactions"):
        """Полная обработка всех данных"""
        print("Loading data...")
        train_df, test_df = self.load_parquet_data()
        
        print("Preprocessing sequences...")
        train_sequences = self.preprocess_sequences(train_df)
        test_sequences = self.preprocess_sequences(test_df)
        
        print("Creating vocabulary...")
        all_sequences = train_sequences + test_sequences
        self.create_vocabulary(all_sequences)
        
        print("Preparing LLM data...")
        train_llm_data = self.prepare_llm_data(train_sequences)
        test_llm_data = self.prepare_llm_data(test_sequences)
        
        print("Splitting train/validation...")
        train_data, val_data = self.split_data(train_llm_data)
        
        print("Saving processed data...")
        self.save_processed_data(train_data, val_data, test_llm_data, output_dir)
        
        return train_data, val_data, test_llm_data


class TransactionDataset(Dataset):
    """Датасет для транзакционных данных"""
    
    def __init__(self, sequences: List[Dict], max_seq_length: int = 50, 
                 small_group_map: Dict = None, amount_scaler: StandardScaler = None):
        self.sequences = sequences
        self.max_seq_length = max_seq_length
        self.small_group_map = small_group_map
        self.amount_scaler = amount_scaler
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        sequence = seq_data['sequence']
        
        # Токенизируем последовательность
        event_times, amounts, small_groups = self.tokenize_sequence(sequence)
        
        # Паддинг до максимальной длины
        pad_length = self.max_seq_length - len(small_groups)
        if pad_length > 0:
            event_times = [0.0] * pad_length + event_times
            amounts = [0.0] * pad_length + amounts
            small_groups = [0] * pad_length + small_groups
        
        # Обрезаем до максимальной длины
        event_times = event_times[-self.max_seq_length:]
        amounts = amounts[-self.max_seq_length:]
        small_groups = small_groups[-self.max_seq_length:]
        
        return {
            'client_id': seq_data['client_id'],
            'event_times': torch.tensor(event_times, dtype=torch.float32),
            'amounts': torch.tensor(amounts, dtype=torch.float32),
            'small_groups': torch.tensor(small_groups, dtype=torch.long),
            'target': torch.tensor(seq_data['target'], dtype=torch.long),
            'sequence_length': torch.tensor(len(sequence), dtype=torch.long)
        }
    
    def tokenize_sequence(self, sequence: List[Dict]) -> Tuple[List[float], List[float], List[int]]:
        """Токенизация последовательности"""
        event_times = []
        amounts = []
        small_groups = []
        
        for trans in sequence:
            event_times.append(trans['event_time'])
            amounts.append(trans['amount_rur'])
            small_groups.append(self.small_group_map[trans['small_group']])
        
        # Нормализуем суммы
        amounts = self.amount_scaler.transform(np.array(amounts).reshape(-1, 1)).flatten()
        
        return event_times, amounts.tolist(), small_groups


if __name__ == "__main__":
    # Пример использования
    processor = TransactionDataProcessor(
        train_parquet_dir="train_trx_file.parquet",
        test_parquet_dir="test_trx_file.parquet",
        max_seq_length=50
    )
    
    train_data, val_data, test_data = processor.process_all_data()
    print("Data processing completed!") 