#!/usr/bin/env python3
"""
Тестовый скрипт для проверки обработки транзакционных данных
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Теперь импортируем модуль
from code.transaction_data_processor import TransactionDataProcessor
import json

def test_data_processing():
    """Тестирование обработки данных"""
    print("=== Testing Transaction Data Processing ===")
    
    # Создаем процессор
    processor = TransactionDataProcessor(
        train_parquet_dir="train_trx_file.parquet",
        test_parquet_dir="test_trx_file.parquet",
        max_seq_length=50,
        val_split=0.1
    )
    
    try:
        # Обрабатываем данные
        print("Processing data...")
        train_data, val_data, test_data = processor.process_all_data("data/transactions")
        
        print(f"✅ Data processing completed successfully!")
        print(f"   Train samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        print(f"   Test samples: {len(test_data)}")
        
        # Проверяем структуру данных
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"✅ Sample data structure:")
            print(f"   Instruction: {sample['instruction'][:100]}...")
            print(f"   Input: {sample['input'][:100]}...")
            print(f"   Output: {sample['output']}")
            print(f"   Client ID: {sample['client_id']}")
            print(f"   Target: {sample['target']}")
        
        # Проверяем словарь
        print(f"✅ Vocabulary created:")
        print(f"   Categories: {len(processor.small_group_map)}")
        print(f"   Sample mapping: {list(processor.small_group_map.items())[:5]}")
        
        # Проверяем файлы
        files_to_check = [
            "data/transactions/train.json",
            "data/transactions/valid.json", 
            "data/transactions/test.json",
            "data/transactions/vocabulary.json"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"✅ File created: {file_path}")
            else:
                print(f"❌ File missing: {file_path}")
        
        print("\n=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_generation():
    """Тестирование генерации few-shot сэмплов"""
    print("\n=== Testing Few-shot Sample Generation ===")
    
    try:
        # Загружаем обработанные данные
        with open("data/transactions/train.json", 'r') as f:
            train_data = json.load(f)
        
        # Создаем few-shot сэмплы
        n_fewshot = 100
        import random
        random.seed(42)
        fewshot_samples = random.sample(train_data, min(n_fewshot, len(train_data)))
        
        # Сохраняем few-shot данные
        os.makedirs("data/transactions/fewshot", exist_ok=True)
        with open(f"data/transactions/fewshot/train-{n_fewshot}.json", 'w') as f:
            json.dump(fewshot_samples, f, indent=2)
        
        with open(f"data/transactions/fewshot/valid-{n_fewshot}.json", 'w') as f:
            json.dump(fewshot_samples[:len(fewshot_samples)//2], f, indent=2)
        
        print(f"✅ Few-shot samples generated:")
        print(f"   Train samples: {len(fewshot_samples)}")
        print(f"   Validation samples: {len(fewshot_samples)//2}")
        print(f"   Files: data/transactions/fewshot/train-{n_fewshot}.json")
        print(f"   Files: data/transactions/fewshot/valid-{n_fewshot}.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during few-shot generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting transaction data processing tests...")
    
    # Тест 1: Обработка данных
    success1 = test_data_processing()
    
    # Тест 2: Генерация few-shot сэмплов
    if success1:
        success2 = test_sample_generation()
    else:
        success2 = False
    
    if success1 and success2:
        print("\n🎉 All tests passed! You can now proceed with the full pipeline.")
        print("\nNext steps:")
        print("1. Update the base model path in the scripts")
        print("2. Run: sh run_transaction_pipeline.sh <your_model_path>")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 