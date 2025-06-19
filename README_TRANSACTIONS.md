# DEALRec для Транзакционных Данных

Этот проект адаптирует метод DEALRec (Data-efficient Fine-tuning for LLM-based Recommendation) для работы с транзакционными данными и извлечения эмбедингов из языковых моделей.

## 🎯 Цель проекта

Обучить LLM на next token prediction для транзакционных данных и получить качественные эмбединги, которые можно использовать для различных задач машинного обучения.

## 🏗️ Архитектура решения

### **Почему SASRec подходит для транзакционных данных:**

1. **Последовательная природа**: Транзакции естественно образуют последовательности во времени
2. **Временные паттерны**: SASRec улавливает зависимости между транзакциями
3. **Многомерные признаки**: Модель работает с несколькими признаками (время, сумма, категория)
4. **Next-item prediction**: Аналогично next token prediction в LLM

### **Пайплайн обработки:**

```
Parquet Data → Preprocessing → SASRec Training → Influence/Effort Scores → 
Data Pruning → LLM Fine-tuning → Embedding Extraction
```

## 📊 Структура данных

### **Входные данные (Parquet):**
- `client_id`: ID клиента
- `event_time`: Временные метки транзакций
- `amount_rur`: Суммы транзакций
- `small_group`: Категории транзакций
- `trans_date`: Даты транзакций
- `trx_count`: Количество транзакций
- `target`: Целевая переменная (класс риска)

### **Пример данных:**
```json
{
  "client_id": "10096",
  "event_time": [7.0, 7.0, 8.0, 9.0, 10.0, ...],
  "amount_rur": [3.01, 3.79, 4.25, 2.33, 1.97, ...],
  "small_group": [3, 8, 2, 2, 3, ...],
  "target": 0
}
```

## 🚀 Быстрый старт

### **1. Подготовка окружения**

```bash
# Установка зависимостей
conda env create -f DEALRec.yaml
conda activate DEALRec

# Дополнительная установка для транзакционных данных
pip install pandas pyarrow scikit-learn
```

### **2. Запуск полного пайплайна**

```bash
# Запуск с вашими параметрами
sh run_transaction_pipeline.sh \
    "/path/to/your/llm/model" \
    0.5 \           # lambda
    25 \            # k (количество групп)
    1024 \          # n_fewshot
    "0,1,2,3"       # GPU IDs
```

### **3. Пошаговый запуск**

#### **Шаг 1: Обработка данных**
```bash
python code/transaction_data_processor.py
```

#### **Шаг 2: Обрезка данных**
```bash
cd code/prune/
sh transaction_prune.sh 0.5 25 1024 0
cd ../../
```

#### **Шаг 3: Дообучение LLM**
```bash
cd code/finetune/
sh transaction_finetune.sh "/path/to/your/llm/model" 1024 "0,1,2,3"
cd ../../
```

#### **Шаг 4: Извлечение эмбедингов**
```bash
python code/finetune/extract_embeddings.py \
    --base_model "/path/to/your/llm/model" \
    --lora_weights ./models/transactions/1024 \
    --parquet_dir train_trx_file.parquet \
    --output_dir embeddings/train
```

## 📁 Структура файлов

```
DEALRec_TRX/
├── code/
│   ├── transaction_data_processor.py    # Обработка транзакционных данных
│   ├── prune/
│   │   ├── transaction_prune.py         # Адаптированная обрезка данных
│   │   ├── transaction_prune.sh         # Скрипт запуска обрезки
│   │   └── ...                          # Оригинальные модули DEALRec
│   └── finetune/
│       ├── transaction_finetune.py      # Адаптированное дообучение LLM
│       ├── transaction_finetune.sh      # Скрипт запуска дообучения
│       ├── extract_embeddings.py        # Извлечение эмбедингов
│       └── ...                          # Оригинальные модули
├── data/
│   └── transactions/                    # Обработанные данные
│       ├── train.json                   # Обучающие данные для LLM
│       ├── valid.json                   # Валидационные данные
│       ├── test.json                    # Тестовые данные
│       ├── vocabulary.json              # Словарь
│       └── fewshot/                     # Отобранные few-shot данные
├── models/
│   └── transactions/                    # Обученные модели
├── embeddings/                          # Извлеченные эмбединги
├── train_trx_file.parquet/             # Исходные обучающие данные
├── test_trx_file.parquet/              # Исходные тестовые данные
└── run_transaction_pipeline.sh         # Основной скрипт
```

## ⚙️ Параметры

### **Параметры обрезки данных:**
- `lamda`: Вес effort score (0.3-1.0)
- `k`: Количество групп для coverage-enhanced selection (25-50)
- `n_fewshot`: Количество отбираемых сэмплов (512-2048)
- `hard_prune`: Процент "сложных" сэмплов для исключения (0.1)

### **Параметры дообучения LLM:**
- `lora_r`: Ранг LoRA (8-16)
- `lora_alpha`: Масштабирующий коэффициент (16-32)
- `learning_rate`: Скорость обучения (1e-4 - 3e-4)
- `num_epochs`: Количество эпох (30-50)

### **Параметры извлечения эмбедингов:**
- `layer_idx`: Индекс слоя для извлечения (-1 для последнего)
- `max_length`: Максимальная длина последовательности (512)

## 📈 Результаты

### **Выходные файлы:**

1. **Отобранные сэмплы**: `code/prune/selected/transactions_1024.pt`
2. **Обученная LLM**: `models/transactions/1024/`
3. **Эмбединги**: 
   - `embeddings/train/transaction_embeddings.npy`
   - `embeddings/test/transaction_embeddings.npy`
   - `embeddings/train/transaction_embeddings_metadata.json`

### **Использование эмбедингов:**

```python
import numpy as np
import json

# Загрузка эмбедингов
embeddings = np.load('embeddings/train/transaction_embeddings.npy')
with open('embeddings/train/transaction_embeddings_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Number of sequences: {metadata['n_sequences']}")
print(f"Embedding dimension: {metadata['embedding_dim']}")

# Использование для downstream задач
# embeddings можно использовать для:
# - Классификации клиентов
# - Кластеризации
# - Аномального детектирования
# - Рекомендательных систем
```

## 🔧 Настройка для вашей модели

### **1. Укажите путь к вашей LLM модели:**
```bash
# В скриптах замените:
--base_model "/path/to/your/llm/model"
```

### **2. Настройте параметры LoRA:**
```python
# В transaction_finetune.py измените:
lora_target_modules = [
    "q_proj",
    "v_proj",
    # Добавьте другие модули если нужно
]
```

### **3. Адаптируйте промпты:**
```python
# В extract_embeddings.py измените create_transaction_prompt():
def create_transaction_prompt(self, sequence: List[Dict]) -> str:
    # Ваша логика создания промптов
    pass
```

## 🎯 Преимущества подхода

1. **Эффективность**: Использует только 1024 сэмпла вместо полного датасета
2. **Качество**: DEALRec отбирает наиболее информативные сэмплы
3. **Гибкость**: Работает с любой LLM архитектурой
4. **Интерпретируемость**: Эмбединги содержат семантическую информацию
5. **Масштабируемость**: LoRA позволяет обучать большие модели

## 🐛 Устранение неполадок

### **Ошибка памяти GPU:**
```bash
# Уменьшите batch_size
--batch_size 64 --micro_batch_size 8
```

### **Ошибка загрузки модели:**
```bash
# Проверьте путь к модели
ls /path/to/your/llm/model/
```

### **Ошибка обработки данных:**
```bash
# Проверьте структуру Parquet файлов
python -c "import pandas as pd; df = pd.read_parquet('train_trx_file.parquet/part-00000-*.parquet'); print(df.columns)"
```

## 📚 Дополнительные ресурсы

- [Оригинальная статья DEALRec](https://arxiv.org/pdf/2401.17197)
- [SASRec архитектура](https://arxiv.org/abs/1808.09781)
- [LoRA метод](https://arxiv.org/abs/2106.09685)

## 🤝 Вклад в проект

Для добавления новых функций или исправления ошибок:
1. Создайте issue с описанием проблемы
2. Сделайте fork репозитория
3. Создайте pull request с вашими изменениями

## 📄 Лицензия

NUS © [NExT++](https://www.nextcenter.org/) 