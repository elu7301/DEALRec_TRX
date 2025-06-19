# 🎯 Резюме: DEALRec для Транзакционных Данных

## �� Что было сделано

Проект DEALRec_TRX полностью адаптирован для работы с транзакционными данными в формате Parquet. Реализован полный пайплайн от обработки данных до оценки качества эмбедингов.

## Ключевые компоненты

### 1. Обработка данных (`transaction_data_processor.py`)
- Чтение parquet-файлов
- Создание словарей и маппингов
- Подготовка данных для LLM
- Нормализация и кодирование

### 2. Обрезка данных (`transaction_prune.py`)
- Использование SASRec как суррогатной модели
- Вычисление influence и effort scores
- DEALRec-методология отбора образцов
- Создание few-shot выборки

### 3. Дообучение LLM (`transaction_finetune.py`)
- LoRA fine-tuning для next token prediction
- Адаптация под транзакционные промпты
- Оптимизация гиперпараметров
- Сохранение обученной модели

### 4. Извлечение эмбедингов (`extract_embeddings.py`)
- Извлечение векторных представлений
- Поддержка многослойных эмбедингов
- Сохранение метаданных
- Обработка больших датасетов

### 5. Оценка качества (`transaction_evaluate.py`) ⭐ **НОВОЕ**
- Анализ косинусного сходства
- Метрики кластеризации (Silhouette, Calinski-Harabasz)
- Анализ по целевым переменным
- t-SNE визуализация

## 🏗️ Архитектура решения

### **Почему SASRec подходит для ваших данных:**

1. **Последовательная природа транзакций** ✅
   - Ваши данные содержат временные последовательности транзакций
   - SASRec специально разработан для последовательных рекомендаций

2. **Временные паттерны** ✅
   - Модель улавливает зависимости между транзакциями
   - Учитывает порядок и временные интервалы

3. **Многомерные признаки** ✅
   - Работает с `event_time`, `amount_rur`, `small_group` одновременно
   - Может обрабатывать несколько типов признаков

4. **Next-item prediction** ✅
   - Предсказывает следующий элемент в последовательности
   - Аналогично next token prediction в LLM

## 🚀 Как использовать

### **1. Быстрый старт:**
```bash
# Тестирование обработки данных
python test_data_processing.py

# Полный пайплайн
sh run_transaction_pipeline.sh "/path/to/your/llm/model" 0.5 25 1024 "0,1,2,3"
```

### **2. Пошаговый запуск:**
```bash
# Шаг 1: Обработка данных
python code/transaction_data_processor.py

# Шаг 2: Обрезка данных
cd code/prune/
sh transaction_prune.sh 0.5 25 1024 0
cd ../../

# Шаг 3: Дообучение LLM
cd code/finetune/
sh transaction_finetune.sh "/path/to/your/llm/model" 1024 "0,1,2,3"
cd ../../

# Шаг 4: Извлечение эмбедингов
python code/finetune/extract_embeddings.py \
    --base_model "/path/to/your/llm/model" \
    --lora_weights ./models/transactions/1024 \
    --parquet_dir train_trx_file.parquet \
    --output_dir embeddings/train
```

## 📊 Входные данные

Ваши Parquet файлы содержат:
- **200 файлов** в `train_trx_file.parquet/` (~741 сэмплов)
- **100 файлов** в `test_trx_file.parquet/` (~86 сэмплов)
- **Структура**: `client_id`, `event_time`, `amount_rur`, `small_group`, `target`, `trx_count`

## 📈 Выходные результаты

1. **Отобранные сэмплы**: `code/prune/selected/transactions_1024.pt`
2. **Обученная LLM**: `models/transactions/1024/`
3. **Эмбединги**: 
   - `embeddings/train/transaction_embeddings.npy`
   - `embeddings/test/transaction_embeddings.npy`

## 🎯 Преимущества подхода

1. **Эффективность** - Использует только 1024 сэмпла вместо полного датасета
2. **Качество** - DEALRec отбирает наиболее информативные сэмплы
3. **Гибкость** - Работает с любой LLM архитектурой
4. **Интерпретируемость** - Эмбединги содержат семантическую информацию
5. **Масштабируемость** - LoRA позволяет обучать большие модели

## 🔧 Настройка для вашей модели

### **Замените пути к модели:**
```bash
# В скриптах замените:
--base_model "/path/to/your/llm/model"
```

### **Настройте параметры LoRA:**
```python
lora_target_modules = [
    "q_proj",
    "v_proj",
    # Добавьте другие модули если нужно
]
```

## 📁 Структура проекта

```
DEALRec_TRX/
├── code/
│   ├── transaction_data_processor.py    # ✅ Новый
│   ├── prune/
│   │   ├── transaction_prune.py         # ✅ Новый
│   │   ├── transaction_prune.sh         # ✅ Новый
│   │   └── ...                          # Оригинальные модули
│   └── finetune/
│       ├── transaction_finetune.py      # ✅ Новый
│       ├── transaction_finetune.sh      # ✅ Новый
│       ├── extract_embeddings.py        # ✅ Новый
│       └── ...                          # Оригинальные модули
├── data/transactions/                   # ✅ Создается автоматически
├── models/transactions/                 # ✅ Создается автоматически
├── embeddings/                          # ✅ Создается автоматически
├── run_transaction_pipeline.sh          # ✅ Новый
├── README_TRANSACTIONS.md               # ✅ Новый
└── test_data_processing.py              # ✅ Новый
```

## 🎉 Готово к использованию!

Проект полностью адаптирован для ваших транзакционных данных. Просто:

1. **Укажите путь к вашей LLM модели**
2. **Запустите тест**: `python test_data_processing.py`
3. **Запустите пайплайн**: `sh run_transaction_pipeline.sh <your_model_path>`

Получите качественные эмбединги для ваших downstream задач! 🚀 

## 🎉 Готово к использованию!

✅ **Проект полностью готов** для обучения LLM на транзакционных данных с автоматической оценкой качества полученных эмбедингов. 