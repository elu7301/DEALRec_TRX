# DEALRec для транзакционных данных

Этот проект адаптирует методологию DEALRec для работы с транзакционными данными в формате Parquet. Проект включает полный пайплайн от обработки данных до извлечения эмбедингов и оценки их качества.

## Структура проекта

```
DEALRec_TRX/
├── code/
│   ├── transaction_data_processor.py    # Обработка parquet-данных
│   ├── prune/
│   │   ├── transaction_prune.py         # Обрезка данных с SASRec
│   │   └── transaction_prune.sh         # Скрипт запуска обрезки
│   └── finetune/
│       ├── transaction_finetune.py      # Дообучение LLM
│       ├── transaction_finetune.sh      # Скрипт запуска дообучения
│       ├── extract_embeddings.py        # Извлечение эмбедингов
│       └── transaction_evaluate.py      # Оценка качества эмбедингов
├── data/
│   └── transactions/                    # Обработанные данные
├── models/
│   └── transactions/                    # Обученные модели
├── embeddings/                          # Извлеченные эмбединги
├── run_transaction_pipeline.sh          # Полный пайплайн
├── evaluate_embeddings.sh               # Оценка качества эмбедингов
└── test_data_processing.py              # Тест обработки данных
```

## Установка и настройка

### 1. Клонирование репозитория
```bash
git clone <URL_репозитория>
cd DEALRec_TRX
```

### 2. Создание окружения
```bash
conda create -n dealrec python=3.10 -y
conda activate dealrec
pip install -r requirements.txt
```

### 3. Подготовка данных
Поместите ваши parquet-файлы в корневую директорию:
- `train_trx_file.parquet/` - обучающие данные
- `test_trx_file.parquet/` - тестовые данные

## Использование

### Полный пайплайн

Запуск всего процесса от обработки данных до оценки качества:

```bash
sh run_transaction_pipeline.sh "/path/to/your/llm/model" 0.5 25 1024 "0,1,2,3"
```

**Параметры:**
- `base_model_path` - путь к базовой LLM-модели
- `lamda` - порог для обрезки данных (0.5)
- `k` - количество групп для DEALRec (25)
- `n_fewshot` - размер few-shot выборки (1024)
- `gpu_ids` - номера GPU через запятую

**Этапы пайплайна:**
1. **Обработка данных** - чтение parquet, создание словарей
2. **Обрезка данных** - выбор наиболее информативных образцов
3. **Дообучение LLM** - fine-tuning на few-shot данных
4. **Извлечение эмбедингов** - получение векторных представлений
5. **Оценка качества** - анализ качества эмбедингов

### Отдельные этапы

#### Тестирование обработки данных
```bash
python test_data_processing.py
```

#### Оценка качества эмбедингов
```bash
sh evaluate_embeddings.sh "/path/to/your/llm/model" "./models/transactions/1024" "embeddings/train/transaction_embeddings.npy"
```

## Выходные файлы

После выполнения пайплайна создаются:

### Данные
- `data/transactions/vocabulary.json` - словари и маппинги
- `data/transactions/processed_data.pt` - обработанные данные

### Модели
- `models/transactions/{n_fewshot}/` - дообученная LLM с LoRA

### Эмбединги
- `embeddings/train/transaction_embeddings.npy` - эмбединги обучающих данных
- `embeddings/test/transaction_embeddings.npy` - эмбединги тестовых данных

### Оценка качества
- `embeddings/train/transaction_embeddings_evaluation.json` - метрики качества для train
- `embeddings/test/transaction_embeddings_evaluation.json` - метрики качества для test

## Метрики качества эмбедингов

Скрипт оценки вычисляет следующие метрики:

### Основные метрики
- **Размерность эмбедингов** - количество признаков
- **Количество эмбедингов** - размер датасета
- **Среднее косинусное сходство** - мера схожести эмбедингов
- **Silhouette score** - качество кластеризации (-1 до 1, чем выше тем лучше)
- **Calinski-Harabasz score** - качество кластеризации (чем выше тем лучше)

### Анализ по целевым переменным
- **Среднее сходство** для каждого класса target
- **Стандартное отклонение** сходства
- **Количество образцов** в каждом классе

### Визуализация
- **t-SNE проекция** эмбедингов в 2D для визуализации

## Примеры использования

### Быстрый старт
```bash
# 1. Проверка данных
python test_data_processing.py

# 2. Полный пайплайн
sh run_transaction_pipeline.sh "/path/to/llama/model" 0.5 25 1024 "0"

# 3. Просмотр результатов
cat embeddings/train/transaction_embeddings_evaluation.json
```

### Настройка параметров
```bash
# Больше данных для обучения
sh run_transaction_pipeline.sh "/path/to/llama/model" 0.3 50 2048 "0,1"

# Более строгая обрезка
sh run_transaction_pipeline.sh "/path/to/llama/model" 0.7 10 512 "0"
```

## Устранение неполадок

### Ошибки импорта
```bash
# Убедитесь, что окружение активировано
conda activate dealrec

# Переустановите зависимости
pip install -r requirements.txt
```

### Ошибки GPU
```bash
# Используйте CPU
sh run_transaction_pipeline.sh "/path/to/llama/model" 0.5 25 1024 "cpu"
```

### Недостаточно памяти
```bash
# Уменьшите размер батча
# Отредактируйте transaction_finetune.sh: --batch_size 64
```

## Требования

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Pandas, NumPy, Scikit-learn
- CUDA (опционально, для GPU)

## Лицензия

[Укажите лицензию проекта] 