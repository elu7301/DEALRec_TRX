import os
import sys
import torch
import numpy as np
import json
import math
from typing import List, Dict, Tuple
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import argparse

class TransactionEvaluator:
    """
    Класс для оценки качества эмбедингов транзакционных данных
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
        
        # Добавляем специальные токены если их нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Loading LoRA weights from {self.lora_weights_path}")
        self.model = LlamaForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Загружаем LoRA веса
        self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)
        self.model.eval()
        
        print("Model loaded successfully")
        
    def create_transaction_prompt(self, sequence: List[Dict]) -> str:
        """Создание промпта для последовательности транзакций"""
        prompt = "Transaction history:\n"
        
        for i, transaction in enumerate(sequence[-10:]):  # Последние 10 транзакций
            category = self.small_group_map_reverse.get(transaction['small_group'], 'Unknown')
            amount = transaction['amount_rur']
            time_str = f"Time: {transaction['event_time']}"
            
            prompt += f"{i+1}. Category: {category}, Amount: {amount:.2f} RUB, {time_str}\n"
        
        prompt += "\nNext transaction prediction:"
        return prompt
        
    def extract_embeddings(self, sequences: List[List[Dict]]) -> np.ndarray:
        """Извлечение эмбедингов из последовательностей"""
        embeddings = []
        
        with torch.no_grad():
            for sequence in tqdm(sequences, desc="Extracting embeddings"):
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
                
                # Берем эмбединг последнего токена
                hidden_states = outputs.hidden_states[-1]
                if hidden_states.shape[1] > 0:
                    embedding = hidden_states[0, -1, :].cpu().numpy()
                else:
                    embedding = np.zeros(hidden_states.shape[-1])
                
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def calculate_similarity_metrics(self, embeddings: np.ndarray, targets: List[int]) -> Dict:
        """Вычисление метрик качества эмбедингов"""
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        print("Calculating similarity metrics...")
        
        # Косинусное сходство
        similarity_matrix = cosine_similarity(embeddings)
        
        # Кластеризация
        n_clusters = min(10, len(embeddings) // 10)  # Адаптивное количество кластеров
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Метрики кластеризации
        silhouette_avg = silhouette_score(embeddings, cluster_labels) if n_clusters > 1 else 0
        calinski_avg = calinski_harabasz_score(embeddings, cluster_labels)
        
        # Анализ по целевым переменным
        target_analysis = {}
        unique_targets = np.unique(targets)
        
        for target in unique_targets:
            target_indices = [i for i, t in enumerate(targets) if t == target]
            if len(target_indices) > 1:
                target_embeddings = embeddings[target_indices]
                target_similarity = cosine_similarity(target_embeddings)
                target_analysis[f'target_{target}'] = {
                    'mean_similarity': np.mean(target_similarity),
                    'std_similarity': np.std(target_similarity),
                    'count': len(target_indices)
                }
        
        # t-SNE для визуализации (если размерность > 2)
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        metrics = {
            'embedding_shape': embeddings.shape,
            'mean_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix),
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_avg,
            'n_clusters': n_clusters,
            'target_analysis': target_analysis,
            'embeddings_2d': embeddings_2d.tolist()
        }
        
        return metrics
    
    def evaluate_embeddings(self, embeddings_path: str, test_data_path: str = None) -> Dict:
        """Основная функция оценки качества эмбедингов"""
        print(f"Loading embeddings from {embeddings_path}")
        
        # Загружаем эмбединги
        embeddings_data = np.load(embeddings_path, allow_pickle=True).item()
        embeddings = embeddings_data['embeddings']
        metadata = embeddings_data.get('metadata', [])
        
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Извлекаем целевые переменные
        targets = [item.get('target', 0) for item in metadata] if metadata else []
        
        # Вычисляем метрики
        metrics = self.calculate_similarity_metrics(embeddings, targets)
        
        # Сохраняем результаты
        results_path = embeddings_path.replace('.npy', '_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation results saved to {results_path}")
        
        # Выводим основные метрики
        print("\n=== Embedding Quality Metrics ===")
        print(f"Embedding dimension: {metrics['embedding_shape'][1]}")
        print(f"Number of embeddings: {metrics['embedding_shape'][0]}")
        print(f"Mean cosine similarity: {metrics['mean_similarity']:.4f}")
        print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.2f}")
        print("=================================")
        
        return metrics


def main():
    """Пример использования"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', required=True, help='Path to base LLM model')
    parser.add_argument('--lora_weights', required=True, help='Path to LoRA weights')
    parser.add_argument('--embeddings_path', required=True, help='Path to embeddings file')
    parser.add_argument('--vocabulary_path', default='data/transactions/vocabulary.json', 
                       help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    # Создаем оценщик
    evaluator = TransactionEvaluator(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights,
        vocabulary_path=args.vocabulary_path
    )
    
    # Оцениваем качество
    metrics = evaluator.evaluate_embeddings(args.embeddings_path)
    
    print(f"Embedding evaluation completed!")


if __name__ == "__main__":
    main() 