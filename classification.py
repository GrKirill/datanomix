import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import torch

def format_items(items):
    formatted_list = []
    for item in items:
        code = item['Код']
        name = item['Наименование с характеристикой']
        formatted_list.append(f"{code}: {name}; \n")
    return formatted_list

class ProductMatcher:
    def __init__(self, file_path, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """
        Инициализация модели BERT и загрузка данных
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.df = self.load_data(file_path)
        self.embeddings = self.generate_embeddings()
        
    @staticmethod
    def preprocess_text(text):
        """
        Базовая очистка текста
        """
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_data(self, file_path):
        """
        Загрузка и предварительная обработка данных
        """
        df = pd.read_csv(
            file_path, 
            usecols=['Код', 'Наименование с характеристикой'],
            dtype={'Код': str}
        )
        df['Наименование с характеристикой'] = df['Наименование с характеристикой'].apply(self.preprocess_text)
        return df

    def generate_embeddings(self):
        """
        Генерация BERT-эмбеддингов для всех товаров
        """
        print("Generating BERT embeddings...")
        texts = self.df['Наименование с характеристикой'].tolist()
        embeddings = self.model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        return embeddings.cpu().numpy() if self.device == 'cuda' else embeddings

    def find_similar_codes(self, new_name, top_n=3, similarity_threshold=0.7):
        """
        Поиск похожих товаров с использованием BERT-эмбеддингов
        """
        # Предобработка и генерация эмбеддинга для нового наименования
        cleaned_name = self.preprocess_text(new_name)
        new_embedding = self.model.encode(
            [cleaned_name],
            convert_to_tensor=True,
            device=self.device
        )
        
        if self.device == 'cuda':
            new_embedding = new_embedding.cpu()
        
        # Расчет косинусной близости
        cosine_similarities = cosine_similarity(
            new_embedding, 
            self.embeddings
        ).flatten()
        
        # Получение индексов топ-N наиболее похожих товаров
        sorted_indices = np.argsort(cosine_similarities)[::-1]
        
        # Фильтрация по порогу схожести и выбор топ-N
        results = []
        for idx in sorted_indices:
            similarity = cosine_similarities[idx]
            if similarity < similarity_threshold or len(results) >= top_n:
                continue
                
            results.append({
                'Код': self.df.iloc[idx]['Код'],
                'Наименование с характеристикой': self.df.iloc[idx]['Наименование с характеристикой'],
                'Схожесть': float(similarity)
            })
        
        return results if results else [{"Статус": "Совпадения не найдены"}]

# Пример использования
if __name__ == "__main__":
    # Инициализация модели
    matcher = ProductMatcher("./data/enstru.csv")
    
    # Тестовый запрос
    test_queries = pd.read_csv('./data/test_data.csv')[15:]['Задание:']
    prediction_1 = []
    prediction_3 = []
    for query in test_queries:
        print(f"\nПоиск для: '{query}'")
        similar_products = matcher.find_similar_codes(query)
        try:
            prediction_1.append(similar_products[0]['Код'])
            prediction_3.append(''.join(format_items(similar_products)))
        except:
            prediction_1.append('-')
            prediction_3.append('-')
        print("Результаты:")
        for i, product in enumerate(similar_products, 1):
            if 'Код' in product:
                print(f"{i}. Код: {product['Код']} | Схожесть: {product['Схожесть']:.4f}")
                print(f"   Наименование с характеристикой: {product['Наименование с характеристикой']}")
            else:
                print(product['Статус'])
        print("-" * 80)
    
    result_df = pd.DataFrame({'Prediction@1': prediction_1, 'Prediction@3': prediction_3})
    result_df.to_csv('result.csv', index=False)