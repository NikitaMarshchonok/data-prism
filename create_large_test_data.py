#!/usr/bin/env python3
"""
Создание большого тестового файла для проверки обработки больших данных
"""
import pandas as pd
import numpy as np
import os

def create_large_test_data():
    """Создает большой тестовый CSV файл"""
    print("🔄 Creating large test dataset...")
    
    # Параметры
    n_rows = 100000  # 100k строк
    n_cols = 20
    
    # Создаем данные
    data = {}
    
    # Числовые колонки
    for i in range(10):
        data[f'numeric_{i}'] = np.random.normal(100, 50, n_rows)
    
    # Категориальные колонки
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(5):
        data[f'category_{i}'] = np.random.choice(categories, n_rows)
    
    # Временные колонки
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='H')
    data['timestamp'] = np.random.choice(dates, n_rows)
    
    # ID колонки
    data['id'] = range(1, n_rows + 1)
    
    # Дополнительные колонки
    data['score'] = np.random.uniform(0, 100, n_rows)
    data['status'] = np.random.choice(['active', 'inactive', 'pending'], n_rows)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], n_rows)
    data['priority'] = np.random.choice(['low', 'medium', 'high'], n_rows)
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Сохраняем
    output_file = 'data/large_test_data.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Created {output_file}")
    print(f"📊 Rows: {len(df):,}")
    print(f"📊 Columns: {len(df.columns)}")
    print(f"💾 Size: {file_size_mb:.1f} MB")
    
    return output_file

if __name__ == "__main__":
    create_large_test_data()
