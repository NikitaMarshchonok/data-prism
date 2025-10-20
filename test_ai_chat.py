#!/usr/bin/env python3
"""
Тест AI чата VibeDash
"""
import requests
import json
import time

def test_ai_chat():
    """Тестирует AI чат функциональность"""
    base_url = "http://localhost:5001/vibedash"
    
    print("🧪 Тестирование AI чата VibeDash...")
    
    # 1. Загружаем данные
    print("\n1. Загружаем тестовые данные...")
    with open('data/demo_sales.csv', 'rb') as f:
        files = {'datafile': f}
        data = {'prompt': 'Test data upload'}
        response = requests.post(f"{base_url}/preview", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Ошибка загрузки данных: {response.status_code}")
        return
    
    print("✅ Данные загружены успешно")
    
    # 2. Извлекаем session_id из ответа
    session_id = "test_session_123"  # Для тестирования
    
    # 3. Тестируем AI анализ
    test_questions = [
        "What is the data overview?",
        "Show me correlations",
        "Find clusters in the data",
        "Build a prediction model"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Тестируем вопрос: '{question}'")
        
        payload = {
            'question': question,
            'session_id': session_id
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/analyze",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    analysis = result.get('analysis', {})
                    print(f"✅ Анализ выполнен")
                    print(f"   Ответ: {analysis.get('answer', '')[:100]}...")
                    print(f"   Графиков: {len(analysis.get('charts', []))}")
                    print(f"   Инсайтов: {len(analysis.get('insights', []))}")
                else:
                    print(f"❌ Ошибка анализа: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP ошибка: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Исключение: {e}")
        
        time.sleep(1)  # Пауза между запросами
    
    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_ai_chat()
