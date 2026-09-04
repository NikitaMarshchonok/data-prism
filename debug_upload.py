#!/usr/bin/env python3
"""
Отладка загрузки файла в VibeDash
"""
import requests
import os

def test_upload():
    print("🔍 Тестируем загрузку файла в VibeDash...")
    
    # URL сервера
    url = "http://localhost:5001/vibedash/preview"
    
    # Тестовый файл
    test_file = "data/uploads/NHL 2004-2018 Player Data.csv"
    
    if not os.path.exists(test_file):
        print(f"❌ Файл {test_file} не найден!")
        return
    
    print(f"📁 Файл найден: {test_file}")
    print(f"📊 Размер файла: {os.path.getsize(test_file) / 1024 / 1024:.1f} MB")
    
    # Данные для отправки
    files = {
        'datafile': (os.path.basename(test_file), open(test_file, 'rb'), 'text/csv')
    }
    
    data = {
        'prompt': 'Create a hockey player analysis dashboard with player statistics, team performance, and career trends'
    }
    
    print("🚀 Отправляем запрос...")
    
    try:
        response = requests.post(url, files=files, data=data, allow_redirects=False)
        
        print(f"📡 Статус ответа: {response.status_code}")
        print(f"📋 Заголовки: {dict(response.headers)}")
        
        if response.status_code == 302:
            print("🔄 Получен редирект - это означает ошибку в обработке!")
            print(f"📍 Редирект на: {response.headers.get('Location', 'Unknown')}")
        elif response.status_code == 200:
            print("✅ Успешный ответ!")
            print(f"📄 Размер ответа: {len(response.content)} байт")
            
            # Проверяем содержимое
            content = response.text
            if "VibeDash Preview" in content:
                print("🎯 Дашборд успешно сгенерирован!")
            elif "VibeDash" in content:
                print("⚠️ Получена страница VibeDash, но не дашборд")
            else:
                print("❌ Неожиданное содержимое ответа")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"📄 Ответ: {response.text[:500]}...")
            
    except Exception as e:
        print(f"❌ Ошибка при отправке запроса: {e}")
    finally:
        files['datafile'][1].close()

if __name__ == "__main__":
    test_upload()
