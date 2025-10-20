"""
Ollama клиент для локального LLM
"""
import os
import requests
import json
from typing import Optional


def ollama_generate(model: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Генерирует ответ через локальный Ollama
    """
    if not os.getenv('USE_OLLAMA', 'false').lower() == 'true':
        return None
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}",
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('response', '').strip()
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Ошибка подключения к Ollama: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Ошибка при генерации через Ollama: {e}")
        return None


def is_ollama_available() -> bool:
    """Проверяет доступность Ollama сервера"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False
