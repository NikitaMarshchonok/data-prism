#!/usr/bin/env python3
"""
Тест функции parse_prompt_to_viz_spec
"""
import pandas as pd
import sys
sys.path.append('.')

from vibedash.spec import parse_prompt_to_viz_spec

def test_nhl_spec():
    print("🔍 Тестируем parse_prompt_to_viz_spec с NHL данными...")
    
    try:
        # Загружаем NHL данные с правильной кодировкой
        try:
            df = pd.read_csv("data/uploads/nhl_test.csv", encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv("data/uploads/nhl_test.csv", encoding='latin-1')
            except:
                df = pd.read_csv("data/uploads/nhl_test.csv", encoding='cp1252')
        print(f"📊 DataFrame загружен: {df.shape}")
        print(f"📋 Колонки: {list(df.columns)}")
        
        # Тестируем parse_prompt_to_viz_spec
        prompt = "Create a hockey analysis dashboard"
        print(f"💭 Промпт: {prompt}")
        
        viz_spec = parse_prompt_to_viz_spec(prompt, list(df.columns))
        print(f"✅ VizSpec создан успешно!")
        print(f"📊 Метрики: {len(viz_spec.metrics)}")
        print(f"📈 Графики: {len(viz_spec.charts)}")
        print(f"🔍 Фильтры: {len(viz_spec.filters)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_nhl_spec()
