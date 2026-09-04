#!/usr/bin/env python3
"""
Тест VibeDash без Flask
"""
import sys
import os
sys.path.append('.')

# Тестируем основные компоненты
def test_spec_parsing():
    """Тест парсинга спецификации"""
    try:
        from vibedash.spec import parse_prompt_to_viz_spec
        print("✅ Модуль spec импортирован")
        
        # Тестовые колонки
        columns = ['Date', 'Region', 'Product', 'Category', 'Sales', 'Quantity', 'Price']
        
        # Тестовый промпт
        prompt = "Создай дашборд продаж с основными KPI, топ-10 категорий по выручке, трендом по месяцам и фильтром по регионам"
        
        # Парсим промпт
        viz_spec = parse_prompt_to_viz_spec(prompt, columns)
        print(f"✅ Спецификация создана: {viz_spec.title}")
        print(f"   Метрики: {len(viz_spec.metrics)}")
        print(f"   Графики: {len(viz_spec.charts)}")
        print(f"   Фильтры: {len(viz_spec.filters)}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в spec: {e}")
        return False

def test_generator_bridge():
    """Тест генератора дашборда"""
    try:
        import pandas as pd
        from vibedash.generator_bridge import generate_dashboard_data
        from vibedash.spec import VizSpec, Metric, Chart
        
        print("✅ Модуль generator_bridge импортирован")
        
        # Создаем тестовые данные
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Region': ['North', 'South', 'East'],
            'Sales': [1000, 1500, 800],
            'Category': ['Electronics', 'Furniture', 'Electronics']
        })
        
        # Создаем простую спецификацию
        viz_spec = VizSpec(
            title="Тестовый дашборд",
            metrics=[Metric(title="Общая сумма", expr="sum(Sales)", fmt="currency")],
            charts=[Chart(type="bar", x="Category", y="Sales", agg="sum")],
            filters=[],
            comments=["Тестовый дашборд"]
        )
        
        # Генерируем данные дашборда
        dashboard_data = generate_dashboard_data(df, viz_spec)
        print(f"✅ Дашборд сгенерирован")
        print(f"   KPIs: {len(dashboard_data['kpis'])}")
        print(f"   Графики: {len(dashboard_data['charts'])}")
        print(f"   Таблицы: {len(dashboard_data['tables'])}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в generator_bridge: {e}")
        return False

def test_ollama_client():
    """Тест Ollama клиента"""
    try:
        from vibedash.ollama_client import is_ollama_available, ollama_generate
        print("✅ Модуль ollama_client импортирован")
        
        # Проверяем доступность Ollama
        available = is_ollama_available()
        print(f"   Ollama доступен: {available}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в ollama_client: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование VibeDash компонентов...")
    print("=" * 50)
    
    tests = [
        test_spec_parsing,
        test_generator_bridge,
        test_ollama_client
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"✅ Пройдено тестов: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 Все тесты прошли успешно!")
        return True
    else:
        print("❌ Некоторые тесты не прошли")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
