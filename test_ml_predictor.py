# test_ml_predictor.py

import pandas as pd
from src.ml_predictor import predict_target

# Загрузим тестовый CSV (можно использовать свой)
df = pd.read_csv("data/uploads/latest_uploaded.csv")  # или подставь путь к своему файлу

# Вызов функции
result = predict_target(df)

# Выводим результат
print("🎯 Целевая переменная:", result.get("target_col"))
print("📊 Метрика:", result.get("metric"))

# Сохраним график признаков
if result.get("feature_importance_plot"):
    with open("feature_importance.png", "wb") as f:
        import base64
        f.write(base64.b64decode(result["feature_importance_plot"].split(",")[1]))
    print("✅ График важности признаков сохранён как feature_importance.png")
else:
    print("❌ График не найден.")
