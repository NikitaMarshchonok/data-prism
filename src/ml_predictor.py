# src/ml_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

def predict_target(df: pd.DataFrame, target_column: str = None):
    report = {}

    # 🔍 1. Целевая переменная — вручную или авто
    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        for col in df.columns[::-1]:
            if df[col].nunique() <= 20 and df[col].dtype in [np.int64, np.int32, object]:
                target_col = col
                break
            elif df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 20:
                target_col = col
                break
        else:
            return {"target_col": "Не определена", "metric": "⚠️ Не удалось выбрать цель", "feature_importance_plot": None}

    report["target_col"] = target_col

    # 🔄 2. Очистка данных
    df_clean = df.dropna(subset=[target_col]).copy()
    df_clean = df_clean.dropna(axis=1, thresh=0.9 * len(df_clean))  # удалим колонки с >10% пропусков
    df_clean = df_clean.select_dtypes(include=[np.number, object])  # только числа и строки

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 🎨 3. Кодировка категориальных признаков
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        try:
            X[col] = le.fit_transform(X[col].astype(str))
        except:
            X = X.drop(columns=[col])

    is_classification = y.nunique() <= 20
    if is_classification and y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    # ✂️ 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier() if is_classification else RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 📊 5. Метрики
    if is_classification:
        score = accuracy_score(y_test, y_pred)
        report["metric"] = f"Accuracy: {score:.2%}"
    else:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        report["metric"] = f"MAE: {mae:.2f}, R²: {r2:.2%}"

    # 📈 6. Важность признаков
    importances = model.feature_importances_
    feature_names = X.columns

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importances)[::-1]
    ax.barh(feature_names[sorted_idx][:10][::-1], importances[sorted_idx][:10][::-1])
    ax.set_title("🔬 Feature Importance")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    report["feature_importance_plot"] = f"data:image/png;base64,{plot_url}"

    return report
