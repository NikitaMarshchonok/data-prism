# src/data_analyzer.py
def analyze_data(data):
    """
    Вычисляет базовые статистические показатели для числовых столбцов.
    :param data: DataFrame с данными
    :return: словарь с показателями и список числовых столбцов
    """
    # Выбираем столбцы с числовыми данными
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    stats = {}
    for col in numeric_columns:
        stats[col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max()
        }
    return stats, numeric_columns
# src/data_analyzer.py

def analyze_missing_values(data):
    """
    Анализ пропущенных значений по каждому столбцу.
    :param data: DataFrame
    :return: словарь {столбец: (кол-во, %)}
    """
    total_rows = len(data)
    missing_info = {}
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_percent = round((missing_count / total_rows) * 100, 2)
        if missing_count > 0:
            missing_info[col] = {
                'count': missing_count,
                'percent': missing_percent
            }
    return missing_info


import plotly.figure_factory as ff

def plot_correlation_heatmap(data):
    """
    Строит интерактивную матрицу корреляций.
    :param data: DataFrame
    :return: HTML график тепловой карты
    """
    corr = data.corr(numeric_only=True)
    z = corr.values.round(2).tolist()
    x = corr.columns.tolist()
    y = corr.index.tolist()

    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
    fig.update_layout(
        title="🧠 Correlation matrix",
        title_font_size=22,
        margin=dict(t=50, l=0, r=0, b=0),
        width=800, height=600
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def detect_categorical_columns(data, max_unique=15):
    """Определяет категориальные столбцы"""
    return [col for col in data.columns if data[col].dtype == 'object' or data[col].nunique() <= max_unique]


import plotly.express as px

def analyze_categorical_column(data, column, top_n=5):
    """Создаёт график и таблицу топ значений"""
    value_counts = data[column].value_counts().nlargest(top_n)
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': column, 'y': 'Frequency'},
        title=f'🔠 Distribution: {column}',
        color_discrete_sequence=['#00AEEF']
    )
    fig.update_layout(height=400, template='plotly_white')
    return value_counts.to_dict(), fig.to_html(full_html=False, include_plotlyjs='cdn')



def describe_column_types(data):
    result = {}
    for col in data.columns:
        dtype = str(data[col].dtype)
        nunique = data[col].nunique()
        result[col] = {
            "dtype": dtype,
            "unique": nunique
        }
    return result


# src/data_analyzer.py

import numpy as np

def generate_description_for_column(data, column):
    """
    Автоматически генерирует понятное описание распределения числового признака.
    """
    series = data[column].dropna()
    desc = ""

    if not np.issubdtype(series.dtype, np.number):
        return f"A numeric summary is unavailable for <b>{column}</b>."

    mean = series.mean()
    median = series.median()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    count = len(series)

    skewness = series.skew()
    outliers = ((series < (mean - 3 * std)) | (series > (mean + 3 * std))).sum()

    desc += f"<b>{column}</b> contains {count} observations. "
    desc += f"Mean: {mean:.2f}, median: {median:.2f}, standard deviation: {std:.2f}. "
    desc += f"Minimum: {min_val}, maximum: {max_val}. "

    # Добавим оценку симметрии
    if skewness < -1:
        desc += "The distribution is strongly left-skewed. "
    elif skewness < -0.5:
        desc += "The distribution is slightly left-skewed. "
    elif skewness < 0.5:
        desc += "The distribution is approximately symmetric. "
    elif skewness < 1:
        desc += "The distribution is slightly right-skewed. "
    else:
        desc += "The distribution is strongly right-skewed. "

    if outliers > 0:
        desc += f"Potential outliers: {outliers} (values more than 3 standard deviations from the mean)."

    return desc

import pandas as pd
import re

'''def detect_time_columns(df):
    """
    Автоматически определяет временные колонки (года, даты, сезоны).
    Возвращает словарь: {имя_колонки: тип ['year', 'date', 'season']}
    """
    time_columns = {}

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_columns[col] = 'date'
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].between(1900, 2100).mean() > 0.8:
                time_columns[col] = 'year'
                continue

        if df[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().mean() > 0.8:
                    time_columns[col] = 'date'
                    continue
            except:
                pass

            if re.search(r'season|год|year', col.lower()):
                time_columns[col] = 'season'

    return time_columns'''
def detect_time_columns(df):
    time_cols = []

    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            non_null_ratio = parsed.notnull().mean()

            # Условие: хотя бы 70% значений должны быть распознаны как datetime
            if non_null_ratio > 0.7:
                # И фильтрация подозрительных: год, число или строки длиной меньше 4 символов
                if not pd.api.types.is_numeric_dtype(df[col]) and df[col].astype(str).str.len().median() >= 4:
                    time_cols.append(col)
        except Exception:
            continue

    return time_cols


# src/data_analyzer.py

def convert_time_columns(df):
    """
    Автоматически ищет и преобразует временные колонки в формат datetime.
    Возвращает обновлённый DataFrame и список найденных колонок.
    """
    time_cols = []
    for col in df.columns:
        if any(x in col.lower() for x in ['date', 'time', 'timestamp', 'year']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                time_cols.append(col)
            except Exception as e:
                print(f"⚠️ Не удалось преобразовать колонку '{col}' в datetime: {e}")
    return df, time_cols


def analyze_data_quality(data):
    """Оценивает качество датасета и возвращает понятные рекомендации.

    Итоговая оценка находится в диапазоне от 0 до 100. Она учитывает
    пропуски, дублирующиеся строки, постоянные столбцы и числовые выбросы,
    найденные методом межквартильного размаха (IQR).
    """
    row_count, column_count = data.shape
    total_cells = row_count * column_count

    if row_count == 0 or column_count == 0:
        return {
            "score": 0,
            "status": "No data",
            "missing_cells": 0,
            "missing_percent": 0.0,
            "duplicate_rows": 0,
            "duplicate_percent": 0.0,
            "constant_columns": [],
            "outliers": {},
            "outlier_cells": 0,
            "recommendations": [
                "Upload a dataset with at least one row to run the quality audit."
            ],
        }

    missing_cells = int(data.isna().sum().sum())
    missing_percent = 100 * missing_cells / total_cells

    try:
        duplicate_rows = int(data.duplicated().sum())
    except TypeError:
        # JSON может содержать списки и словари, которые нельзя хешировать.
        comparable_data = data.map(repr) if hasattr(data, "map") else data.applymap(repr)
        duplicate_rows = int(comparable_data.duplicated().sum())
    duplicate_percent = 100 * duplicate_rows / row_count

    constant_columns = [
        str(column)
        for column in data.columns
        if data[column].nunique(dropna=False) <= 1
    ]

    outliers = {}
    numeric_data = data.select_dtypes(include="number")
    numeric_cells = int(numeric_data.notna().sum().sum())
    outlier_cells = 0
    for column in numeric_data.columns:
        series = numeric_data[column].dropna()
        if series.empty:
            continue

        first_quartile = series.quantile(0.25)
        third_quartile = series.quantile(0.75)
        iqr = third_quartile - first_quartile
        if iqr <= 0:
            continue

        lower_bound = first_quartile - 1.5 * iqr
        upper_bound = third_quartile + 1.5 * iqr
        count = int(((series < lower_bound) | (series > upper_bound)).sum())
        if count:
            outliers[str(column)] = count
            outlier_cells += count

    constant_percent = 100 * len(constant_columns) / column_count
    outlier_percent = 100 * outlier_cells / numeric_cells if numeric_cells else 0.0
    penalty = (
        min(40.0, missing_percent * 0.4)
        + min(25.0, duplicate_percent * 0.25)
        + min(15.0, constant_percent * 0.15)
        + min(20.0, outlier_percent * 0.2)
    )
    score = max(0, round(100 - penalty))

    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    elif score >= 50:
        status = "Needs attention"
    else:
        status = "Poor"

    recommendations = []
    if missing_cells:
        recommendations.append(
            "Review missing values. Consider imputing them or excluding affected rows or columns where appropriate."
        )
    if duplicate_rows:
        recommendations.append(
            "Review duplicate rows and remove any that represent the same observation."
        )
    if constant_columns:
        recommendations.append(
            "Consider excluding constant columns from analysis and modeling."
        )
    if outlier_cells:
        recommendations.append(
            "Review numeric outliers for data entry errors and meaningful rare observations."
        )
    if not recommendations:
        recommendations.append("The quality checks found no critical issues.")

    return {
        "score": score,
        "status": status,
        "missing_cells": missing_cells,
        "missing_percent": round(missing_percent, 2),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round(duplicate_percent, 2),
        "constant_columns": constant_columns,
        "outliers": outliers,
        "outlier_cells": outlier_cells,
        "recommendations": recommendations,
    }
