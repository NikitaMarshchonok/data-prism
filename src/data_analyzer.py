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
        title="🧠 Корреляционная матрица",
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
        labels={'x': column, 'y': 'Частота'},
        title=f'🔠 Распределение значений: {column}',
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
        return f"Столбец <b>{column}</b> не является числовым, поэтому график построен без числового описания."

    mean = series.mean()
    median = series.median()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    count = len(series)

    skewness = series.skew()
    outliers = ((series < (mean - 3 * std)) | (series > (mean + 3 * std))).sum()

    desc += f"Признак <b>{column}</b> содержит {count} наблюдений. "
    desc += f"Среднее значение: {mean:.2f}, медиана: {median:.2f}, стандартное отклонение: {std:.2f}. "
    desc += f"Минимум: {min_val}, максимум: {max_val}. "

    # Добавим оценку симметрии
    if skewness < -1:
        desc += "Распределение имеет сильную левостороннюю асимметрию. "
    elif skewness < -0.5:
        desc += "Распределение немного скошено влево. "
    elif skewness < 0.5:
        desc += "Распределение близко к симметричному. "
    elif skewness < 1:
        desc += "Распределение немного скошено вправо. "
    else:
        desc += "Распределение имеет сильную правостороннюю асимметрию. "

    if outliers > 0:
        desc += f"Обнаружено выбросов: {outliers} (значения вне 3σ от среднего)."

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

