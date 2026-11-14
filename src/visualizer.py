# src/visualizer.py
# src/visualizer.py
import os
import matplotlib
import pandas as pd
matplotlib.use('Agg')  # Устанавливаем неинтерактивный бэкенд
import matplotlib.pyplot as plt
import seaborn as sns

# src/visualizer.py
import plotly.express as px

# src/visualizer.py
import plotly.express as px

import plotly.express as px


def plot_histogram_interactive(data, column):
    nbins = min(10, len(data[column].unique()))

    fig = px.histogram(
        data,
        x=column,
        nbins=nbins,
        marginal="box",
        title=f"📊 Distribution: {column}",
        template="plotly_white",
        color_discrete_sequence=["#5470C6"]  # Приятный синий цвет
    )

    fig.update_layout(
        title_font_size=20,
        xaxis_title=f"{column} (values)",  # перевод оси X
        yaxis_title="Frequency",  # перевод оси Y
        bargap=0.05,
        height=450
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def plot_histogram(data, column, output_path):
    """
    Строит гистограмму с наложенной кривой плотности для указанного столбца и сохраняет её как изображение.
    :param data: DataFrame с данными
    :param column: имя столбца для построения графика
    :param output_path: путь для сохранения изображения
    """
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    sns.histplot(data[column].dropna(), kde=True)
    plt.title(f'Distribution: {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()
    print(f"График для столбца {column} сохранён в {output_path}")


import plotly.express as px


def plot_time_trend(df, time_col, value_col=None):
    """
    Строит график тренда по временной колонке.
    Если value_col не указана — строит количество записей по времени.
    """
    if time_col not in df.columns:
        return None

    df_copy = df.copy()
    series = df_copy[time_col].dropna()

    # Если это не дата — проверим, возможно это просто "год"
    if pd.api.types.is_numeric_dtype(series) and series.max() < 2100:
        df_copy[time_col] = series.astype(int)
        kind = "year"
    else:
        # Попробуем преобразовать в datetime
        try:
            df_copy[time_col] = pd.to_datetime(series, errors='coerce')
            kind = "datetime"
        except Exception:
            return None

    if value_col:
        grouped = df_copy.groupby(time_col)[value_col].mean().reset_index()
        fig = px.line(grouped, x=time_col, y=value_col,
                      title=f"Mean {value_col} over time ({time_col})")
    else:
        grouped = df_copy[time_col].value_counts().sort_index().reset_index()
        grouped.columns = [time_col, "Count"]
        fig = px.bar(grouped, x=time_col, y="Count",
                     title=f"Record count by {time_col}")

    fig.update_layout(template="simple_white", height=400)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
