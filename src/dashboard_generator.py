import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.data_loader import load_data
from src.kpi_generator import generate_dynamic_kpis  # ✅ Кастомные KPI
from pandas.api.types import is_numeric_dtype
from src.ai_summary import generate_ai_summary_openai
from src.ml_predictor import predict_target


import os
from dotenv import load_dotenv
import openai


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_summary(df):
    summary_lines = []

    # 🧾 Общая информация
    summary_lines.append(f"Total rows: {len(df):,}")
    summary_lines.append(f"Columns: {len(df.columns)}")

    # 📉 Пропуски
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        summary_lines.append(f"Columns with missing values: {', '.join(missing_cols)}")
    else:
        summary_lines.append("No missing values.")

    # 🧮 Типы данных
    type_counts = df.dtypes.value_counts()
    summary_lines.append("Data types: " + ", ".join([f"{t}: {c}" for t, c in type_counts.items()]))

    # 📊 Категориальные колонки
    cat_cols = [col for col in df.columns if df[col].nunique() <= 20 and df[col].dtype == 'object']
    for col in cat_cols[:3]:
        summary_lines.append(f"Column '{col}' has {df[col].nunique()} distinct values: {', '.join(map(str, df[col].dropna().unique()[:5])) + ('...' if df[col].nunique() > 5 else '')}")

    return "<br>".join(summary_lines)

def generate_missing_comment(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return "✅ No missing values."

    top_col = missing.idxmax()
    top_pct = (missing.max() / len(df)) * 100
    total_cols = missing.shape[0]

    return f"🔍 '{top_col}' has the highest share of missing values ({top_pct:.1f}%). Columns with missing values: {total_cols}. Consider imputing values or excluding rows where appropriate."



def generate_chart_comment(df, col, chart_type):
    if chart_type == "hist":
        desc = df[col].describe()
        return f"📊 {col}: median = {df[col].median():.1f}, min = {desc['min']:.1f}, max = {desc['max']:.1f}"

    elif chart_type == "cat":
        top_val = df[col].value_counts().idxmax()
        top_count = df[col].value_counts().max()
        total = df[col].count()
        percent = top_count / total * 100
        return f"🔢 Most frequent category: '{top_val}' ({percent:.1f}%)"

    elif chart_type == "trend":
        return f"📉 Range of observed {col} values: {df[col].min():.1f} → {df[col].max():.1f}"

    else:
        return "ℹ️ No description available."


def generate_sparklines(df):
    from plotly.offline import plot
    import plotly.graph_objects as go

    sparklines = {}
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols[:3]:  # или [:5], как хочешь
        values = df[col].dropna().tolist()
        if len(values) < 5:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=values,
            mode='lines',
            line=dict(color='rgba(0, 200, 255, 0.8)', width=1.8),
            hoverinfo='skip'
        ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=45,
            width=120,
            paper_bgcolor='rgba(0,0,0,0)',  # полностью прозрачный фон
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        # 💡 Генерируем HTML-код без plotly.js и без div wrapper'а
        sparkline_html = plot(fig, output_type='div', include_plotlyjs=False)

        # Удалим фоновую рамку через style inline (ещё защита на всякий случай)
        sparkline_html = sparkline_html.replace(
            'class="plot-container"',
            'class="plot-container" style="background:none !important;"'
        )

        sparklines[col] = sparkline_html

    return sparklines


def generate_dashboard_data(df, target_column=None):
    if df is None:
        return {}, [], [], "", {}, "", None



    kpis = {
        'Total rows': f"{df.shape[0]:,}",
        'Columns': str(df.shape[1]),
        'Missing values': str(df.isnull().sum().sum())
    }

    custom_kpis = generate_dynamic_kpis(df)


    # 🧠 Генерация мини-графиков (sparklines) и KPI по первым числовым колонкам
    sparklines = {}
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols[:3]:
        values = df[col].dropna().tolist()
        if len(values) < 5:
            continue

        # Добавим KPI (например, среднее значение)
        kpis[col] = f"{df[col].mean():.1f}"

        # Создаём мини-график
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=values,
            mode='lines',
            line=dict(color='rgba(0, 200, 255, 0.8)', width=2),
            hoverinfo='skip'
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=50,
            width=150,
            paper_bgcolor='#131c2c',
            plot_bgcolor='#131c2c',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        sparklines[col] = fig.to_html(full_html=False, include_plotlyjs=False)


    fill_percent = custom_kpis.pop('📦 Data completeness', None)
    unique_percent = custom_kpis.pop('🔁 Uniqueness', None)
    first_col_name = custom_kpis.pop('🧬 First column', None)

    kpis.update(custom_kpis)

    top_charts = []
    numeric_cols = df.select_dtypes(include='number').columns

    # 📈 Топ-3 колонки с максимальной дисперсией
    if len(numeric_cols) >= 3:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        top3 = variances.head(3)
        top_dispersion = ', '.join([f"{col} ({var:.1f})" for col, var in top3.items()])
        kpis['📈 Highest variance'] = top_dispersion

    # 🔹 Гистограммы с современным стилем
    for col in numeric_cols[:2]:
        fig = px.histogram(df, x=col, title=f"Distribution: {col}")
        fig.update_layout(
            plot_bgcolor='#131c2c',
            paper_bgcolor='#131c2c',
            font=dict(color='white'),
            bargap=0.2,
            margin=dict(t=40, b=30, l=0, r=0)
        )
        fig.update_traces(
            marker_color='rgba(0, 200, 255, 0.7)',
            marker_line_color='rgba(0, 200, 255, 1)',
            marker_line_width=1.5,
            hovertemplate='Value: %{x}<br>Frequency: %{y}<extra></extra>'
        )
        comment = generate_chart_comment(df, col, "hist")
        top_charts.append({
            'title': f'Histogram: {col}',
            'html': fig.to_html(full_html=False),
            'comment': comment
        })

    # 🔹 Категориальные графики с новым стилем
    cat_cols = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns
    filtered_cats = [col for col in cat_cols if df[col].nunique() <= 20]
    for col in filtered_cats[:2]:
        top_values = df[col].value_counts().reset_index()
        top_values.columns = [col, 'count']
        fig = px.bar(top_values, x=col, y='count', title=f'Frequency chart: {col}')
        fig.update_layout(
            plot_bgcolor='#131c2c',
            paper_bgcolor='#131c2c',
            font=dict(color='white'),
            margin=dict(t=40, b=30, l=0, r=0),
            xaxis_tickangle=-45
        )
        fig.update_traces(
            marker_color='rgba(255, 99, 132, 0.7)',
            marker_line_color='rgba(255, 99, 132, 1)',
            marker_line_width=1.5,
            hovertemplate='Category: %{x}<br>Frequency: %{y}<extra></extra>'
        )
        comment = generate_chart_comment(df, col, "cat")
        top_charts.append({
            'title': f'Frequency chart: {col}',
            'html': fig.to_html(full_html=False),
            'comment': comment
        })

    # ✅ Вставляем матрицу корреляции ОДИН
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(2)
        annotation_text = [
            [f"{value:.2f}" for value in row]
            for row in corr_matrix.to_numpy()
        ]
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.to_numpy(),
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                text=annotation_text,
                texttemplate='%{text}',
                colorscale='Tealrose',
                zmin=-1,
                zmax=1,
                zmid=0,
                colorbar=dict(title='Correlation'),
                hovertemplate='%{x} × %{y}: %{z:.2f}<extra></extra>',
            )
        )
        fig.update_layout(
            title='📊 Correlation matrix',
            font=dict(color='white'),
            paper_bgcolor='#131c2c',
            plot_bgcolor='#131c2c',
            margin=dict(t=40, b=30, l=0, r=0)
        )
        top_charts.append({
            'title': '📊 Correlation matrix',
            'html': fig.to_html(full_html=False),
            'comment': '🔗 Correlation coefficients between numeric features'
        })


    # 🔹 Тренды по времени — стилизовано
    time_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or 'date' in col.lower() or 'time' in col.lower():
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notnull().sum() / len(df) > 0.5 and converted.nunique() > 3:
                time_cols.append(col)

    for time_col in time_cols[:1]:
        df['_time'] = pd.to_datetime(df[time_col], errors='coerce')
        for num_col in numeric_cols[:2]:
            fig = px.scatter(df, x='_time', y=num_col, title=f"📉 Trend over time: {num_col}")
            fig.update_layout(
                plot_bgcolor='#131c2c',
                paper_bgcolor='#131c2c',
                font=dict(color='white'),
                margin=dict(t=40, b=30, l=0, r=0)
            )
            fig.update_traces(
                marker=dict(
                    color='rgba(0, 255, 180, 0.6)',
                    size=6,
                    line=dict(width=1, color='rgba(0, 255, 180, 1)')
                ),
                hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>'
            )
            comment = generate_chart_comment(df, num_col, "trend")
            top_charts.append({
                'title': f'📉 Trend over time: {num_col}',
                'html': fig.to_html(full_html=False),
                'comment': comment
            })


    def create_gauge(title, percent):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            title={'text': title, 'font': {'size': 18, 'color': 'white'}},
            number={'suffix': "%", 'font': {'color': 'white'}},
            gauge={
                'shape': "angular",
                'axis': {
                    'range': [0, 100],
                    'tickmode': 'linear',
                    'tick0': 0,
                    'dtick': 20,
                    'tickcolor': '#cccccc',
                    'tickfont': {'color': '#cccccc'}
                },
                'bar': {'color': "rgba(0, 168, 255, 0.6)"},
                'bgcolor': "#131c2c",
                'bordercolor': "#131c2c",
                'steps': [
                    {'range': [0, 50], 'color': '#7f1d1d'},  # тёмно-красный
                    {'range': [50, 75], 'color': '#78350f'},  # коричневый
                    {'range': [75, 100], 'color': '#065f46'}  # тёмно-зелёный
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': percent
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='#131c2c',  # общий фон
            plot_bgcolor='#131c2c',
            margin=dict(t=40, b=40, l=20, r=20),  # ⬅️ Увеличили нижний отступ (b=40)
            height=270  # ⬆️ Чуть больше высота (для шкалы снизу)
        )
        return fig.to_html(full_html=False)

    if fill_percent:
        try:
            percent = float(fill_percent.replace('%', '').replace(',', '.'))
            top_charts.insert(0, {
                'title': '📦 Data completeness indicator',
                'type': 'indicator',
                'html': create_gauge("📦 Data completeness", percent)
            })
        except Exception as e:
            print(f"⚠️ Индикатор заполненности не построен: {e}")

    if unique_percent and first_col_name:
        try:
            percent = float(unique_percent.replace('%', '').replace(',', '.'))
            top_charts.insert(1, {
                'title': f'🔁 Uniqueness of values: {first_col_name}',
                'html': create_gauge(f"🔁 Uniqueness: {first_col_name}", percent)
            })
        except Exception as e:
            print(f"⚠️ Индикатор уникальности не построен: {e}")

    table1 = {
        'title': '🔍 First 5 rows',
        'headers': list(df.columns),
        'rows': df.head(5).values.tolist()
    }

    table2 = {
        'title': '📌 Column data types',
        'headers': ['Column', 'Data type'],
        'rows': [[col, str(dtype)] for col, dtype in df.dtypes.items()]
    }

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        total = len(df)
        table3 = {
            'title': '📉 Missing values by column',
            'headers': ['Column', 'Missing', 'Percent'],
            'rows': [
                [col, int(missing[col]), f"{(missing[col]/total*100):.1f}%"]
                for col in missing.index
            ],
            'comment': generate_missing_comment(df)
        }
    else:
        table3 = {
            'title': '📉 Missing values',
            'headers': ['✅ All values are present'],
            'rows': [],
            'comment': generate_missing_comment(df)
        }

    tables = [table1, table2, table3]

    # 📉 Таблица выбросов по IQR
    outlier_rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        count = outliers.shape[0]
        if count > 0:
            percent = count / df.shape[0] * 100
            outlier_rows.append([col, count, f"{percent:.1f}%"])

    if outlier_rows:
        table4 = {
            'title': '📉 IQR outliers',
            'headers': ['Column', 'Outlier count', 'Outlier percentage'],
            'rows': outlier_rows
        }
        tables.append(table4)

    summary = generate_summary(df)
    # 🧠 Генерация AI Summary
    ai_summary = generate_ai_summary_openai(df)

    # 🎯 ML-предсказание
    try:
        ml_result = predict_target(df, target_column)
        ml_card = {
            "target": ml_result.get("target_col", "Unspecified"),
            "task_type": ml_result.get("task_type"),
            "model_name": ml_result.get("model_name"),
            "metric": ml_result.get("metric", "Metric unavailable"),
            "predictive_support": ml_result.get("predictive_support", {}),
            "plot": ml_result.get("feature_importance_plot", None),
            "model_comparison": ml_result.get("model_comparison", []),
            "cv_folds": ml_result.get("cv_folds", 0),
            "cv_metric": ml_result.get("cv_metric"),
            "diagnostics": ml_result.get("diagnostics", {}),
            "reliability": ml_result.get("reliability", {}),
            "feature_importance_method": ml_result.get("feature_importance_method"),
            "evaluation_notes": ml_result.get("evaluation_notes", []),
        }
    except Exception as e:
        ml_card = None
        print(f"❌ Ошибка при ML-предсказании: {e}")



    return kpis, top_charts, tables, summary, sparklines, ai_summary, ml_card
