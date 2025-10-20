"""
Мост между VibeDash и существующим генератором дашборда
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Dict, List, Any
from .spec import VizSpec, Metric, Chart, Filter


def generate_dashboard_data(df: pd.DataFrame, viz_spec: VizSpec) -> Dict[str, Any]:
    """
    Генерирует данные дашборда на основе VizSpec
    Возвращает данные в формате, совместимом с существующим дашбордом
    """
    if df is None or df.empty:
        return {
            "kpis": [],
            "charts": [],
            "tables": [],
            "ai_summary": "Нет данных для анализа"
        }
    
    # Применяем фильтры
    filtered_df = _apply_filters(df, viz_spec.filters)
    
    # Генерируем KPIs
    kpis = _generate_kpis(filtered_df, viz_spec.metrics)
    
    # Генерируем графики
    charts = _generate_charts(filtered_df, viz_spec.charts)
    
    # Генерируем таблицы
    tables = _generate_tables(filtered_df, viz_spec)
    
    # AI summary
    ai_summary = _generate_ai_summary(filtered_df, viz_spec)
    
    return {
        "kpis": kpis,
        "charts": charts,
        "tables": tables,
        "ai_summary": ai_summary
    }


def _apply_filters(df: pd.DataFrame, filters: List[Filter]) -> pd.DataFrame:
    """Применяет фильтры к DataFrame"""
    filtered_df = df.copy()
    
    for filter_obj in filters:
        if filter_obj.where:
            # Простой DSL парсинг для where условий
            try:
                # Безопасная замена названий колонок
                condition = filter_obj.where
                for col in df.columns:
                    if col in condition:
                        condition = condition.replace(col, f"filtered_df['{col}']")
                
                # Выполняем фильтрацию
                mask = eval(condition)
                filtered_df = filtered_df[mask]
            except Exception as e:
                print(f"⚠️ Ошибка в фильтре {filter_obj.field}: {e}")
                continue
        
        if filter_obj.values:
            # Фильтр по значениям
            if filter_obj.field in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[filter_obj.field].isin(filter_obj.values)]
    
    return filtered_df


def _generate_kpis(df: pd.DataFrame, metrics: List[Metric]) -> List[Dict[str, str]]:
    """Генерирует KPI карточки"""
    kpis = []
    
    for metric in metrics:
        try:
            value = _evaluate_metric(df, metric.expr)
            formatted_value = _format_value(value, metric.fmt)
            
            kpis.append({
                "title": metric.title,
                "value": formatted_value,
                "tooltip": f"Выражение: {metric.expr}"
            })
        except Exception as e:
            print(f"⚠️ Ошибка в метрике {metric.title}: {e}")
            kpis.append({
                "title": metric.title,
                "value": "Ошибка",
                "tooltip": f"Ошибка: {str(e)}"
            })
    
    return kpis


def _evaluate_metric(df: pd.DataFrame, expr: str) -> float:
    """Вычисляет значение метрики"""
    # Простые выражения
    if expr == "count()":
        return len(df)
    
    # Агрегации по колонкам
    if expr.startswith("sum(") and expr.endswith(")"):
        col = expr[4:-1]
        return df[col].sum()
    elif expr.startswith("mean(") and expr.endswith(")"):
        col = expr[5:-1]
        return df[col].mean()
    elif expr.startswith("max(") and expr.endswith(")"):
        col = expr[4:-1]
        return df[col].max()
    elif expr.startswith("min(") and expr.endswith(")"):
        col = expr[4:-1]
        return df[col].min()
    elif expr.startswith("median(") and expr.endswith(")"):
        col = expr[7:-1]
        return df[col].median()
    
    # Fallback - пытаемся вычислить как Python выражение
    try:
        # Безопасная замена названий колонок
        safe_expr = expr
        for col in df.columns:
            if col in safe_expr:
                safe_expr = safe_expr.replace(col, f"df['{col}']")
        return eval(safe_expr)
    except:
        return 0


def _format_value(value: float, fmt: str) -> str:
    """Форматирует значение согласно типу"""
    if fmt == "currency":
        return f"${value:,.2f}"
    elif fmt == "percent":
        return f"{value:.1f}%"
    elif fmt == "number":
        return f"{value:,.0f}"
    else:
        return f"{value:,.2f}"


def _generate_charts(df: pd.DataFrame, charts: List[Chart]) -> List[Dict[str, str]]:
    """Генерирует графики"""
    chart_list = []
    
    for chart in charts:
        try:
            html = _create_chart_html(df, chart)
            chart_list.append({
                "title": chart.title or f"График: {chart.type}",
                "html": html
            })
        except Exception as e:
            print(f"⚠️ Ошибка в графике {chart.title}: {e}")
            chart_list.append({
                "title": chart.title or f"График: {chart.type}",
                "html": f"<div>Ошибка создания графика: {str(e)}</div>"
            })
    
    return chart_list


def _create_chart_html(df: pd.DataFrame, chart: Chart) -> str:
    """Создает HTML для графика"""
    
    if chart.type == "bar":
        if chart.agg and chart.y:
            # Агрегированные данные
            if chart.group:
                grouped = df.groupby(chart.group)[chart.y].agg(chart.agg).reset_index()
                fig = px.bar(grouped, x=chart.group, y=chart.y, title=chart.title)
            else:
                fig = px.bar(df, x=chart.x, y=chart.y, title=chart.title)
        else:
            # Простой подсчет
            value_counts = df[chart.x].value_counts()
            if chart.top:
                value_counts = value_counts.head(chart.top)
            fig = px.bar(x=value_counts.index, y=value_counts.values, title=chart.title)
    
    elif chart.type == "line":
        if chart.y:
            fig = px.line(df, x=chart.x, y=chart.y, title=chart.title)
        else:
            # Временной ряд
            value_counts = df[chart.x].value_counts().sort_index()
            fig = px.line(x=value_counts.index, y=value_counts.values, title=chart.title)
    
    elif chart.type == "scatter":
        if chart.y:
            fig = px.scatter(df, x=chart.x, y=chart.y, title=chart.title)
        else:
            fig = px.scatter(df, x=chart.x, title=chart.title)
    
    elif chart.type == "hist":
        fig = px.histogram(df, x=chart.x, title=chart.title)
    
    elif chart.type == "pie":
        if chart.y and chart.agg:
            # Агрегированные данные для pie chart
            if chart.group:
                grouped = df.groupby(chart.group)[chart.y].agg(chart.agg).reset_index()
                fig = px.pie(grouped, names=chart.group, values=chart.y, title=chart.title)
            else:
                fig = px.pie(df, names=chart.x, values=chart.y, title=chart.title)
        else:
            # Простой подсчет
            value_counts = df[chart.x].value_counts()
            if chart.top:
                value_counts = value_counts.head(chart.top)
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=chart.title)
    
    else:  # area
        if chart.y:
            fig = px.area(df, x=chart.x, y=chart.y, title=chart.title)
        else:
            value_counts = df[chart.x].value_counts().sort_index()
            fig = px.area(x=value_counts.index, y=value_counts.values, title=chart.title)
    
    # Применяем стиль
    fig.update_layout(
        plot_bgcolor='#131c2c',
        paper_bgcolor='#131c2c',
        font=dict(color='white'),
        margin=dict(t=40, b=30, l=0, r=0)
    )
    
    # Просто генерируем HTML с Plotly.js
    return fig.to_html(full_html=False, include_plotlyjs=True)


def _generate_tables(df: pd.DataFrame, viz_spec: VizSpec) -> List[Dict[str, Any]]:
    """Генерирует таблицы"""
    tables = []
    
    # Основная таблица данных
    tables.append({
        "title": "Data Overview",
        "headers": list(df.columns),
        "rows": df.head(10).values.tolist()
    })
    
    # Статистика по числовым колонкам
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe()
        tables.append({
            "title": "Numeric Columns Statistics",
            "headers": ["Metric"] + list(stats_df.columns),
            "rows": [[idx] + row.tolist() for idx, row in stats_df.iterrows()]
        })
    
    # Топ значения по категориальным колонкам
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols[:3]:  # Первые 3 категориальные колонки
        value_counts = df[col].value_counts().head(10)
        tables.append({
            "title": f"Top Values: {col}",
            "headers": [col, "Count"],
            "rows": [[val, count] for val, count in value_counts.items()]
        })
    
    return tables


def _generate_ai_summary(df: pd.DataFrame, viz_spec: VizSpec) -> str:
    """Генерирует AI summary"""
    summary_parts = []
    
    # Основная информация
    summary_parts.append(f"📊 Dashboard Analysis: {viz_spec.title}")
    summary_parts.append(f"📈 Analyzed {len(df):,} records and {len(df.columns)} columns")
    
    # Информация о метриках
    if viz_spec.metrics:
        summary_parts.append(f"📋 Created {len(viz_spec.metrics)} KPI metrics")
    
    # Информация о графиках
    if viz_spec.charts:
        chart_types = [chart.type for chart in viz_spec.charts]
        summary_parts.append(f"📊 Created {len(viz_spec.charts)} charts: {', '.join(set(chart_types))}")
    
    # Информация о фильтрах
    if viz_spec.filters:
        summary_parts.append(f"🔍 Applied {len(viz_spec.filters)} filters")
    
    # Комментарии из спецификации
    if viz_spec.comments:
        summary_parts.extend(viz_spec.comments)
    
    return "<br>".join(summary_parts)
