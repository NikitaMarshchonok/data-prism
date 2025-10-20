"""
VibeDash VizSpec Models
Pydantic модели для спецификации визуализации дашборда
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
import json
import re


class Metric(BaseModel):
    """Метрика для KPI карточки"""
    title: str
    expr: str  # e.g. "sum(Revenue)" or "count()"
    fmt: Optional[str] = None  # e.g. "currency", "percent", "number"


class Chart(BaseModel):
    """Спецификация графика"""
    type: Literal["bar", "line", "area", "scatter", "hist", "pie"]
    x: str
    y: Optional[Union[str, List[str]]] = None
    agg: Optional[str] = None   # "sum", "mean", "count", "max", "min"
    top: Optional[int] = None   # для топ-N записей
    resample: Optional[str] = None  # "D", "W", "M" для временных рядов
    group: Optional[str] = None  # группировка по категории
    title: Optional[str] = None


class Filter(BaseModel):
    """Фильтр для данных"""
    field: str
    values: Optional[List[str]] = None
    where: Optional[str] = None  # simple DSL: e.g. "Revenue > 5000 and Region == 'EMEA'"


class VizSpec(BaseModel):
    """Полная спецификация дашборда"""
    title: str
    metrics: List[Metric] = []
    charts: List[Chart] = []
    filters: List[Filter] = []
    comments: List[str] = []


def parse_prompt_to_viz_spec(prompt: str, df_columns: List[str]) -> VizSpec:
    """
    Парсинг текстового промпта в VizSpec
    Сначала пробуем эвристики, потом Ollama (если доступен)
    """
    # Очищаем промпт
    prompt = prompt.strip().lower()
    
    # Эвристический парсинг
    spec = _parse_heuristics(prompt, df_columns)
    
    # Пробуем улучшить через Ollama
    try:
        from .ollama_client import ollama_generate
        if ollama_generate:
            improved_spec = _improve_with_ollama(prompt, spec, df_columns)
            if improved_spec:
                return improved_spec
    except Exception as e:
        print(f"⚠️ Ollama недоступен, используем эвристики: {e}")
    
    return spec


def _parse_heuristics(prompt: str, df_columns: List[str]) -> VizSpec:
    """Эвристический парсинг промпта"""
    
    # Определяем тип дашборда по ключевым словам
    title = "Аналитический дашборд"
    if any(word in prompt for word in ["продаж", "sales", "revenue", "доход"]):
        title = "Дашборд продаж"
    elif any(word in prompt for word in ["финанс", "finance", "расход", "expense"]):
        title = "Финансовый дашборд"
    elif any(word in prompt for word in ["недвижим", "real estate", "property", "квартир"]):
        title = "Анализ недвижимости"
    
    # Ищем числовые колонки для метрик
    numeric_cols = [col for col in df_columns if _is_numeric_column(col)]
    categorical_cols = [col for col in df_columns if not _is_numeric_column(col)]
    
    metrics = []
    charts = []
    filters = []
    comments = []
    
    # Базовые метрики - создаем больше KPI
    if numeric_cols:
        # Ищем колонки с суммами/доходами
        revenue_cols = [col for col in numeric_cols if any(word in col.lower() for word in ["revenue", "sales", "доход", "сумма", "amount", "price", "cost"])]
        if revenue_cols:
            metrics.append(Metric(title="Total Amount", expr=f"sum({revenue_cols[0]})", fmt="currency"))
            metrics.append(Metric(title="Average Value", expr=f"mean({revenue_cols[0]})", fmt="currency"))
        
        # Считаем записи
        metrics.append(Metric(title="Record Count", expr="count()", fmt="number"))
        
        # Основные статистики для первых 3 числовых колонок
        for i, col in enumerate(numeric_cols[:3]):
            metrics.append(Metric(title=f"Avg {col}", expr=f"mean({col})", fmt="number"))
            metrics.append(Metric(title=f"Max {col}", expr=f"max({col})", fmt="number"))
            metrics.append(Metric(title=f"Min {col}", expr=f"min({col})", fmt="number"))
    
    # Базовые графики - создаем больше визуализаций
    if numeric_cols:
        # Гистограмма для первой числовой колонки
        charts.append(Chart(
            type="hist",
            x=numeric_cols[0],
            title=f"Distribution of {numeric_cols[0]}"
        ))
        
        # Если есть вторая числовая колонка - scatter plot
        if len(numeric_cols) > 1:
            charts.append(Chart(
                type="scatter",
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"Correlation: {numeric_cols[0]} vs {numeric_cols[1]}"
            ))
        
        # Если есть третья числовая колонка - еще один scatter
        if len(numeric_cols) > 2:
            charts.append(Chart(
                type="scatter",
                x=numeric_cols[1],
                y=numeric_cols[2],
                title=f"Correlation: {numeric_cols[1]} vs {numeric_cols[2]}"
            ))
    
    # Категориальные графики
    if categorical_cols:
        # Столбчатый график топ-10 категорий
        charts.append(Chart(
            type="bar",
            x=categorical_cols[0],
            y=numeric_cols[0] if numeric_cols else None,
            agg="count",
            top=10,
            title=f"Top 10 {categorical_cols[0]}"
        ))
        
        # Если есть вторая категориальная колонка
        if len(categorical_cols) > 1:
            charts.append(Chart(
                type="bar",
                x=categorical_cols[1],
                y=numeric_cols[0] if numeric_cols else None,
                agg="count",
                top=10,
                title=f"Top 10 {categorical_cols[1]}"
            ))
    
    # Дополнительные графики по ключевым словам
    if "тренд" in prompt or "trend" in prompt:
        if len(numeric_cols) > 1:
            charts.append(Chart(
                type="line",
                x=numeric_cols[0],
                y=numeric_cols[1],
                title="Trend Analysis"
            ))
    
    if "корреляц" in prompt or "correlation" in prompt:
        if len(numeric_cols) > 1:
            charts.append(Chart(
                type="scatter",
                x=numeric_cols[0],
                y=numeric_cols[1],
                title="Correlation Analysis"
            ))
    
    # Фильтры
    if categorical_cols:
        filters.append(Filter(field=categorical_cols[0], values=None))
    
    # Комментарии
    comments.append("Дашборд создан автоматически на основе вашего запроса")
    if numeric_cols:
        comments.append(f"Проанализировано {len(numeric_cols)} числовых колонок")
    if categorical_cols:
        comments.append(f"Найдено {len(categorical_cols)} категориальных колонок")
    
    return VizSpec(
        title=title,
        metrics=metrics,
        charts=charts,
        filters=filters,
        comments=comments
    )


def _is_numeric_column(col_name: str) -> bool:
    """Проверяет, является ли колонка числовой по названию"""
    numeric_keywords = ["id", "count", "number", "amount", "price", "cost", "revenue", "sales", "age", "year", "month", "day", "value", "score", "rate", "percent", "ratio"]
    return any(keyword in col_name.lower() for keyword in numeric_keywords) or col_name.isdigit()


def _improve_with_ollama(prompt: str, spec: VizSpec, df_columns: List[str]) -> Optional[VizSpec]:
    """Улучшает спецификацию через Ollama"""
    try:
        from .ollama_client import ollama_generate
        
        # Создаем промпт для Ollama
        system_prompt = f"""
        Ты эксперт по анализу данных. Создай JSON спецификацию дашборда на основе:
        - Промпт пользователя: "{prompt}"
        - Доступные колонки: {df_columns}
        
        Верни JSON в формате:
        {{
            "title": "Название дашборда",
            "metrics": [{{"title": "Название", "expr": "выражение", "fmt": "формат"}}],
            "charts": [{{"type": "тип", "x": "колонка_x", "y": "колонка_y", "title": "название"}}],
            "filters": [{{"field": "поле", "values": ["значения"]}}],
            "comments": ["комментарий"]
        }}
        """
        
        response = ollama_generate(system_prompt, prompt)
        if response:
            # Парсим JSON ответ
            data = json.loads(response)
            return VizSpec(**data)
            
    except Exception as e:
        print(f"⚠️ Ошибка Ollama: {e}")
    
    return None