"""
VibeDash VizSpec Models
Pydantic модели для спецификации визуализации дашборда
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import json
import re

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


MAX_AUTOMATIC_METRICS = 8
MAX_AUTOMATIC_CHARTS = 6

_CURRENCY_TERMS = {
    "amount",
    "cost",
    "expense",
    "income",
    "margin",
    "price",
    "profit",
    "revenue",
    "sales",
    "spend",
}
_PERCENT_TERMS = {"percentage", "percent", "pct", "rate", "ratio", "share"}
_ADDITIVE_TERMS = {
    "amount",
    "cnt",
    "count",
    "expense",
    "income",
    "quantity",
    "revenue",
    "sales",
    "spend",
    "total",
    "units",
    "volume",
}
_MEASURE_TERMS = _CURRENCY_TERMS | _PERCENT_TERMS | _ADDITIVE_TERMS | {
    "age",
    "atemp",
    "distance",
    "duration",
    "hum",
    "humidity",
    "score",
    "speed",
    "temp",
    "temperature",
    "weight",
    "windspeed",
}
_TEMPORAL_TERMS = {"date", "datetime", "day", "month", "quarter", "time", "timestamp", "week", "year"}
_IDENTIFIER_TERMS = {
    "guid",
    "id",
    "identifier",
    "key",
    "uuid",
}
_ABBREVIATIONS = {"api": "API", "arpu": "ARPU", "id": "ID", "kpi": "KPI", "mrr": "MRR", "nps": "NPS", "roi": "ROI"}
_COLUMN_LABEL_OVERRIDES = {
    "atemp": "Apparent Temperature",
    "cnt": "Count",
    "dteday": "Date",
    "hum": "Humidity",
    "mnth": "Month",
    "weathersit": "Weather Situation",
    "workingday": "Working Day",
    "yr": "Year",
}


class Metric(BaseModel):
    """Метрика для KPI карточки"""
    title: str
    expr: str  # e.g. "sum(Revenue)" or "count()"
    fmt: Optional[str] = None  # e.g. "currency", "percent", "number"


class Chart(BaseModel):
    """Спецификация графика"""
    type: Literal["bar", "line", "area", "scatter", "hist", "pie", "gauge"]
    x: Optional[str] = None  # Опционально для gauge charts
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
    metrics: List[Metric] = Field(default_factory=list)
    charts: List[Chart] = Field(default_factory=list)
    filters: List[Filter] = Field(default_factory=list)
    comments: List[str] = Field(default_factory=list)


def create_saas_demo_viz_spec() -> VizSpec:
    """Return the stable dashboard contract used by the built-in product demo."""
    return VizSpec(
        title="SaaS Growth Evidence Dashboard",
        metrics=[
            Metric(
                title="Total Revenue",
                expr="sum(MonthlyRevenue)",
                fmt="currency",
            ),
            Metric(
                title="Average Daily Revenue",
                expr="mean(MonthlyRevenue)",
                fmt="currency",
            ),
            Metric(
                title="Average Customers",
                expr="mean(CustomerCount)",
                fmt="number",
            ),
            Metric(
                title="Average Churn Rate",
                expr="mean(ChurnRate)",
                fmt="percent",
            ),
        ],
        charts=[
            Chart(
                type="line",
                x="Date",
                y="MonthlyRevenue",
                title="Daily Revenue Trend",
            ),
            Chart(
                type="bar",
                y="MonthlyRevenue",
                agg="mean",
                group="Region",
                title="Average Revenue by Region",
            ),
            Chart(
                type="bar",
                y="ChurnRate",
                agg="mean",
                group="Plan",
                title="Average Churn by Plan",
            ),
            Chart(
                type="scatter",
                x="CustomerCount",
                y="MonthlyRevenue",
                title="Customer Count vs Revenue",
            ),
            Chart(
                type="hist",
                x="TicketCount",
                title="Support Ticket Distribution",
            ),
        ],
        comments=[
            "Built-in synthetic dataset with deterministic analytical signals.",
            "Evidence, statistics, and anomaly results are calculated locally.",
        ],
    )


def parse_prompt_to_viz_spec(
    prompt: str,
    df_columns: List[str],
    *,
    dataframe: Optional[Any] = None,
) -> VizSpec:
    """
    Парсинг текстового промпта в VizSpec
    Сначала пробуем эвристики, потом Ollama (если доступен)
    """
    # Очищаем промпт
    prompt = prompt.strip().lower()
    
    column_profiles = _profile_dataframe_columns(dataframe, df_columns)
    excluded_columns = _prompt_excluded_columns(prompt, df_columns)

    # Эвристический парсинг
    spec = _parse_heuristics(
        prompt,
        df_columns,
        column_profiles,
        excluded_columns=excluded_columns,
    )
    
    # Пробуем улучшить через Ollama
    try:
        from .ollama_client import ollama_generate
        if ollama_generate:
            improved_spec = _improve_with_ollama(prompt, spec, df_columns)
            if improved_spec:
                if _prompt_mentioned_columns(prompt, df_columns):
                    improved_spec = improved_spec.model_copy(
                        update={
                            "metrics": spec.metrics + improved_spec.metrics,
                            "charts": spec.charts + improved_spec.charts,
                            "filters": spec.filters + improved_spec.filters,
                        }
                    )
                return _sanitize_generated_spec(
                    improved_spec,
                    df_columns,
                    column_profiles,
                    excluded_columns=excluded_columns,
                )
    except Exception as e:
        print(f"⚠️ Ollama недоступен, используем эвристики: {e}")
    
    return _sanitize_generated_spec(
        spec,
        df_columns,
        column_profiles,
        excluded_columns=excluded_columns,
    )


def _column_terms(column: str) -> set[str]:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(column))
    return {
        part
        for part in re.split(r"[^a-z0-9]+", normalized.lower())
        if part
    }


def _humanize_column(column: str) -> str:
    override = _COLUMN_LABEL_OVERRIDES.get(str(column).strip().lower())
    if override:
        return override
    normalized = re.sub(r"[_\-]+", " ", str(column).strip())
    normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", normalized)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    return " ".join(
        _ABBREVIATIONS.get(word.lower(), word.capitalize())
        for word in normalized.split()
    )


def _column_aliases(column: str) -> set[str]:
    """Return conservative prompt aliases for an exact dataset column."""
    original = str(column).strip().lower()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(column)).lower()
    spaced = re.sub(r"[^a-z0-9а-яё]+", " ", spaced).strip()
    aliases = {original, spaced, spaced.replace(" ", "_")}
    return {alias for alias in aliases if alias}


def _prompt_contains_alias(prompt: str, alias: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9а-яё]){re.escape(alias)}(?![a-z0-9а-яё])",
            prompt,
            flags=re.IGNORECASE,
        )
    )


def _prompt_mentioned_columns(prompt: str, df_columns: List[str]) -> List[str]:
    """Find explicitly named columns without fuzzy guesses or invented fields."""
    return [
        column
        for column in df_columns
        if any(
            _prompt_contains_alias(prompt, alias)
            for alias in _column_aliases(column)
        )
    ]


def _prompt_excluded_columns(prompt: str, df_columns: List[str]) -> set[str]:
    """Honor explicit English and Russian requests to ignore a column."""
    exclusion_prefix = (
        r"(?:не\s+(?:используй|использовать|учитывай|учитывать)|"
        r"исключи|исключить|игнорируй|игнорировать|"
        r"do\s+not\s+use|don't\s+use|exclude|ignore|without)"
    )
    excluded = set()
    for column in df_columns:
        for alias in _column_aliases(column):
            if re.search(
                rf"{exclusion_prefix}[^.!?\n]{{0,40}}"
                rf"(?<![a-z0-9а-яё]){re.escape(alias)}(?![a-z0-9а-яё])",
                prompt,
                flags=re.IGNORECASE,
            ):
                excluded.add(column)
                break
    return excluded


def _is_temporal_column(column: str) -> bool:
    return bool(_column_terms(column) & _TEMPORAL_TERMS)


def _is_identifier_column(column: str) -> bool:
    """Recognize fields whose values identify or order rows, not cohorts."""
    terms = _column_terms(column)
    normalized = re.sub(r"[^a-z0-9]+", "", str(column).lower())
    return (
        normalized in {
            "index",
            "rank",
            "ranking",
            "recordid",
            "rk",
            "rowid",
            "rowindex",
            "rownumber",
        }
        or bool(terms & (_IDENTIFIER_TERMS - {"id", "key"}))
        or "id" in terms
        or (
            "key" in terms
            and str(column).lower().rstrip().endswith("key")
        )
    )


def _profile_dataframe_columns(
    dataframe: Optional[Any],
    df_columns: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build a bounded, aggregate-only profile for automatic semantics."""
    if dataframe is None:
        return {}

    profiles: Dict[str, Dict[str, Any]] = {}
    for column in df_columns:
        if column not in dataframe.columns:
            continue
        series = dataframe[column]
        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        is_numeric = bool(is_numeric_dtype(series.dtype))
        is_temporal = bool(is_datetime64_any_dtype(series.dtype))
        if not is_temporal and not is_numeric and non_missing_count:
            sample = series.dropna().astype(str).head(200)
            if not sample.empty:
                date_like = sample.str.match(
                    r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
                    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4})(?:[T\s].*)?\s*$"
                )
                if date_like.mean() >= 0.9:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    is_temporal = bool(parsed.notna().mean() >= 0.9)

        numeric_values = (
            pd.to_numeric(series.dropna(), errors="coerce")
            if is_numeric
            else None
        )
        is_integer_like = bool(
            is_numeric
            and numeric_values is not None
            and not numeric_values.empty
            and ((numeric_values - numeric_values.round()).abs() <= 1e-9).all()
        )
        is_monotonic_unique = bool(
            is_integer_like
            and non_missing_count >= 3
            and unique_count == non_missing_count
            and (
                numeric_values.is_monotonic_increasing
                or numeric_values.is_monotonic_decreasing
            )
        )
        is_sequential_identifier = bool(
            is_monotonic_unique
            and numeric_values.diff().dropna().abs().eq(1).all()
        )
        low_cardinality_integer = bool(
            is_integer_like
            and unique_count > 1
            and (
                unique_count <= 20
                or unique_count / non_missing_count <= 0.02
            )
        )

        if is_temporal or _is_temporal_column(column):
            role = "temporal"
        elif _is_identifier_column(column) or is_sequential_identifier:
            role = "identifier"
        elif is_numeric and _column_terms(column) & _MEASURE_TERMS:
            role = "measure"
        elif is_numeric and low_cardinality_integer:
            role = "category"
        elif is_numeric:
            role = "measure"
        else:
            role = "category"

        profiles[column] = {
            "is_numeric": is_numeric,
            "is_temporal": is_temporal,
            "is_integer_like": is_integer_like,
            "is_monotonic_unique": is_monotonic_unique,
            "is_sequential_identifier": is_sequential_identifier,
            "non_missing_count": non_missing_count,
            "unique_count": unique_count,
            "unique_ratio": (
                unique_count / non_missing_count if non_missing_count else 0.0
            ),
            "role": role,
        }
    return profiles


def _column_role(
    column: str,
    column_profiles: Mapping[str, Mapping[str, Any]],
) -> str:
    profile = column_profiles.get(column, {})
    role = profile.get("role")
    if role in {"temporal", "identifier", "category", "measure"}:
        return str(role)
    if _is_temporal_column(column):
        return "temporal"
    if _is_identifier_column(column):
        return "identifier"
    return "measure" if _is_numeric_column(column) else "category"


def _is_useful_automatic_category(
    column: str,
    column_profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Keep repeatable business cohorts; reject IDs and near-unique labels."""
    if _column_role(column, column_profiles) != "category":
        return False
    profile = column_profiles.get(column)
    if not profile:
        return True
    non_missing_count = int(profile.get("non_missing_count", 0))
    unique_count = int(profile.get("unique_count", 0))
    if non_missing_count == 0 or unique_count <= 1:
        return False
    return unique_count < 20 or unique_count / non_missing_count <= 0.5


def _metric_format_for_column(column: str) -> str:
    terms = _column_terms(column)
    if terms & _CURRENCY_TERMS:
        return "currency"
    if terms & _PERCENT_TERMS:
        return "percent"
    return "number"


def _metric_column(expression: str) -> Optional[str]:
    match = re.fullmatch(
        r"\s*(?:sum|mean|median|min|max|std|count|nunique)\(\s*`?([^()`]+?)`?\s*\)\s*",
        expression or "",
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _sanitize_generated_spec(
    spec: VizSpec,
    df_columns: List[str],
    column_profiles: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
    *,
    excluded_columns: Optional[set[str]] = None,
) -> VizSpec:
    """Apply conservative product rules to automatically generated specs.

    Automatic dashboards must not imply targets that the user never supplied.
    The sanitizer therefore removes gauges, deduplicates output, caps visual
    volume, and repairs obvious semantic units. Explicit built-in demo specs do
    not pass through this function.
    """
    known_columns = set(df_columns)
    profiles = column_profiles or {}
    excluded = excluded_columns or set()
    metrics: List[Metric] = []
    metric_expressions = set()
    for metric in spec.metrics:
        expression_key = re.sub(r"\s+", "", metric.expr).lower()
        if expression_key in metric_expressions:
            continue
        column = _metric_column(metric.expr)
        if column and column not in known_columns:
            continue
        if column and column in excluded:
            continue
        aggregate = (metric.expr or "").strip().split("(", 1)[0].lower()
        if (
            column
            and _column_role(column, profiles) != "measure"
            and aggregate not in {"count", "nunique"}
        ):
            continue
        if (
            column
            and profiles
            and not bool(profiles.get(column, {}).get("is_numeric", False))
            and not re.match(r"\s*(?:count|nunique)\s*\(", metric.expr, re.IGNORECASE)
        ):
            continue
        if column:
            expected_format = _metric_format_for_column(column)
            if metric.fmt in {None, "number"} and expected_format != "number":
                metric = metric.model_copy(update={"fmt": expected_format})
        metrics.append(metric)
        metric_expressions.add(expression_key)
        if len(metrics) >= MAX_AUTOMATIC_METRICS:
            break

    charts: List[Chart] = []
    chart_signatures = set()
    for chart in spec.charts:
        if chart.type == "gauge":
            continue
        referenced = [chart.x, chart.group]
        if isinstance(chart.y, list):
            referenced.extend(chart.y)
        else:
            referenced.append(chart.y)
        if any(column and column not in known_columns for column in referenced):
            continue
        if any(column and column in excluded for column in referenced):
            continue
        if (
            chart.type == "hist"
            and chart.x
            and _column_role(chart.x, profiles) != "measure"
        ):
            continue
        if chart.type == "scatter" and any(
            column and _column_role(column, profiles) != "measure"
            for column in [
                chart.x,
                chart.y if isinstance(chart.y, str) else None,
            ]
        ):
            continue
        if chart.type in {"line", "area"}:
            if chart.x and _column_role(chart.x, profiles) == "identifier":
                continue
            if (
                isinstance(chart.y, str)
                and _column_role(chart.y, profiles) != "measure"
            ):
                continue
        if chart.type in {"bar", "pie"} and isinstance(chart.y, str):
            if _column_role(chart.y, profiles) != "measure":
                continue
        category = chart.group
        if not category and chart.type in {"bar", "pie"} and not chart.y:
            category = chart.x
        if category and not _is_useful_automatic_category(category, profiles):
            continue
        if (
            chart.type in {"bar", "pie"}
            and chart.x
            and _is_temporal_column(chart.x)
            and (
                not chart.y
                or chart.top is not None
                or "top" in (chart.title or "").lower()
                or "rank" in (chart.title or "").lower()
            )
        ):
            continue
        signature = (
            chart.type,
            chart.x,
            tuple(chart.y) if isinstance(chart.y, list) else chart.y,
            chart.group,
            chart.agg,
        )
        if signature in chart_signatures:
            continue
        charts.append(chart)
        chart_signatures.add(signature)
        if len(charts) >= MAX_AUTOMATIC_CHARTS:
            break

    filters = [
        filter_obj
        for filter_obj in spec.filters
        if filter_obj.field in known_columns
        and filter_obj.field not in excluded
        and (
            filter_obj.where
            or _is_useful_automatic_category(filter_obj.field, profiles)
        )
    ]

    return spec.model_copy(
        update={"metrics": metrics, "charts": charts, "filters": filters}
    )


def _parse_heuristics(
    prompt: str,
    df_columns: List[str],
    column_profiles: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
    *,
    excluded_columns: Optional[set[str]] = None,
) -> VizSpec:
    """Эвристический парсинг промпта"""
    
    # Определяем тип дашборда по ключевым словам
    title = "Аналитический дашборд"
    if any(word in prompt for word in ["продаж", "sales", "revenue", "доход"]):
        title = "Дашборд продаж"
    elif any(word in prompt for word in ["финанс", "finance", "расход", "expense"]):
        title = "Финансовый дашборд"
    elif any(word in prompt for word in ["недвижим", "real estate", "property", "квартир"]):
        title = "Анализ недвижимости"
    
    # Разделяем роли по реальным значениям, затем учитываем явные указания
    # пользователя. Числовой dtype сам по себе не делает поле метрикой.
    numeric_cols = []
    categorical_cols = []

    profiles = column_profiles or {}
    excluded = excluded_columns or set()
    mentioned_columns = [
        column
        for column in _prompt_mentioned_columns(prompt, df_columns)
        if column not in excluded
    ]
    for col in df_columns:
        if col in excluded:
            continue
        role = _column_role(col, profiles)
        if role == "identifier":
            continue
        if role == "temporal":
            categorical_cols.append(col)
            continue
        if role == "measure":
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    mentioned_measures = [col for col in mentioned_columns if col in numeric_cols]
    mentioned_categories = [
        col for col in mentioned_columns if col in categorical_cols
    ]
    numeric_cols = mentioned_measures + [
        col for col in numeric_cols if col not in mentioned_measures
    ]

    metrics = []
    charts = []
    filters = []
    comments = []
    revenue_cols = []
    
    # Базовые метрики: небольшой набор без дубликатов и с явными единицами.
    if numeric_cols:
        metric_columns = mentioned_measures or numeric_cols
        revenue_cols = [
            col for col in metric_columns
            if _column_terms(col) & _CURRENCY_TERMS
        ]
        if revenue_cols:
            revenue = revenue_cols[0]
            revenue_label = _humanize_column(revenue)
            metrics.append(Metric(
                title=f"Total {revenue_label}",
                expr=f"sum({revenue})",
                fmt="currency",
            ))
            metrics.append(Metric(
                title=f"Average {revenue_label}",
                expr=f"mean({revenue})",
                fmt="currency",
            ))
        
        additive_cols = [
            col for col in mentioned_measures
            if _column_terms(col) & _ADDITIVE_TERMS
            and col not in revenue_cols
        ]
        for column in additive_cols[:2]:
            metrics.append(Metric(
                title=f"Total {_humanize_column(column)}",
                expr=f"sum({column})",
                fmt=_metric_format_for_column(column),
            ))

        metrics.append(Metric(title="Record Count", expr="count()", fmt="number"))

        represented_expressions = {metric.expr for metric in metrics}
        for col in metric_columns:
            expression = f"mean({col})"
            if expression in represented_expressions:
                continue
            metrics.append(Metric(
                title=f"Average {_humanize_column(col)}",
                expr=expression,
                fmt=_metric_format_for_column(col),
            ))
            represented_expressions.add(expression)
            if len(metrics) >= MAX_AUTOMATIC_METRICS:
                break
    
    # Базовые графики - создаем больше визуализаций
    if numeric_cols:
        chart_measure_cols = mentioned_measures or numeric_cols
        # Гистограмма для первой числовой колонки
        charts.append(Chart(
            type="hist",
            x=chart_measure_cols[0],
            title=f"Distribution of {chart_measure_cols[0]}"
        ))
        
        # Если есть вторая числовая колонка - scatter plot
        if len(chart_measure_cols) > 1:
            charts.append(Chart(
                type="scatter",
                x=chart_measure_cols[0],
                y=chart_measure_cols[1],
                title=f"Correlation: {chart_measure_cols[0]} vs {chart_measure_cols[1]}"
            ))
        
        # Если есть третья числовая колонка - еще один scatter
        if len(chart_measure_cols) > 2:
            charts.append(Chart(
                type="scatter",
                x=chart_measure_cols[1],
                y=chart_measure_cols[2],
                title=f"Correlation: {chart_measure_cols[1]} vs {chart_measure_cols[2]}"
            ))

    temporal_cols = [
        col for col in categorical_cols
        if _column_role(col, profiles) == "temporal"
    ]
    business_categories = [
        col
        for col in categorical_cols
        if _is_useful_automatic_category(col, profiles)
    ]
    mentioned_temporals = [col for col in mentioned_columns if col in temporal_cols]
    mentioned_business_categories = [
        col for col in mentioned_categories if col in business_categories
    ]
    temporal_cols = mentioned_temporals + [
        col for col in temporal_cols if col not in mentioned_temporals
    ]
    business_categories = mentioned_business_categories + [
        col for col in business_categories if col not in mentioned_business_categories
    ]

    # Временная динамика строится по оси времени, а не как рейтинг дат.
    if temporal_cols and numeric_cols and (
        mentioned_temporals
        or any(word in prompt for word in ["динамик", "тренд", "trend", "time"])
    ):
        trend_metric = mentioned_measures[0] if mentioned_measures else (
            revenue_cols[0] if revenue_cols else numeric_cols[0]
        )
        charts.append(Chart(
            type="line",
            x=temporal_cols[0],
            y=trend_metric,
            title=f"{_humanize_column(trend_metric)} over time",
        ))

    # Категориальные графики показывают честную агрегацию, а не сумму строк
    # под вводящим в заблуждение названием Top 10.
    chart_categories = mentioned_business_categories or business_categories[:2]
    for category in chart_categories:
        if mentioned_measures or revenue_cols:
            value_column = (
                mentioned_measures[0] if mentioned_measures else revenue_cols[0]
            )
            aggregation = (
                "sum"
                if _column_terms(value_column) & _ADDITIVE_TERMS
                else "mean"
            )
            charts.append(Chart(
                type="bar",
                y=value_column,
                agg=aggregation,
                group=category,
                title=(
                    f"{'Total' if aggregation == 'sum' else 'Average'} "
                    f"{_humanize_column(value_column)} by {_humanize_column(category)}"
                ),
            ))
        else:
            charts.append(Chart(
                type="bar",
                x=category,
                top=10,
                title=f"Record count by {_humanize_column(category)}",
            ))
    
    # Фильтры
    if business_categories:
        filters.append(Filter(field=business_categories[0], values=None))
    
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
    # Исключаем ID колонки, которые обычно содержат строки
    if "id" in col_name.lower() and "customer" in col_name.lower():
        return False
    if "id" in col_name.lower() and "user" in col_name.lower():
        return False
    if "id" in col_name.lower() and "product" in col_name.lower():
        return False
    
    numeric_keywords = ["count", "number", "amount", "price", "cost", "revenue", "sales", "age", "year", "month", "day", "value", "score", "rate", "percent", "ratio", "total", "sum", "avg", "mean", "max", "min"]
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
        
        response = ollama_generate("llama3.1:8b", system_prompt, prompt)
        if response:
            # Парсим JSON ответ
            data = json.loads(response)
            return VizSpec(**data)
            
    except Exception as e:
        print(f"⚠️ Ошибка Ollama: {e}")
    
    return None
