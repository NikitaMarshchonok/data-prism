"""
Мост между VibeDash и существующим генератором дашборда
"""
import ast
import operator
import re
from functools import reduce
from html import escape
from numbers import Number
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .spec import VizSpec, Metric, Chart, Filter
from .insight_engine import EvidenceBasedInsightEngine
from .statistical_engine import StatisticalValidationEngine
from .anomaly_segmentation_engine import AnomalySegmentationEngine


MAX_DSL_LENGTH = 1000
MAX_DSL_NODES = 200


class UnsafeExpressionError(ValueError):
    """Выражение содержит операцию, запрещённую в VibeDash DSL."""


def _parse_dsl_expression(expression: str, columns) -> tuple[ast.Expression, Dict[str, str]]:
    """Разбирает выражение без выполнения Python-кода.

    Колонки с пробелами поддерживаются в обратных кавычках:
    `` `Sales Amount` > 100 ``.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("Expression must be a non-empty string")
    if len(expression) > MAX_DSL_LENGTH:
        raise UnsafeExpressionError("Expression is too long")

    column_names = {str(column) for column in columns}
    aliases = {}

    def replace_backtick_column(match):
        column = match.group(1)
        if column not in column_names:
            raise UnsafeExpressionError(f"Column '{column}' not found")
        alias = f"__column_{len(aliases)}"
        aliases[alias] = column
        return alias

    normalized = re.sub(r"`([^`]+)`", replace_backtick_column, expression.strip())
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as error:
        raise UnsafeExpressionError("Invalid expression syntax") from error

    if sum(1 for _ in ast.walk(tree)) > MAX_DSL_NODES:
        raise UnsafeExpressionError("Expression is too complex")
    return tree, aliases


def _resolve_column(df: pd.DataFrame, node: ast.AST, aliases: Dict[str, str]):
    if not isinstance(node, ast.Name):
        raise UnsafeExpressionError("A column name is required")
    column = aliases.get(node.id, node.id)
    if column not in df.columns:
        raise UnsafeExpressionError(f"Column '{column}' not found")
    return column


def _evaluate_filter_condition(df: pd.DataFrame, expression: str) -> pd.Series:
    """Вычисляет булеву маску с помощью ограниченного DSL."""
    tree, aliases = _parse_dsl_expression(expression, df.columns)

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name):
            return df[_resolve_column(df, node, aliases)]
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool, type(None))):
                return node.value
            raise UnsafeExpressionError("Unsupported literal")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [evaluate(element) for element in node.elts]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [evaluate(value) for value in node.values]
            operation = operator.and_ if isinstance(node.op, ast.And) else operator.or_
            return reduce(operation, values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            operation = operator.and_ if isinstance(node.op, ast.BitAnd) else operator.or_
            return operation(evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~operand if isinstance(operand, pd.Series) else not operand
            if isinstance(node.op, ast.USub) and isinstance(operand, Number):
                return -operand
            if isinstance(node.op, ast.UAdd) and isinstance(operand, Number):
                return operand
            raise UnsafeExpressionError("Unsupported unary operator")
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            result = None
            for comparison_operator, comparator in zip(node.ops, node.comparators):
                right = evaluate(comparator)
                comparison = _compare_filter_values(left, comparison_operator, right)
                result = comparison if result is None else operator.and_(result, comparison)
                left = right
            return result

        raise UnsafeExpressionError(
            f"Operation '{type(node).__name__}' is not allowed in filter expressions"
        )

    mask = evaluate(tree)
    if not isinstance(mask, pd.Series) or not pd.api.types.is_bool_dtype(mask.dtype):
        raise UnsafeExpressionError("Filter expression must produce a boolean mask")
    return mask.reindex(df.index, fill_value=False).fillna(False)


def _compare_filter_values(left, comparison_operator, right):
    if isinstance(comparison_operator, ast.In):
        return left.isin(right) if isinstance(left, pd.Series) else left in right
    if isinstance(comparison_operator, ast.NotIn):
        result = left.isin(right) if isinstance(left, pd.Series) else left in right
        return ~result if isinstance(result, pd.Series) else not result
    if isinstance(comparison_operator, ast.Is):
        if right is not None:
            raise UnsafeExpressionError("'is' is only supported with None")
        return left.isna() if isinstance(left, pd.Series) else left is None
    if isinstance(comparison_operator, ast.IsNot):
        if right is not None:
            raise UnsafeExpressionError("'is not' is only supported with None")
        return left.notna() if isinstance(left, pd.Series) else left is not None

    operations = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    operation = operations.get(type(comparison_operator))
    if operation is None:
        raise UnsafeExpressionError("Unsupported comparison operator")
    return operation(left, right)


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
            "ai_summary": "Нет данных для анализа",
            "insights": [],
            "statistical_validation": StatisticalValidationEngine(
                pd.DataFrame() if df is None else df
            ).analyze(),
            "pattern_analysis": AnomalySegmentationEngine(
                pd.DataFrame() if df is None else df
            ).analyze(),
            "df_shape": (0, 0) if df is None else df.shape,
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

    # Доказательные выводы рассчитываются детерминированно, без LLM.
    insights = EvidenceBasedInsightEngine(filtered_df).generate()
    statistical_validation = StatisticalValidationEngine(filtered_df).analyze()
    pattern_analysis = AnomalySegmentationEngine(filtered_df).analyze()
    
    return {
        "kpis": kpis,
        "charts": charts,
        "tables": tables,
        "ai_summary": ai_summary,
        "insights": insights,
        "statistical_validation": statistical_validation,
        "pattern_analysis": pattern_analysis,
        "df_shape": filtered_df.shape,
    }


def _apply_filters(df: pd.DataFrame, filters: List[Filter]) -> pd.DataFrame:
    """Применяет фильтры к DataFrame"""
    filtered_df = df.copy()
    
    for filter_obj in filters:
        if filter_obj.where:
            try:
                mask = _evaluate_filter_condition(filtered_df, filter_obj.where)
                filtered_df = filtered_df.loc[mask]
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
    """Вычисляет KPI через безопасный белый список агрегатов и арифметики."""
    tree, aliases = _parse_dsl_expression(expr, df.columns)

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            operations = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            operation = operations.get(type(node.op))
            if operation is None:
                raise UnsafeExpressionError("Unsupported arithmetic operator")
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise UnsafeExpressionError("Exponent is too large")
            return operation(left, right)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise UnsafeExpressionError("Only named aggregate functions are allowed")

            function_name = node.func.id
            if function_name == "count" and not node.args:
                return len(df)
            if len(node.args) != 1:
                raise UnsafeExpressionError(
                    f"Function '{function_name}' expects exactly one column"
                )

            column = _resolve_column(df, node.args[0], aliases)
            series = df[column]
            if function_name == "count":
                return series.count()
            if function_name == "nunique":
                return series.nunique()
            if function_name in {"sum", "mean", "median", "std"}:
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"Column '{column}' is not numeric")
                return getattr(series, function_name)()
            if function_name in {"min", "max"}:
                if not pd.api.types.is_numeric_dtype(series):
                    return series.nunique()
                return getattr(series, function_name)()
            raise UnsafeExpressionError(f"Function '{function_name}' is not allowed")

        raise UnsafeExpressionError(
            f"Operation '{type(node).__name__}' is not allowed in metric expressions"
        )

    result = evaluate(tree)
    if not isinstance(result, Number):
        raise UnsafeExpressionError("Metric expression must produce a number")
    return result


def _format_value(value: float, fmt: str) -> str:
    """Форматирует значение согласно типу"""
    try:
        if fmt == "currency":
            return f"${value:,.2f}"
        elif fmt == "percent":
            return f"{value:.1f}%"
        elif fmt == "number":
            return f"{value:,.0f}"
        else:
            return f"{value:,.2f}"
    except (ValueError, TypeError):
        # Если форматирование не удается, возвращаем как строку
        return str(value)


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
                "html": f"<div>Ошибка создания графика: {escape(str(e))}</div>"
            })
    
    return chart_list


def _create_chart_html(df: pd.DataFrame, chart: Chart) -> str:
    """Создает HTML графика; аналитические выводы формируются отдельно."""
    
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
    
    elif chart.type == "gauge":
        # Gauge chart для KPI метрик
        if chart.y:
            # Вычисляем значение для gauge
            if chart.agg == "mean":
                value = df[chart.y].mean()
            elif chart.agg == "sum":
                value = df[chart.y].sum()
            elif chart.agg == "count":
                value = len(df)
            else:
                value = df[chart.y].mean()

            # Определяем максимальное значение
            max_value = df[chart.y].max() * 1.2 if chart.agg in ["mean", "sum"] else len(df) * 1.2

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': chart.title},
                delta = {'reference': max_value * 0.8},
                gauge = {
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, max_value * 0.4], 'color': "lightgray"},
                        {'range': [max_value * 0.4, max_value * 0.6], 'color': "gray"},
                        {'range': [max_value * 0.6, max_value * 0.8], 'color': "orange"},
                        {'range': [max_value * 0.8, max_value], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value * 0.9
                    }
                }
            ))
        else:
            # Простой gauge для подсчета записей
            value = len(df)
            max_value = value * 1.2

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': chart.title},
                gauge = {
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, max_value * 0.4], 'color': "lightgray"},
                        {'range': [max_value * 0.4, max_value * 0.6], 'color': "gray"},
                        {'range': [max_value * 0.6, max_value * 0.8], 'color': "orange"},
                        {'range': [max_value * 0.8, max_value], 'color': "green"}
                    ]
                }
            ))

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

    # Plotly загружается один раз шаблоном страницы.
    return fig.to_html(full_html=False, include_plotlyjs=False)


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
    
    return "\n".join(summary_parts)
