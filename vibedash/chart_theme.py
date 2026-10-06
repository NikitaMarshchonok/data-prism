"""Shared visual contract for VibeDash evidence charts.

The theme intentionally favours legibility and restrained semantic colour over
decorative effects.  Analytical conclusions remain in the evidence panels;
charts are interactive diagnostics that help a reviewer inspect those claims.
"""

from __future__ import annotations

import math
import re
from html import escape
from numbers import Number
from typing import Any, Iterable, Optional

import pandas as pd
import plotly.graph_objects as go

from .spec import Chart


ACCENT = "#72e0b6"
BLUE = "#8eb6f7"
AMBER = "#efc477"
VIOLET = "#c6a6f7"
CORAL = "#ef8f8f"
CHART_COLORS = [ACCENT, BLUE, AMBER, VIOLET, CORAL]

CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

_CHART_KIND_LABELS = {
    "area": "Trend",
    "bar": "Comparison",
    "gauge": "Indicator",
    "hist": "Distribution",
    "line": "Trend",
    "pie": "Composition",
    "scatter": "Relationship",
}

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
_ABBREVIATIONS = {
    "api": "API",
    "arpu": "ARPU",
    "id": "ID",
    "kpi": "KPI",
    "mrr": "MRR",
    "nps": "NPS",
    "roi": "ROI",
}


def chart_kind_label(chart_type: str) -> str:
    """Return a user-facing diagnostic category for a chart type."""
    return _CHART_KIND_LABELS.get(chart_type, "Diagnostic")


def humanize_chart_label(value: Any) -> str:
    """Turn a data column identifier into a safe, readable axis label."""
    if value is None:
        return ""
    label = str(value).strip()
    label = re.sub(r"[_\-]+", " ", label)
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    words = [
        _ABBREVIATIONS.get(word.lower(), word.capitalize())
        for word in label.split()
    ]
    return escape(" ".join(words))


def _column_name(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _column_terms(column: Optional[str]) -> set[str]:
    if not column:
        return set()
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column)
    return {part for part in re.split(r"[^a-z0-9]+", normalized.lower()) if part}


def _numeric_values(frame: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if not column or column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def semantic_value_kind(frame: pd.DataFrame, column: Optional[str]) -> str:
    """Infer a display format from a column name and its observed values."""
    terms = _column_terms(column)
    if terms & _PERCENT_TERMS:
        values = _numeric_values(frame, column)
        if values.empty or float(values.abs().max()) <= 1.5:
            return "ratio"
        return "percent"
    if terms & _CURRENCY_TERMS:
        return "currency"
    return "number"


def _axis_format(frame: pd.DataFrame, column: Optional[str]) -> dict[str, Any]:
    if _numeric_values(frame, column).empty:
        return {}
    kind = semantic_value_kind(frame, column)
    if kind == "ratio":
        return {"tickformat": ".1%"}
    if kind == "percent":
        return {"tickformat": ".1f", "ticksuffix": "%"}
    if kind == "currency":
        return {"tickformat": "~s", "tickprefix": "$"}
    return {"tickformat": "~s", "separatethousands": True}


def _hover_value(frame: pd.DataFrame, column: Optional[str], axis: str) -> str:
    kind = semantic_value_kind(frame, column)
    if kind == "ratio":
        return f"%{{{axis}:.2%}}"
    if kind == "percent":
        return f"%{{{axis}:.1f}}%"
    if kind == "currency":
        return f"$%{{{axis}:,.2f}}"
    return f"%{{{axis}:,.2f}}"


def _display_value(frame: pd.DataFrame, column: Optional[str], value: Any) -> str:
    if not isinstance(value, Number) or not math.isfinite(float(value)):
        return str(value)
    numeric = float(value)
    kind = semantic_value_kind(frame, column)
    if kind == "ratio":
        return f"{numeric * 100:.1f}%"
    if kind == "percent":
        return f"{numeric:.1f}%"
    if kind == "currency":
        if abs(numeric) >= 1_000_000:
            return f"${numeric / 1_000_000:.1f}M"
        if abs(numeric) >= 1_000:
            return f"${numeric / 1_000:.1f}K"
        return f"${numeric:,.0f}"
    return f"{numeric:,.1f}".rstrip("0").rstrip(".")


def _bar_labels(
    frame: pd.DataFrame,
    column: Optional[str],
    values: Optional[Iterable[Any]],
) -> Optional[list[str]]:
    if values is None:
        return None
    values = list(values)
    if len(values) > 12:
        return None
    return [_display_value(frame, column, value) for value in values]


def _chart_axes(chart: Chart) -> tuple[Optional[str], Optional[str]]:
    y_column = _column_name(chart.y)
    if chart.type == "bar" and chart.group:
        return chart.group, y_column
    if chart.type == "hist":
        return chart.x, None
    return chart.x, y_column


def apply_evidence_chart_theme(
    fig: go.Figure,
    chart: Chart,
    frame: pd.DataFrame,
) -> go.Figure:
    """Apply the product chart contract to a Plotly figure."""
    x_column, y_column = _chart_axes(chart)
    x_label = humanize_chart_label(x_column)
    y_label = humanize_chart_label(y_column)
    if chart.type in {"bar", "line", "area"} and not y_column:
        y_label = "Count"
    if chart.type == "hist":
        y_label = "Count"

    fig.update_layout(
        title=None,
        autosize=True,
        height=350,
        margin={"t": 12, "r": 18, "b": 54, "l": 58},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_COLORS,
        font={"color": "#b9cbc5", "family": "DM Sans, sans-serif", "size": 12},
        hoverlabel={
            "bgcolor": "#07130f",
            "bordercolor": "#315047",
            "font": {"color": "#edf7f3", "family": "DM Sans, sans-serif"},
        },
        hovermode="x unified" if chart.type in {"line", "area"} else "closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#8fa69e", "size": 11},
        },
    )

    axis_style = {
        "automargin": True,
        "showline": True,
        "linecolor": "rgba(153, 184, 173, .24)",
        "linewidth": 1,
        "ticks": "",
        "tickfont": {"color": "#8fa69e", "size": 11},
        "title_font": {"color": "#a9bdb6", "size": 12},
        "zeroline": False,
    }
    fig.update_xaxes(
        **axis_style,
        title_text=x_label or None,
        showgrid=chart.type == "scatter",
        gridcolor="rgba(153, 184, 173, .10)",
    )
    fig.update_yaxes(
        **axis_style,
        title_text=y_label or None,
        showgrid=True,
        gridcolor="rgba(153, 184, 173, .13)",
        gridwidth=1,
    )

    if x_column:
        fig.update_xaxes(**_axis_format(frame, x_column))
    if y_column:
        fig.update_yaxes(**_axis_format(frame, y_column))
    elif chart.type in {"bar", "hist", "line", "area"}:
        fig.update_yaxes(tickformat=",.0f")

    if chart.type == "bar":
        fig.update_layout(bargap=0.28, barcornerradius=6)
        for trace in fig.data:
            trace.update(
                marker={"color": ACCENT, "line": {"width": 0}},
                opacity=0.9,
                text=_bar_labels(frame, y_column, trace.y),
                textposition="outside",
                textfont={"color": "#b9cbc5", "size": 10},
                cliponaxis=False,
                hovertemplate=(
                    f"{x_label}: %{{x}}<br>{y_label}: "
                    f"{_hover_value(frame, y_column, 'y')}<extra></extra>"
                ),
            )
    elif chart.type == "line":
        fig.update_traces(
            line={"color": ACCENT, "width": 3},
            marker={"color": "#d8fff0", "size": 5, "line": {"color": ACCENT, "width": 1}},
            hovertemplate=(
                f"{x_label}: %{{x}}<br>{y_label}: "
                f"{_hover_value(frame, y_column, 'y')}<extra></extra>"
            ),
        )
    elif chart.type == "area":
        fig.update_traces(
            line={"color": ACCENT, "width": 2.5},
            fillcolor="rgba(114,224,182,.18)",
            hovertemplate=(
                f"{x_label}: %{{x}}<br>{y_label}: "
                f"{_hover_value(frame, y_column, 'y')}<extra></extra>"
            ),
        )
    elif chart.type == "scatter":
        fig.update_traces(
            marker={
                "color": BLUE,
                "size": 7,
                "opacity": 0.72,
                "line": {"color": "rgba(216,232,226,.42)", "width": 0.7},
            },
            hovertemplate=(
                f"{x_label}: {_hover_value(frame, x_column, 'x')}<br>"
                f"{y_label}: {_hover_value(frame, y_column, 'y')}<extra></extra>"
            ),
        )
    elif chart.type == "hist":
        fig.update_layout(bargap=0.08, barcornerradius=3)
        fig.update_traces(
            marker={"color": BLUE, "line": {"color": "rgba(142,182,247,.35)", "width": 1}},
            opacity=0.88,
            hovertemplate=(
                f"{x_label}: {_hover_value(frame, x_column, 'x')}<br>"
                "Count: %{y:,.0f}<extra></extra>"
            ),
        )
    elif chart.type == "pie":
        fig.update_traces(
            hole=0.54,
            marker={"colors": CHART_COLORS, "line": {"color": "#10201c", "width": 2}},
            textinfo="percent+label",
            textfont={"color": "#edf7f3", "size": 11},
            hovertemplate="%{label}<br>%{value:,.2f} · %{percent}<extra></extra>",
        )
    elif chart.type == "gauge":
        fig.update_traces(
            number_font={"color": "#edf7f3", "family": "Space Grotesk, sans-serif"},
            gauge_bar_color=ACCENT,
            gauge_bgcolor="rgba(153,184,173,.08)",
            gauge_bordercolor="rgba(153,184,173,.20)",
            gauge_borderwidth=1,
            selector={"type": "indicator"},
        )

    return fig
