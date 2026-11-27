"""Deterministic plain-language explanations for the Data Audit report.

The evidence engines describe results in statistical vocabulary.  This module
maps each stable result identifier (readiness ``check_id``, insight
``insight_id``, decision-brief ``category``, statistical test ``family``) to a
short explanation for a business reader: what was found, why it matters and
what to do.  Numbers come only from metrics the engines already computed.

Nothing here generates new conclusions.  When an identifier is unknown, or a
metric the wording needs is missing, the explanation is ``None`` and the
report falls back to the engine's original text instead of guessing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional


Explanation = dict[str, str]

MAX_NAME_LENGTH = 60


def _name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) > MAX_NAME_LENGTH:
        text = text[: MAX_NAME_LENGTH - 1].rstrip() + "…"
    return f"“{text}”"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _count(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _percent(rate: Any) -> Optional[str]:
    number = _number(rate)
    if number is None or number < 0:
        return None
    percent = number * 100
    if 0 < percent < 0.1:
        return "less than 0.1%"
    if percent < 10:
        return f"{percent:.1f}%"
    return f"{percent:.0f}%"


def _explanation(headline: str, meaning: str, action: str) -> Explanation:
    return {"headline": headline, "meaning": meaning, "action": action}


def _strength(absolute_r: float) -> str:
    if absolute_r >= 0.9:
        return "very strong"
    if absolute_r >= 0.7:
        return "strong"
    if absolute_r >= 0.5:
        return "moderate"
    if absolute_r >= 0.3:
        return "weak"
    return "very weak"


# --------------------------------------------------------------------------
# Data readiness checks
# --------------------------------------------------------------------------

def _missingness(check: Mapping[str, Any], metrics: Mapping[str, Any]) -> Optional[Explanation]:
    share = _percent(metrics.get("missing_rate"))
    if share is None:
        return None
    empty = _count(metrics.get("empty_column_count")) or 0
    empty_note = f" {empty} column(s) are completely empty." if empty else ""
    if check.get("status") == "blocked":
        return _explanation(
            f"{share} of the table is empty, too much to analyze safely.",
            "With this much missing data, totals, averages and comparisons describe the gaps more than the business."
            + empty_note,
            "Check that the export is complete (all columns, the full period) and upload it again.",
        )
    if (_number(metrics.get("missing_rate")) or 0) < 0.01 and not empty:
        return _explanation(
            f"{share} of the values are missing.",
            "A share this small rarely changes overall numbers, but it can matter for the specific columns where the gaps are.",
            "Confirm that the gaps are expected (for example, a field that does not apply to every row).",
        )
    return _explanation(
        f"{share} of the values are missing.",
        "Missing values quietly change totals and averages, and rows with gaps may drop out of comparisons."
        + empty_note,
        "Confirm why the values are missing (not collected, not applicable, export error) before trusting affected numbers.",
    )


def _duplicates(check: Mapping[str, Any], metrics: Mapping[str, Any]) -> Optional[Explanation]:
    share = _percent(metrics.get("duplicate_rate"))
    rows = _count(metrics.get("duplicate_rows"))
    if share is None or rows is None:
        return None
    if check.get("status") == "blocked":
        return _explanation(
            f"{share} of rows are exact copies of other rows.",
            "Duplicates inflate totals and counts. At this level, most aggregate numbers will be overstated.",
            "Find out why rows repeat (double export, joins) and remove copies using a unique business key, such as an order ID.",
        )
    return _explanation(
        f"{rows} row(s) ({share}) are exact copies of other rows.",
        "If these are accidental copies, totals and counts are overstated. If they are real repeated events, they are fine.",
        "Check a few repeated rows in the source system before removing anything.",
    )


_CHECK_TEXT: dict[tuple[str, str], Explanation] = {
    ("dataset-shape", "blocked"): _explanation(
        "There is not enough data to analyze.",
        "With fewer than eight rows, any pattern is as likely to be chance as a real effect.",
        "Upload a table with a header row and more records, ideally covering a longer period.",
    ),
    ("dataset-shape", "warning"): _explanation(
        "The dataset is small.",
        "Patterns found in a small sample often disappear when more data arrives, so treat them as hints, not facts.",
        "Collect more records before making decisions that are expensive to reverse.",
    ),
    ("column-identity", "blocked"): _explanation(
        "Several columns have the same name.",
        "When two columns share a name, it is unclear which one a number comes from, so calculations can be wrong.",
        "Give every column a unique, descriptive name and upload again.",
    ),
    ("column-identity", "warning"): _explanation(
        "Some columns have no real name.",
        "Columns such as “Unnamed: 0” are usually leftover row numbers from a spreadsheet and carry no business meaning.",
        "Delete these columns or give them a meaningful name.",
    ),
    ("usable-signal", "blocked"): _explanation(
        "No column changes from row to row.",
        "If every value is the same, there is nothing to compare, so no trend or difference can be found.",
        "Upload data that includes the measures and categories you want to compare.",
    ),
    ("usable-signal", "warning"): _explanation(
        "Part of the table carries no information.",
        "Columns with the same value in every row cannot explain any difference, and with too few varying columns comparisons are limited.",
        "Remove constant columns, or add the dimensions and outcomes your question is about.",
    ),
    ("finite-numeric-values", "warning"): _explanation(
        "Some numbers are infinite.",
        "Infinite values usually come from dividing by zero in a spreadsheet formula and break averages and charts.",
        "Fix the formula in the source (for example, return blank when the divisor is zero) and upload again.",
    ),
    ("sensitive-schema", "warning"): _explanation(
        "Some column names suggest personal data.",
        "Names, emails, phone numbers or ID numbers are rarely needed for analysis, and processing them may require consent.",
        "Remove personal columns that the question does not need before sharing the data or this report.",
    ),
    ("high-cardinality-fields", "warning"): _explanation(
        "Some text columns have a different value in almost every row.",
        "Columns like IDs, names or free text cannot be used to group results, so they are excluded from comparisons.",
        "Keep them for reference, but group by a broader category (for example, region instead of address) if you need comparisons.",
    ),
}

_CHECK_BUILDERS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Optional[Explanation]]] = {
    "missingness": _missingness,
    "duplicate-rows": _duplicates,
}

#: Every (check_id, status) the readiness engine can emit for a non-passing check.
KNOWN_CHECKS = frozenset(_CHECK_TEXT) | frozenset(
    (check_id, status)
    for check_id in _CHECK_BUILDERS
    for status in ("warning", "blocked")
)


def explain_check(check: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain one non-passing readiness check, or ``None`` if unknown."""
    if not isinstance(check, Mapping):
        return None
    check_id, status = check.get("check_id"), check.get("status")
    if status not in {"warning", "blocked"}:
        return None
    metrics = check.get("metrics") if isinstance(check.get("metrics"), Mapping) else {}
    builder = _CHECK_BUILDERS.get(check_id)
    if builder is not None:
        return builder(check, metrics)
    text = _CHECK_TEXT.get((check_id, status))
    return dict(text) if text else None


# --------------------------------------------------------------------------
# Evidence insights
# --------------------------------------------------------------------------

def _quality_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    missing = _percent(metrics.get("missing_rate"))
    duplicates = _count(metrics.get("duplicate_rows"))
    if missing is None or duplicates is None:
        return None
    return _explanation(
        f"{missing} of values are missing and {duplicates} row(s) are duplicated.",
        "Small gaps rarely change the big picture, but they can distort numbers for the specific columns where they occur.",
        "Look at which columns have gaps (see Data quality) before relying on them.",
    )


def _relationship_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    x, y = _name(metrics.get("column_x")), _name(metrics.get("column_y"))
    r = _number(metrics.get("pearson_r"))
    if not x or not y or r is None:
        return None
    direction = "higher" if r > 0 else "lower"
    return _explanation(
        f"When {x} is higher, {y} is usually {direction} too.",
        f"The link is {_strength(abs(r))} (correlation {r:.2f} on a scale from -1 to 1). "
        "A link does not prove that one causes the other: both may depend on something else, such as company size or season.",
        "Before acting on it, ask whether there is a business reason for the link and check it again on next month's data.",
    )


def _trend_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    column, time = _name(metrics.get("value_column")), _name(metrics.get("time_column"))
    change = _number(metrics.get("relative_change"))
    if not column or not time or change is None:
        return None
    verb = "grew" if change > 0 else "fell"
    return _explanation(
        f"{column} {verb} by {_percent(abs(change))} over the period.",
        f"This compares the average of the earliest fifth of the timeline ({time}) with the latest fifth. "
        "It does not separate a lasting trend from seasonality or a one-off event.",
        "Compare the same months of the previous year, and check whether a campaign, price change or data change explains the shift.",
    )


def _outlier_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    column = _name(metrics.get("column"))
    count, share = _count(metrics.get("outlier_count")), _percent(metrics.get("outlier_rate"))
    if not column or count is None or share is None:
        return None
    return _explanation(
        f"{count} value(s) of {column} ({share}) are far outside the usual range.",
        "Extreme values can be data errors, but they can also be your biggest customers or rare important events. "
        "Either way, they pull averages up or down.",
        "Look at these rows in the source. Fix them if they are errors; keep them, and consider reporting medians, if they are real.",
    )


def _segment_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    column, value = _name(metrics.get("column")), _name(metrics.get("top_value"))
    share = _percent(metrics.get("top_share"))
    if not column or not value or share is None:
        return None
    return _explanation(
        f"{value} makes up {share} of {column}.",
        "When one group dominates, overall numbers mostly describe that group, and smaller groups can be hidden.",
        "Look at results for each group separately before drawing conclusions about the whole business.",
    )


def _structural_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    column = _name(metrics.get("column"))
    components = metrics.get("component_columns")
    if not column or not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        return None
    parts = [part for part in (_name(item) for item in components) if part]
    if not parts:
        return None
    return _explanation(
        f"{column} is simply the sum of {', '.join(parts)}.",
        "Because it is calculated from other columns, it is not an independent driver. "
        "Saying that its parts “explain” it would be circular.",
        "Treat it as a total. Analyze its components instead if you want to know what drives it.",
    )


def _small_sample_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    rows = _count(metrics.get("row_count"))
    minimum = _count(metrics.get("recommended_minimum"))
    if rows is None or minimum is None:
        return None
    return _explanation(
        f"Only {rows} rows were available.",
        f"Below about {minimum} rows, patterns can easily be chance, so every finding in this report is a hint, not proof.",
        "Collect more data and run the analysis again before making expensive decisions.",
    )


def _empty_insight(metrics: Mapping[str, Any]) -> Optional[Explanation]:
    return _explanation(
        "The file contains no data rows.",
        "There is nothing to analyze.",
        "Check the export and upload a file with data.",
    )


_INSIGHT_BUILDERS: dict[str, Callable[[Mapping[str, Any]], Optional[Explanation]]] = {
    "dataset-empty": _empty_insight,
    "data-quality": _quality_insight,
    "observed-additive-relationship": _structural_insight,
    "strongest-correlation": _relationship_insight,
    "strongest-trend": _trend_insight,
    "potential-outliers": _outlier_insight,
    "category-concentration": _segment_insight,
    "small-sample": _small_sample_insight,
}

#: Insight ids the report does not explain because it never shows them.
UNEXPLAINED_INSIGHT_IDS = frozenset({"dataset-overview"})
KNOWN_INSIGHT_IDS = frozenset(_INSIGHT_BUILDERS) | UNEXPLAINED_INSIGHT_IDS


def explain_insight(insight: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain one evidence insight, or ``None`` if unknown or incomplete."""
    if not isinstance(insight, Mapping):
        return None
    builder = _INSIGHT_BUILDERS.get(insight.get("insight_id"))
    if builder is None:
        return None
    metrics = insight.get("metrics") if isinstance(insight.get("metrics"), Mapping) else {}
    return builder(metrics)


# --------------------------------------------------------------------------
# Statistical validation
# --------------------------------------------------------------------------

def explain_test(test: Mapping[str, Any]) -> Optional[str]:
    """One sentence describing a statistical test result in plain words."""
    if not isinstance(test, Mapping):
        return None
    family = test.get("family")
    significant = test.get("significant") is True
    if family == "correlation":
        columns = test.get("columns")
        effect = test.get("effect_size") if isinstance(test.get("effect_size"), Mapping) else {}
        r = _number(effect.get("value"))
        if not isinstance(columns, Sequence) or len(columns) != 2 or r is None:
            return None
        x, y = _name(columns[0]), _name(columns[1])
        if not significant:
            return f"No reliable link between {x} and {y}."
        direction = "move together" if r > 0 else "move in opposite directions"
        return f"{x} and {y} {direction} ({_strength(abs(r))} link). Reliable, but not proof of cause."
    if family == "group-comparison":
        estimates = test.get("estimates") if isinstance(test.get("estimates"), Mapping) else {}
        value, group = _name(estimates.get("value_column")), _name(estimates.get("group_column"))
        a, b = _name(estimates.get("group_a")), _name(estimates.get("group_b"))
        mean_a, mean_b = _number(estimates.get("group_a_mean")), _number(estimates.get("group_b_mean"))
        if not all((value, group, a, b)) or mean_a is None or mean_b is None:
            return None
        if not significant:
            return f"No reliable difference in {value} between {a} and {b}."
        higher, lower = (a, b) if mean_a > mean_b else (b, a)
        return f"{value} is reliably higher for {higher} than for {lower} (by {group})."
    if family == "multi-group-comparison":
        return _explain_multi_group(test, significant)
    if family == "rate-comparison":
        return _explain_rate_comparison(test, significant)
    return None


KNOWN_TEST_FAMILIES = frozenset({"correlation", "group-comparison", "multi-group-comparison", "rate-comparison"})


def _small_effect_note(test: Mapping[str, Any]) -> str:
    effect = test.get("effect_size") if isinstance(test.get("effect_size"), Mapping) else {}
    if effect.get("magnitude") in {"negligible", "small"}:
        return " The difference is real but small; check whether it matters for the business."
    return ""


def _explain_multi_group(test: Mapping[str, Any], significant: bool) -> Optional[str]:
    estimates = test.get("estimates") if isinstance(test.get("estimates"), Mapping) else {}
    value, group = _name(estimates.get("value_column")), _name(estimates.get("group_column"))
    top, bottom = _name(estimates.get("highest_group")), _name(estimates.get("lowest_group"))
    unit = estimates.get("unit")
    high = _format_value(estimates.get("highest_mean"), unit)
    low = _format_value(estimates.get("lowest_mean"), unit)
    if not all((value, group, top, bottom, high, low)):
        return None
    if not significant:
        return f"No reliable difference in {value} between the {group} groups."
    sentence = f"{value} differs between {group} groups: highest for {top} (average {high}), lowest for {bottom} (average {low})."
    return sentence + _small_effect_note(test)


def _explain_rate_comparison(test: Mapping[str, Any], significant: bool) -> Optional[str]:
    estimates = test.get("estimates") if isinstance(test.get("estimates"), Mapping) else {}
    outcome, segment = _name(estimates.get("outcome_column")), _name(estimates.get("segment_column"))
    level = _name(estimates.get("positive_level"))
    top, bottom = _name(estimates.get("highest_segment")), _name(estimates.get("lowest_segment"))
    high, low = _percent(estimates.get("highest_rate")), _percent(estimates.get("lowest_rate"))
    if not all((outcome, segment, level, top, bottom, high, low)):
        return None
    if not significant:
        return f"The share of {outcome} = {level} is about the same across {segment}; no reliable difference."
    sentence = f"The share of {outcome} = {level} depends on {segment}: {high} for {top} versus {low} for {bottom}."
    return sentence + _small_effect_note(test)


def explain_statistics(
    validation: Mapping[str, Any],
    *,
    temporal_checked: bool = False,
    segments_checked: bool = False,
) -> Optional[Explanation]:
    """Explain why statistics were run or withheld, or ``None`` if unknown.

    ``temporal_checked`` states that time-aware trend tests did run, so the
    reader is pointed to them instead of being told to find such a method.
    """
    if not isinstance(validation, Mapping):
        return None
    status = validation.get("status")
    if status == "inference_withheld" and validation.get("reason") == "temporal_design_unverified":
        if segments_checked:
            return _explanation(
                "Segments were compared with a method built for data over time; other links were not tested.",
                "Values on neighbouring days or weeks depend on each other, and standard tests then report far more "
                "“discoveries” than are real. Segment differences below compare groups within the same period and "
                "allow for that dependence. Links between two numeric columns are still not tested.",
                "Treat the segment gaps below as reliable signals of where to look, not proof of cause.",
            )
        if temporal_checked:
            return _explanation(
                "Links between columns were not tested, because the data runs over time.",
                "Values on neighbouring days or weeks depend on each other, and standard tests of links between columns "
                "then report far more “discoveries” than are real. Trends over time were checked with methods built "
                "for this situation; see “Trends over time”.",
                "If you need proven drivers, compare independent groups such as customers or stores, or run a controlled test.",
            )
        return _explanation(
            "We did not test for cause-and-effect because the data runs over time.",
            "Values on neighbouring days or months depend on each other, and standard significance tests then report "
            "far more “discoveries” than are real. The descriptions of your data in this report are still accurate.",
            "If you need proven drivers, use a method built for time series, or compare independent groups such as customers or stores.",
        )
    if status == "ok":
        candidates = _count(validation.get("candidate_test_count"))
        survived = _count(validation.get("significant_test_count"))
        if candidates is None or survived is None:
            return None
        return _explanation(
            f"We checked {candidates} possible patterns; {survived} held up under strict testing.",
            "Testing many patterns at once guarantees some lucky hits. We corrected for that, so the patterns that "
            "remain are unlikely to be chance. They are still associations, not proof of cause.",
            "Use the confirmed patterns to choose what to investigate or test next, not as final answers.",
        )
    return None


# --------------------------------------------------------------------------
# Trends over time
# --------------------------------------------------------------------------

_GRAIN_WORDS = {"day": ("day", "days"), "week": ("week", "weeks"), "month": ("month", "months")}
_SHORT_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _signed_percent(rate: Any) -> Optional[str]:
    number = _number(rate)
    if number is None:
        return None
    magnitude = _percent(abs(number))
    if magnitude is None:
        return None
    if magnitude == "0.0%":
        return magnitude
    return f"{'+' if number > 0 else '−'}{magnitude}"


def _date_parts(value: Any) -> Optional[tuple[int, int, int]]:
    try:
        year, month, day = (int(part) for part in str(value).split("-"))
    except (TypeError, ValueError):
        return None
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    return year, month, day


def _day_label(parts: tuple[int, int, int]) -> str:
    return f"{parts[2]} {_SHORT_MONTHS[parts[1] - 1]} {parts[0]}"


def _period_range(grain: str, first: Any, last: Any) -> Optional[str]:
    start, end = _date_parts(first), _date_parts(last)
    if start is None or end is None:
        return None
    if grain == "month":
        return f"{_SHORT_MONTHS[start[1] - 1]} {start[0]} to {_SHORT_MONTHS[end[1] - 1]} {end[0]}"
    if grain == "week":
        return f"the week of {_day_label(start)} to the week of {_day_label(end)}"
    return f"{_day_label(start)} to {_day_label(end)}"


def explain_temporal_series(item: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain one measure's trend result, or ``None`` if a needed value is missing."""
    if not isinstance(item, Mapping):
        return None
    column = _name(item.get("column"))
    grain = item.get("grain")
    periods = _count(item.get("period_count"))
    trend = item.get("trend") if isinstance(item.get("trend"), Mapping) else {}
    direction = trend.get("direction")
    if not column or grain not in _GRAIN_WORDS or periods is None or direction not in {"increasing", "decreasing", "no_trend"}:
        return None
    singular, plural = _GRAIN_WORDS[grain]
    span = _period_range(grain, item.get("first_period"), item.get("last_period"))
    scope = f"{periods} complete {plural}" + (f", from {span}" if span else "")
    change = item.get("change_over_period") if isinstance(item.get("change_over_period"), Mapping) else {}
    relative = _signed_percent(change.get("relative_change"))
    interval = change.get("relative_change_ci")
    low = high = None
    if isinstance(interval, Sequence) and not isinstance(interval, str) and len(interval) == 2:
        low, high = _signed_percent(interval[0]), _signed_percent(interval[1])

    likely_range = f"{low} to {high}" if low and high else None
    if direction == "no_trend":
        change_value = _number(change.get("relative_change"))
        if relative and change_value is not None and abs(change_value) >= 0.05:
            moved = (
                f" The values moved by about {relative}, but the swings from {singular} to {singular} are large, "
                "so this could still be chance"
                + (f" (the likely range runs from {likely_range})." if likely_range else ".")
            )
        elif relative:
            moved = f" Overall the level stayed roughly flat (about {relative})."
        else:
            moved = ""
        return _explanation(
            f"No reliable trend in {column}.",
            f"Over {scope}, the changes in {column} are consistent with normal ups and downs.{moved}",
            f"Treat a single good or bad {singular} as noise unless the change continues for several more {plural}.",
        )

    rising = direction == "increasing"
    verb = "risen" if rising else "fallen"
    magnitude = relative[1:] if relative and relative[0] in "+−" else relative
    headline = (
        f"{column} has {verb} by about {magnitude} over the analyzed period."
        if magnitude
        else f"{column} has {verb} consistently over the analyzed period."
    )
    range_note = (
        f" Allowing for normal ups and downs, the change was most likely between {low} and {high}."
        if likely_range
        else ""
    )
    return _explanation(
        headline,
        f"Measured {singular} by {singular} over {scope}.{range_note} "
        "The test allows for neighbouring periods resembling each other, so this is unlikely to be random fluctuation. "
        "It describes the past and is not a forecast.",
        f"Break {column} down by product, channel or region to see where the change comes from, "
        "then decide whether it is a change to encourage or to fix.",
    )


def _period_label(grain: Any, value: Any) -> Optional[str]:
    parts = _date_parts(value)
    if parts is None or grain not in _GRAIN_WORDS:
        return None
    if grain == "month":
        return f"{_SHORT_MONTHS[parts[1] - 1]} {parts[0]}"
    if grain == "week":
        return f"the week of {_day_label(parts)}"
    return _day_label(parts)


def _significant(result: Any) -> bool:
    return isinstance(result, Mapping) and result.get("significant") is True


def explain_seasonality(item: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain a reliable weekly or yearly rhythm, or ``None``."""
    result = item.get("seasonality") if isinstance(item, Mapping) else None
    column = _name(item.get("column")) if isinstance(item, Mapping) else None
    if not column or not _significant(result):
        return None
    peak, trough = result.get("peak"), result.get("trough")
    if not isinstance(peak, Mapping) or not isinstance(trough, Mapping):
        return None
    high, low = _signed_percent(peak.get("relative")), _signed_percent(trough.get("relative"))
    cycles = _count(result.get("cycles_used"))
    if not high or not low or cycles is None or not peak.get("label") or not trough.get("label"):
        return None
    if result.get("cycle") == "weekday":
        return _explanation(
            f"{column} follows a weekly rhythm: highest on {peak['label']}s, lowest on {trough['label']}s.",
            f"A typical {peak['label']} runs about {high} against the weekly average and a {trough['label']} about "
            f"{low} (measured over {cycles} full weeks, after removing the overall trend).",
            "Compare each day with the same weekday of earlier weeks, not with the day before, and plan staff, "
            "stock and promotions around this rhythm.",
        )
    if result.get("cycle") == "month":
        caveat = " Two years is the minimum for this check, so confirm it with next year's data." if cycles < 3 else ""
        return _explanation(
            f"{column} follows a yearly rhythm: highest in {peak['label']}, lowest in {trough['label']}.",
            f"A typical {peak['label']} runs about {high} against the yearly average and {trough['label']} about "
            f"{low} (measured over {cycles} years, after removing the overall trend).{caveat}",
            "Compare each month with the same month of the previous year, and plan budgets and stock for the "
            "seasonal peak and dip.",
        )
    return None


def explain_level_shift(item: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain a reliable abrupt change in level, or ``None``."""
    result = item.get("level_shift") if isinstance(item, Mapping) else None
    column = _name(item.get("column")) if isinstance(item, Mapping) else None
    if not column or not _significant(result):
        return None
    when = _period_label(item.get("grain"), result.get("change_period"))
    shift = _number(result.get("relative_shift"))
    if when is None or shift is None:
        return None
    direction = "up" if shift > 0 else "down"
    return _explanation(
        f"{column} shifted {direction} by about {_percent(abs(shift))} around {when}.",
        "Allowing for the overall trend, the level changed abruptly at that point instead of gradually. Abrupt "
        "shifts usually have a specific cause, such as a price change, a lost or won client, a new competitor, "
        "or a change in how the data is recorded.",
        "Find out what changed around that date. If it was a data or tracking change, compare only the periods after it.",
    )


def explain_last_period(item: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain an unusual latest period, or ``None`` when it is normal or unknown."""
    result = item.get("latest_period") if isinstance(item, Mapping) else None
    column = _name(item.get("column")) if isinstance(item, Mapping) else None
    grain = item.get("grain") if isinstance(item, Mapping) else None
    if not column or grain not in _GRAIN_WORDS or not _significant(result):
        return None
    singular, plural = _GRAIN_WORDS[grain]
    when = _period_label(grain, result.get("period"))
    score = _number(result.get("robust_z"))
    if when is None or score is None:
        return None
    above = score > 0
    deviation = _number(result.get("relative_deviation"))
    size = f"about {_percent(abs(deviation))} {'above' if above else 'below'}" if deviation is not None else (
        "well above" if above else "well below"
    )
    cause = (
        "a duplicate import or a one-off large order can cause a sudden spike"
        if above
        else "late or missing records are the most common cause of a sudden drop"
    )
    return _explanation(
        f"The latest {singular} of {column} ({when}) was {size} what recent {plural} suggested.",
        f"A gap this large is rare for this measure, which normally moves much less from one {singular} to the next. "
        f"Before reacting, check that the {singular} is complete: {cause}.",
        "Confirm the numbers in the source system, then look for the cause (campaign, outage, holiday, large order).",
    )


def explain_temporal_patterns(item: Mapping[str, Any]) -> list[Explanation]:
    """Explanations for every reliable pattern found in one measure, most urgent first."""
    explanations = (explain_last_period(item), explain_level_shift(item), explain_seasonality(item))
    return [explanation for explanation in explanations if explanation]


def explain_temporal(analysis: Mapping[str, Any]) -> Optional[Explanation]:
    """Summarize the time-aware analysis, or ``None`` when it does not apply."""
    if not isinstance(analysis, Mapping):
        return None
    status = analysis.get("status")
    if status == "insufficient_data":
        return _explanation(
            "There is not enough history yet to judge trends.",
            "A trend needs at least 12 complete days, weeks or months with data. Shorter or patchy histories "
            "cannot separate a real trend from normal ups and downs.",
            "Upload a longer period, or come back when more history has accumulated.",
        )
    if status != "ok":
        return None
    series = analysis.get("series") if isinstance(analysis.get("series"), Sequence) else ()
    checked = len(series)
    reliable = sum(
        1 for item in series if isinstance(item, Mapping) and isinstance(item.get("trend"), Mapping)
        and item["trend"].get("significant") is True
    )
    if checked == 0:
        return None
    measures = "1 measure" if checked == 1 else f"{checked} measures"
    if reliable == 0:
        result = "none shows a reliable trend"
    elif reliable == 1:
        result = "1 shows a reliable trend"
    else:
        result = f"{reliable} show a reliable trend"
    headline = f"We checked {measures} over time; {result}."
    return _explanation(
        headline,
        "Neighbouring days and weeks tend to resemble each other, which makes ordinary methods see trends that are "
        "not there. These checks allow for that and for testing several measures at once. Each measure was also "
        "checked for a weekly or yearly rhythm, a sudden shift in level, and an unusual latest period.",
        "Focus on the measures with a reliable trend; for the rest, recent ups and downs are most likely noise.",
    )


# --------------------------------------------------------------------------
# Why a KPI changed
# --------------------------------------------------------------------------

_WINDOW_SHORT = {
    "day": "last 7 days vs the 7 before",
    "week": "last 4 weeks vs the 4 before",
    "month": "last month vs the same month a year earlier",
}


def _format_value(value: Any, unit: Any) -> Optional[str]:
    number = _number(value)
    if number is None:
        return None
    if unit == "ratio":
        return f"{number * 100:.1f}%"
    if unit == "percent":
        return f"{number:.1f}%"
    magnitude = abs(number)
    if magnitude >= 1000:
        return f"{number:,.0f}"
    if magnitude >= 10:
        return f"{number:,.1f}"
    return f"{number:.3g}"


def _format_change(value: Any, unit: Any) -> Optional[str]:
    number = _number(value)
    if number is None:
        return None
    if unit == "ratio":
        return f"{'+' if number >= 0 else '−'}{abs(number) * 100:.1f} points"
    if unit == "percent":
        return f"{'+' if number >= 0 else '−'}{abs(number):.1f} points"
    formatted = _format_value(abs(number), unit)
    return f"{'+' if number >= 0 else '−'}{formatted}" if formatted else None


def _unusual_sentence(measure: Mapping[str, Any]) -> str:
    unusual = measure.get("unusual")
    if not isinstance(unusual, Mapping):
        return "There is not enough history yet to say whether a swing of this size is unusual."
    if unusual.get("significant") is True:
        return "This is larger than the usual swing between comparable periods."
    return "Swings of this size are normal for this measure, so it may reverse on its own."


def explain_driver(measure: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain the latest change of one measure and where it came from, or ``None``."""
    if not isinstance(measure, Mapping):
        return None
    column = _name(measure.get("column"))
    window = _WINDOW_SHORT.get(measure.get("grain"))
    unit = measure.get("unit")
    if not column or not window:
        return None
    relative = _number(measure.get("relative_change"))
    change = _number(measure.get("change"))
    if change is None:
        return None
    before, after = _format_value(measure.get("before"), unit), _format_value(measure.get("after"), unit)
    points = unit in {"ratio", "percent"}
    level = ""
    if before and after:
        level = f", from an average of {before} to {after}" if measure.get("kind") == "average" else f", from {before} to {after}"
    dimensions = [entry for entry in measure.get("dimensions") or [] if isinstance(entry, Mapping)]
    top = dimensions[0] if dimensions else None

    sized = relative is not None and abs(relative) >= 0.05 or (
        relative is None and top is not None and (top.get("driver") or top.get("mix_story"))
    )
    if not sized:
        shown = _signed_percent(relative) if relative is not None else _format_change(change, unit)
        headline = f"{column} was roughly flat ({shown}, {window})."
        offsetting = next(
            (entry for entry in dimensions if len(entry.get("offsetting") or []) >= 2
             and entry["offsetting"][0]["contribution"] * entry["offsetting"][1]["contribution"] < 0),
            None,
        )
        if offsetting is None:
            return _explanation(headline, "No segment moved enough to matter.", "Nothing to act on for this measure.")
        dimension = _name(offsetting.get("dimension"))
        first, second = offsetting["offsetting"][0], offsetting["offsetting"][1]
        first_name, second_name = _name(first.get("segment")), _name(second.get("segment"))
        first_change, second_change = _format_change(first.get("contribution"), unit), _format_change(second.get("contribution"), unit)
        if not all((dimension, first_name, second_name, first_change, second_change)):
            return _explanation(headline, "No segment moved enough to matter.", "Nothing to act on for this measure.")
        return _explanation(
            headline,
            f"The flat total hides opposite moves in {dimension}: {first_name} changed by {first_change} and "
            f"{second_name} by {second_change}, which cancel out.",
            f"Look at {first_name} ({dimension}) on its own; the overall total hides what happened there.",
        )

    verb = "rose" if change > 0 else "fell"
    if points and before and after:
        # Rates move in percentage points; "fell by 22%" of a 5% rate confuses.
        headline = f"{column} {verb} from an average of {before} to {after} ({window})."
    else:
        size = _percent(abs(relative)) if relative is not None else _format_change(change, unit)
        headline = f"{column} {verb} by {size}{level} ({window})."
    unusual = _unusual_sentence(measure)
    if top is None:
        return _explanation(headline, unusual, "Check whether prices, volumes or the data feed changed in this period.")
    dimension = _name(top.get("dimension"))
    if top.get("mix_story") and isinstance(top.get("mix_driver"), Mapping):
        segment = top["mix_driver"]
        name = _name(segment.get("segment"))
        share_before, share_after = _percent(segment.get("share_before")), _percent(segment.get("share_after"))
        level_word = "below" if (_number(segment.get("before")) or 0) < (_number(measure.get("before")) or 0) else "above"
        if dimension and name and share_before and share_after:
            return _explanation(
                headline,
                f"This is mostly a mix effect: rows where {dimension} is {name}, which sit {level_word} the average, "
                f"went from {share_before} to {share_after} of the total. Within each {dimension}, {column} itself changed "
                f"much less. {unusual}",
                f"Track {column} per {dimension} as well as overall; the overall average can move even when no "
                "segment changed.",
            )
    driver = top.get("driver")
    if isinstance(driver, Mapping):
        name = _name(driver.get("segment"))
        share = _number(driver.get("share_of_change"))
        base = _percent(driver.get("base_share"))
        if dimension and name and share is not None and base:
            portion = "more than the whole change" if share > 1 else f"{_percent(share)} of the change"
            word = "drop" if change < 0 else "increase"
            own = _signed_percent(driver.get("relative_change")) if driver.get("relative_change") is not None else None
            if measure.get("kind") == "average":
                moved = _format_value(driver.get("before"), unit), _format_value(driver.get("after"), unit)
                own_text = f" ({column} there went from {moved[0]} to {moved[1]})" if all(moved) else ""
            else:
                own_text = f" (it changed by {own} on its own)" if own else ""
            return _explanation(
                headline,
                f"Most of the {word} comes from {name} ({dimension}): {portion}, although it makes up only {base} "
                f"of the total{own_text}. {unusual}",
                f"Start with {name} ({dimension}): what changed there in this period (prices, staff, campaigns, stock, "
                "data feed)?",
            )
    if dimension:
        return _explanation(
            headline,
            f"The change is spread across {dimension} roughly in proportion to size; no single segment stands out. {unusual}",
            "Look for a business-wide cause, such as pricing, the season or the market, rather than one segment.",
        )
    return _explanation(headline, unusual, "Check whether prices, volumes or the data feed changed in this period.")


def explain_drivers(analysis: Mapping[str, Any]) -> Optional[Explanation]:
    """Summarize the driver analysis, or ``None`` when it does not apply."""
    if not isinstance(analysis, Mapping) or analysis.get("status") != "ok":
        return None
    measures = [item for item in analysis.get("measures") or [] if isinstance(item, Mapping)]
    if not measures:
        return None
    flagged = sum(1 for item in measures if item.get("material") is True)
    count = "1 key measure" if len(measures) == 1 else f"{len(measures)} key measures"
    result = (
        "none changed more than usual" if flagged == 0
        else "1 changed more than usual" if flagged == 1
        else f"{flagged} changed more than usual"
    )
    return _explanation(
        f"We compared the latest period with a comparable earlier one for {count}; {result}.",
        "Each change is split exactly across the categories in your data, so you can see which segment it came "
        "from. This shows where a change happened, not why.",
        "Start any investigation with the segment named for each measure.",
    )


# --------------------------------------------------------------------------
# Segments compared over time
# --------------------------------------------------------------------------

def explain_segment_gap(test: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain one time-aware segment comparison, or ``None`` if incomplete."""
    if not isinstance(test, Mapping):
        return None
    measure, segment = _name(test.get("measure")), _name(test.get("segment_column"))
    top, bottom = _name(test.get("highest_segment")), _name(test.get("lowest_segment"))
    unit = test.get("unit")
    grain = _GRAIN_WORDS.get(test.get("grain"))
    periods = _count(test.get("periods_compared"))
    high, low = _format_value(test.get("highest_average"), unit), _format_value(test.get("lowest_average"), unit)
    if not all((measure, segment, top, bottom, grain, high, low)) or periods is None:
        return None
    rate = test.get("kind") == "rate"
    level = _name(test.get("positive_level")) if rate else None
    subject = f"The share of {measure} = {level}" if rate and level else measure
    span = f"{periods} {grain[1] if periods != 1 else grain[0]}"
    if test.get("significant") is not True:
        return _explanation(
            f"{subject} does not differ reliably between {segment} groups.",
            f"We compared the {segment} groups side by side in {span}. The gaps we saw are within what changes "
            "from period to period would produce by chance.",
            f"Do not plan around a difference between {segment} groups for {measure}.",
        )
    consistency = _number(test.get("consistency"))
    share = f", and it was ahead in {_percent(consistency)} of them" if consistency is not None else ""
    small = test.get("magnitude") in {"negligible", "small"}
    return _explanation(
        f"{subject} is reliably higher for {top} ({high} on average) than for {bottom} ({low}).",
        f"We compared the {segment} groups within the same {grain[0]} over {span}{share}. Comparing within each "
        f"{grain[0]} removes shared ups and downs such as seasons and campaigns, and the test allows for one "
        f"{grain[0]} depending on the previous one."
        + (" The gap is real but small." if small else ""),
        f"Find out what {top} does differently, such as its customers, prices or product mix, before acting on it.",
    )


def explain_segments(analysis: Mapping[str, Any]) -> Optional[Explanation]:
    """Summarize the time-aware segment comparisons, or ``None`` when they did not run."""
    if not isinstance(analysis, Mapping) or analysis.get("status") != "ok":
        return None
    tests = [test for test in analysis.get("tests") or [] if isinstance(test, Mapping)]
    if not tests:
        return None
    reliable = sum(1 for test in tests if test.get("significant") is True)
    found = "none differ reliably" if reliable == 0 else "1 differs reliably" if reliable == 1 else f"{reliable} differ reliably"
    return _explanation(
        f"We compared segments over time in {len(tests)} measure-and-segment pairs; {found}.",
        "Each segment is compared with the others in the same period, which cancels shared seasons, trends and "
        "campaigns, and the test allows for neighbouring periods depending on each other.",
        "Use reliable gaps to decide where to look next; they show where results differ, not why.",
    )


# --------------------------------------------------------------------------
# Decision-brief priorities
# --------------------------------------------------------------------------

def _anomaly_priority(dashboard: Mapping[str, Any]) -> Optional[Explanation]:
    patterns = dashboard.get("pattern_analysis") if isinstance(dashboard.get("pattern_analysis"), Mapping) else {}
    anomaly = patterns.get("anomaly_detection") if isinstance(patterns.get("anomaly_detection"), Mapping) else {}
    flagged, evaluated = _count(anomaly.get("flagged_count")), _count(anomaly.get("evaluated_rows"))
    if flagged is None or evaluated is None:
        return None
    return _explanation(
        f"{flagged} of {evaluated} rows look unusual compared with the rest.",
        "Unusual does not mean wrong. These rows can be data errors, rare events or your most important customers, "
        "and they are the fastest place to find problems or opportunities.",
        "Review the flagged rows with someone who knows the business and the source system.",
    )


def _signal_priority(priority: Mapping[str, Any], dashboard: Mapping[str, Any]) -> Optional[Explanation]:
    validation = dashboard.get("statistical_validation")
    tests = validation.get("tests") if isinstance(validation, Mapping) else None
    title = priority.get("title")
    for test in tests if isinstance(tests, Sequence) else ():
        if isinstance(test, Mapping) and test.get("title") == title and test.get("significant") is True:
            sentence = explain_test(test)
            if sentence:
                return _explanation(
                    sentence,
                    "This pattern passed strict statistical checks, so it is unlikely to be chance. "
                    "It does not show what causes what.",
                    "Find the business reason behind it and confirm it on a new period before changing plans.",
                )
    return None


def _temporal_item(priority: Mapping[str, Any], dashboard: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    temporal = dashboard.get("temporal_analysis")
    series = temporal.get("series") if isinstance(temporal, Mapping) else None
    columns = priority.get("columns")
    if not isinstance(columns, Sequence) or isinstance(columns, str) or len(columns) != 1:
        return None
    for item in series if isinstance(series, Sequence) else ():
        if isinstance(item, Mapping) and item.get("column") == columns[0]:
            return item
    return None


def _trend_priority(priority: Mapping[str, Any], dashboard: Mapping[str, Any]) -> Optional[Explanation]:
    item = _temporal_item(priority, dashboard)
    if item is None or not _significant(item.get("trend")):
        return None
    return explain_temporal_series(item)


def _readiness_priority(priority: Mapping[str, Any], dashboard: Mapping[str, Any]) -> Optional[Explanation]:
    readiness = dashboard.get("readiness")
    checks = readiness.get("checks") if isinstance(readiness, Mapping) else None
    for check in checks if isinstance(checks, Sequence) else ():
        if (
            isinstance(check, Mapping)
            and check.get("status") in {"warning", "blocked"}
            and check.get("title") == priority.get("title")
        ):
            return explain_check(check)
    return None


def _insight_priority(priority: Mapping[str, Any], dashboard: Mapping[str, Any]) -> Optional[Explanation]:
    category = str(priority.get("category", ""))[len("evidence-"):]
    insights = dashboard.get("insights")
    for insight in insights if isinstance(insights, Sequence) else ():
        if (
            isinstance(insight, Mapping)
            and insight.get("category") == category
            and insight.get("title") == priority.get("title")
        ):
            return explain_insight(insight)
    return None


#: Decision-brief categories with a dedicated explanation.  ``evidence-*``
#: categories reuse the matching insight explanation.
KNOWN_PRIORITY_CATEGORIES = frozenset(
    {
        "assumption-review",
        "validated-signal",
        "validated-trend",
        "level-shift",
        "last-period-alert",
        "change-driver",
        "segment-gap",
        "anomaly-review",
        "data-readiness",
    }
)


def temporal_checked(dashboard: Mapping[str, Any]) -> bool:
    """True when time-aware trend tests ran on at least one measure."""
    temporal = dashboard.get("temporal_analysis") if isinstance(dashboard, Mapping) else None
    return isinstance(temporal, Mapping) and temporal.get("status") == "ok" and bool(temporal.get("series"))


def segments_checked(dashboard: Mapping[str, Any]) -> bool:
    """True when time-aware segment comparisons ran on at least one pair."""
    segments = dashboard.get("segment_analysis") if isinstance(dashboard, Mapping) else None
    return isinstance(segments, Mapping) and segments.get("status") == "ok" and bool(segments.get("tests"))


def explain_priority(priority: Mapping[str, Any], dashboard: Mapping[str, Any]) -> Optional[Explanation]:
    """Explain one decision-brief priority, or ``None`` if it cannot be matched."""
    if not isinstance(priority, Mapping) or not isinstance(dashboard, Mapping):
        return None
    category = priority.get("category")
    if category == "assumption-review":
        validation = dashboard.get("statistical_validation")
        return explain_statistics(
            validation,
            temporal_checked=temporal_checked(dashboard),
            segments_checked=segments_checked(dashboard),
        ) if isinstance(validation, Mapping) else None
    if category == "validated-signal":
        return _signal_priority(priority, dashboard)
    if category == "validated-trend":
        return _trend_priority(priority, dashboard)
    if category == "level-shift":
        item = _temporal_item(priority, dashboard)
        return explain_level_shift(item) if item is not None else None
    if category == "change-driver":
        drivers = dashboard.get("driver_analysis")
        columns = priority.get("columns")
        for measure in (drivers.get("measures") or []) if isinstance(drivers, Mapping) else ():
            if isinstance(measure, Mapping) and isinstance(columns, Sequence) and [measure.get("column")] == list(columns):
                return explain_driver(measure)
        return None
    if category == "segment-gap":
        segments = dashboard.get("segment_analysis")
        columns = priority.get("columns")
        for test in (segments.get("tests") or []) if isinstance(segments, Mapping) else ():
            if (isinstance(test, Mapping) and isinstance(columns, Sequence)
                    and [test.get("measure"), test.get("segment_column")] == list(columns)):
                return explain_segment_gap(test)
        return None
    if category == "last-period-alert":
        item = _temporal_item(priority, dashboard)
        return explain_last_period(item) if item is not None else None
    if category == "anomaly-review":
        return _anomaly_priority(dashboard)
    if category == "data-readiness":
        return _readiness_priority(priority, dashboard)
    if isinstance(category, str) and category.startswith("evidence-"):
        return _insight_priority(priority, dashboard)
    return None
