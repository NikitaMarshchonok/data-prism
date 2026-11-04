"""Synthetic rolling-origin evaluation; execution integrity is not model quality.

Outer actuals are sliced separately from the input to the production forecaster.
Simple references and errors are calculated here without its private predictors
or metric functions. Neither negative results nor failed runs may be filtered.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.forecasting import (
    LEGACY_CANDIDATES as CANDIDATES, LEGACY_CANDIDATE_SUITE, CANDIDATE_SUITE,
    CANDIDATES as EXTENDED_CANDIDATES, BASELINES as EXTENDED_BASELINES,
    ForecastTask, forecast_series,
)
from src.forecast_scoring import score_predictions, baseline_predictions


CONTRACT = "forecast-benchmark-v1"
EXTENDED_CONTRACT = "forecast-benchmark-v2"
SCENARIOS = ("constant", "weekly_exact", "noisy_weekly", "trend", "intermittent",
             "white_noise", "regime_change")
DETERMINISTIC = ("constant", "weekly_exact")
BASELINES = ("last_value", "seasonal_naive")
PERIOD = 7
MAX_RUNS = 128


def _methods(config):
    if config['contract'] == CONTRACT:
        return LEGACY_CANDIDATE_SUITE, CANDIDATES, BASELINES
    return CANDIDATE_SUITE, EXTENDED_CANDIDATES, EXTENDED_BASELINES


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_benchmark_config(config):
    fields = {"contract", "generator_version", "scenarios", "seeds",
              "initial_history_days", "horizon_days", "outer_windows", "reporting_delays"}
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ValueError("Benchmark configuration requires exactly the versioned protocol fields.")
    if config['contract'] not in (CONTRACT, EXTENDED_CONTRACT) or type(config['generator_version']) is not int or config['generator_version'] != 1:
        raise ValueError("Unsupported benchmark or generator version.")
    scenarios = config['scenarios']
    if not isinstance(scenarios, list) or scenarios != list(SCENARIOS):
        raise ValueError("v1 requires all seven scenarios in protocol order; do not remove adverse cases.")
    for name, minimum, maximum in (("initial_history_days", 60, 365), ("horizon_days", 1, 14),
                                   ("outer_windows", 1, 5)):
        value = config[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    for name, minimum, maximum, length in (("seeds", 0, 2**32 - 1, 5),
                                          ("reporting_delays", 0, 30, 3)):
        values = config[name]
        if (not isinstance(values, list) or not 1 <= len(values) <= length
                or any(type(value) is not int or not minimum <= value <= maximum for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must be a bounded list of distinct integers.")
    if config['initial_history_days'] < 30 + 4 * config['horizon_days'] + 2 * max(config['reporting_delays']):
        raise ValueError("Initial history is insufficient for all declared delayed evaluations.")
    count = (2 + 5 * len(config['seeds'])) * config['outer_windows'] * len(config['reporting_delays'])
    if count > MAX_RUNS:
        raise ValueError(f"Protocol exceeds the {MAX_RUNS}-run compute budget.")
    # Copy even nested lists so a caller cannot change the recorded protocol mid-run.
    return json.loads(json.dumps(dict(config), allow_nan=False))


def generate_series(scenario, seed, config):
    """v1 generators are fixed in source, not fitted to observed benchmark scores."""
    size = config['initial_history_days'] + max(config['reporting_delays']) + config['outer_windows'] * config['horizon_days']
    index = np.arange(size, dtype=float)
    rng = np.random.default_rng(0 if seed is None else seed)
    weekly = np.asarray([12., 18., 22., 35., 60., 45., 25.])[np.arange(size) % PERIOD]
    if scenario == 'constant':
        values = np.full(size, 40.)
    elif scenario == 'weekly_exact':
        values = weekly
    elif scenario == 'noisy_weekly':
        values = np.maximum(0., weekly + rng.normal(0., 5., size))
    elif scenario == 'trend':
        values = 30. + .6 * index + rng.normal(0., 3., size)
    elif scenario == 'intermittent':
        values = (rng.random(size) < .15) * rng.poisson(20., size)
    elif scenario == 'white_noise':
        values = 50. + rng.normal(0., 10., size)
    elif scenario == 'regime_change':
        # The first outer forecast faces an unannounced shift. Later origins may
        # learn from facts released since then; no configuration is retuned.
        boundary = config['initial_history_days'] + max(config['reporting_delays'])
        values = 50. + .3 * index + rng.normal(0., 3., size)
        values[boundary:] += 50. - 1.2 * (index[boundary:] - boundary)
    else:
        raise ValueError("Unknown synthetic scenario.")
    return pd.DataFrame({'date': pd.date_range('2020-01-01', periods=size).strftime('%Y-%m-%d'),
                         'target': np.asarray(values, dtype=float)})


def compare_mae(model_mae, baseline_mae):
    difference = model_mae - baseline_mae
    tie = math.isclose(model_mae, baseline_mae, rel_tol=1e-9, abs_tol=1e-9)
    return {'outcome': 'tie' if tie else 'better' if difference < 0 else 'worse',
            'mae_difference': difference,
            'relative_mae_improvement': 1 - model_mae / baseline_mae if baseline_mae > 0 else None,
            'relative_improvement_defined': baseline_mae > 0}


def _engine_evidence(report, task, cutoff, dates, config):
    """Reject incomplete outputs and selection inconsistent with validation MAE.

    Metadata checks do not prove that the implementation respected its cutoffs;
    separate prefix/mutation tests are still required.
    """
    if not isinstance(report, dict) or report.get('status') != 'ok':
        message = report.get('message', 'No successful engine result.') if isinstance(report, dict) else 'Malformed engine result.'
        raise ValueError(str(message))
    suite, expected_candidates, expected_baselines = _methods(config)
    if (report.get('candidate_suite') != suite or report.get('contract') != 'daily-forecast-v3'
            or report.get('future_forecast_origin') != task.forecast_origin
            or report.get('future_fit_through') != cutoff
            or report.get('availability', {}).get('history_mode') != 'available_history_only'
            or report.get('availability', {}).get('reporting_delay_days') != task.reporting_delay_days):
        raise ValueError("Engine output does not match the scheduled issue date and available cutoff.")
    predictions = report['future_forecast']
    if not isinstance(predictions, list) or [row['date'] for row in predictions] != dates:
        raise ValueError("Engine must return exactly the scheduled outer dates in order.")
    forecast = [row['prediction'] for row in predictions]
    score_predictions([0.] * len(dates), forecast)  # type/finite/length check
    rows = report['model_comparison']
    if not isinstance(rows, list) or [row['model_name'] for row in rows] != list(expected_candidates):
        raise ValueError("All configured candidates, including failures, must be reported in order.")
    candidates = []
    for row in rows:
        candidate = {'model_name': row['model_name'], 'status': row['status'], 'selected': row['selected']}
        if type(row['selected']) is not bool:
            raise ValueError("Candidate selection must be explicit.")
        if row['status'] == 'ok':
            folds = row['fold_metrics']
            if len(folds) != 3:
                raise ValueError("Candidate must complete all three validation folds.")
            scores = [fold['mae'] for fold in folds] + [row['mean_mae'], row['mean_objective_loss']]
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) or value < 0 for value in scores):
                raise ValueError("Validation errors must be finite and non-negative.")
            mean = math.fsum(fold['mae'] for fold in folds) / 3
            if not all(math.isclose(mean, value, rel_tol=1e-9, abs_tol=1e-9)
                       for value in (row['mean_mae'], row['mean_objective_loss'])):
                raise ValueError("Unweighted protocol requires mean validation MAE selection.")
            # Check the mean independently but preserve the engine's reported
            # float for ordering: fsum vs numpy summation can differ at roundoff.
            candidate.update(mean_validation_mae=float(row['mean_mae']), fold_mae=[fold['mae'] for fold in folds])
        elif row['status'] == 'error' and not row['selected']:
            candidate.update(mean_validation_mae=None, reason=str(row.get('reason', 'Candidate failed.')))
        else:
            raise ValueError("Invalid candidate status or selected failed candidate.")
        candidates.append(candidate)
    valid = [row for row in candidates if row['status'] == 'ok']
    references = [row for row in valid if row['model_name'] in expected_baselines]
    if not valid or not references:
        raise ValueError("No completed validation candidate or baseline.")
    chosen = min(valid, key=lambda row: row['mean_validation_mae'])['model_name']
    baseline = min(references, key=lambda row: row['mean_validation_mae'])['model_name']
    if (report['model_name'] != chosen or report['baseline_name'] != baseline
            or [row['model_name'] for row in candidates if row['selected']] != [chosen]):
        raise ValueError("Method and baseline must be selected by inner validation, not outer outcomes.")
    return forecast, {'model_name': chosen, 'baseline_name': baseline, 'candidates': candidates}


def evaluate_window(data, config, window_index, delay):
    h = config['horizon_days']
    target_start = config['initial_history_days'] + max(config['reporting_delays']) + window_index * h
    origin = str(data.date.iloc[target_start - 1])
    available = data.iloc[:target_start - delay][['date', 'target']].copy(deep=True)
    cutoff = str(available.date.iloc[-1])
    # Freeze reference predictions and actuals before passing a private copy to
    # the engine; even accidental engine mutation cannot change the scorer.
    dates = data.date.iloc[target_start:target_start + h].tolist()
    actual = data.target.iloc[target_start:target_start + h].astype(float).tolist()
    suite, _, _ = _methods(config)
    references = baseline_predictions(available.target.tolist(), h, delay, candidate_suite=suite)
    result = {'window_index': window_index, 'forecast_origin': origin, 'available_through': cutoff,
              'reporting_delay_days': delay, 'history_rows': len(available),
              'input_sha256': _hash(available.to_dict(orient='list')), 'dates': dates, 'actual': actual,
              'baselines': {name: {'forecast': values, 'metrics': score_predictions(actual, values)}
                            for name, values in references.items()},
              'status': 'error', 'forecast': [], 'model_metrics': None, 'comparisons': {}, 'engine': None}
    task = ForecastTask('target', 'date', h, PERIOD, delay, origin)
    try:
        report = forecast_series(available, task, candidate_suite=suite)
        forecast, evidence = _engine_evidence(report, task, cutoff, dates, config)
        metrics = score_predictions(actual, forecast)
        result.update(forecast=forecast, model_metrics=metrics, engine=evidence,
                      comparisons={name: compare_mae(metrics['mae'], base['metrics']['mae'])
                                   for name, base in result['baselines'].items()})
        result['selected_baseline_comparison'] = result['comparisons'][evidence['baseline_name']]
        failed = sum(row['status'] == 'error' for row in evidence['candidates'])
        result['candidate_failures'] = failed
        result['status'] = 'scored_with_candidate_failures' if failed else 'scored'
    except Exception as error:
        # An individual failed origin must remain in the report and fail the
        # integrity gate, not stop the report before other scheduled runs.
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    return result


def _summarize(windows):
    scored = [row for row in windows if row['model_metrics'] is not None]
    counts = Counter(row['selected_baseline_comparison']['outcome'] for row in scored)
    return {'scheduled_windows': len(windows), 'scored_windows': len(scored),
            'failed_windows': sum(row['status'] == 'error' for row in windows),
            'candidate_failures': sum(row.get('candidate_failures', 0) for row in windows),
            'vs_validation_selected_baseline': {name: counts[name] for name in ('better', 'tie', 'worse')},
            'undefined_relative_improvement_windows': sum(
                not row['selected_baseline_comparison']['relative_improvement_defined'] for row in scored)}


def run_forecast_benchmark(config):
    protocol = validate_benchmark_config(config)
    series = []
    for scenario in protocol['scenarios']:
        seeds = [None] if scenario in DETERMINISTIC else protocol['seeds']
        for seed in seeds:
            data = generate_series(scenario, seed, protocol)
            windows = [evaluate_window(data, protocol, index, delay)
                       for delay in protocol['reporting_delays'] for index in range(protocol['outer_windows'])]
            series.append({'scenario': scenario, 'seed': seed,
                           'data_sha256': _hash(data.to_dict(orient='list')),
                           'windows': windows, 'summary': _summarize(windows)})
    totals = _summarize([window for row in series for window in row['windows']])
    complete = totals['failed_windows'] == 0 and totals['candidate_failures'] == 0
    suite, candidates, baselines = _methods(protocol)
    return {'contract': protocol['contract'], 'status': 'completed' if complete else 'completed_with_failures',
            'protocol_integrity': 'passed' if complete else 'failed',
            'predictive_acceptance': 'not_established', 'protocol': protocol,
            'protocol_sha256': _hash(protocol), 'candidate_suite': suite,
            'candidates': list(candidates), 'baseline_candidates': list(baselines),
            'implementation': {
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('forecast_benchmark.py', 'forecast_scoring.py', 'forecasting.py', 'forecast_policy.py')},
                'dependencies': {'python': platform.python_version(), 'numpy': np.__version__,
                                 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}},
            'series_count': len(series), 'totals': totals, 'series': series,
            'limitations': [
                'Synthetic regression cases only, not untouched client data or professional certification.',
                'Protocol and generator version must be reviewed before examining results; do not tune them to erase losses.',
                'Outer windows do not overlap within a series/delay; seeds, delay variants and successive origins are not independent trials.',
                'Each origin reruns the existing inner-validation selector on facts available then; no outer score selects candidates or baselines.',
                'Every declared reference is scored independently; the reported reference is selected by inner validation, not the best outer error.',
                'v1 freezes the two original baselines; v2 adds historical mean/median. Neither gains on noise nor comparisons on these inspected cases prove a temporal signal or unseen performance.',
                'Later origins may use released earlier outcomes, as in an expanding-history deployment; they are not fresh unseen datasets.',
                'Counts are descriptive, not significance, confidence intervals, business value, an overall accuracy score or a production gate.',
                'Fixed delays are simulated; revisions, real release timestamps, external predictors, multiple entities and live monitoring are not evaluated.',
            ]}
