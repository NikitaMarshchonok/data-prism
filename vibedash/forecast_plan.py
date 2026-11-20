"""Input eligibility and date plan for a supported daily evaluation, no fitting."""
import pandas as pd

from src.forecasting import forecast_history_requirements
from .forecast_tasks import forecast_task_from_form, ForecastTaskError


CONTRACT = 'daily-forecast-plan-v1'
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_COLUMNS = 512


def review_forecast_plan(frame, semantics, form, *, preview_row_limit):
    task = forecast_task_from_form(frame, semantics, form)
    if task is None:
        raise ForecastTaskError('Choose the daily forecast workflow to review a forecast plan.')
    if len(frame) > preview_row_limit:
        raise ForecastTaskError('Forecast history cannot be truncated to the dashboard preview limit.')
    settings = task['settings']
    requirements = forecast_history_requirements(settings)
    h, delay = settings['horizon'], settings.get('reporting_delay_days', 0)
    dates = pd.to_datetime(frame[settings['time_column']]).sort_values().dt.strftime('%Y-%m-%d').tolist()
    explicit = 'forecast_origin' in settings
    end = len(frame) if explicit else len(frame) - delay
    origin = settings.get('forecast_origin', dates[-1])
    future = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=h).strftime('%Y-%m-%d').tolist()
    return {'contract': CONTRACT, 'status': 'eligible_for_evaluation', 'training_started': False,
            'provided_rows': len(frame), 'minimum_evaluation_rows': requirements['main_evaluation'],
            'minimum_historical_rows': requirements['historical_backtest'],
            'historical_status': ('eligible' if len(frame) >= requirements['historical_backtest']
                                  else 'insufficient_history'),
            'horizon_days': h, 'seasonal_period_days': settings['seasonal_period'],
            'reporting_delay_days': delay, 'validation_windows': 3,
            'history_mode': 'available_history_only' if explicit else 'retrospective_complete_series',
            'forecast_origin': origin, 'available_targets_through': dates[end - 1],
            'holdout_origin': dates[end - h - 1], 'holdout_training_through': dates[end - h - delay - 1],
            'holdout_start': dates[end - h], 'holdout_end': dates[end - 1],
            'forecast_start': future[0], 'forecast_end': future[-1],
            'selection_objective': 'weighted_absolute_error' if task.get('decision_policy') else 'mae',
            'unused_column_count': len(frame.columns) - 2,
            'production_readiness': 'not_established'}
