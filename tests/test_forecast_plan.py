"""Independent calendar and admission checks for the no-fit forecast review."""
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from werkzeug.datastructures import MultiDict

import web_app
from src.forecasting import ForecastTask, forecast_history_requirements
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests.test_forecast_policy import POLICY_FORM
from tests import test_vibedash_job_routes as job_fixture
from vibedash.forecast_plan import review_forecast_plan


class ForecastPlanTests(unittest.TestCase):
    def plan(self, frame=None, **changes):
        frame = daily_frame() if frame is None else frame
        return review_forecast_plan(frame, confirmed(frame), FORM | changes, preview_row_limit=10000)

    @patch('src.forecasting._forecast', side_effect=AssertionError('Review must never fit'))
    def test_calendar_is_sorted_and_matches_independent_dates(self, fit):
        plan = self.plan(daily_frame().iloc[::-1])
        self.assertEqual(plan['forecast_origin'], '2020-06-08')
        self.assertEqual(plan['available_targets_through'], '2020-06-08')
        self.assertEqual((plan['holdout_start'], plan['holdout_end']), ('2020-06-02', '2020-06-08'))
        self.assertEqual(plan['holdout_training_through'], '2020-06-01')
        self.assertEqual((plan['forecast_start'], plan['forecast_end']), ('2020-06-09', '2020-06-15'))
        self.assertFalse(plan['training_started'])
        fit.assert_not_called()

    def test_reporting_delay_and_explicit_issue_date_change_calendar(self):
        historical = self.plan(forecast_reporting_delay='2')
        available = self.plan(forecast_reporting_delay='2', forecast_origin='2020-06-10')
        self.assertEqual(historical['available_targets_through'], '2020-06-06')
        self.assertEqual(historical['holdout_training_through'], '2020-05-28')
        self.assertEqual(available['available_targets_through'], '2020-06-08')
        self.assertEqual(available['holdout_training_through'], '2020-05-30')
        self.assertEqual(available['forecast_start'], '2020-06-11')
        self.assertEqual(available['history_mode'], 'available_history_only')
        self.assertEqual((historical['minimum_evaluation_rows'], historical['minimum_historical_rows']), (64, 87))
        self.assertEqual((available['minimum_evaluation_rows'], available['minimum_historical_rows']), (62, 85))

    def test_history_boundaries_include_nonweekly_period_and_long_delay(self):
        for h, p, d, origin, expected in [(7, 7, 0, None, (58, 79)), (3, 3, 10, None, (72, 91)),
                                        (3, 3, 10, '2020-06-18', (62, 81)), (5, 14, 0, None, (76, 91))]:
            with self.subTest(h=h, p=p, d=d, origin=origin):
                rows = forecast_history_requirements(ForecastTask('cnt', 'dteday', h, p, d, origin))
                self.assertEqual((rows['main_evaluation'], rows['historical_backtest']), expected)
        for count, status in [(58, 'insufficient_history'), (78, 'insufficient_history'), (79, 'eligible')]:
            self.assertEqual(self.plan(daily_frame().iloc[:count])['historical_status'], status)
        with self.assertRaisesRegex(ValueError, 'at least 58'):
            self.plan(daily_frame().iloc[:57])

    @patch('src.forecasting._forecast', side_effect=AssertionError('Review must never fit'))
    def test_invalid_history_and_unconfirmed_settings_are_rejected(self, fit):
        invalid = [daily_frame().drop(index=30), daily_frame().assign(cnt=float('inf')),
                   daily_frame().assign(cnt=-1), daily_frame().assign(dteday='2020-01-01')]
        missing = daily_frame().astype({'cnt': 'float64'})
        missing.loc[20, 'cnt'] = float('nan')
        invalid.append(missing)
        for index, frame in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.plan(frame)
        for changes in ({'forecast_available': ''}, {'forecast_horizon': '0'}, {'forecast_reporting_delay': '31'},
                        {'forecast_origin': '2020-06-09', 'forecast_reporting_delay': '2'},
                        {'forecast_origin': '2020-06-11', 'forecast_reporting_delay': '2'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)
        fit.assert_not_called()


class ForecastPlanRouteTests(unittest.TestCase):
    setUp = job_fixture.VibeDashJobRouteTests.setUp
    tearDown = job_fixture.VibeDashJobRouteTests.tearDown

    def fields(self, client, frame=None, **changes):
        frame = daily_frame() if frame is None else frame
        client.get('/vibedash/')
        with client.session_transaction() as browser:
            token = browser['vibedash_decision_csrf_token']
        return FORM | POLICY_FORM | {'task_type': 'daily_forecast', 'prompt': 'Evaluate daily cnt forecast',
            'csrf_token': token, 'semantic_contract': json.dumps(confirmed(frame)),
            'datafile': (BytesIO(frame.to_csv(index=False).encode()), 'private-customer.csv')} | changes

    def post(self, client, fields):
        response = client.post('/vibedash/forecast-plan', data=fields)
        self.assertEqual(response.headers.get('Cache-Control'), 'no-store')
        return response

    @patch('src.forecasting._forecast', side_effect=AssertionError('No fit'))
    @patch('vibedash.routes.analysis_job_dispatcher.submit', side_effect=AssertionError('No job'))
    def test_success_is_aggregate_only_and_leaves_no_upload_or_job(self, submit, fit):
        with web_app.app.test_client() as client:
            fields = self.fields(client)
            with patch('vibedash.analysis_jobs.AnalysisJobStore.create', side_effect=AssertionError('No job creation')):
                response = self.post(client, fields)
            self.assertEqual(response.status_code, 200, response.get_json())
            plan = response.get_json()
            self.assertEqual(plan['status'], 'eligible_for_evaluation')
            self.assertEqual(plan['selection_objective'], 'weighted_absolute_error')
            self.assertEqual(plan['provided_rows'], 160)
            self.assertEqual(plan['production_readiness'], 'not_established')
            for private in ('private-customer', 'staffing', 'casual', 'cnt', 'predictions'):
                self.assertNotIn(private, response.get_data(as_text=True))
            self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])
        fit.assert_not_called()
        submit.assert_not_called()

    def test_routing_and_csrf_are_checked_before_csv_read(self):
        with web_app.app.test_client() as client, patch('vibedash.routes._load_vibedash_table') as read:
            for changes, code in [({'csrf_token': ''}, 400), ({'task_type': 'classification'}, 422),
                                  ({'task_type': 'descriptive_analysis', 'analysis_mode': 'evidence'}, 422),
                                  ({'demo_dataset': 'saas_growth'}, 422)]:
                self.assertEqual(self.post(client, self.fields(client, **changes)).status_code, code)
            read.assert_not_called()

    def test_policy_semantics_and_preview_limit_must_be_confirmed(self):
        with web_app.app.test_client() as client:
            for changes in ({'forecast_policy_confirmed': ''}, {'forecast_available': ''},
                            {'semantic_contract': '{}'}, {'forecast_horizon': '91'}):
                self.assertEqual(self.post(client, self.fields(client, **changes)).status_code, 422)
            with patch.dict(web_app.app.config, MAX_ROWS_PREVIEW=100):
                self.assertEqual(self.post(client, self.fields(client)).status_code, 422)

    def test_duplicate_fields_and_files_are_rejected(self):
        with web_app.app.test_client() as client:
            fields = MultiDict(list(self.fields(client).items()))
            fields.add('forecast_horizon', '8')
            self.assertEqual(self.post(client, fields).status_code, 422)
            fields = MultiDict(list(self.fields(client).items()))
            fields.add('datafile', (BytesIO(b'day,value\n'), 'second.csv'))
            self.assertEqual(self.post(client, fields).status_code, 400)

    def test_parser_refuses_corruption_and_column_row_byte_overflow(self):
        with web_app.app.test_client() as client:
            for raw in (b'\xff\x00\xfe', (','.join(f'c{i}' for i in range(513)) + '\n').encode()):
                self.assertEqual(self.post(client, self.fields(client, datafile=(BytesIO(raw), 'bad.csv'))).status_code, 422)
            frame = pd.DataFrame({'dteday': pd.date_range('2000-01-01', periods=5001).strftime('%Y-%m-%d'),
                                  'cnt': range(5001)})
            self.assertEqual(self.post(client, self.fields(client, frame=frame)).status_code, 422)
            with patch('vibedash.routes.FORECAST_PLAN_MAX_BYTES', 500):
                self.assertEqual(self.post(client, self.fields(client)).status_code, 413)

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_job_rechecks_changed_history_after_successful_plan(self, submit):
        with web_app.app.test_client() as client:
            self.assertEqual(self.post(client, self.fields(client)).status_code, 200)
            response = client.post('/vibedash/jobs', data=self.fields(client, frame=daily_frame().drop(index=25)))
            self.assertEqual(response.status_code, 422)
        submit.assert_not_called()
