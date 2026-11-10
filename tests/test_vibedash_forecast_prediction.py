"""Frozen web inference and explicitly retained forecast outcomes."""
from datetime import datetime, timezone
from io import BytesIO
import json
import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd
from werkzeug.datastructures import FileStorage

import web_app
from src.forecast_artifact import ForecastHistoryError, predict_artifact
from tests import test_vibedash_forecast_model_web as web_fixture
from tests.test_forecast_tasks import daily_frame
from vibedash.forecast_models import ForecastModelNotFound
from vibedash.forecast_outcomes import MAX_ACTUALS_BYTES
from vibedash.forecast_prediction import MAX_HISTORY_BYTES, ForecastHistoryTooLarge, load_prediction_history


class PredictionHistoryParserTests(unittest.TestCase):
    def test_utf8_bom_and_quoted_columns_preserve_csv_semantics(self):
        raw = '\ufeffdate,"target, units"\r\n2020-01-01,1.25\r\n'.encode()
        frame = load_prediction_history(FileStorage(stream=BytesIO(raw), filename='history.CSV'))
        self.assertEqual(list(frame.columns), ['date', 'target, units'])
        self.assertEqual(frame.iloc[0, 1], 1.25)

    def test_rejects_ambiguous_headers_and_rows_before_pandas(self):
        for raw in (b'date,target,target\n2020-01-01,1,2\n',
                    b'date,target, target \n2020-01-01,1,2\n',
                    b'date,\n2020-01-01,1\n',
                    b'date,target\n2020-01-01,1,2\n',
                    b'date,target\n2020-01-01\n',
                    b'date,target\n2020-01-01,"1\n',
                    b'date,target\n\x00,1\n', b'\xff', b'date,target\n'):
            with self.subTest(raw=raw), self.assertRaises(ForecastHistoryError):
                load_prediction_history(FileStorage(stream=BytesIO(raw), filename='history.csv'))

    def test_byte_row_and_column_limits_are_not_silent_truncation(self):
        with self.assertRaises(ForecastHistoryTooLarge):
            load_prediction_history(FileStorage(stream=BytesIO(b'x' * (MAX_HISTORY_BYTES + 1)), filename='x.csv'))
        for raw in (b'date,target\n' + b'2020-01-01,1\n' * 5001,
                    ','.join(f'column{i}' for i in range(65)).encode()):
            with self.assertRaises(ForecastHistoryError):
                load_prediction_history(FileStorage(stream=BytesIO(raw), filename='x.csv'))


class ForecastPredictionWebTests(unittest.TestCase):
    fixture = web_fixture.ForecastModelWebTests.fixture
    PASSWORD = web_fixture.ForecastModelWebTests.PASSWORD
    _restore_app = web_fixture.ForecastModelWebTests._restore_app
    _csrf = staticmethod(web_fixture.ForecastModelWebTests._csrf)
    _register = web_fixture.ForecastModelWebTests._register
    request_data = web_fixture.ForecastModelWebTests.request_data
    _queue = web_fixture.ForecastModelWebTests._queue
    _complete = web_fixture.ForecastModelWebTests._complete
    _export = web_fixture.ForecastModelWebTests._export

    def setUp(self):
        web_fixture.ForecastModelWebTests.setUp(self)
        self.model_id = self._queue(forecast_model_retention_days='7', forecast_reporting_delay='2',
                                    forecast_origin='2020-06-10')
        self.assertEqual(self._complete(self.model_id)['status'], 'completed')
        self.url = f'/vibedash/account/models/{self.model_id}/predict'
        self.frame = pd.concat([daily_frame(), pd.DataFrame({
            'dteday': ['2020-06-09', '2020-06-10', '2020-06-11'],
            'cnt': [310, 315, 320], 'casual': [20, 40, 20],
        })], ignore_index=True)

    def _post(self, *, client=None, frame=None, raw=None, **fields):
        client = self.owner if client is None else client
        client.get('/vibedash/')
        with client.session_transaction() as session:
            token = session['vibedash_auth_csrf_token']
        if raw is None:
            raw = (self.frame if frame is None else frame).to_csv(index=False).encode()
        response = client.post(self.url, data={
            'csrf_token': token, 'forecast_origin': '2020-06-13', 'history_available': 'yes',
            'history_csv': (BytesIO(raw), 'updated.csv'), **fields,
        })
        # The test client can spool its encoded request body to a temporary
        # file; close that synthetic WSGI input independently of the upload.
        response.request.input_stream.close()
        return response

    def _snapshot(self):
        with sqlite3.connect(self.job_path) as connection:
            models = connection.execute('SELECT * FROM forecast_models').fetchall()
            jobs = connection.execute('SELECT count(*) FROM analysis_jobs').fetchone()
        paths = {path.relative_to(self.root) for path in self.root.rglob('*') if path.is_file()}
        return models, jobs, paths, self._outcomes()

    def _outcomes(self):
        with sqlite3.connect(self.job_path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute('SELECT * FROM forecast_outcomes')]

    def _save(self, **fields):
        response = self._post(retain_forecast='yes', **fields)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('Open saved forecast and add actuals', html)
        for row in self._outcomes():
            url = '/vibedash/account/forecasts/' + row['prediction_id']
            if f'href="{url}#forecast-quality">Open saved forecast' in html:
                return url
        self.fail('The saved forecast link was not found.')

    def _series_rows(self, table='forecast_series'):
        self.assertIn(table, ('forecast_series', 'forecast_series_links', 'forecast_series_actuals'))
        with sqlite3.connect(self.job_path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(f'SELECT * FROM {table}')]

    def _post_actuals(self, url, rows, *, client=None, **fields):
        client = self.owner if client is None else client
        client.get('/vibedash/')
        with client.session_transaction() as session:
            csrf = session['vibedash_auth_csrf_token']
        raw = rows if isinstance(rows, bytes) else ('date,actual\n' + ''.join(
            f'{day},{value}\n' for day, value in rows)).encode()
        response = client.post(url, data={
            'csrf_token': csrf, 'actuals_csv': (BytesIO(raw), 'daily-facts.csv'), **fields,
        })
        response.request.input_stream.close()
        return response

    def _correct(self, url, *, client=None, **fields):
        client = self.owner if client is None else client
        client.get('/vibedash/')
        with client.session_transaction() as session:
            csrf = session['vibedash_auth_csrf_token']
        return client.post(url + '/correct', data={
            'csrf_token': csrf, 'correction_date': '2020-06-14:1',
            'correction_value': '12.5', 'correction_reason': 'Reconciled source',
            **fields,
        })

    def test_owned_forecast_matches_frozen_model_and_retains_nothing(self):
        artifact = self._export(self.owner, self.model_id).get_json()
        expected = predict_artifact(artifact, history=self.frame, forecast_origin='2020-06-13')
        before = self._snapshot()
        with patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('must not fit')), \
                patch('vibedash.routes.run_confirmed_forecast', side_effect=AssertionError('must not reselect')):
            response = self._post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        self.assertEqual(response.headers['Referrer-Policy'], 'no-referrer')
        html = response.get_data(as_text=True)
        self.assertIn('New forecast ready', html)
        self.assertIn('Accuracy on new outcomes is unknown', html)
        self.assertIn('History available through 2020-06-11; model trained through 2020-06-08', html)
        for point in expected['predictions']:
            self.assertIn(f'<th scope="row">{point["date"]}</th><td>{point["prediction"]:.8g}</td><td>{point["baseline"]:.8g}</td>', html)
        self.assertNotIn('updated.csv', html)
        self.assertNotIn(artifact['sha256'], html)
        self.assertEqual(self._snapshot(), before)
        account = self.owner.get('/vibedash/account').get_data(as_text=True)
        self.assertNotIn('New forecast ready', account)
        self.assertIn('new issue date minus 2 days', account)

    def test_guest_other_account_and_disabled_feature_cannot_parse_input(self):
        stranger = web_app.app.test_client()
        self.assertEqual(self._register(stranger, 'stranger@example.com').status_code, 302)
        with patch('vibedash.routes.load_prediction_history') as parse:
            for client in (web_app.app.test_client(), stranger):
                response = self._post(client=client)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.get_json(), {'error': 'Saved model is unavailable.'})
            with patch.dict(web_app.app.config, VIBEDASH_MODEL_STORAGE_ENABLED=False):
                self.assertEqual(self._post().status_code, 404)
                self.assertNotIn('/predict#saved-model-', self.owner.get('/vibedash/account').get_data(as_text=True))
        parse.assert_not_called()

    def test_expired_model_and_model_lost_during_request_are_unavailable(self):
        with patch('vibedash.forecast_models.ForecastModelStore.predict', side_effect=ForecastModelNotFound):
            self.assertEqual(self._post().status_code, 404)
        with sqlite3.connect(self.job_path) as connection:
            connection.execute("UPDATE forecast_models SET expires_at='2000-01-01T00:00:00+00:00'")
        with patch('vibedash.routes.load_prediction_history') as parse:
            self.assertEqual(self._post().status_code, 404)
        parse.assert_not_called()

    def test_csrf_and_explicit_input_confirmation_are_required(self):
        with patch('vibedash.routes.load_prediction_history') as parse:
            self.assertEqual(self._post(csrf_token='bad').status_code, 400)
            for fields, marker in (({'forecast_origin': ['2020-06-13', '2020-06-14']}, '-origin'),
                                   ({'forecast_origin': 'June 13'}, '-origin'),
                                   ({'history_available': ''}, '-availability')):
                response = self._post(**fields)
                self.assertEqual(response.status_code, 422)
                html = response.get_data(as_text=True)
                self.assertIn(f'href="#forecast-{self.model_id}{marker}"', html)
                self.assertIn('aria-invalid="true"', html)
        parse.assert_not_called()

    def test_rejects_prior_dates_unavailable_tail_stale_prefix_and_incomplete_history(self):
        modified = self.frame.copy()
        modified.loc[0, 'cnt'] += 1
        invalid = [
            (self.frame.iloc[:-3], '2020-06-10', 'later than'),
            (self.frame, '2020-06-12', 'reporting delay'),
            (self.frame, '2020-06-14', 'reporting delay'),
            (modified, '2020-06-13', 'complete original training prefix'),
            (self.frame.iloc[1:], '2020-06-13', 'complete original training prefix'),
            (self.frame.drop(index=159), '2020-06-13', 'without gaps'),
        ]
        for frame, origin, message in invalid:
            with self.subTest(origin=origin, message=message), patch('src.forecast_artifact._predict') as predict:
                response = self._post(frame=frame, forecast_origin=origin)
                self.assertEqual(response.status_code, 422)
                self.assertIn(message, response.get_data(as_text=True))
                self.assertNotIn('New forecast ready', response.get_data(as_text=True))
                predict.assert_not_called()

    def test_negative_count_output_is_not_displayed_as_a_forecast(self):
        with patch('src.forecast_artifact._predict', side_effect=lambda state, history, steps, period: [-1.] * steps):
            response = self._post()
        self.assertEqual(response.status_code, 422)
        self.assertIn('produced negative counts', response.get_data(as_text=True))
        self.assertNotIn('<caption>Daily predictions', response.get_data(as_text=True))

    def test_malformed_and_oversize_uploads_are_rejected_without_echo(self):
        response = self._post(raw=b'dteday,cnt\n2020-06-11,123,private extra cell\n')
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('private extra cell', response.get_data(as_text=True))
        response = self._post(raw=b'x' * (MAX_HISTORY_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')

    def test_opt_in_is_immutable_and_an_identical_retry_reuses_the_record(self):
        self.assertEqual(self._post(retain_forecast=['yes', 'yes']).status_code, 422)
        self.assertEqual(self._outcomes(), [])
        with patch('vibedash.forecast_models._now', return_value=datetime(2020, 6, 12, tzinfo=timezone.utc)):
            self.assertEqual(self._post(retain_forecast='yes').status_code, 422)
        self.assertEqual(self._outcomes(), [])
        url = self._save()
        initial = self._outcomes()
        self.assertEqual(self._save(), url)
        self.assertEqual(self._outcomes(), initial)
        with patch('src.forecast_artifact._predict', side_effect=lambda state, history, steps, period: [1.] * steps):
            response = self._post(retain_forecast='yes')
        self.assertEqual(response.status_code, 422)
        self.assertIn('already exists for this issue date', response.get_data(as_text=True))
        self.assertEqual(self._outcomes(), initial)
        with patch('src.forecast_artifact._predict', side_effect=lambda state, history, steps, period: [-1.] * steps):
            self.assertEqual(self._post(retain_forecast='yes').status_code, 422)
        self.assertEqual(self._outcomes(), initial)

    def test_scores_paired_dates_and_persists_partial_and_full_coverage(self):
        url = self._save()
        saved, = self._outcomes()
        report = json.loads(saved['forecast_json'])['report']
        first, *rest = report['predictions']
        actual = first['prediction'] + 12.5
        before = self.owner.get(url).get_data(as_text=True)
        self.assertIn('Accuracy is unknown until actuals are added', before)
        self.assertNotIn('Errors on the same matched dates', before)
        self.assertEqual(self._post_actuals(url, [(first['date'], actual)]).status_code, 303)
        response = self.owner.get(url)
        html = response.get_data(as_text=True)
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        self.assertIn(f'1 of {len(rest) + 1} forecast dates matched', html)
        self.assertIn('Partial results cover the matched dates only', html)
        baseline_error = abs(actual - first['baseline'])
        self.assertIn(f'<th scope="row">MAE</th><td>12.5</td><td>{baseline_error:.8g}</td>', html)
        self.assertIn(f'<th scope="row">RMSE</th><td>12.5</td><td>{baseline_error:.8g}</td>', html)
        self.assertNotIn('daily-facts.csv', html)
        initial = self._outcomes()
        self.assertEqual(self._post_actuals(url, [(first['date'], actual)]).status_code, 303)
        self.assertEqual(self._outcomes(), initial)
        response = self._post_actuals(url, [(p['date'], p['prediction']) for p in rest])
        self.assertEqual(response.status_code, 303)
        self.assertIn('Errors cover all forecast dates', self.owner.get(url).get_data(as_text=True))
        updated, = self._outcomes()
        self.assertEqual(updated['forecast_json'], saved['forecast_json'])
        self.assertEqual(updated['expires_at'], saved['expires_at'])

    def test_utc_issue_day_distinguishes_live_and_retrospective_forecast_scores(self):
        # The fixture's issue dates are historical relative to the test runner.
        # Use a simulated application clock and skip real-time expiry cleanup.
        with patch('vibedash.analysis_jobs.AnalysisJobStore.purge_terminal', return_value=0), \
                patch('vibedash.forecast_models._now') as clock:
            clock.return_value = datetime(2020, 6, 13, 23, 59, 59, tzinfo=timezone.utc)
            prospective = self._save()
            before = self._outcomes()
            clock.return_value = datetime(2020, 6, 14, tzinfo=timezone.utc)
            self.assertEqual(self._save(), prospective)
            self.assertEqual(self._outcomes(), before)
            retrospective = self._save(frame=self.frame.iloc[:-1], forecast_origin='2020-06-12')
            account = self.owner.get('/vibedash/account').get_data(as_text=True)
            for url, label in ((prospective, 'Prospective'), (retrospective, 'Retrospective')):
                self.assertIn(f'· {url.rsplit("/", 1)[1][:12]} · {label}</a>', account)
                html = self.owner.get(url).get_data(as_text=True)
                self.assertIn(f'<strong>{label}</strong>', html)

            clock.return_value = datetime(2020, 6, 16, tzinfo=timezone.utc)
            for url, label in ((prospective, 'Prospective'), (retrospective, 'Retrospective')):
                self.assertEqual(self._post_actuals(url, [('2020-06-14', 315)]).status_code, 303)
                html = self.owner.get(url).get_data(as_text=True)
                self.assertIn(f'<strong>{label}</strong>', html)
                self.assertIn('Errors on the same matched dates', html)

    def test_bad_or_premature_actuals_leave_the_entire_batch_unapplied(self):
        url = self._save()
        self.assertEqual(self._post_actuals(url, [('2020-06-14', 10)]).status_code, 303)
        initial = self._outcomes()
        for rows in ([('2020-06-15', 20), ('2020-06-14', 11)],
                     [('2020-06-15', 20), ('2020-06-13', 20)],
                     [('2020-06-15', -1)], [('2020-06-15', 'nan')],
                     [('2020-06-15', '1e99999')], [('2020-06-15', '9007199254740992')],
                     [('2020-06-15', 10), ('2020-06-15', 10)],
                     b'date,wrong\n2020-06-15,private-cell\n'):
            with self.subTest(rows=rows):
                response = self._post_actuals(url, rows)
                self.assertEqual(response.status_code, 422)
                self.assertIn('aria-invalid="true"', response.get_data(as_text=True))
                self.assertNotIn('private-cell', response.get_data(as_text=True))
        with patch('vibedash.forecast_models._now', return_value=datetime(2020, 6, 16, tzinfo=timezone.utc)):
            response = self._post_actuals(url, [('2020-06-15', 20)])
            self.assertEqual(response.status_code, 422)
            self.assertIn('too early', response.get_data(as_text=True))
        self.assertEqual(self._post_actuals(url, b'x' * (MAX_ACTUALS_BYTES + 1)).status_code, 413)
        self.assertEqual(self._outcomes(), initial)

    def test_actuals_require_owner_and_csrf_and_survive_model_expiry(self):
        url = self._save()
        stranger = web_app.app.test_client()
        self.assertEqual(self._register(stranger, 'stranger@example.com').status_code, 302)
        with patch('vibedash.routes.load_actuals') as parse:
            for client in (stranger, web_app.app.test_client()):
                self.assertEqual(client.get(url).status_code, 404)
                self.assertEqual(self._post_actuals(url, [], client=client).status_code, 404)
            self.assertEqual(self._post_actuals(url, [], csrf_token=['a', 'b']).status_code, 400)
        parse.assert_not_called()
        with sqlite3.connect(self.job_path) as connection:
            connection.execute("UPDATE forecast_models SET expires_at='2000-01-01T00:00:00+00:00'")
        self.assertEqual(self.owner.get(url).status_code, 200)
        self.assertEqual(self._post_actuals(url, [('2020-06-14', 0)]).status_code, 303)
        self.assertIn('1 of ', self.owner.get(url).get_data(as_text=True))
        with sqlite3.connect(self.job_path) as connection:
            connection.execute("UPDATE forecast_outcomes SET expires_at='2000-01-01T00:00:00+00:00'")
        with patch.dict(web_app.app.config, VIBEDASH_MODEL_STORAGE_ENABLED=False):
            self.assertEqual(self.owner.get(url).status_code, 404)
        self.assertEqual(self._outcomes(), [])

    def test_correction_preserves_prior_actual_and_rescores_both_predictions(self):
        url = self._save()
        saved, = self._outcomes()
        first = json.loads(saved['forecast_json'])['report']['predictions'][0]
        self.assertEqual(self._post_actuals(url, [(first['date'], 10)]).status_code, 303)
        original = json.loads(self._outcomes()[0]['actuals_json'])[first['date']]
        html = self.owner.get(url).get_data(as_text=True)
        self.assertIn('Correct a saved actual', html)
        self.assertIn(f'value="{first["date"]}:1"', html)
        self.assertNotIn('Actual correction history', html)

        actual = first['prediction'] + 7.5
        response = self._correct(url, correction_value=str(actual), correction_reason='Reconciled <ticket>')
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        page = self.owner.get(url)
        self.assertEqual(page.headers['Cache-Control'], 'private, no-store')
        html = page.get_data(as_text=True)
        baseline_error = abs(actual - first['baseline'])
        self.assertIn(f'<th scope="row">MAE</th><td>7.5</td><td>{baseline_error:.8g}</td>', html)
        self.assertIn(f'<th scope="row">RMSE</th><td>7.5</td><td>{baseline_error:.8g}</td>', html)
        self.assertIn(f'value="{first["date"]}:2"', html)
        self.assertIn('Actual correction history', html)
        self.assertIn('Reconciled &lt;ticket&gt;', html)
        self.assertIn(f'Model error {abs(10 - first["prediction"]):.8g}', html)
        self.assertIn(f'baseline error {abs(10 - first["baseline"]):.8g}', html)
        self.assertIn('Model error 7.5', html)
        updated, = self._outcomes()
        history = json.loads(updated['actuals_json'])[first['date']]
        self.assertEqual(history['value'], actual)
        self.assertEqual(history['superseded'], [original])
        self.assertEqual(updated['forecast_json'], saved['forecast_json'])
        self.assertEqual(updated['expires_at'], saved['expires_at'])
        self.assertEqual(self._post_actuals(url, [(first['date'], 10)]).status_code, 422)
        self.assertEqual(self._outcomes(), [updated])

    def test_correction_replays_and_stale_or_invalid_updates(self):
        url = self._save()
        self.assertEqual(self._post_actuals(url, [('2020-06-14', 10)]).status_code, 303)
        before = self._outcomes()
        for fields in (
            {'correction_value': '-1'}, {'correction_value': 'nan'}, {'correction_value': '10'},
            {'correction_reason': ' '}, {'correction_reason': 'x' * 301},
            {'correction_reason': ['first', 'second']}, {'correction_date': '2020-06-15:1'},
        ):
            with self.subTest(fields=fields):
                response = self._correct(url, **fields)
                self.assertEqual(response.status_code, 422)
                self.assertIn('aria-invalid="true"', response.get_data(as_text=True))
        self.assertEqual(self._outcomes(), before)

        self.assertEqual(self._correct(url).status_code, 303)
        after = self._outcomes()
        self.assertEqual(self._correct(url).status_code, 303)
        self.assertEqual(self._correct(url, correction_value='13').status_code, 409)
        self.assertEqual(self._correct(url, correction_reason='Another reason').status_code, 409)
        with patch('vibedash.forecast_outcomes.MAX_ACTUAL_CORRECTIONS', 1):
            self.assertEqual(self._correct(url, correction_date='2020-06-14:2',
                                           correction_value='13').status_code, 422)
        self.assertEqual(self._outcomes(), after)
        self.assertEqual(self._correct(url, correction_date='2020-06-14:2',
                                       correction_value='13').status_code, 303)
        actual = json.loads(self._outcomes()[0]['actuals_json'])['2020-06-14']
        self.assertEqual([version['value'] for version in actual['superseded']], [10., 12.5])
        self.assertEqual(actual['value'], 13.)

    def test_correction_requires_owner_csrf_and_unexpired_forecast(self):
        url = self._save()
        self.assertEqual(self._post_actuals(url, [('2020-06-14', 10)]).status_code, 303)
        stranger = web_app.app.test_client()
        self.assertEqual(self._register(stranger, 'stranger@example.com').status_code, 302)
        before = self._outcomes()
        with patch('vibedash.routes.correction_from_form') as parse:
            self.assertEqual(self._correct(url, client=stranger).status_code, 404)
            self.assertEqual(self._correct(url, client=web_app.app.test_client()).status_code, 404)
            self.assertEqual(self._correct(url, csrf_token=['a', 'b']).status_code, 400)
            self.assertEqual(self._correct(url, correction_reason='x' * 9000).status_code, 413)
            with patch.dict(web_app.app.config, VIBEDASH_MODEL_STORAGE_ENABLED=False):
                self.assertEqual(self._correct(url).status_code, 404)
            self.assertEqual(self._outcomes(), before)
            with sqlite3.connect(self.job_path) as connection:
                connection.execute("UPDATE forecast_outcomes SET expires_at='2000-01-01T00:00:00+00:00'")
            self.assertEqual(self._correct(url).status_code, 404)
        parse.assert_not_called()
        self.assertEqual(self._outcomes(), [])

    def test_account_erasure_purges_saved_forecasts_and_facts(self):
        url = self._save()
        self.assertEqual(self._post_actuals(url, [('2020-06-14', 10)]).status_code, 303)
        self.assertEqual(self._correct(url).status_code, 303)
        with self.owner.session_transaction() as session:
            csrf = session['vibedash_auth_csrf_token']
        response = self.owner.post('/vibedash/account/delete', data={
            'csrf_token': csrf, 'current_password': self.PASSWORD, 'delete_confirmation': 'DELETE',
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self._outcomes(), [])
        self.assertEqual(self.owner.get(url).status_code, 404)

    def test_shared_actuals_and_corrections_rescore_overlapping_forecasts_only(self):
        first_url = self._save(new_series_label='Store A orders', series_confirmed='yes')
        series, = self._series_rows()
        self.assertEqual(self._post_actuals(first_url, [('2020-06-14', 300), ('2020-06-15', 301)]).status_code, 303)
        updated_history = pd.concat([self.frame, pd.DataFrame({
            'dteday': ['2020-06-12'], 'cnt': [325], 'casual': [30],
        })], ignore_index=True)
        second_model = self._queue(forecast_model_retention_days='7', forecast_reporting_delay='2',
                                   forecast_origin='2020-06-10')
        self.assertEqual(self._complete(second_model)['status'], 'completed')
        with patch.object(self, 'url', f'/vibedash/account/models/{second_model}/predict'):
            second_url = self._save(frame=updated_history, forecast_origin='2020-06-14',
                                    series_id=series['series_id'], series_confirmed='yes')
            isolated_url = self._save()
        self.assertIn('2 of 7 forecast dates matched', self.owner.get(first_url).get_data(as_text=True))
        linked = self.owner.get(second_url).get_data(as_text=True)
        self.assertIn('1 of 7 forecast dates matched', linked)
        self.assertIn('Store A orders', linked)
        self.assertIn('0 of 7 forecast dates matched', self.owner.get(isolated_url).get_data(as_text=True))
        self.assertEqual(self._post_actuals(second_url, [('2020-06-14', 300)]).status_code, 422)
        response = self._correct(second_url, correction_date='2020-06-15:1', correction_value='310')
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self._correct(first_url, correction_date='2020-06-15:1',
                                       correction_value='311').status_code, 409)
        for url, values in ((first_url, {'2020-06-14': 300, '2020-06-15': 310}),
                            (second_url, {'2020-06-15': 310})):
            row = next(row for row in self._outcomes() if url.endswith(row['prediction_id']))
            points = [p for p in json.loads(row['forecast_json'])['report']['predictions'] if p['date'] in values]
            model = sum(abs(values[p['date']] - p['prediction']) for p in points) / len(points)
            baseline = sum(abs(values[p['date']] - p['baseline']) for p in points) / len(points)
            html = self.owner.get(url).get_data(as_text=True)
            self.assertIn(f'<th scope="row">MAE</th><td>{model:.8g}</td><td>{baseline:.8g}</td>', html)
            self.assertIn('Reconciled source', html)
            self.assertEqual(row['actuals_json'], '{}')
        self.assertIn('0 of 7 forecast dates matched', self.owner.get(isolated_url).get_data(as_text=True))
        self.assertEqual(len(self._series_rows('forecast_series_actuals')), 2)

    def test_series_selection_is_explicit_scoped_compatible_and_immutable(self):
        for fields in ({'new_series_label': 'No opt-in', 'series_confirmed': 'yes'},
                       {'retain_forecast': 'yes', 'new_series_label': 'Not confirmed'},
                       {'retain_forecast': 'yes', 'series_id': 'a' * 32,
                        'new_series_label': 'Both', 'series_confirmed': 'yes'}):
            self.assertEqual(self._post(**fields).status_code, 422)
        self.assertEqual(self._outcomes(), [])
        first_url = self._save(new_series_label='Store A orders', series_confirmed='yes')
        series, = self._series_rows()
        self.assertEqual(self._save(new_series_label='store a orders', series_confirmed='yes'), first_url)
        self.assertEqual(self._save(series_id=series['series_id'], series_confirmed='yes'), first_url)
        self.assertEqual(self._post(retain_forecast='yes').status_code, 422)
        self.assertEqual(self._post(retain_forecast='yes', new_series_label='Different',
                                    series_confirmed='yes').status_code, 422)
        with sqlite3.connect(self.job_path) as connection:
            foreign_id = 'b' * 32
            connection.execute('INSERT INTO forecast_series VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                               ('c' * 64, foreign_id, 'Foreign label', 'foreign label', 'dteday', 'cnt', 'count', 2))
        response = self._post(retain_forecast='yes', series_id=foreign_id, series_confirmed='yes')
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('Foreign label', response.get_data(as_text=True))
        incompatible = self._queue(forecast_model_retention_days='7', forecast_reporting_delay='1',
                                   forecast_origin='2020-06-09')
        self.assertEqual(self._complete(incompatible)['status'], 'completed')
        with patch.object(self, 'url', f'/vibedash/account/models/{incompatible}/predict'):
            response = self._post(retain_forecast='yes', forecast_origin='2020-06-12',
                                  series_id=series['series_id'], series_confirmed='yes')
        self.assertEqual(response.status_code, 422)
        self.assertIn('unavailable or incompatible', response.get_data(as_text=True))
        self.assertEqual(len(self._outcomes()), 1)
        self.assertEqual(len(self._series_rows('forecast_series_links')), 1)

    def test_series_accuracy_matches_due_dates_by_lead_for_one_prospective_model(self):
        other_model = self._queue(forecast_model_retention_days='7', forecast_reporting_delay='2',
                                  forecast_origin='2020-06-10')
        self.assertEqual(self._complete(other_model)['status'], 'completed')
        frame = pd.concat([self.frame, pd.DataFrame({
            'dteday': ['2020-06-12'], 'cnt': [325], 'casual': [30],
        })], ignore_index=True)
        with patch('vibedash.analysis_jobs.AnalysisJobStore.purge_terminal', return_value=0), \
                patch('vibedash.forecast_models._now') as clock:
            clock.return_value = datetime(2020, 6, 13, tzinfo=timezone.utc)
            first = self._save(new_series_label='Store A orders', series_confirmed='yes')
            series, = self._series_rows()
            shared = {'series_id': series['series_id'], 'series_confirmed': 'yes'}
            self.assertIn('Actuals are still pending', self.owner.get(first).get_data(as_text=True))
            self._save(frame=self.frame.iloc[:-1], forecast_origin='2020-06-12', **shared)
            with patch.object(self, 'url', f'/vibedash/account/models/{other_model}/predict'):
                other = self._save(**shared)
            clock.return_value = datetime(2020, 6, 14, tzinfo=timezone.utc)
            second = self._save(frame=frame, forecast_origin='2020-06-14', **shared)
            clock.return_value = datetime(2020, 6, 17, tzinfo=timezone.utc)

            self.assertEqual(self._post_actuals(first, [('2020-06-14', 300)]).status_code, 303)
            partial = ' '.join(self.owner.get(second).get_data(as_text=True).split())
            self.assertIn('2 saved prospective forecasts.', partial)
            self.assertIn('<th scope="row">1</th> <td>1 / 2</td>', partial)
            self.assertIn('<th scope="row">2</th> <td>0 / 1</td> <td>Pending</td>', partial)
            self.assertIn('1 saved prospective forecast.', self.owner.get(other).get_data(as_text=True))

            self.assertEqual(self._post_actuals(second, [('2020-06-15', 301)]).status_code, 303)
            self.assertEqual(self._correct(second, correction_date='2020-06-15:1',
                                           correction_value='310').status_code, 303)
            reports = {row['prediction_id']: json.loads(row['forecast_json'])['report']['predictions']
                       for row in self._outcomes()}
            first_points = reports[first.rsplit('/', 1)[1]]
            second_points = reports[second.rsplit('/', 1)[1]]
            html = ' '.join(self.owner.get(first).get_data(as_text=True).split())
            for lead, coverage, pairs in (
                (1, '2 / 2', [(first_points[0], 300), (second_points[0], 310)]),
                (2, '1 / 1', [(first_points[1], 310)]),
            ):
                model = sum(abs(point['prediction'] - actual) for point, actual in pairs) / len(pairs)
                baseline = sum(abs(point['baseline'] - actual) for point, actual in pairs) / len(pairs)
                self.assertIn(f'<th scope="row">{lead}</th> <td>{coverage}</td> '
                              f'<td>{model:.8g}</td> <td>{baseline:.8g}</td>', html)
            with sqlite3.connect(self.job_path) as connection:
                connection.execute('UPDATE forecast_outcomes SET expires_at=? WHERE prediction_id=?',
                                   ('2020-06-17T00:00:00.000000+00:00', first.rsplit('/', 1)[1]))
            self.assertIn('1 saved prospective forecast.', self.owner.get(second).get_data(as_text=True))

    def test_shared_facts_follow_remaining_date_coverage_and_account_erasure(self):
        first_url = self._save(new_series_label='Store A orders', series_confirmed='yes')
        series, = self._series_rows()
        frame = pd.concat([self.frame, pd.DataFrame({
            'dteday': ['2020-06-12'], 'cnt': [325], 'casual': [30],
        })], ignore_index=True)
        second_url = self._save(frame=frame, forecast_origin='2020-06-14',
                                series_id=series['series_id'], series_confirmed='yes')
        self.assertEqual(self._post_actuals(first_url, [('2020-06-14', 300), ('2020-06-15', 301)]).status_code, 303)
        self.assertEqual(self._correct(first_url, correction_date='2020-06-15:1',
                                       correction_value='305').status_code, 303)
        with sqlite3.connect(self.job_path) as connection:
            connection.execute("UPDATE forecast_outcomes SET expires_at='2000-01-01T00:00:00+00:00' WHERE prediction_id=?",
                               (first_url.rsplit('/', 1)[1],))
        with patch.dict(web_app.app.config, VIBEDASH_MODEL_STORAGE_ENABLED=False):
            self.assertEqual(self.owner.get(first_url).status_code, 404)
        self.assertEqual([row['day'] for row in self._series_rows('forecast_series_actuals')], ['2020-06-15'])
        self.assertIn('Reconciled source', self.owner.get(second_url).get_data(as_text=True))
        with self.owner.session_transaction() as session:
            csrf = session['vibedash_auth_csrf_token']
        response = self.owner.post('/vibedash/account/delete', data={
            'csrf_token': csrf, 'current_password': self.PASSWORD, 'delete_confirmation': 'DELETE',
        })
        self.assertEqual(response.status_code, 302)
        for table in ('forecast_series', 'forecast_series_links', 'forecast_series_actuals'):
            self.assertEqual(self._series_rows(table), [])
