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

    def _save(self):
        response = self._post(retain_forecast='yes')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Open saved forecast and add actuals', response.get_data(as_text=True))
        saved, = self._outcomes()
        return '/vibedash/account/forecasts/' + saved['prediction_id']

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

    def test_account_erasure_purges_saved_forecasts_and_facts(self):
        url = self._save()
        self.assertEqual(self._post_actuals(url, [('2020-06-14', 10)]).status_code, 303)
        with self.owner.session_transaction() as session:
            csrf = session['vibedash_auth_csrf_token']
        response = self.owner.post('/vibedash/account/delete', data={
            'csrf_token': csrf, 'current_password': self.PASSWORD, 'delete_confirmation': 'DELETE',
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self._outcomes(), [])
        self.assertEqual(self.owner.get(url).status_code, 404)
