"""
VibeDash Flask routes
"""
import csv
import io
import os
import re
import secrets
import threading
import uuid
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path

try:
    from flask import (
        current_app,
        flash,
        jsonify,
        make_response,
        redirect,
        render_template,
        request,
        send_file,
        session,
        url_for,
    )
    from werkzeug.utils import secure_filename
    import pandas as pd
    from src.demo_data import (
        DEMO_DATASET_ID,
        DEMO_FILENAME,
        DEMO_PROMPT,
        create_saas_growth_demo,
    )
    from .spec import create_saas_demo_viz_spec, parse_prompt_to_viz_spec
    from .generator_bridge import generate_dashboard_data as bridge_generate_dashboard_data
    from .exporter import (
        cleanup_expired_artifacts,
        load_session_data,
        make_single_file_html,
        save_export,
        save_session_data,
    )
    from .ollama_client import is_ollama_available
    from .analysis_jobs import (
        AnalysisJobDispatcher,
        AnalysisJobCapacityError,
        AnalysisJobStore,
        JOB_ID_PATTERN,
        MAX_HISTORY_JOBS,
    )
    from .audit_manifest import (
        build_audit_manifest,
        build_period_comparison_manifest,
    )
    from .decision_brief import build_decision_brief
    from .comparison_report import build_comparison_decision_guidance
    from .pilot_metrics import feedback_available, forget_scope, record_feedback, scope_token
    from .decision_cases import (
        DecisionCaseCapacityError,
        DecisionCaseConflictError,
        DecisionCaseStore,
        IDENTIFIER_PATTERN as DECISION_CASE_ID_PATTERN,
        MAX_DECISION_CASES,
    )
    from .readiness_engine import DatasetReadinessEngine
    from .accounts import (
        ACCOUNT_ID_PATTERN,
        AccountStore,
        AccountValidationError,
        account_scope_id,
        normalize_email,
    )
    from .period_comparison import (
        MAX_COLUMNS as COMPARISON_MAX_COLUMNS,
        MAX_MEMORY_BYTES as COMPARISON_MAX_MEMORY_BYTES,
        MAX_ROWS as COMPARISON_MAX_ROWS,
        MAX_REQUEST_BYTES as COMPARISON_MAX_REQUEST_BYTES,
        MAX_TOTAL_COLUMNS as COMPARISON_MAX_TOTAL_COLUMNS,
        MAX_TOTAL_ROWS as COMPARISON_MAX_TOTAL_ROWS,
        build_period_comparison,
    )
    from . import vibedash_bp
except ImportError:
    # Flask не установлен, создаем заглушки
    vibedash_bp = None


if vibedash_bp:
    analysis_job_dispatcher = AnalysisJobDispatcher(max_workers=1)
    _CSV_HEADER_LOCK = threading.Lock()
    _ACCOUNT_STORE_INIT_LOCK = threading.Lock()
    _UNSET_SCOPE = object()
    _ACCOUNT_SESSION_KEY = 'vibedash_account_id'
    _ACCOUNT_CREDENTIAL_SESSION_KEY = 'vibedash_account_credential'
    _VIBEDASH_IDENTITY_SESSION_KEYS = (
        _ACCOUNT_SESSION_KEY,
        _ACCOUNT_CREDENTIAL_SESSION_KEY,
        'vibedash_analysis_scope_id',
        'vibedash_auth_csrf_token',
        'vibedash_decision_csrf_token',
    )

    class DatasetReadinessBlocked(ValueError):
        """The uploaded table failed one or more safe readiness contracts."""

        def __init__(self, report):
            super().__init__(report['summary'])
            self.report = report

    def _account_store():
        """Return the process-cached account store, failing closed on errors."""
        application = current_app._get_current_object()
        store = application.extensions.get('vibedash_account_store')
        if store is not None:
            return store
        with _ACCOUNT_STORE_INIT_LOCK:
            store = application.extensions.get('vibedash_account_store')
            if store is not None:
                return store
            try:
                store = AccountStore(application.config['VIBEDASH_ACCOUNT_STORE_PATH'])
            except Exception:
                application.logger.warning(
                    'VibeDash account store unavailable',
                    extra={'event': 'vibedash_account_store_unavailable'},
                )
                return None
            application.extensions['vibedash_account_store'] = store
            return store


    def _current_account():
        """Resolve an account only from one current id/token snapshot.

        The signed Flask cookie is only a transport envelope.  The account
        store's credential check binds the id and opaque token to the current
        password hash in one read, so copied cookies stop working after a
        password change without a server-side session table.
        """
        account_id = session.get(_ACCOUNT_SESSION_KEY)
        credential = session.get(_ACCOUNT_CREDENTIAL_SESSION_KEY)
        if (
            not isinstance(account_id, str)
            or ACCOUNT_ID_PATTERN.fullmatch(account_id) is None
            or not isinstance(credential, str)
            or not re.fullmatch(r'[0-9a-f]{64}', credential)
        ):
            if account_id is not None or credential is not None:
                _rotate_vibedash_identity()
            return None
        store = _account_store()
        if store is None:
            _rotate_vibedash_identity()
            return None
        try:
            account = store.account_for_credential(
                account_id, credential, current_app.secret_key
            )
        except Exception:
            account = None
        if not isinstance(account, dict) or account.get('id') != account_id:
            _rotate_vibedash_identity()
            return None
        try:
            if normalize_email(account.get('email')) != account.get('email'):
                raise ValueError('non-canonical account record')
            # The account must also be usable with this deployment's secret.
            # Otherwise the landing page must not advertise an identity whose
            # account-owned scope cannot be resolved.
            account_scope_id(account_id, current_app.secret_key)
        except Exception:
            _rotate_vibedash_identity()
            return None
        return account


    def _establish_account(account):
        """Replace VibeDash identity state and issue a fresh credential."""
        try:
            store = _account_store()
            if store is None or not isinstance(account, dict):
                raise RuntimeError('account establishment is unavailable')
            # Login, registration, and password-change operations attach a
            # credential derived from the exact password-hash snapshot they
            # verified or wrote while holding the store transaction.  There is
            # deliberately no fallback to credential_token(): a later hash
            # read would reintroduce the verification-to-session TOCTOU race
            # and let a public account record establish an identity without a
            # transaction-bound authentication result.
            credential = account.get('_vibedash_session_credential')
            if not isinstance(credential, str) or not re.fullmatch(r'[0-9a-f]{64}', credential):
                raise RuntimeError('account credential is unavailable')
            _rotate_vibedash_identity()
            session.permanent = True
            session[_ACCOUNT_SESSION_KEY] = account['id']
            session[_ACCOUNT_CREDENTIAL_SESSION_KEY] = credential
            _auth_csrf_token()
            _decision_csrf_token()
            return True
        except Exception:
            # Establishment must never leave an account id, credential, or
            # CSRF token behind when any part of minting or session mutation
            # fails.  Preserve only unrelated classic/monitoring state.
            try:
                _rotate_vibedash_identity()
            except Exception:
                # A broken session implementation is already fail-closed for
                # this request; do not surface a secondary exception here.
                pass
            return False


    def _auth_csrf_token():
        token = session.get('vibedash_auth_csrf_token')
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{64}', token):
            token = secrets.token_hex(32)
            session['vibedash_auth_csrf_token'] = token
        return token


    def _valid_auth_csrf_token(token):
        expected = _auth_csrf_token()
        return (
            isinstance(token, str)
            and len(token) == len(expected)
            and secrets.compare_digest(token, expected)
        )


    def _rotate_vibedash_identity():
        """Rotate VibeDash identity state without destroying other app state.

        Flask's signed-cookie session has no server-side session identifier to
        rotate. Removing the VibeDash identity, guest scope, and CSRF tokens
        prevents guest-scope claiming and replay through the browser's next
        cookie while leaving the classic upload/report workflow and its
        independent monitoring scope intact in the same browser. Credential
        rotation on password change additionally invalidates separately copied
        old VibeDash cookies because their token is derived from the old hash.
        """
        for key in _VIBEDASH_IDENTITY_SESSION_KEYS:
            session.pop(key, None)


    def _render_auth(mode, *, error=None, status=200):
        # Resolve the account before minting the form token.  A deleted or
        # malformed account identity rotates all VibeDash keys; generating the
        # token first would render a token that the replacement session no
        # longer accepts, making the next form submission fail once more.
        account = _current_account()
        return render_template(
            'vibedash_auth.html',
            mode=mode,
            error=error,
            csrf_token=_auth_csrf_token(),
            account=account,
        ), status


    def _analysis_scope_id():
        account = _current_account()
        if account is not None:
            try:
                return account_scope_id(account['id'], current_app.secret_key)
            except Exception:
                # A malformed secret must never make an account appear to own
                # a guest scope.  Discard the account identity and force a
                # fresh guest scope instead of silently mixing identities.
                _rotate_vibedash_identity()
        scope_id = session.get('vibedash_analysis_scope_id')
        if not isinstance(scope_id, str) or not JOB_ID_PATTERN.fullmatch(scope_id):
            scope_id = uuid.uuid4().hex
            session['vibedash_analysis_scope_id'] = scope_id
        return scope_id


    def _load_owned_session_data(session_id, scope_id=_UNSET_SCOPE):
        """Load one retained result only when it belongs to this VibeDash scope."""
        expected_scope = (
            _analysis_scope_id() if scope_id is _UNSET_SCOPE else scope_id
        )
        # A scope explicitly carried by a durable job is an ownership
        # invariant, not a fallback hint.  Missing or malformed job metadata
        # must fail closed instead of falling back to request state.
        if not isinstance(expected_scope, str) or not JOB_ID_PATTERN.fullmatch(
            expected_scope
        ):
            return None
        return load_session_data(session_id, owner_id=expected_scope)


    def _session_persist_scope(scope_id=_UNSET_SCOPE):
        """Resolve the owner for request and worker persistence paths."""
        expected_scope = (
            _analysis_scope_id() if scope_id is _UNSET_SCOPE else scope_id
        )
        if not isinstance(expected_scope, str) or not JOB_ID_PATTERN.fullmatch(
            expected_scope
        ):
            raise ValueError('The analysis scope is invalid.')
        return expected_scope


    def _analysis_job_store():
        return AnalysisJobStore(current_app.config['VIBEDASH_JOB_STORE_PATH'])


    def _decision_case_store():
        return DecisionCaseStore(current_app.config['VIBEDASH_JOB_STORE_PATH'])


    def _decision_csrf_token():
        token = session.get('vibedash_decision_csrf_token')
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{64}', token):
            token = secrets.token_hex(32)
            session['vibedash_decision_csrf_token'] = token
        return token


    def _valid_decision_csrf_token(token):
        expected = _decision_csrf_token()
        return (
            isinstance(token, str)
            and len(token) == len(expected)
            and secrets.compare_digest(token, expected)
        )


    def _decision_evidence_snapshot(job, session_data, priority_number):
        dashboard_data = session_data.get('dashboard_data') or {}
        decision_brief = dashboard_data.get('decision_brief') or {}
        priorities = decision_brief.get('priorities') or []
        selected = next(
            (
                priority
                for priority in priorities
                if priority.get('priority') == priority_number
            ),
            None,
        )
        if selected is None:
            raise ValueError('The selected decision priority is unavailable.')

        manifest = job.get('manifest') or {}
        dataset = manifest.get('dataset') or {}
        specification = manifest.get('specification') or {}

        def bounded(value, maximum=500):
            return ' '.join(str(value or '').split())[:maximum]

        evidence = [
            bounded(item, 300)
            for item in (selected.get('evidence') or [])[:5]
            if bounded(item, 300)
        ]
        return {
            'contract': 'decision-case-source-v1',
            'analysis_job_id': job['id'],
            'analysis_contract': bounded(manifest.get('analysis_contract'), 80),
            'dataset_sha256': bounded(dataset.get('content_sha256'), 64),
            'analysis_title': bounded(
                specification.get('title')
                or session_data.get('filename')
                or 'Analysis',
                200,
            ),
            'priority': {
                'number': priority_number,
                'category': bounded(selected.get('category'), 80),
                'title': bounded(selected.get('title'), 200),
                'finding': bounded(selected.get('finding'), 800),
                'recommended_action': bounded(selected.get('action'), 800),
                'confidence': bounded(selected.get('confidence'), 40),
                'evidence': evidence,
            },
        }


    def _decision_case_view(decision_case):
        review_date = date.fromisoformat(decision_case['review_date'])
        return {
            **decision_case,
            'short_id': decision_case['id'][:10],
            'created_label': datetime.fromisoformat(
                decision_case['created_at']
            ).astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'review_label': review_date.strftime('%Y-%m-%d'),
            'overdue': (
                decision_case['status'] == 'tracking'
                and review_date < datetime.now(timezone.utc).date()
            ),
        }


    def _load_vibedash_csv(source, *, nrows=None, max_columns=None):
        last_error = None
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                if hasattr(source, 'seek'):
                    source.seek(0)
                if max_columns is not None:
                    header = _read_original_csv_header(source, encoding)
                    if len(header) > max_columns:
                        raise ValueError('The CSV exceeds the column limit.')
                read_options = {'encoding': encoding}
                if nrows is not None:
                    read_options['nrows'] = nrows
                dataframe = pd.read_csv(source, **read_options)
                header = _read_original_csv_header(source, encoding)
                normalized = [str(column).strip() for column in header]
                if len(normalized) == len(dataframe.columns) and (
                    len(normalized) != len(set(normalized))
                ):
                    dataframe.columns = normalized
                return dataframe
            except UnicodeDecodeError as error:
                last_error = error
            except csv.Error as error:
                raise ValueError('The CSV header could not be read.') from error
        raise ValueError('The CSV encoding is not supported.') from last_error


    def _comparison_label(value, field):
        if not isinstance(value, str):
            raise ValueError(f'{field} is invalid.')
        value = value.strip()
        if (
            not value
            or len(value) > 80
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError(f'{field} is invalid.')
        return value


    def _comparison_filename(value, field):
        if not isinstance(value, str):
            raise ValueError(f'{field} is invalid.')
        normalized = secure_filename(value)
        if (
            not normalized
            or normalized != value
            or len(value) > 255
            or Path(value).suffix.lower() != '.csv'
        ):
            raise ValueError(f'{field} is invalid.')
        return value


    def _cleanup_comparison_inputs(paths):
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                current_app.logger.warning(
                    'Comparison input cleanup failed',
                    extra={'event': 'vibedash_comparison_input_cleanup_failed'},
                )


    def _save_comparison_upload(upload, path):
        """Create an upload without following a pre-existing symlink."""
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, 'wb') as destination:
                descriptor = None
                upload.save(destination)
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            path.unlink(missing_ok=True)
            raise


    def _comparison_input_path(stored_filename):
        if not isinstance(stored_filename, str) or not re.fullmatch(
            r'vibedash-[0-9a-f]{32}\.csv', stored_filename
        ):
            raise ValueError('The comparison input reference is invalid.')
        upload_directory = Path(current_app.config['UPLOAD_FOLDER'])
        path = upload_directory / stored_filename
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError('The comparison input is unavailable.')
        return path


    def _comparison_payload_paths(payload):
        """Resolve only the two generated comparison input references."""
        if not isinstance(payload, dict) or payload.get('analysis_kind') != 'period_comparison':
            return []
        paths = []
        for key in ('baseline', 'current'):
            item = payload.get(key)
            if not isinstance(item, dict):
                continue
            stored_filename = item.get('stored_filename')
            try:
                paths.append(_comparison_input_path(stored_filename))
            except (FileNotFoundError, ValueError):
                # Cleanup should also be attempted for a missing input and
                # must never trust arbitrary paths from a persisted payload.
                if isinstance(stored_filename, str) and re.fullmatch(
                    r'vibedash-[0-9a-f]{32}\.csv', stored_filename
                ):
                    paths.append(Path(current_app.config['UPLOAD_FOLDER']) / stored_filename)
        return paths


    def _build_period_comparison_session(
        payload, *, run_id=None, scope_id=_UNSET_SCOPE
    ):
        """Build and persist an aggregate-only two-period comparison."""
        if not isinstance(payload, dict) or payload.get('analysis_kind') != 'period_comparison':
            raise ValueError('The comparison request is invalid.')
        baseline = payload.get('baseline')
        current = payload.get('current')
        if not isinstance(baseline, dict) or not isinstance(current, dict):
            raise ValueError('The comparison inputs are invalid.')
        paths = []
        try:
            if baseline.get('stored_filename') == current.get('stored_filename'):
                raise ValueError('The comparison inputs must be distinct.')
            # Register both deterministic paths before checking existence so a
            # missing first input cannot strand the second upload on a worker
            # failure/restart.
            stored_names = (
                baseline.get('stored_filename'),
                current.get('stored_filename'),
            )
            for stored_name in stored_names:
                if not isinstance(stored_name, str) or not re.fullmatch(
                    r'vibedash-[0-9a-f]{32}\.csv', stored_name
                ):
                    raise ValueError('The comparison input reference is invalid.')
            upload_directory = Path(current_app.config['UPLOAD_FOLDER'])
            paths.extend(upload_directory / stored_name for stored_name in stored_names)
            baseline_path, current_path = paths
            for path in paths:
                if path.is_symlink() or not path.is_file():
                    raise FileNotFoundError('The comparison input is unavailable.')
            baseline_label = _comparison_label(baseline.get('label'), 'baseline_label')
            current_label = _comparison_label(current.get('label'), 'current_label')
            if baseline_label.casefold() == current_label.casefold():
                raise ValueError('The comparison labels must be distinct.')

            # nrows=100001 is intentional: one extra row is enough to reject a
            # file over the hard 100,000-row contract without loading it all.
            baseline_df = _load_vibedash_csv(
                baseline_path,
                nrows=min(COMPARISON_MAX_ROWS, COMPARISON_MAX_TOTAL_ROWS) + 1,
                max_columns=COMPARISON_MAX_COLUMNS,
            )
            if len(baseline_df) > COMPARISON_MAX_ROWS or len(baseline_df) > COMPARISON_MAX_TOTAL_ROWS:
                raise ValueError('The comparison inputs exceed the row limit.')
            remaining_rows = COMPARISON_MAX_TOTAL_ROWS - len(baseline_df)
            remaining_columns = COMPARISON_MAX_TOTAL_COLUMNS - baseline_df.shape[1]
            if remaining_columns < 1:
                raise ValueError('The comparison inputs exceed the combined column limit.')
            current_df = _load_vibedash_csv(
                current_path,
                nrows=min(COMPARISON_MAX_ROWS, remaining_rows) + 1,
                max_columns=min(COMPARISON_MAX_COLUMNS, remaining_columns),
            )
            if len(baseline_df) > COMPARISON_MAX_ROWS or len(current_df) > COMPARISON_MAX_ROWS:
                raise ValueError('The comparison inputs exceed the row limit.')
            if len(baseline_df) + len(current_df) > COMPARISON_MAX_TOTAL_ROWS:
                raise ValueError('The comparison inputs exceed the combined row limit.')
            if baseline_df.shape[1] + current_df.shape[1] > COMPARISON_MAX_TOTAL_COLUMNS:
                raise ValueError('The comparison inputs exceed the combined column limit.')
            combined_memory = sum(
                int(frame.memory_usage(index=True, deep=True).sum())
                for frame in (baseline_df, current_df)
            )
            if combined_memory > COMPARISON_MAX_MEMORY_BYTES:
                raise ValueError('The comparison inputs exceed the combined memory limit.')

            report = build_period_comparison(
                baseline_df,
                current_df,
                baseline_label=baseline_label,
                current_label=current_label,
            )
            baseline_filename = _comparison_filename(
                baseline.get('filename'), 'baseline_filename'
            )
            current_filename = _comparison_filename(
                current.get('filename'), 'current_filename'
            )
            manifest = build_period_comparison_manifest(
                baseline_path=baseline_path,
                current_path=current_path,
                baseline_dataframe=baseline_df,
                current_dataframe=current_df,
                baseline_filename=baseline_filename,
                current_filename=current_filename,
                baseline_label=baseline_label,
                current_label=current_label,
                comparison_result=report,
                run_id=run_id,
                engine_version=current_app.config['SERVICE_VERSION'],
                retention_hours=current_app.config['VIBEDASH_RETENTION_HOURS'],
            )
            session_id = str(uuid.uuid4())
            session_data = {
                'analysis_kind': 'period_comparison',
                'comparison': report,
                'report': report,
                'comparison_result': report,
                'baseline_filename': baseline_filename,
                'current_filename': current_filename,
                'baseline_label': baseline_label,
                'current_label': current_label,
                'filenames': {
                    'baseline': baseline_filename,
                    'current': current_filename,
                },
                'labels': {
                    'baseline': baseline_label,
                    'current': current_label,
                },
                'filename': f'{baseline_filename} vs {current_filename}',
                'audit_manifest': manifest,
            }
            owner_scope = _session_persist_scope(scope_id)
            if not save_session_data(
                session_id,
                session_data,
                owner_id=owner_scope,
            ):
                raise RuntimeError('The comparison session could not be persisted.')
            return {'session_id': session_id, 'manifest': manifest}
        finally:
            # Uploads are worker inputs only. Retention cleanup remains the
            # fallback for an interrupted process where this finally is skipped.
            _cleanup_comparison_inputs(paths)


    def _read_original_csv_header(source, encoding):
        # ``read(64 KiB)`` is not a valid CSV-header parser: a single quoted
        # field or a wide header can legally exceed that buffer and would make
        # the column-limit check see only a partial row. Parse the complete
        # first record instead, while bounding csv's field parser to the same
        # request-size contract used by comparison uploads.
        # ``csv.field_size_limit`` is process-global. Keep the enlarged limit
        # scoped to this parser call and serialize callers so another request
        # never observes a permanently changed parser setting.
        with _CSV_HEADER_LOCK:
            previous_limit = csv.field_size_limit()
            csv.field_size_limit(
                max(previous_limit, COMPARISON_MAX_REQUEST_BYTES)
            )
            try:
                if hasattr(source, 'seek'):
                    source.seek(0)
                    if isinstance(source, io.TextIOBase):
                        try:
                            return next(csv.reader(source), [])
                        finally:
                            source.seek(0)
                    wrapper = io.TextIOWrapper(source, encoding=encoding, newline='')
                    try:
                        return next(csv.reader(wrapper), [])
                    finally:
                        # Keep the caller's file object usable; callers seek it
                        # before handing it to pandas and after this returns.
                        wrapper.detach()
                        source.seek(0)
                with Path(source).open('r', encoding=encoding, newline='') as handle:
                    return next(csv.reader(handle), [])
            finally:
                csv.field_size_limit(previous_limit)


    def _build_dashboard_session(
        payload, *, run_id=None, scope_id=_UNSET_SCOPE
    ):
        """Build and persist one dashboard through the shared analysis path."""
        stored_filename = payload.get('stored_filename', '')
        if not isinstance(stored_filename, str) or not re.fullmatch(
            r'vibedash-[0-9a-f]{32}\.csv', stored_filename
        ):
            raise ValueError('The analysis input reference is invalid.')

        upload_path = Path(current_app.config['UPLOAD_FOLDER']) / stored_filename
        if upload_path.is_symlink() or not upload_path.is_file():
            raise FileNotFoundError('The analysis input is unavailable.')

        prompt = payload.get('prompt', '')
        filename = payload.get('filename', 'dataset.csv')
        demo_dataset = payload.get('demo_dataset', '')
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > 4000
        ):
            raise ValueError('The analysis request is invalid.')
        if not isinstance(filename, str) or not filename:
            raise ValueError('The dataset name is invalid.')

        source_df = _load_vibedash_csv(upload_path)
        if source_df.empty:
            raise ValueError('The dataset does not contain any rows.')
        source_row_count = len(source_df)
        max_rows = current_app.config['MAX_ROWS_PREVIEW']
        truncated = source_row_count > max_rows
        df = source_df.head(max_rows) if truncated else source_df
        readiness = DatasetReadinessEngine(df).assess()
        if not readiness['analysis_allowed']:
            raise DatasetReadinessBlocked(readiness)

        if demo_dataset:
            if demo_dataset != DEMO_DATASET_ID:
                raise ValueError('The demonstration dataset is invalid.')
            viz_spec = create_saas_demo_viz_spec()
        else:
            viz_spec = parse_prompt_to_viz_spec(prompt, list(df.columns))
        dashboard_data = bridge_generate_dashboard_data(df, viz_spec)
        dashboard_data['readiness'] = readiness
        dashboard_data['decision_brief'] = build_decision_brief(
            dashboard_data,
            readiness,
        )

        viz_spec_data = viz_spec.model_dump()
        audit_manifest = build_audit_manifest(
            dataset_path=upload_path,
            dataframe=df,
            source_row_count=source_row_count,
            truncated=truncated,
            source_name=filename,
            prompt=prompt,
            demo_dataset=demo_dataset,
            run_id=run_id,
            viz_spec=viz_spec_data,
            dashboard_data=dashboard_data,
            engine_version=current_app.config['SERVICE_VERSION'],
            retention_hours=current_app.config['VIBEDASH_RETENTION_HOURS'],
        )
        session_id = str(uuid.uuid4())
        session_data = {
            'viz_spec': viz_spec_data,
            'dashboard_data': dashboard_data,
            'filename': filename,
            'stored_filename': stored_filename,
            'prompt': prompt,
            'df_shape': df.shape,
            'file_path': str(upload_path),
            'audit_manifest': audit_manifest,
        }
        owner_scope = _session_persist_scope(scope_id)
        if not save_session_data(
            session_id,
            session_data,
            owner_id=owner_scope,
        ):
            raise RuntimeError('VibeDash session could not be persisted.')
        return {
            'session_id': session_id,
            'session_data': session_data,
            'viz_spec': viz_spec,
            'truncated': truncated,
            'manifest': audit_manifest,
        }


    def _process_analysis_job(job):
        scope_id = job.get('scope_id')
        # Workers have no request session to fall back to.  Validate the
        # durable owner before touching inputs or writing a retained record.
        _session_persist_scope(scope_id)
        if (job.get('payload') or {}).get('analysis_kind') == 'period_comparison':
            return _build_period_comparison_session(
                job.get('payload', {}),
                run_id=job['id'],
                scope_id=scope_id,
            )
        result = _build_dashboard_session(
            job.get('payload', {}),
            run_id=job['id'],
            scope_id=scope_id,
        )
        return {
            'session_id': result['session_id'],
            'manifest': result['manifest'],
        }


    def _history_entry(job):
        manifest = job.get('manifest') or {}
        payload = job.get('payload') or {}
        manifest_dataset = manifest.get('dataset') or {}
        manifest_inputs = manifest.get('inputs') or {}
        manifest_specification = manifest.get('specification') or {}
        manifest_evidence = manifest.get('evidence') or {}
        created_at = job.get('created_at')
        started_at = job.get('started_at')
        completed_at = job.get('completed_at')

        def parse_timestamp(value):
            if not value:
                return None
            try:
                return datetime.fromisoformat(value).astimezone(timezone.utc)
            except (TypeError, ValueError):
                return None

        created = parse_timestamp(created_at)
        started = parse_timestamp(started_at)
        completed = parse_timestamp(completed_at)
        duration_seconds = None
        if started and completed:
            duration_seconds = max(0.0, (completed - started).total_seconds())

        if payload.get('analysis_kind') == 'period_comparison':
            baseline = payload.get('baseline') or {}
            current = payload.get('current') or {}
            baseline_manifest = manifest_inputs.get('baseline') or {}
            current_manifest = manifest_inputs.get('current') or {}
            return {
                'id': job['id'],
                'short_id': job['id'][:10],
                'status': job['status'],
                'filename': (
                    f"{baseline.get('filename') or baseline_manifest.get('filename') or 'baseline.csv'}"
                    f" vs {current.get('filename') or current_manifest.get('filename') or 'current.csv'}"
                ),
                'prompt': 'Period comparison',
                'is_demo': False,
                'title': (
                    f"{baseline.get('label') or baseline_manifest.get('label') or 'Baseline'}"
                    f" vs {current.get('label') or current_manifest.get('label') or 'Current'}"
                ),
                'created_at': (
                    created.strftime('%Y-%m-%d %H:%M UTC')
                    if created else 'Unknown time'
                ),
                'duration_seconds': duration_seconds,
                'manifest': manifest,
                'audit_available': bool(
                    (baseline_manifest.get('content_sha256'))
                    and (current_manifest.get('content_sha256'))
                ),
                'analyzed_rows': (
                    f"{baseline_manifest.get('analyzed_rows', 0)} / "
                    f"{current_manifest.get('analyzed_rows', 0)}"
                ) if manifest_inputs else None,
                'column_count': (
                    f"{baseline_manifest.get('column_count', 0)} / "
                    f"{current_manifest.get('column_count', 0)}"
                ) if manifest_inputs else None,
                'insight_count': None,
                'statistical_test_count': (
                    (manifest.get('comparison') or {}).get('tested_metric_count')
                    or ((manifest.get('comparison') or {}).get('counts') or {}).get(
                        'tested_metrics'
                    )
                ),
                'dataset_sha256': None,
                'result_url': (
                    url_for('vibedash.analysis_job_result', job_id=job['id'])
                    if job['status'] == 'completed' and job.get('session_id')
                    else None
                ),
                'manifest_url': (
                    url_for('vibedash.analysis_job_manifest', job_id=job['id'])
                    if job['status'] == 'completed' and manifest
                    else None
                ),
            }

        return {
            'id': job['id'],
            'short_id': job['id'][:10],
            'status': job['status'],
            'filename': payload.get('filename') or 'dataset.csv',
            'prompt': payload.get('prompt') or 'Analysis request',
            'is_demo': bool(payload.get('demo_dataset')),
            'title': (
                manifest_specification.get('title')
                or payload.get('filename')
                or 'Analysis'
            ),
            'created_at': (
                created.strftime('%Y-%m-%d %H:%M UTC')
                if created else 'Unknown time'
            ),
            'duration_seconds': duration_seconds,
            'manifest': manifest,
            'audit_available': bool(manifest_dataset.get('content_sha256')),
            'analyzed_rows': manifest_dataset.get('analyzed_rows'),
            'column_count': manifest_dataset.get('column_count'),
            'insight_count': manifest_evidence.get('insight_count'),
            'statistical_test_count': manifest_evidence.get(
                'statistical_test_count'
            ),
            'dataset_sha256': manifest_dataset.get('content_sha256'),
            'result_url': (
                url_for('vibedash.analysis_job_result', job_id=job['id'])
                if job['status'] == 'completed' and job.get('session_id')
                else None
            ),
            'manifest_url': (
                url_for('vibedash.analysis_job_manifest', job_id=job['id'])
                if job['status'] == 'completed' and manifest
                else None
            ),
        }


    @vibedash_bp.before_request
    def cleanup_runtime_artifacts():
        """Apply the configured retention window before serving VibeDash."""
        result = cleanup_expired_artifacts(
            current_app.config['UPLOAD_FOLDER'],
            current_app.config['VIBEDASH_RETENTION_HOURS'],
        )
        removed = sum(
            result[key]
            for key in (
                'removed_sessions',
                'removed_uploads',
                'removed_exports',
            )
        )
        if removed:
            current_app.logger.info(
                "Expired VibeDash artifacts removed",
                extra={
                    "event": "vibedash_retention_cleanup",
                    **result,
                },
            )
        if result['errors']:
            current_app.logger.warning(
                "VibeDash artifact cleanup completed with errors",
                extra={
                    "event": "vibedash_retention_cleanup_error",
                    **result,
                },
            )

        store = _analysis_job_store()
        stale_job_records = store.fail_stale_running_jobs(
            current_app.config['VIBEDASH_JOB_TIMEOUT_SECONDS']
        )
        # A process restart can leave a comparison job in running state while
        # its worker is gone.  Remove both temporary inputs as part of the
        # stale-job transition instead of waiting for retention cleanup.
        for stale_job in stale_job_records:
            _cleanup_comparison_inputs(
                _comparison_payload_paths(stale_job.get('payload'))
            )
        removed_jobs = store.purge_terminal(
            current_app.config['VIBEDASH_RETENTION_HOURS']
        )
        removed_decisions = _decision_case_store().purge_terminal(
            current_app.config['VIBEDASH_DECISION_RETENTION_DAYS']
        )
        if stale_job_records:
            current_app.logger.warning(
                "Interrupted VibeDash jobs marked as failed",
                extra={"event": "vibedash_jobs_interrupted"},
            )
        if removed_jobs:
            current_app.logger.info(
                "Expired VibeDash jobs removed",
                extra={"event": "vibedash_jobs_retention_cleanup"},
            )
        if removed_decisions:
            current_app.logger.info(
                "Expired closed decision cases removed",
                extra={"event": "vibedash_decisions_retention_cleanup"},
            )

        application = current_app._get_current_object()
        jobs = store.list_for_scope(
            _analysis_scope_id(),
            limit=MAX_HISTORY_JOBS,
        )
        for job in jobs:
            if job['status'] == 'queued':
                analysis_job_dispatcher.submit(
                    application,
                    job['id'],
                    _process_analysis_job,
                )


    @vibedash_bp.route('/')
    def index():
        """Главная страница VibeDash"""
        # Проверяем доступность Ollama
        ollama_available = is_ollama_available()
        
        # Предустановленные промпты
        preset_prompts = {
            "saas_review": "Review weekly SaaS performance: compare revenue, customer count and churn across periods and available segments. Check data quality and show supporting numbers and limitations. Identify what needs investigation before making a business decision.",
            "sales": "Sales dashboard for a monthly CSV: main KPIs (Total Sales, Orders, AOV), top 10 categories, revenue trend by week, bar by region, filter by region, highlight YoY growth.",
            "finance": "Финансы: сумма дохода и расходов, дельта, тренд по неделям, топ-категории расходов, фильтр по отделу, комментарий по выбросам.",
            "real_estate": "Real-estate listing analysis: median price by city, distribution by rooms, time trend by posting date (W), filter by city, show top 10 streets by average price."
        }
        
        # Resolve account state before minting VibeDash CSRF tokens.  Invalid
        # or deleted account identities rotate those keys and must not leave
        # stale tokens in the rendered forms.
        account = _current_account()
        return render_template('vibedash_landing.html',
                             preset_prompts=preset_prompts,
                             demo_prompt=DEMO_PROMPT,
                             retention_hours=current_app.config['VIBEDASH_RETENTION_HOURS'],
                             pilot_csrf_token=_decision_csrf_token(),
                             auth_csrf_token=_auth_csrf_token(),
                             account=account,
                             ollama_available=ollama_available)


    @vibedash_bp.route('/register', methods=['GET', 'POST'])
    def register():
        """Create a pilot account and establish its durable analysis scope."""
        if request.method == 'GET':
            return _render_auth('register')
        if not _valid_auth_csrf_token(request.form.get('csrf_token')):
            return _render_auth('register', error='This form has expired. Please try again.', status=400)
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        store = _account_store()
        if store is None:
            return _render_auth('register', error='Account storage is temporarily unavailable.', status=503)
        try:
            account = store.register(
                email,
                password,
                credential_secret=current_app.secret_key,
            )
        except AccountValidationError as error:
            return _render_auth('register', error=str(error), status=400)
        except Exception:
            current_app.logger.warning(
                'VibeDash account registration failed',
                extra={'event': 'vibedash_account_registration_failed'},
            )
            return _render_auth('register', error='Account storage is temporarily unavailable.', status=503)
        if account is None:
            return _render_auth(
                'register',
                error='Unable to create an account with those details.',
                status=400,
            )
        if not _establish_account(account):
            return _render_auth(
                'register', error='Account storage is temporarily unavailable.', status=503
            )
        flash('Your pilot account is ready.', 'success')
        return redirect(url_for('vibedash.index'))


    @vibedash_bp.route('/login', methods=['GET', 'POST'])
    def login():
        """Authenticate a pilot account without disclosing account existence."""
        if request.method == 'GET':
            return _render_auth('login')
        if not _valid_auth_csrf_token(request.form.get('csrf_token')):
            return _render_auth('login', error='This form has expired. Please try again.', status=400)
        store = _account_store()
        if store is None:
            return _render_auth('login', error='Login is temporarily unavailable.', status=503)
        try:
            account = store.authenticate(
                request.form.get('email', ''),
                request.form.get('password', ''),
                credential_secret=current_app.secret_key,
            )
        except Exception:
            current_app.logger.warning(
                'VibeDash account login failed',
                extra={'event': 'vibedash_account_login_failed'},
            )
            return _render_auth('login', error='Login is temporarily unavailable.', status=503)
        if account is None:
            return _render_auth('login', error='Email or password is incorrect.', status=401)
        if not _establish_account(account):
            return _render_auth(
                'login', error='Login is temporarily unavailable.', status=503
            )
        flash('Welcome back.', 'success')
        return redirect(url_for('vibedash.index'))


    @vibedash_bp.post('/logout')
    def logout():
        """End the account session; a fresh guest scope is created later."""
        if not _valid_auth_csrf_token(request.form.get('csrf_token')):
            return 'This form has expired. Please try again.', 400
        _rotate_vibedash_identity()
        _auth_csrf_token()
        _decision_csrf_token()
        flash('You are signed out. Guest analyses remain browser-scoped.', 'success')
        return redirect(url_for('vibedash.index'))


    def _render_account_settings(account, *, error=None, success=None, status=200):
        return render_template(
            'vibedash_account.html',
            account=account,
            auth_csrf_token=_auth_csrf_token(),
            error=error,
            success=success,
        ), status


    @vibedash_bp.route('/account', methods=['GET'])
    def account_settings():
        """Show the authenticated pilot account settings."""
        account = _current_account()
        if account is None:
            return redirect(url_for('vibedash.login'))
        return _render_account_settings(account)


    @vibedash_bp.route('/account/password', methods=['POST'])
    def account_password():
        """Change the authenticated account password with CSRF protection."""
        account = _current_account()
        if account is None:
            return redirect(url_for('vibedash.login'))
        if not _valid_auth_csrf_token(request.form.get('csrf_token')):
            return _render_account_settings(
                account,
                error='This form has expired. Please try again.',
                status=400,
            )

        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirmation = request.form.get('new_password_confirmation', '')
        if new_password != confirmation:
            return _render_account_settings(
                account,
                error='New passwords do not match.',
                status=400,
            )
        store = _account_store()
        if store is None:
            return _render_account_settings(
                account,
                error='Password change is temporarily unavailable.',
                status=503,
            )
        try:
            changed = store.change_password(
                account['id'],
                current_password,
                new_password,
                credential_secret=current_app.secret_key,
            )
        except AccountValidationError as error:
            # Password policy failures are intentionally explicit; current
            # password and lockout failures below remain generic.
            return _render_account_settings(account, error=str(error), status=400)
        except Exception:
            current_app.logger.warning(
                'VibeDash password change failed',
                extra={'event': 'vibedash_account_password_change_failed'},
            )
            return _render_account_settings(
                account,
                error='Password change is temporarily unavailable.',
                status=503,
            )
        if changed is None:
            return _render_account_settings(
                account,
                error='The current password or account state is invalid.',
                status=400,
            )
        if not _establish_account(changed):
            # The password hash has already rotated, so the old credential is
            # no longer usable.  Do not render the old account record with a
            # stale identity; send the browser through a clean login instead.
            flash('Your password was changed. Please sign in again.', 'success')
            return redirect(url_for('vibedash.login'))
        flash('Your password was changed. Other signed-in browsers must sign in again.', 'success')
        return redirect(url_for('vibedash.account_settings'))


    @vibedash_bp.get('/history')
    def analysis_history():
        """Show recent analysis jobs owned by this signed VibeDash scope."""
        jobs = _analysis_job_store().list_for_scope(
            _analysis_scope_id(),
            limit=MAX_HISTORY_JOBS,
        )
        return render_template(
            'vibedash_history.html',
            jobs=[_history_entry(job) for job in jobs],
            retention_hours=current_app.config['VIBEDASH_RETENTION_HOURS'],
        )


    @vibedash_bp.get('/decisions')
    def decision_cases():
        """Show evidence-linked decisions for this signed VibeDash scope."""
        list_limit = min(
            current_app.config['VIBEDASH_MAX_DECISION_CASES_PER_SCOPE'],
            MAX_DECISION_CASES,
        )
        cases = _decision_case_store().list_for_scope(
            _analysis_scope_id(),
            limit=list_limit,
        )
        return render_template(
            'vibedash_decisions.html',
            decision_cases=[_decision_case_view(case) for case in cases],
            selected_case=None,
            csrf_token=_decision_csrf_token(),
            retention_days=current_app.config[
                'VIBEDASH_DECISION_RETENTION_DAYS'
            ],
        )


    @vibedash_bp.get('/decisions/<case_id>')
    def decision_case_detail(case_id):
        """Open one decision case owned by this signed VibeDash scope."""
        if not DECISION_CASE_ID_PATTERN.fullmatch(case_id):
            return jsonify({'error': 'Decision case not found.'}), 404
        scope_id = _analysis_scope_id()
        decision_case = _decision_case_store().get(case_id, scope_id)
        if decision_case is None:
            return jsonify({'error': 'Decision case not found.'}), 404
        list_limit = min(
            current_app.config['VIBEDASH_MAX_DECISION_CASES_PER_SCOPE'],
            MAX_DECISION_CASES,
        )
        cases = _decision_case_store().list_for_scope(
            scope_id,
            limit=list_limit,
        )
        return render_template(
            'vibedash_decisions.html',
            decision_cases=[_decision_case_view(case) for case in cases],
            selected_case=_decision_case_view(decision_case),
            csrf_token=_decision_csrf_token(),
            retention_days=current_app.config[
                'VIBEDASH_DECISION_RETENTION_DAYS'
            ],
        )


    @vibedash_bp.post('/readiness')
    def dataset_readiness():
        """Assess an uploaded CSV without retaining its rows."""
        uploaded_file = request.files.get('datafile')
        if uploaded_file is None or not uploaded_file.filename:
            return jsonify({'error': 'A CSV file is required.'}), 400
        filename = secure_filename(uploaded_file.filename)
        if not filename or Path(filename).suffix.lower() != '.csv':
            return jsonify({'error': 'VibeDash currently accepts CSV files only.'}), 400
        try:
            dataframe = _load_vibedash_csv(uploaded_file.stream)
            report = DatasetReadinessEngine(dataframe).assess()
        except Exception:
            current_app.logger.info(
                "VibeDash readiness input rejected",
                extra={"event": "vibedash_readiness_rejected"},
            )
            return jsonify({
                'error': 'The CSV could not be read. Check its delimiter, header, and encoding.'
            }), 422

        current_app.logger.info(
            "VibeDash readiness completed",
            extra={"event": "vibedash_readiness_completed"},
        )
        return jsonify({'readiness': report})


    @vibedash_bp.route('/preview', methods=['POST'])
    def preview():
        """Предварительный просмотр дашборда"""
        try:
            current_app.logger.info(
                "VibeDash preview started",
                extra={"event": "vibedash_preview_started"},
            )

            prompt = request.form.get('prompt', '').strip()
            if not prompt:
                current_app.logger.warning(
                    "VibeDash preview rejected: prompt missing",
                    extra={"event": "vibedash_preview_rejected"},
                )
                flash('Please enter a dashboard description!', 'error')
                return redirect(url_for('vibedash.index'))

            demo_dataset = request.form.get('demo_dataset', '').strip()
            upload_directory = Path(current_app.config['UPLOAD_FOLDER'])
            upload_directory.mkdir(parents=True, exist_ok=True)
            stored_filename = f"vibedash-{uuid.uuid4().hex}.csv"
            upload_path = upload_directory / stored_filename

            if demo_dataset:
                if demo_dataset != DEMO_DATASET_ID:
                    flash('Unknown demonstration dataset.', 'error')
                    return redirect(url_for('vibedash.index'))
                filename = DEMO_FILENAME
                df = create_saas_growth_demo()
                df.to_csv(upload_path, index=False)
                current_app.logger.info(
                    "Built-in demonstration dataset generated",
                    extra={"event": "vibedash_demo_generated"},
                )
            else:
                file = request.files.get('datafile')
                if file is None or not file.filename:
                    current_app.logger.warning(
                        "VibeDash preview rejected: file missing",
                        extra={"event": "vibedash_preview_rejected"},
                    )
                    flash('No file selected!', 'error')
                    return redirect(url_for('vibedash.index'))

                filename = secure_filename(file.filename)
                if not filename or Path(filename).suffix.lower() != '.csv':
                    flash('VibeDash currently accepts CSV files only.', 'error')
                    return redirect(url_for('vibedash.index'))
                file.save(upload_path)
            current_app.logger.info(
                "VibeDash dataset loaded",
                extra={"event": "vibedash_dataset_loaded"},
            )

            analysis_result = _build_dashboard_session({
                'stored_filename': stored_filename,
                'filename': filename,
                'prompt': prompt,
                'demo_dataset': demo_dataset,
            })
            if analysis_result['truncated']:
                max_rows = current_app.config['MAX_ROWS_PREVIEW']
                flash(f'Data limited to {max_rows:,} rows for preview', 'info')
                current_app.logger.info(
                    "VibeDash dataset truncated to preview limit",
                    extra={"event": "vibedash_dataset_truncated"},
                )

            # Проверяем размер файла
            file_size_mb = os.path.getsize(upload_path) / (1024 * 1024)
            if file_size_mb > 100:  # Больше 100MB
                flash(f'Large file detected ({file_size_mb:.1f}MB). Processing may take longer...', 'warning')
                current_app.logger.warning(
                    "VibeDash large dataset detected",
                    extra={"event": "vibedash_large_dataset"},
                )

            # Рендерим предварительный просмотр
            current_app.logger.info(
                "VibeDash preview completed",
                extra={"event": "vibedash_preview_completed"},
            )
            return render_template('vibedash_evidence.html',
                                 session_id=analysis_result['session_id'],
                                 viz_spec=analysis_result['viz_spec'],
                                 dashboard_data=analysis_result['session_data']['dashboard_data'],
                                 filename=filename,
                                 prompt=prompt,
                                 audit_manifest=analysis_result['manifest'],
                                 manifest_url=None)
        
        except DatasetReadinessBlocked as error:
            upload_path.unlink(missing_ok=True)
            current_app.logger.info(
                "VibeDash preview blocked by dataset readiness",
                extra={"event": "vibedash_readiness_blocked"},
            )
            flash(error.report['summary'], 'error')
            return redirect(url_for('vibedash.index'))
        except Exception:
            current_app.logger.exception(
                "VibeDash preview failed",
                extra={"event": "vibedash_preview_failed"},
            )
            flash(
                'Could not create the dashboard. Check the CSV and analysis request.',
                'error',
            )
            return redirect(url_for('vibedash.index'))


    @vibedash_bp.post('/jobs')
    def create_analysis_job():
        """Queue one bounded VibeDash analysis and return its status URL."""
        pilot_opt_in = request.form.get('pilot_metrics') == 'yes'
        if pilot_opt_in and not _valid_decision_csrf_token(request.form.get('csrf_token')):
            return jsonify({'error': 'Refresh the page before opting into pilot measurement.'}), 400
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Analysis description is required.'}), 400
        if len(prompt) > 4000:
            return jsonify({'error': 'Analysis description is too long.'}), 400

        demo_dataset = request.form.get('demo_dataset', '').strip()
        upload_directory = Path(current_app.config['UPLOAD_FOLDER'])
        upload_directory.mkdir(parents=True, exist_ok=True)
        stored_filename = f"vibedash-{uuid.uuid4().hex}.csv"
        upload_path = upload_directory / stored_filename

        if demo_dataset:
            if demo_dataset != DEMO_DATASET_ID:
                return jsonify({'error': 'Unknown demonstration dataset.'}), 400
            filename = DEMO_FILENAME
            try:
                create_saas_growth_demo().to_csv(upload_path, index=False)
            except Exception:
                upload_path.unlink(missing_ok=True)
                current_app.logger.exception(
                    "VibeDash job input could not be prepared",
                    extra={"event": "vibedash_job_input_failed"},
                )
                return jsonify({'error': 'The analysis input could not be prepared.'}), 500
        else:
            uploaded_file = request.files.get('datafile')
            if uploaded_file is None or not uploaded_file.filename:
                return jsonify({'error': 'A CSV file is required.'}), 400
            filename = secure_filename(uploaded_file.filename)
            if not filename or Path(filename).suffix.lower() != '.csv':
                return jsonify({'error': 'VibeDash currently accepts CSV files only.'}), 400
            try:
                uploaded_file.save(upload_path)
            except Exception:
                upload_path.unlink(missing_ok=True)
                current_app.logger.exception(
                    "VibeDash job upload could not be stored",
                    extra={"event": "vibedash_job_input_failed"},
                )
                return jsonify({'error': 'The uploaded file could not be stored.'}), 500

        try:
            readiness = DatasetReadinessEngine(
                _load_vibedash_csv(upload_path)
            ).assess()
        except Exception:
            upload_path.unlink(missing_ok=True)
            return jsonify({
                'error': 'The CSV could not be read. Check its delimiter, header, and encoding.'
            }), 422
        if not readiness['analysis_allowed']:
            upload_path.unlink(missing_ok=True)
            current_app.logger.info(
                "VibeDash job blocked by dataset readiness",
                extra={"event": "vibedash_readiness_blocked"},
            )
            return jsonify({
                'error': readiness['summary'],
                'readiness': readiness,
            }), 422

        try:
            store = _analysis_job_store()
            job = store.create(
                _analysis_scope_id(),
                {
                    'stored_filename': stored_filename,
                    'filename': filename,
                    'prompt': prompt,
                    'demo_dataset': demo_dataset,
                },
                max_active_per_scope=current_app.config[
                    'VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE'
                ],
                max_active_total=current_app.config[
                    'VIBEDASH_MAX_ACTIVE_JOBS'
                ],
                pilot_scope_token=(
                    scope_token(_analysis_scope_id(), current_app.secret_key)
                    if pilot_opt_in else None
                ),
            )
        except AnalysisJobCapacityError:
            upload_path.unlink(missing_ok=True)
            return jsonify({
                'error': 'The analysis queue is busy. Please wait and try again.'
            }), 429
        except Exception:
            upload_path.unlink(missing_ok=True)
            current_app.logger.exception(
                "VibeDash analysis job could not be created",
                extra={"event": "vibedash_job_creation_failed"},
            )
            return jsonify({'error': 'The analysis job could not be created.'}), 500

        analysis_job_dispatcher.submit(
            current_app._get_current_object(),
            job['id'],
            _process_analysis_job,
        )
        current_app.logger.info(
            "VibeDash analysis job queued",
            extra={"event": "vibedash_job_queued"},
        )
        return jsonify({
            'job_id': job['id'],
            'status': job['status'],
            'readiness': readiness,
            'status_url': url_for(
                'vibedash.analysis_job_status',
                job_id=job['id'],
            ),
        }), 202


    @vibedash_bp.post('/comparisons/jobs')
    def create_period_comparison_job():
        """Queue one bounded comparison between two uploaded CSV periods."""
        if not _valid_decision_csrf_token(request.form.get('csrf_token')):
            return jsonify({'error': 'Invalid form token.'}), 400
        if (
            request.content_length is not None
            and request.content_length > COMPARISON_MAX_REQUEST_BYTES
        ):
            return jsonify({'error': 'The comparison request exceeds the upload limit.'}), 413
        baseline_upload = request.files.get('baseline_file')
        current_upload = request.files.get('current_file')
        if (
            baseline_upload is None
            or not baseline_upload.filename
            or current_upload is None
            or not current_upload.filename
        ):
            return jsonify({'error': 'Two CSV files are required.'}), 400

        try:
            baseline_label = _comparison_label(
                request.form.get('baseline_label', ''), 'baseline_label'
            )
            current_label = _comparison_label(
                request.form.get('current_label', ''), 'current_label'
            )
            if baseline_label.casefold() == current_label.casefold():
                raise ValueError('The comparison labels must be distinct.')
        except ValueError:
            return jsonify({'error': 'Comparison labels must be distinct, non-empty, and at most 80 characters.'}), 400

        filenames = [
            secure_filename(baseline_upload.filename),
            secure_filename(current_upload.filename),
        ]
        if any(
            not name
            or len(name) > 255
            or Path(name).suffix.lower() != '.csv'
            for name in filenames
        ):
            return jsonify({'error': 'VibeDash currently accepts CSV files only.'}), 400

        upload_directory = Path(current_app.config['UPLOAD_FOLDER'])
        upload_directory.mkdir(parents=True, exist_ok=True)
        stored_filenames = [
            f'vibedash-{uuid.uuid4().hex}.csv',
            f'vibedash-{uuid.uuid4().hex}.csv',
        ]
        paths = [upload_directory / name for name in stored_filenames]
        keep_uploads = False
        try:
            # Save sequentially so a failure in the second stream can still
            # remove the first file deterministically.
            _save_comparison_upload(baseline_upload, paths[0])
            _save_comparison_upload(current_upload, paths[1])
            if sum(path.stat().st_size for path in paths) > COMPARISON_MAX_REQUEST_BYTES:
                return jsonify({'error': 'The comparison inputs exceed the upload limit.'}), 413
            baseline_df = _load_vibedash_csv(
                paths[0],
                nrows=min(COMPARISON_MAX_ROWS, COMPARISON_MAX_TOTAL_ROWS) + 1,
                max_columns=COMPARISON_MAX_COLUMNS,
            )
            if len(baseline_df) > COMPARISON_MAX_ROWS or len(baseline_df) > COMPARISON_MAX_TOTAL_ROWS:
                return jsonify({'error': 'The comparison inputs exceed the row limit.'}), 422
            remaining_rows = COMPARISON_MAX_TOTAL_ROWS - len(baseline_df)
            remaining_columns = COMPARISON_MAX_TOTAL_COLUMNS - baseline_df.shape[1]
            if remaining_columns < 1:
                return jsonify({'error': 'The comparison inputs exceed the combined column limit.'}), 422
            current_df = _load_vibedash_csv(
                paths[1],
                nrows=min(COMPARISON_MAX_ROWS, remaining_rows) + 1,
                max_columns=min(COMPARISON_MAX_COLUMNS, remaining_columns),
            )
            if len(baseline_df) > COMPARISON_MAX_ROWS or len(current_df) > COMPARISON_MAX_ROWS:
                return jsonify({'error': 'The comparison inputs exceed the row limit.'}), 422
            if len(baseline_df) + len(current_df) > COMPARISON_MAX_TOTAL_ROWS:
                return jsonify({'error': 'The comparison inputs exceed the combined row limit.'}), 422
            if baseline_df.shape[1] + current_df.shape[1] > COMPARISON_MAX_TOTAL_COLUMNS:
                return jsonify({'error': 'The comparison inputs exceed the combined column limit.'}), 422
            combined_memory = sum(
                int(frame.memory_usage(index=True, deep=True).sum())
                for frame in (baseline_df, current_df)
            )
            if combined_memory > COMPARISON_MAX_MEMORY_BYTES:
                return jsonify({'error': 'The comparison inputs exceed the combined memory limit.'}), 422
            baseline_readiness = DatasetReadinessEngine(baseline_df).assess()
            current_readiness = DatasetReadinessEngine(current_df).assess()
            readiness = {
                'baseline': baseline_readiness,
                'current': current_readiness,
            }
            if not baseline_readiness['analysis_allowed'] or not current_readiness['analysis_allowed']:
                return jsonify({
                    'error': 'One or both comparison inputs are not ready for analysis.',
                    'readiness': readiness,
                }), 422

            store = _analysis_job_store()
            job = store.create(
                _analysis_scope_id(),
                {
                    'analysis_kind': 'period_comparison',
                    'baseline': {
                        'stored_filename': stored_filenames[0],
                        'filename': filenames[0],
                        'label': baseline_label,
                    },
                    'current': {
                        'stored_filename': stored_filenames[1],
                        'filename': filenames[1],
                        'label': current_label,
                    },
                },
                max_active_per_scope=current_app.config[
                    'VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE'
                ],
                max_active_total=current_app.config['VIBEDASH_MAX_ACTIVE_JOBS'],
            )
            keep_uploads = True
        except AnalysisJobCapacityError:
            return jsonify({
                'error': 'The analysis queue is busy. Please wait and try again.'
            }), 429
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            current_app.logger.info(
                'VibeDash comparison input rejected',
                extra={'event': 'vibedash_comparison_input_rejected'},
            )
            return jsonify({
                'error': 'The CSV could not be read. Check its delimiter, header, and encoding.'
            }), 422
        except Exception:
            current_app.logger.exception(
                'VibeDash comparison job could not be created',
                extra={'event': 'vibedash_comparison_job_creation_failed'},
            )
            return jsonify({'error': 'The comparison job could not be created.'}), 500
        finally:
            if not keep_uploads:
                _cleanup_comparison_inputs(paths)

        try:
            analysis_job_dispatcher.submit(
                current_app._get_current_object(), job['id'], _process_analysis_job
            )
        except Exception:
            # A submission exception is a route-side failure: fail the durable
            # record and remove both temporary inputs before responding.
            current_app.logger.exception(
                'VibeDash comparison dispatcher submission failed',
                extra={'event': 'vibedash_comparison_dispatch_failed'},
            )
            _cleanup_comparison_inputs(paths)
            _analysis_job_store().fail(job['id'], 'dispatch_failed')
            return jsonify({'error': 'The comparison job could not be queued.'}), 500
        return jsonify({
            'job_id': job['id'],
            'status': job['status'],
            'readiness': readiness,
            'status_url': url_for('vibedash.analysis_job_status', job_id=job['id']),
        }), 202


    @vibedash_bp.get('/jobs/<job_id>')
    def analysis_job_status(job_id):
        """Return a scoped, non-sensitive representation of one job."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None:
            return jsonify({'error': 'Analysis job not found.'}), 404

        response = {
            'job_id': job['id'],
            'status': job['status'],
            'created_at': job['created_at'],
            'updated_at': job['updated_at'],
        }
        if job['status'] == 'completed':
            response['result_url'] = url_for(
                'vibedash.analysis_job_result',
                job_id=job['id'],
            )
            if job.get('manifest'):
                response['manifest_url'] = url_for(
                    'vibedash.analysis_job_manifest',
                    job_id=job['id'],
                )
            payload = job.get('payload')
            if (
                job.get('session_id')
                and isinstance(payload, Mapping)
                and payload.get('analysis_kind') == 'period_comparison'
            ):
                response['comparison_report_url'] = url_for(
                    'vibedash.analysis_job_comparison_report',
                    job_id=job['id'],
                )
        elif job['status'] == 'failed':
            response['error'] = (
                'The analysis was interrupted. Please submit it again.'
                if job['error_code'] == 'worker_interrupted'
                else 'The analysis could not be completed.'
            )
        return jsonify(response)


    @vibedash_bp.get('/jobs/<job_id>/result')
    def analysis_job_result(job_id):
        """Render a completed job only for its signed VibeDash scope."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None:
            return jsonify({'error': 'Analysis job not found.'}), 404
        if job['status'] != 'completed' or not job['session_id']:
            return jsonify({'error': 'Analysis result is not ready.'}), 409

        session_data = _load_owned_session_data(
            job['session_id'], job.get('scope_id')
        )
        if not session_data:
            return jsonify({'error': 'Analysis result is no longer available.'}), 410
        if not isinstance(session_data, Mapping):
            return jsonify({'error': 'Analysis result is unavailable for this analysis.'}), 409
        if session_data.get('analysis_kind') == 'period_comparison':
            report = session_data.get('report') or session_data.get('comparison')
            if not isinstance(report, Mapping):
                return jsonify({'error': 'Comparison report is unavailable for this analysis.'}), 409
            return render_template(
                'vibedash_comparison.html',
                report=report,
                filenames={
                    'baseline': session_data.get('baseline_filename'),
                    'current': session_data.get('current_filename'),
                },
                labels={
                    'baseline': session_data.get('baseline_label'),
                    'current': session_data.get('current_label'),
                },
                baseline_filename=session_data.get('baseline_filename'),
                current_filename=session_data.get('current_filename'),
                baseline_label=session_data.get('baseline_label'),
                current_label=session_data.get('current_label'),
                audit_manifest=session_data.get('audit_manifest'),
                manifest_url=(
                    url_for('vibedash.analysis_job_manifest', job_id=job['id'])
                    if job.get('manifest') else None
                ),
                analysis_job_id=job['id'],
                comparison_export_url=url_for(
                    'vibedash.analysis_job_comparison_report',
                    job_id=job['id'],
                ),
                decision_guidance=build_comparison_decision_guidance(report),
            )
        if (
            not isinstance(session_data.get('viz_spec'), Mapping)
            or not isinstance(session_data.get('dashboard_data'), Mapping)
            or not isinstance(session_data.get('filename'), str)
            or not isinstance(session_data.get('prompt'), str)
        ):
            return jsonify({'error': 'Analysis result is unavailable for this analysis.'}), 409
        return render_template(
            'vibedash_evidence.html',
            session_id=job['session_id'],
            viz_spec=session_data['viz_spec'],
            dashboard_data=session_data['dashboard_data'],
            filename=session_data['filename'],
            prompt=session_data['prompt'],
            audit_manifest=session_data.get('audit_manifest'),
            manifest_url=(
                url_for('vibedash.analysis_job_manifest', job_id=job['id'])
                if job.get('manifest') else None
            ),
            analysis_job_id=job['id'],
            decision_csrf_token=_decision_csrf_token(),
            pilot_feedback_available=feedback_available(
                current_app.config['VIBEDASH_JOB_STORE_PATH'], job['id'],
            ),
        )


    @vibedash_bp.post('/jobs/<job_id>/feedback')
    def pilot_feedback(job_id):
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        if not _valid_decision_csrf_token(request.form.get('csrf_token')):
            return jsonify({'error': 'Invalid form token.'}), 400
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None:
            return jsonify({'error': 'Analysis job not found.'}), 404
        if job['status'] != 'completed':
            return jsonify({'error': 'Analysis result is not ready.'}), 409
        try:
            saved = record_feedback(
                current_app.config['VIBEDASH_JOB_STORE_PATH'], job_id,
                request.form.get('usefulness'), request.form.get('blocker'),
            )
            message = (
                'Thank you. Your feedback was saved.' if saved else
                'Feedback is unavailable for this run or its measurement has been removed.'
            )
            flash(message, 'info')
        except ValueError:
            flash('Please select a usefulness rating and a listed blocker.', 'error')
        return redirect(url_for('vibedash.analysis_job_result', job_id=job_id), code=303)


    @vibedash_bp.post('/pilot/forget')
    def forget_pilot_metrics():
        if not _valid_decision_csrf_token(request.form.get('csrf_token')):
            return jsonify({'error': 'Invalid form token.'}), 400
        forget_scope(
            current_app.config['VIBEDASH_JOB_STORE_PATH'],
            scope_token(_analysis_scope_id(), current_app.secret_key),
        )
        flash('Pilot measurement for this VibeDash scope was removed. Your analyses and decisions are still available.', 'info')
        return redirect(url_for('vibedash.index'), code=303)


    @vibedash_bp.post('/jobs/<job_id>/decisions')
    def create_decision_case(job_id):
        """Create one bounded, evidence-linked decision from a completed job."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        if not _valid_decision_csrf_token(request.form.get('csrf_token')):
            return jsonify({'error': 'Invalid form token.'}), 400

        scope_id = _analysis_scope_id()
        job = _analysis_job_store().get(job_id, scope_id)
        if job is None:
            return jsonify({'error': 'Analysis job not found.'}), 404
        if job['status'] != 'completed' or not job.get('session_id'):
            return jsonify({'error': 'Analysis result is not ready.'}), 409
        session_data = _load_owned_session_data(
            job['session_id'], job.get('scope_id')
        )
        if not session_data:
            return jsonify({'error': 'Analysis result is no longer available.'}), 410
        if (
            not isinstance(session_data, Mapping)
            or not isinstance(session_data.get('dashboard_data'), Mapping)
        ):
            return jsonify({'error': 'Analysis result is unavailable for this analysis.'}), 409

        try:
            priority = int(request.form.get('priority', ''))
            snapshot = _decision_evidence_snapshot(job, session_data, priority)
            decision_case = _decision_case_store().create(
                scope_id,
                job_id,
                priority=priority,
                owner=request.form.get('owner', ''),
                decision=request.form.get('decision', ''),
                success_metric=request.form.get('success_metric', ''),
                target_outcome=request.form.get('target_outcome', ''),
                review_date=request.form.get('review_date', ''),
                evidence_snapshot=snapshot,
                max_cases_per_scope=current_app.config[
                    'VIBEDASH_MAX_DECISION_CASES_PER_SCOPE'
                ],
            )
        except DecisionCaseConflictError:
            existing = _decision_case_store().find_for_job_priority(
                scope_id,
                job_id,
                priority,
            )
            if existing:
                flash('This analysis priority is already being tracked.', 'info')
                return redirect(
                    url_for(
                        'vibedash.decision_case_detail',
                        case_id=existing['id'],
                    ),
                    code=303,
                )
            return jsonify({'error': 'The decision case already exists.'}), 409
        except DecisionCaseCapacityError:
            return jsonify({
                'error': 'The decision-case limit for this VibeDash scope has been reached.'
            }), 429
        except (TypeError, ValueError) as error:
            return jsonify({'error': str(error)}), 400

        current_app.logger.info(
            "Evidence-linked decision case created",
            extra={"event": "vibedash_decision_created"},
        )
        flash('Decision case created. Track the outcome after the review date.', 'info')
        return redirect(
            url_for(
                'vibedash.decision_case_detail',
                case_id=decision_case['id'],
            ),
            code=303,
        )


    @vibedash_bp.post('/decisions/<case_id>/outcome')
    def update_decision_case_outcome(case_id):
        """Record the observed result of an owned decision case."""
        if not DECISION_CASE_ID_PATTERN.fullmatch(case_id):
            return jsonify({'error': 'Decision case not found.'}), 404
        if not _valid_decision_csrf_token(request.form.get('csrf_token')):
            return jsonify({'error': 'Invalid form token.'}), 400
        try:
            decision_case = _decision_case_store().update_outcome(
                case_id,
                _analysis_scope_id(),
                status=request.form.get('status', ''),
                actual_outcome=request.form.get('actual_outcome', ''),
            )
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        if decision_case is None:
            return jsonify({'error': 'Decision case not found.'}), 404

        current_app.logger.info(
            "Decision case outcome updated",
            extra={"event": "vibedash_decision_outcome_updated"},
        )
        flash('Decision outcome updated.', 'info')
        return redirect(
            url_for('vibedash.decision_case_detail', case_id=case_id),
            code=303,
        )


    @vibedash_bp.get('/jobs/<job_id>/manifest')
    def analysis_job_manifest(job_id):
        """Download the bounded audit manifest for a completed owned job."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None:
            return jsonify({'error': 'Analysis job not found.'}), 404
        if job['status'] != 'completed' or not job.get('manifest'):
            return jsonify({'error': 'Analysis manifest is not ready.'}), 409
        response = jsonify(job['manifest'])
        response.headers['Content-Disposition'] = (
            f'attachment; filename="data-prism-manifest-{job_id[:12]}.json"'
        )
        return response


    @vibedash_bp.get('/jobs/<job_id>/comparison-report.html')
    def analysis_job_comparison_report(job_id):
        """Download an owned, in-memory standalone period-comparison report."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None:
            return jsonify({'error': 'Analysis job not found.'}), 404
        if job['status'] != 'completed' or not job.get('session_id'):
            return jsonify({'error': 'Comparison report is not ready.'}), 409
        payload = job.get('payload')
        if not isinstance(payload, Mapping) or payload.get('analysis_kind') != 'period_comparison':
            return jsonify({'error': 'Comparison report is unavailable for this analysis.'}), 409

        session_data = _load_owned_session_data(
            job['session_id'], job.get('scope_id')
        )
        if not session_data:
            return jsonify({'error': 'Analysis result is no longer available.'}), 410
        if not isinstance(session_data, Mapping):
            return jsonify({'error': 'Comparison report is unavailable for this analysis.'}), 409
        if session_data.get('analysis_kind') != 'period_comparison':
            return jsonify({'error': 'Comparison report is unavailable for this analysis.'}), 409
        report = session_data.get('report') or session_data.get('comparison')
        if not isinstance(report, Mapping):
            return jsonify({'error': 'Comparison report is unavailable for this analysis.'}), 409

        css_path = Path(current_app.static_folder or '') / 'vibedash_comparison.css'
        try:
            comparison_css = css_path.read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            current_app.logger.exception(
                'VibeDash comparison report stylesheet could not be read',
                extra={'event': 'vibedash_comparison_export_css_failed'},
            )
            return jsonify({'error': 'The comparison report could not be exported.'}), 500

        # CSS is trusted application code, but an accidental or compromised
        # stylesheet containing ``</style`` would terminate the raw-text
        # element and invalidate the standalone document. Escape the closing
        # tag boundary while preserving the stylesheet's rendered behavior.
        comparison_css = re.sub(
            r'</style', '<\\/style', comparison_css, flags=re.IGNORECASE
        )
        html = render_template(
            'vibedash_comparison.html',
            report=report,
            baseline_filename=session_data.get('baseline_filename'),
            current_filename=session_data.get('current_filename'),
            baseline_label=session_data.get('baseline_label'),
            current_label=session_data.get('current_label'),
            audit_manifest=None,
            manifest_url=None,
            analysis_job_id=None,
            comparison_export_url=None,
            decision_guidance=build_comparison_decision_guidance(report),
            comparison_css=comparison_css,
            export_mode=True,
        )
        response = make_response(html)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['Content-Disposition'] = (
            f'attachment; filename="data-prism-comparison-{job_id[:12]}.html"'
        )
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = (
            "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; "
            "style-src-attr 'none'; img-src data:; font-src 'none'; "
            "base-uri 'none'; form-action 'none'; "
            "object-src 'none'; frame-ancestors 'none';"
        )
        return response


    @vibedash_bp.route('/export/<session_id>')
    def export(session_id):
        """Экспорт дашборда в single-file HTML"""
        try:
            # Загружаем данные сессии
            session_data = _load_owned_session_data(session_id)
            if not session_data:
                flash('Session not found!', 'error')
                return redirect(url_for('vibedash.index'))
            
            # Рендерим HTML дашборда
            html_content = render_template('vibedash_evidence.html',
                                         session_id=session_id,
                                         viz_spec=session_data['viz_spec'],
                                         dashboard_data=session_data['dashboard_data'],
                                         filename=session_data['filename'],
                                         prompt=session_data['prompt'],
                                         audit_manifest=session_data.get('audit_manifest'),
                                         manifest_url=None,
                                         export_mode=True)
            
            # Создаем single-file HTML
            single_file_html = make_single_file_html(
                html_content,
                css_paths=['static/vibedash_preview.css'],
                js_paths=[]
            )
            
            # Сохраняем файл
            filepath = save_export(single_file_html, session_id)
            
            # Отправляем файл пользователю
            return send_file(filepath, as_attachment=True, 
                            download_name=f"vibedash_export_{session_id}.html")
        
        except Exception:
            current_app.logger.error(
                'VibeDash export failed',
                extra={'event': 'vibedash_export_failed'},
            )
            flash('Export is unavailable for this session.', 'error')
            return redirect(url_for('vibedash.index'))


    @vibedash_bp.route('/api/ollama-status')
    def ollama_status():
        """API для проверки статуса Ollama"""
        return jsonify({
            'available': is_ollama_available(),
            'use_ollama': os.getenv('USE_OLLAMA', 'false').lower() == 'true'
        })


    @vibedash_bp.route('/chat')
    def chat():
        """AI Data Science Chat"""
        session_id = request.args.get('session_id', '')
        return render_template('vibedash_chat.html', session_id=session_id)


    @vibedash_bp.route('/api/analyze', methods=['POST'])
    def analyze_data():
        """API для анализа данных через AI"""
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            session_id = data.get('session_id', '')
            
            if not question:
                return jsonify({'error': 'Question is required'}), 400
            
            # Загружаем данные сессии
            session_data = _load_owned_session_data(session_id)
            if not session_data:
                return jsonify({'error': 'Session not found'}), 404
            
            # Загружаем данные из файла
            stored_filename = session_data.get('stored_filename', '')
            if not isinstance(stored_filename, str) or not re.fullmatch(
                r'vibedash-[0-9a-f]{32}\.csv',
                stored_filename,
            ):
                return jsonify({'error': 'No data file found'}), 400

            file_path = Path(current_app.config['UPLOAD_FOLDER']) / stored_filename
            if not file_path.is_file():
                return jsonify({'error': 'Data file not found'}), 400
            
            import pandas as pd
            # Загружаем данные с правильной кодировкой
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except:
                    df = pd.read_csv(file_path, encoding='cp1252')
            
            if df.empty:
                return jsonify({'error': 'No data available'}), 400
            
            # Инициализируем AI анализатор
            from .ai_analyzer import DataScienceAI
            ai_analyzer = DataScienceAI(df)
            
            # Анализируем вопрос
            analysis = ai_analyzer.analyze_question(question)
            
            return jsonify({
                'success': True,
                'analysis': analysis
            })
            
        except Exception:
            current_app.logger.error(
                'VibeDash chat analysis failed',
                extra={'event': 'vibedash_chat_analysis_failed'},
            )
            return jsonify({'error': 'Analysis is unavailable.'}), 500
