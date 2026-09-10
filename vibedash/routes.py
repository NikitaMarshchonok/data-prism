"""
VibeDash Flask routes
"""
import os
import re
import uuid
from pathlib import Path

try:
    from flask import (
        current_app,
        flash,
        jsonify,
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
        MAX_OWNED_JOBS,
    )
    from . import vibedash_bp
except ImportError:
    # Flask не установлен, создаем заглушки
    vibedash_bp = None


if vibedash_bp:
    analysis_job_dispatcher = AnalysisJobDispatcher(max_workers=1)

    def _analysis_scope_id():
        scope_id = session.get('vibedash_analysis_scope_id')
        if not isinstance(scope_id, str) or not JOB_ID_PATTERN.fullmatch(scope_id):
            scope_id = uuid.uuid4().hex
            session['vibedash_analysis_scope_id'] = scope_id
        return scope_id


    def _owned_analysis_job_ids():
        job_ids = session.get('vibedash_analysis_job_ids', [])
        if not isinstance(job_ids, list):
            return []
        return [
            job_id for job_id in job_ids
            if isinstance(job_id, str) and JOB_ID_PATTERN.fullmatch(job_id)
        ][-MAX_OWNED_JOBS:]


    def _remember_analysis_job(job_id):
        job_ids = [
            existing for existing in _owned_analysis_job_ids()
            if existing != job_id
        ]
        session['vibedash_analysis_job_ids'] = (
            job_ids + [job_id]
        )[-MAX_OWNED_JOBS:]


    def _analysis_job_store():
        return AnalysisJobStore(current_app.config['VIBEDASH_JOB_STORE_PATH'])


    def _load_vibedash_csv(path):
        last_error = None
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError('The CSV encoding is not supported.') from last_error


    def _build_dashboard_session(payload):
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
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('The analysis request is invalid.')
        if not isinstance(filename, str) or not filename:
            raise ValueError('The dataset name is invalid.')

        df = _load_vibedash_csv(upload_path)
        if df.empty:
            raise ValueError('The dataset does not contain any rows.')
        max_rows = current_app.config['MAX_ROWS_PREVIEW']
        truncated = len(df) > max_rows
        if len(df) > max_rows:
            df = df.head(max_rows)

        if demo_dataset:
            if demo_dataset != DEMO_DATASET_ID:
                raise ValueError('The demonstration dataset is invalid.')
            viz_spec = create_saas_demo_viz_spec()
        else:
            viz_spec = parse_prompt_to_viz_spec(prompt, list(df.columns))
        dashboard_data = bridge_generate_dashboard_data(df, viz_spec)

        session_id = str(uuid.uuid4())
        session_data = {
            'viz_spec': viz_spec.model_dump(),
            'dashboard_data': dashboard_data,
            'filename': filename,
            'stored_filename': stored_filename,
            'prompt': prompt,
            'df_shape': df.shape,
            'file_path': str(upload_path),
        }
        if not save_session_data(session_id, session_data):
            raise RuntimeError('VibeDash session could not be persisted.')
        return {
            'session_id': session_id,
            'session_data': session_data,
            'viz_spec': viz_spec,
            'truncated': truncated,
        }


    def _process_analysis_job(job):
        result = _build_dashboard_session(job.get('payload', {}))
        return result['session_id']


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
        stale_jobs = store.fail_stale_running(
            current_app.config['VIBEDASH_JOB_TIMEOUT_SECONDS']
        )
        removed_jobs = store.purge_terminal(
            current_app.config['VIBEDASH_RETENTION_HOURS']
        )
        if stale_jobs:
            current_app.logger.warning(
                "Interrupted VibeDash jobs marked as failed",
                extra={"event": "vibedash_jobs_interrupted"},
            )
        if removed_jobs:
            current_app.logger.info(
                "Expired VibeDash jobs removed",
                extra={"event": "vibedash_jobs_retention_cleanup"},
            )

        application = current_app._get_current_object()
        for job_id in _owned_analysis_job_ids():
            job = store.get(job_id, _analysis_scope_id())
            if job and job['status'] == 'queued':
                analysis_job_dispatcher.submit(
                    application,
                    job_id,
                    _process_analysis_job,
                )


    @vibedash_bp.route('/')
    def index():
        """Главная страница VibeDash"""
        # Проверяем доступность Ollama
        ollama_available = is_ollama_available()
        
        # Предустановленные промпты
        preset_prompts = {
            "sales": "Sales dashboard for a monthly CSV: main KPIs (Total Sales, Orders, AOV), top 10 categories, revenue trend by week, bar by region, filter by region, highlight YoY growth.",
            "finance": "Финансы: сумма дохода и расходов, дельта, тренд по неделям, топ-категории расходов, фильтр по отделу, комментарий по выбросам.",
            "real_estate": "Real-estate listing analysis: median price by city, distribution by rooms, time trend by posting date (W), filter by city, show top 10 streets by average price."
        }
        
        return render_template('vibedash_landing.html',
                             preset_prompts=preset_prompts,
                             demo_prompt=DEMO_PROMPT,
                             retention_hours=current_app.config['VIBEDASH_RETENTION_HOURS'],
                             ollama_available=ollama_available)


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
                                 prompt=prompt)
        
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

        _remember_analysis_job(job['id'])
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
            'status_url': url_for(
                'vibedash.analysis_job_status',
                job_id=job['id'],
            ),
        }), 202


    @vibedash_bp.get('/jobs/<job_id>')
    def analysis_job_status(job_id):
        """Return a scoped, non-sensitive representation of one job."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None or job_id not in _owned_analysis_job_ids():
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
        elif job['status'] == 'failed':
            response['error'] = (
                'The analysis was interrupted. Please submit it again.'
                if job['error_code'] == 'worker_interrupted'
                else 'The analysis could not be completed.'
            )
        return jsonify(response)


    @vibedash_bp.get('/jobs/<job_id>/result')
    def analysis_job_result(job_id):
        """Render a completed job only for its signed browser session."""
        if not JOB_ID_PATTERN.fullmatch(job_id):
            return jsonify({'error': 'Analysis job not found.'}), 404
        job = _analysis_job_store().get(job_id, _analysis_scope_id())
        if job is None or job_id not in _owned_analysis_job_ids():
            return jsonify({'error': 'Analysis job not found.'}), 404
        if job['status'] != 'completed' or not job['session_id']:
            return jsonify({'error': 'Analysis result is not ready.'}), 409

        session_data = load_session_data(job['session_id'])
        if not session_data:
            return jsonify({'error': 'Analysis result is no longer available.'}), 410
        return render_template(
            'vibedash_evidence.html',
            session_id=job['session_id'],
            viz_spec=session_data['viz_spec'],
            dashboard_data=session_data['dashboard_data'],
            filename=session_data['filename'],
            prompt=session_data['prompt'],
        )


    @vibedash_bp.route('/export/<session_id>')
    def export(session_id):
        """Экспорт дашборда в single-file HTML"""
        try:
            # Загружаем данные сессии
            session_data = load_session_data(session_id)
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
        
        except Exception as e:
            flash(f'Export error: {str(e)}', 'error')
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
            session_data = load_session_data(session_id)
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
            
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
