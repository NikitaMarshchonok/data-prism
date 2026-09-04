"""
VibeDash Flask routes
"""
import os
import uuid
try:
    from flask import render_template, request, redirect, url_for, send_file, jsonify, flash
    from werkzeug.utils import secure_filename
    import pandas as pd
    from .spec import parse_prompt_to_viz_spec
    from .generator_bridge import generate_dashboard_data as bridge_generate_dashboard_data
    from .exporter import make_single_file_html, save_export, load_session_data, save_session_data
    from .ollama_client import is_ollama_available
    from . import vibedash_bp
except ImportError:
    # Flask не установлен, создаем заглушки
    vibedash_bp = None


if vibedash_bp:
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
        
        return render_template('vibedash_index.html', 
                             preset_prompts=preset_prompts,
                             ollama_available=ollama_available)


    @vibedash_bp.route('/preview', methods=['POST'])
    def preview():
        """Предварительный просмотр дашборда"""
        try:
            import logging
            logging.info("=== VibeDash Preview Request ===")
            logging.info(f"Request method: {request.method}")
            logging.info(f"Request files: {list(request.files.keys())}")
            logging.info(f"Request form: {dict(request.form)}")
            
            # Получаем файл и промпт
            if 'datafile' not in request.files:
                logging.error("No file part in request")
                flash('No file selected!', 'error')
                return redirect(url_for('vibedash.index'))
            
            file = request.files['datafile']
            if file.filename == '':
                logging.error("No file selected")
                flash('No file selected!', 'error')
                return redirect(url_for('vibedash.index'))
            
            prompt = request.form.get('prompt', '').strip()
            if not prompt:
                logging.error("No prompt provided")
                flash('Please enter a dashboard description!', 'error')
                return redirect(url_for('vibedash.index'))
            
            logging.info(f"File: {file.filename}, Prompt: {prompt[:50]}...")
            
            # Сохраняем файл
            filename = secure_filename(file.filename)
            upload_path = os.path.join('data/uploads', filename)
            file.save(upload_path)
            logging.info(f"File saved to: {upload_path}")
            
            # Загружаем данные с правильной кодировкой
            try:
                df = pd.read_csv(upload_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                try:
                    df = pd.read_csv(upload_path, encoding='latin-1')
                except:
                    df = pd.read_csv(upload_path, encoding='cp1252')
            logging.info(f"DataFrame loaded: {df.shape}")
            
            # Ограничиваем размер для предварительного просмотра
            max_rows = int(os.getenv('MAX_ROWS_PREVIEW', '100000'))
            if len(df) > max_rows:
                df = df.head(max_rows)
                flash(f'Data limited to {max_rows:,} rows for preview', 'info')
                logging.info(f"Data limited to {max_rows:,} rows")
            
            # Проверяем размер файла
            file_size_mb = os.path.getsize(upload_path) / (1024 * 1024)
            if file_size_mb > 100:  # Больше 100MB
                flash(f'Large file detected ({file_size_mb:.1f}MB). Processing may take longer...', 'warning')
                logging.info(f"Large file detected: {file_size_mb:.1f}MB")
            
            # Парсим промпт в VizSpec
            logging.info("Parsing prompt to VizSpec...")
            viz_spec = parse_prompt_to_viz_spec(prompt, list(df.columns))
            logging.info(f"VizSpec generated: {len(viz_spec.metrics)} metrics, {len(viz_spec.charts)} charts")
            
            # Генерируем данные дашборда
            logging.info("Generating dashboard data...")
            dashboard_data = bridge_generate_dashboard_data(df, viz_spec)
            logging.info("Dashboard data generated successfully")
            
            # Создаем сессию
            session_id = str(uuid.uuid4())
            session_data = {
                'viz_spec': viz_spec.dict(),
                'dashboard_data': dashboard_data,
                'filename': filename,
                'prompt': prompt,
                'df_shape': df.shape,
                'file_path': upload_path  # Сохраняем путь для AI чата
            }
            
            # Сохраняем данные сессии
            save_session_data(session_id, session_data)
            logging.info(f"Session data saved: {session_id}")
            
            # Рендерим предварительный просмотр
            logging.info("Rendering preview template...")
            return render_template('vibedash_preview.html',
                                 session_id=session_id,
                                 viz_spec=viz_spec,
                                 dashboard_data=dashboard_data,
                                 filename=filename,
                                 prompt=prompt)
        
        except Exception as e:
            import logging
            import traceback
            logging.exception("Error in preview route:")
            print(f"❌ ERROR in preview route: {str(e)}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            flash(f'Error creating dashboard: {str(e)}', 'error')
            return redirect(url_for('vibedash.index'))


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
            html_content = render_template('vibedash_preview.html',
                                         session_id=session_id,
                                         viz_spec=session_data['viz_spec'],
                                         dashboard_data=session_data['dashboard_data'],
                                         filename=session_data['filename'],
                                         prompt=session_data['prompt'],
                                         export_mode=True)
            
            # Создаем single-file HTML
            single_file_html = make_single_file_html(
                html_content,
                css_paths=['static/vibedash.css'],
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
            filename = session_data.get('filename', '')
            if not filename:
                return jsonify({'error': 'No data file found'}), 400
            
            file_path = os.path.join('data/uploads', filename)
            if not os.path.exists(file_path):
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