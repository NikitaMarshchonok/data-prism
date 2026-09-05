import os
import re
import secrets
import sqlite3
import uuid

from flask import Flask, request, redirect, url_for, send_from_directory, render_template, session
from werkzeug.utils import secure_filename

from src.data_loader import SUPPORTED_DATA_EXTENSIONS, is_supported_data_file, load_data
from src.data_analyzer import (
    analyze_data,
    analyze_missing_values,
    plot_correlation_heatmap,
    detect_categorical_columns,
    analyze_categorical_column,
    describe_column_types,
    generate_description_for_column,
    detect_time_columns,
    convert_time_columns,
    analyze_data_quality
)
from src.visualizer import plot_histogram_interactive, plot_time_trend
from src.report_generator import generate_report
from src.pdf_exporter import export_report_to_pdf
from src.data_drift import (
    compare_to_baseline,
    create_baseline_profile,
    load_baseline_profile,
    save_baseline_profile,
)
from src.drift_store import DriftStore
from markupsafe import Markup
import markdown as md

# 📁 Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
BASELINE_FOLDER = os.path.join(BASE_DIR, 'data', 'baselines')
DRIFT_STORE_PATH = os.path.join(BASE_DIR, 'data', 'drift', 'drift_history.sqlite3')
IMAGE_FOLDER = 'images'

# ✅ Создаём папки, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY') or secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['BASELINE_FOLDER'] = BASELINE_FOLDER
app.config['DRIFT_STORE_PATH'] = DRIFT_STORE_PATH
app.config['DRIFT_HISTORY_RETENTION'] = int(os.getenv('DRIFT_HISTORY_RETENTION', '100'))
app.config['DATA_PRISM_API_KEY'] = os.getenv('DATA_PRISM_API_KEY')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_MB', '100')) * 1024 * 1024

# Регистрируем VibeDash Blueprint
from vibedash import vibedash_bp
from src.monitoring_api import monitoring_api_bp
app.register_blueprint(vibedash_bp)
app.register_blueprint(monitoring_api_bp)


@app.template_filter("markdown")
def markdown_filter(text):
    return Markup(md.markdown(text))


def save_uploaded_dataset(uploaded_file):
    """Безопасно сохраняет исходник и нормализованную CSV-копию датасета."""
    original_filename = secure_filename(uploaded_file.filename or '')
    if not original_filename:
        raise ValueError('Некорректное имя файла.')
    if not is_supported_data_file(original_filename):
        supported = ', '.join(sorted(SUPPORTED_DATA_EXTENSIONS))
        raise ValueError(f'Неподдерживаемый формат. Допустимые расширения: {supported}.')

    dataset_id = uuid.uuid4().hex
    source_filename = f'{dataset_id}_{original_filename}'
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
    uploaded_file.save(source_path)

    data, truncated = load_data(source_path)
    if data is None:
        os.remove(source_path)
        raise ValueError('Не удалось прочитать файл. Проверьте его формат и содержимое.')

    dataset_filename = f'{dataset_id}.csv'
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
    temporary_path = f'{dataset_path}.tmp'
    try:
        data.to_csv(temporary_path, index=False)
        os.replace(temporary_path, dataset_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        if os.path.exists(source_path):
            os.remove(source_path)
        raise

    return data, truncated, dataset_id, dataset_filename


def baseline_profile_path(filename):
    """Resolve only server-generated baseline profile names."""
    if not isinstance(filename, str) or not re.fullmatch(r'[0-9a-f]{32}\.json', filename):
        raise ValueError('Некорректная ссылка на baseline-профиль.')
    return os.path.join(app.config['BASELINE_FOLDER'], filename)


def monitoring_scope_id():
    """Return an isolated server-signed monitoring scope for this session."""
    scope_id = session.get('monitoring_scope_id')
    if not isinstance(scope_id, str) or not re.fullmatch(r'[0-9a-f]{32}', scope_id):
        scope_id = uuid.uuid4().hex
        session['monitoring_scope_id'] = scope_id
    return scope_id


def get_drift_store():
    return DriftStore(
        app.config['DRIFT_STORE_PATH'],
        monitoring_scope_id(),
        retention=app.config['DRIFT_HISTORY_RETENTION'],
    )


@app.errorhandler(413)
def upload_too_large(_error):
    max_upload_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return f'Файл слишком большой. Максимальный размер: {max_upload_mb} МБ.', 413


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'datafile' not in request.files:
            return 'Нет файла в запросе!', 400
        file = request.files['datafile']
        if file.filename == '':
            return 'Файл не выбран!', 400
        if file:
            try:
                data, truncated, dataset_id, dataset_filename = save_uploaded_dataset(file)
            except ValueError as error:
                return str(error), 400
            except Exception:
                app.logger.exception('Не удалось сохранить загруженный датасет')
                return 'Внутренняя ошибка при сохранении файла.', 500

            report_filename = f'{dataset_id}.html'
            report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
            session['dataset_filename'] = dataset_filename
            session['dataset_name'] = secure_filename(file.filename or dataset_filename)
            session['report_filename'] = report_filename
            session['dataset_truncated'] = truncated

            # ⏱️ Временные колонки
            data, detected_time_cols = convert_time_columns(data.copy())
            if detected_time_cols:
                print(f"🕒 Найдены временные колонки: {', '.join(detected_time_cols)}")

            # 📊 Анализ
            stats, numeric_cols = analyze_data(data)
            missing_data = analyze_missing_values(data)
            data_quality = analyze_data_quality(data)
            column_overview = describe_column_types(data)
            corr_chart = plot_correlation_heatmap(data) if len(numeric_cols) >= 2 else None

            interactive_charts = {}
            column_descriptions = {}
            for col in numeric_cols:
                interactive_charts[col] = plot_histogram_interactive(data, col)
                column_descriptions[col] = generate_description_for_column(data, col)

            categorical_summaries = {}
            for col in detect_categorical_columns(data):
                table_data, chart_html = analyze_categorical_column(data, col)
                categorical_summaries[col] = {
                    'table': table_data,
                    'chart': chart_html
                }

            time_trends = {}
            for col in detect_time_columns(data):
                trend_chart = plot_time_trend(data, col)
                if trend_chart:
                    time_trends[col] = trend_chart

            # 📝 Генерация HTML отчета
            generate_report(
                stats=stats,
                interactive_charts=interactive_charts,
                missing_data=missing_data,
                corr_chart=corr_chart,
                categorical_data=categorical_summaries,
                column_overview=column_overview,
                column_descriptions=column_descriptions,
                time_trends=time_trends,
                data_quality=data_quality,
                output_file=report_path
            )

            return redirect(url_for('show_report', filename=report_filename))

    # 🌐 Интерфейс загрузки
    max_upload_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return f'''
    <!doctype html>
    <html lang="ru">
    <head><meta charset="UTF-8"><title>Загрузите файл с данными</title></head>
    <body>
        <h1>Загрузите файл с данными</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="datafile" accept=".csv,.tsv,.xlsx,.xls,.json,.parquet" required>
            <p>Поддерживаются CSV, TSV, Excel, JSON и Parquet до {max_upload_mb} МБ.</p>
            <input type="submit" value="Загрузить">
        </form>
    </body>
    </html>
    '''


@app.route('/reports/<filename>')
def show_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename)


# ✅ Маршрут: генерация PDF вручную
@app.route('/download-pdf')
def download_pdf():
    report_filename = session.get('report_filename')
    if not report_filename:
        return 'Сначала загрузите датасет и создайте HTML-отчёт!', 400

    html_report = os.path.join(app.config['REPORT_FOLDER'], report_filename)
    pdf_filename = f'{os.path.splitext(report_filename)[0]}.pdf'
    pdf_report = os.path.join(app.config['REPORT_FOLDER'], pdf_filename)
    if not os.path.exists(html_report):
        return 'HTML-отчёт не найден. Загрузите датасет повторно.', 404

    try:
        export_report_to_pdf(html_report, pdf_report)
        print("✅ PDF успешно создан вручную.")
        return send_from_directory(app.config['REPORT_FOLDER'], pdf_filename, as_attachment=True)
    except Exception:
        app.logger.exception('Не удалось создать PDF-отчёт')
        return 'Не удалось создать PDF-отчёт.', 500


# ✅ Маршрут: BI-дэшборд
@app.route('/dashboard', methods=['GET', 'POST'])
def show_dashboard():
    from src.data_loader import load_data
    from src.dashboard_generator import generate_dashboard_data

    dataset_filename = session.get('dataset_filename')
    if not dataset_filename:
        return '<h2>Сначала загрузите датасет на главной странице.</h2>', 400

    # Загружаем датасет только из текущей пользовательской сессии
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
    df, _ = load_data(dataset_path)
    if df is None:
        return "<h2>❌ Ошибка загрузки данных</h2>", 404

    drift_store = None
    drift_store_error = None
    try:
        drift_store = get_drift_store()
    except (OSError, sqlite3.Error, ValueError):
        app.logger.exception('Не удалось открыть журнал drift-мониторинга')
        drift_store_error = 'Журнал drift-мониторинга временно недоступен.'

    if request.method == 'POST' and request.form.get('dashboard_action') == 'acknowledge_drift_alert':
        if drift_store is None:
            return 'Журнал drift-мониторинга недоступен.', 503
        try:
            alert_id = int(request.form.get('alert_id', ''))
            if not drift_store.acknowledge_alert(alert_id):
                return 'Alert не найден или уже подтверждён.', 404
        except (TypeError, ValueError):
            return 'Некорректный идентификатор alert.', 400
        except sqlite3.Error:
            app.logger.exception('Не удалось подтвердить drift alert')
            return 'Журнал drift-мониторинга недоступен.', 503
        return redirect(url_for('show_dashboard'))

    # Явно фиксируем текущий датасет как эталон.
    # Последующие загрузки его не перезаписывают.
    if request.method == 'POST' and request.form.get('dashboard_action') == 'set_drift_baseline':
        baseline_filename = session.get('baseline_profile_filename')
        if not baseline_filename:
            baseline_filename = f'{uuid.uuid4().hex}.json'
        try:
            baseline_profile = create_baseline_profile(
                df,
                dataset_name=session.get('dataset_name', dataset_filename),
            )
            save_baseline_profile(
                baseline_profile,
                baseline_profile_path(baseline_filename),
            )
            session['baseline_profile_filename'] = baseline_filename
        except (OSError, ValueError):
            app.logger.exception('Не удалось сохранить baseline-профиль')
            return (
                'Не удалось сохранить baseline-профиль текущего датасета.',
                400,
            )
        return redirect(url_for('show_dashboard'))

    drift_report = None
    drift_error = None
    drift_history = []
    drift_alerts = []
    baseline_filename = session.get('baseline_profile_filename')
    if baseline_filename:
        try:
            baseline_profile = load_baseline_profile(baseline_profile_path(baseline_filename))
            drift_report = compare_to_baseline(df, baseline_profile)
        except (OSError, ValueError):
            app.logger.exception(
                'Не удалось сравнить датасет с baseline-профилем'
            )
            drift_error = (
                'Baseline-профиль недоступен или повреждён. '
                'Сохраните новый эталон.'
            )

    if drift_report is not None and drift_store is not None:
        try:
            drift_store.record_run(
                drift_report,
                batch_id=dataset_filename,
                dataset_name=session.get('dataset_name', dataset_filename),
            )
        except (sqlite3.Error, TypeError, ValueError):
            app.logger.exception('Не удалось записать drift-событие')
            drift_store_error = (
                'Не удалось записать результат в журнал '
                'drift-мониторинга.'
            )

    if drift_store is not None:
        try:
            drift_history = drift_store.list_runs(limit=10)
            drift_alerts = drift_store.list_alerts(limit=10)
        except sqlite3.Error:
            app.logger.exception(
                'Не удалось прочитать журнал drift-мониторинга'
            )
            drift_store_error = (
                'Журнал drift-мониторинга временно недоступен.'
            )

    # Получаем список подходящих колонок для целевой переменной
    selectable_columns = [col for col in df.columns if df[col].nunique() >= 2]

    # Обработка выбора пользователя
    selected_target = request.form.get('target_column')
    if not selected_target or selected_target not in df.columns:
        selected_target = selectable_columns[0] if selectable_columns else None

    try:
        # Генерация данных для дашборда
        kpis, top_charts, tables, summary, sparklines, ai_summary, ml_card = generate_dashboard_data(df, selected_target)

        return render_template(
            'dashboard.html',
            kpis=kpis,
            top_charts=top_charts,
            tables=tables,
            summary=summary,
            sparklines=sparklines,
            ai_summary=ai_summary,
            ml_card=ml_card,
            drift_report=drift_report,
            drift_error=drift_error,
            drift_store_error=drift_store_error,
            drift_history=drift_history,
            drift_alerts=drift_alerts,
            baseline_configured=bool(baseline_filename),
            selectable_columns=selectable_columns,
            selected_target=selected_target
        )
    except Exception:
        app.logger.exception('Не удалось сформировать BI-дашборд')
        return "<h2>❌ Не удалось сформировать дашборд.</h2>", 500



if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, port=int(os.getenv('PORT', '5001')))
