import os
import shutil
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from src.data_loader import load_data
from src.data_analyzer import (
    analyze_data,
    analyze_missing_values,
    plot_correlation_heatmap,
    detect_categorical_columns,
    analyze_categorical_column,
    describe_column_types,
    generate_description_for_column,
    detect_time_columns,
    convert_time_columns
)
from src.visualizer import plot_histogram_interactive, plot_time_trend
from src.report_generator import generate_report
from src.pdf_exporter import export_report_to_pdf
from markupsafe import Markup
import markdown as md

# 📁 Пути
UPLOAD_FOLDER = 'data/uploads'
REPORT_FOLDER = 'reports'
IMAGE_FOLDER = 'images'
HTML_REPORT = os.path.join(REPORT_FOLDER, 'final_report.html')
PDF_REPORT = os.path.join(REPORT_FOLDER, 'final_report.pdf')

# ✅ Создаём папки, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.template_filter("markdown")
def markdown_filter(text):
    return Markup(md.markdown(text))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'datafile' not in request.files:
            return 'Нет файла в запросе!'
        file = request.files['datafile']
        if file.filename == '':
            return 'Файл не выбран!'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # 💾 Сохраняем копию для дашборда
            shutil.copy(file_path, os.path.join(app.config['UPLOAD_FOLDER'], 'latest_uploaded.csv'))

            data, _ = load_data(file_path)
            if data is None:
                return 'Ошибка при загрузке данных'

            # ⏱️ Временные колонки
            data, detected_time_cols = convert_time_columns(data)
            if detected_time_cols:
                print(f"🕒 Найдены временные колонки: {', '.join(detected_time_cols)}")

            # 📊 Анализ
            stats, numeric_cols = analyze_data(data)
            missing_data = analyze_missing_values(data)
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
                output_file=HTML_REPORT
            )

            return redirect(url_for('show_report', filename='final_report.html'))

    # 🌐 Интерфейс загрузки
    return '''
    <!doctype html>
    <html lang="ru">
    <head><meta charset="UTF-8"><title>Загрузите файл с данными</title></head>
    <body>
        <h1>Загрузите файл с данными</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="datafile">
            <input type="submit" value="Загрузить">
        </form>
    </body>
    </html>
    '''


@app.route('/reports/<filename>')
def show_report(filename):
    return send_from_directory(REPORT_FOLDER, filename)


@app.route('/reports/<path:filename>')
def download_file(filename):
    return send_from_directory(REPORT_FOLDER, filename)


# ✅ Маршрут: генерация PDF вручную
@app.route('/download-pdf')
def download_pdf():
    if not os.path.exists(HTML_REPORT):
        return 'Сначала создайте HTML-отчёт!'

    try:
        export_report_to_pdf(HTML_REPORT, PDF_REPORT)
        print("✅ PDF успешно создан вручную.")
        return send_from_directory(REPORT_FOLDER, 'final_report.pdf', as_attachment=True)
    except Exception as e:
        return f"❌ Ошибка при создании PDF: {e}"


# ✅ Маршрут: BI-дэшборд
@app.route('/dashboard')
@app.route('/dashboard', methods=['GET', 'POST'])
def show_dashboard():
    from src.data_loader import load_data
    from src.dashboard_generator import generate_dashboard_data

    # Загружаем данные
    df, _ = load_data('data/uploads/latest_uploaded.csv')
    if df is None:
        return "<h2>❌ Ошибка загрузки данных</h2>"

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
            selectable_columns=selectable_columns,
            selected_target=selected_target
        )
    except Exception as e:
        return f"<h2>❌ Ошибка при загрузке дашборда:</h2><pre>{e}</pre>"



if __name__ == '__main__':
    app.run(debug=True)
