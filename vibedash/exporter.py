"""
Экспорт дашборда в single-file HTML
"""
import os
import re
from typing import List
from datetime import datetime


def make_single_file_html(html: str, css_paths: List[str] = None, js_paths: List[str] = None) -> str:
    """
    Создает самодостаточный HTML файл с встроенными CSS/JS
    """
    if css_paths is None:
        css_paths = []
    if js_paths is None:
        js_paths = []
    
    # Встраиваем CSS
    inline_css = ""
    for css_path in css_paths:
        if os.path.exists(css_path):
            with open(css_path, 'r', encoding='utf-8') as f:
                inline_css += f"<style>\n{f.read()}\n</style>\n"
    
    # Встраиваем JS
    inline_js = ""
    for js_path in js_paths:
        if os.path.exists(js_path):
            with open(js_path, 'r', encoding='utf-8') as f:
                inline_js += f"<script>\n{f.read()}\n</script>\n"
    
    # Встраиваем Plotly.js
    plotly_js = _get_plotly_js()
    
    # Создаем полный HTML
    full_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VibeDash Export - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    {inline_css}
    {plotly_js}
    {inline_js}
</head>
<body>
    {html}
</body>
</html>"""
    
    return full_html


def _get_plotly_js() -> str:
    """
    Возвращает встроенный Plotly.js
    Для production лучше использовать CDN или локальную копию
    """
    return """
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    """


def save_export(html_content: str, session_id: str, exports_dir: str = "exports") -> str:
    """
    Сохраняет экспортированный HTML файл
    """
    # Создаем папку exports если не существует
    os.makedirs(exports_dir, exist_ok=True)
    
    # Генерируем имя файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vibedash_export_{session_id}_{timestamp}.html"
    filepath = os.path.join(exports_dir, filename)
    
    # Сохраняем файл
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return filepath


def load_session_data(session_id: str, sessions_dir: str = "tmp/vibedash") -> dict:
    """
    Загружает данные сессии из временного хранилища
    """
    session_file = os.path.join(sessions_dir, f"{session_id}.json")
    
    if not os.path.exists(session_file):
        return None
    
    try:
        import json
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Ошибка загрузки сессии {session_id}: {e}")
        return None


def save_session_data(session_id: str, data: dict, sessions_dir: str = "tmp/vibedash") -> bool:
    """
    Сохраняет данные сессии во временное хранилище
    """
    os.makedirs(sessions_dir, exist_ok=True)
    session_file = os.path.join(sessions_dir, f"{session_id}.json")
    
    try:
        import json
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"⚠️ Ошибка сохранения сессии {session_id}: {e}")
        return False
