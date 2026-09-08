"""
Экспорт дашборда в single-file HTML
"""
import os
import json
import uuid
from pathlib import Path
from typing import List
from datetime import datetime

from plotly.utils import PlotlyJSONEncoder


DEFAULT_SESSIONS_DIR = Path("tmp") / "vibedash"
DEFAULT_EXPORTS_DIR = Path("exports")


def _resolve_sessions_dir(sessions_dir=None) -> Path:
    """Resolve VibeDash session storage inside the configured runtime state."""
    if sessions_dir is not None:
        return Path(sessions_dir)

    state_dir = os.getenv("DATA_PRISM_STATE_DIR")
    if state_dir:
        return Path(state_dir).expanduser() / "sessions" / "vibedash"
    return DEFAULT_SESSIONS_DIR


def _resolve_exports_dir(exports_dir=None) -> Path:
    """Resolve generated exports inside writable runtime storage."""
    if exports_dir is not None:
        return Path(exports_dir)

    state_dir = os.getenv("DATA_PRISM_STATE_DIR")
    if state_dir:
        return Path(state_dir).expanduser() / "exports" / "vibedash"
    return DEFAULT_EXPORTS_DIR


def _session_file(session_id: str, sessions_dir=None) -> Path:
    """Build a safe session path for server-generated UUID identifiers."""
    try:
        normalized_session_id = str(uuid.UUID(session_id))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Invalid VibeDash session identifier.") from error
    if normalized_session_id != session_id.lower():
        raise ValueError("Invalid VibeDash session identifier.")
    return _resolve_sessions_dir(sessions_dir) / f"{normalized_session_id}.json"


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


def save_export(html_content: str, session_id: str, exports_dir=None) -> str:
    """
    Сохраняет экспортированный HTML файл
    """
    normalized_session_id = str(uuid.UUID(session_id))
    export_directory = _resolve_exports_dir(exports_dir)
    export_directory.mkdir(parents=True, exist_ok=True)

    # Генерируем имя файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vibedash_export_{normalized_session_id}_{timestamp}.html"
    filepath = export_directory / filename

    # Сохраняем файл
    with filepath.open('w', encoding='utf-8') as f:
        f.write(html_content)

    return str(filepath)


def load_session_data(session_id: str, sessions_dir=None) -> dict:
    """
    Загружает данные сессии из временного хранилища
    """
    try:
        session_file = _session_file(session_id, sessions_dir)
    except ValueError:
        return None

    if not session_file.exists():
        return None

    try:
        with session_file.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Ошибка загрузки сессии {session_id}: {e}")
        return None


def save_session_data(session_id: str, data: dict, sessions_dir=None) -> bool:
    """
    Сохраняет данные сессии во временное хранилище
    """
    try:
        session_file = _session_file(session_id, sessions_dir)
        session_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = session_file.with_suffix(".json.tmp")
        with temporary_file.open('w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
                cls=PlotlyJSONEncoder,
            )
        os.replace(temporary_file, session_file)
        return True
    except Exception as e:
        if 'temporary_file' in locals() and temporary_file.exists():
            temporary_file.unlink()
        print(f"⚠️ Ошибка сохранения сессии {session_id}: {e}")
        return False
