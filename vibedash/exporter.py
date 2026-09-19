"""
Экспорт дашборда в single-file HTML
"""
import os
import json
import secrets
import tempfile
import uuid
from pathlib import Path
import re
import time
from typing import List
from datetime import datetime

from plotly.utils import PlotlyJSONEncoder


DEFAULT_SESSIONS_DIR = Path("tmp") / "vibedash"
DEFAULT_EXPORTS_DIR = Path("exports")
SESSION_FILE_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\.json$"
)
ANALYSIS_SCOPE_PATTERN = re.compile(r"^[0-9a-f]{32}$")
UPLOAD_FILE_PATTERN = re.compile(r"^vibedash-[0-9a-f]{32}\.csv$")
EXPORT_FILE_PATTERN = re.compile(
    r"^vibedash_export_[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}_[0-9]{8}_[0-9]{6}\.html$"
)


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


def _delete_expired_files(directory, pattern, cutoff_timestamp: float):
    """Delete only recognized, regular files older than the cutoff."""
    removed = 0
    errors = 0
    directory = Path(directory)
    try:
        candidates = list(directory.iterdir())
    except FileNotFoundError:
        return removed, errors
    except OSError:
        return removed, 1

    for candidate in candidates:
        if not pattern.fullmatch(candidate.name):
            continue
        try:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if candidate.stat().st_mtime >= cutoff_timestamp:
                continue
            candidate.unlink()
            removed += 1
        except OSError:
            errors += 1
    return removed, errors


def cleanup_expired_artifacts(
    upload_dir,
    retention_hours: int,
    *,
    sessions_dir=None,
    exports_dir=None,
    now: float | None = None,
):
    """Remove expired VibeDash artifacts without touching unrelated files."""
    if isinstance(retention_hours, bool) or not isinstance(retention_hours, int):
        raise ValueError("retention_hours must be an integer")
    if retention_hours < 1:
        raise ValueError("retention_hours must be at least 1")

    current_timestamp = time.time() if now is None else float(now)
    cutoff_timestamp = current_timestamp - (retention_hours * 60 * 60)
    locations = (
        ("sessions", _resolve_sessions_dir(sessions_dir), SESSION_FILE_PATTERN),
        ("uploads", Path(upload_dir), UPLOAD_FILE_PATTERN),
        ("exports", _resolve_exports_dir(exports_dir), EXPORT_FILE_PATTERN),
    )
    result = {
        "removed_sessions": 0,
        "removed_uploads": 0,
        "removed_exports": 0,
        "errors": 0,
    }

    for label, directory, pattern in locations:
        removed, errors = _delete_expired_files(
            directory,
            pattern,
            cutoff_timestamp,
        )
        result[f"removed_{label}"] = removed
        result["errors"] += errors
    return result


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


def _validated_analysis_scope_id(scope_id: str) -> str | None:
    """Return a normalized VibeDash scope or ``None`` for malformed input."""
    if not isinstance(scope_id, str) or not ANALYSIS_SCOPE_PATTERN.fullmatch(scope_id):
        return None
    return scope_id


def load_session_data(
    session_id: str,
    sessions_dir=None,
    *,
    owner_id: str | None = None,
) -> dict:
    """
    Загружает данные сессии из временного хранилища
    """
    try:
        session_file = _session_file(session_id, sessions_dir)
    except ValueError:
        return None

    # Ownership is mandatory for every retained-session read.  Keeping an
    # ownerless compatibility mode here would make a future route call a
    # one-argument helper and silently reintroduce an IDOR.
    expected_owner = _validated_analysis_scope_id(owner_id)
    if expected_owner is None:
        return None

    # A session record must be a regular application-owned file.  In
    # particular, do not follow a symlink planted at a predictable session
    # filename while loading retained data.
    if session_file.is_symlink() or not session_file.is_file():
        return None

    try:
        with session_file.open('r', encoding='utf-8') as f:
            session_data = json.load(f)
        if not isinstance(session_data, dict):
            return None
        stored_owner = session_data.get('analysis_scope_id')
        if (
            _validated_analysis_scope_id(stored_owner) is None
            or not secrets.compare_digest(stored_owner, expected_owner)
        ):
            return None
        return session_data
    except Exception:
        # Do not put session identifiers or stored/user data in logs.  A
        # malformed/partially-written record is simply unavailable.
        return None


def save_session_data(
    session_id: str,
    data: dict,
    sessions_dir=None,
    *,
    owner_id: str | None = None,
) -> bool:
    """
    Сохраняет данные сессии во временное хранилище
    """
    temporary_file = None
    try:
        session_file = _session_file(session_id, sessions_dir)
        if not isinstance(data, dict):
            return False
        normalized_owner = _validated_analysis_scope_id(owner_id)
        if normalized_owner is None:
            return False
        # The owner is always assigned here, after route validation, and
        # never accepted from request payload/session data.
        data = dict(data)
        data['analysis_scope_id'] = normalized_owner
        # Use an exclusive, randomly named file in the same directory.  A
        # predictable ``<session>.json.tmp`` can be replaced with a symlink
        # between requests and would otherwise be followed by ``open('w')``.
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=session_file.parent,
            prefix=f'.{session_file.name}.',
            suffix='.tmp',
            delete=False,
        ) as f:
            temporary_file = Path(f.name)
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
                cls=PlotlyJSONEncoder,
            )
        os.replace(temporary_file, session_file)
        return True
    except Exception:
        if (
            isinstance(temporary_file, Path)
            and temporary_file.exists()
            and not temporary_file.is_symlink()
        ):
            temporary_file.unlink()
        return False
