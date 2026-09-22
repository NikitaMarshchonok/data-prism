"""
Экспорт дашборда в single-file HTML
"""
import os
import json
import errno
import secrets
import stat
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
    """Delete only recognized, regular files older than the cutoff.

    Keep the directory open and use descriptor-relative operations throughout
    the scan.  A pathname ``iterdir``/``unlink`` pair would follow a swapped
    directory symlink and could remove a matching file outside runtime state.
    """
    removed = 0
    errors = 0
    directory = Path(directory)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(directory, flags)
    except FileNotFoundError:
        return removed, errors
    except OSError:
        return removed, 1
    try:
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                return removed, 1
            candidates = os.listdir(descriptor)
        except OSError:
            return removed, 1

        for name in candidates:
            if not pattern.fullmatch(name):
                continue
            try:
                candidate_stat = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(candidate_stat.st_mode):
                    continue
                if candidate_stat.st_mtime >= cutoff_timestamp:
                    continue
                os.unlink(name, dir_fd=descriptor)
                removed += 1
            except FileNotFoundError:
                # Another retention pass may have won the race.  The desired
                # end state is already true, so this is not an error.
                continue
            except OSError:
                errors += 1
    finally:
        os.close(descriptor)
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

    # Keep the final directory open and use descriptor-relative operations.
    # Opening the predictable export pathname with ``w`` would follow a
    # symlink planted there and could overwrite an arbitrary local file; using
    # a path-based temporary file also allows the directory itself to be
    # swapped between creation and replacement.
    directory_flags = os.O_RDONLY
    if hasattr(os, 'O_DIRECTORY'):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, 'O_NOFOLLOW'):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = None
    temporary_name = None
    temporary_fd = None
    try:
        directory_fd = os.open(export_directory, directory_flags)
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise OSError('Export storage is not a directory.')
        temporary_name = f'.{filename}.{secrets.token_hex(16)}.tmp'
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            temporary_flags |= os.O_NOFOLLOW
        temporary_fd = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(temporary_fd, 'w', encoding='utf-8') as handle:
            temporary_fd = None
            handle.write(html_content)
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
    except Exception:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

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

    # Open the final directory and file without following symlinks.  A
    # separate ``is_symlink``/``is_file`` check followed by ``open`` leaves a
    # replacement window in which a symlink can be swapped in and private
    # content outside the sessions directory read.
    directory_fd = None
    file_fd = None
    try:
        directory_flags = os.O_RDONLY
        if hasattr(os, 'O_DIRECTORY'):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, 'O_NOFOLLOW'):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(session_file.parent, directory_flags)
        file_flags = os.O_RDONLY
        if hasattr(os, 'O_NOFOLLOW'):
            file_flags |= os.O_NOFOLLOW
        file_fd = os.open(session_file.name, file_flags, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            return None
        with os.fdopen(file_fd, 'r', encoding='utf-8') as f:
            file_fd = None
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
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as error:
        if getattr(error, 'errno', None) == errno.ELOOP:
            return None
        return None
    except Exception:
        # Do not put session identifiers or stored/user data in logs.  A
        # malformed/partially-written record is simply unavailable.
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


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
    directory_fd = None
    temporary_name = None
    temporary_fd = None
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
        # Keep the final directory open and use descriptor-relative operations
        # throughout. A path-based temporary file and ``os.replace`` can be
        # redirected if the sessions directory is swapped for a symlink while
        # this write is in progress.
        session_file.parent.mkdir(parents=True, exist_ok=True)
        directory_flags = os.O_RDONLY
        if hasattr(os, 'O_DIRECTORY'):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, 'O_NOFOLLOW'):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(session_file.parent, directory_flags)
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            return False
        temporary_name = f'.{session_file.name}.{secrets.token_hex(16)}.tmp'
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, 'O_NOFOLLOW'):
            temporary_flags |= os.O_NOFOLLOW
        temporary_fd = os.open(
            temporary_name,
            temporary_flags,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(temporary_fd, 'w', encoding='utf-8') as f:
            temporary_fd = None
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
                cls=PlotlyJSONEncoder,
            )
        os.replace(
            temporary_name,
            session_file.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        return True
    except Exception:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None and directory_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        return False
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
