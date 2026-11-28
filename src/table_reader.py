"""Read uploaded tables from any field and record exactly how they were read.

Spreadsheet exports differ by country and tool: Excel saves CSV in the
system code page (cp1251 in Russian, cp1255 in Hebrew, cp1252 in Western
locales) with ``;`` separators and decimal commas, "Unicode text" is
tab-separated UTF-16, and many people simply upload the workbook. A reader
that assumes UTF-8 and commas does not fail on such files; it silently returns
one column of mojibake. Every decision below is therefore detected from the
bytes, applied deterministically, and reported back so a person can check it.
"""

from __future__ import annotations

import csv
import io
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = frozenset({".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".parquet"})
SUPPORTED_ACCEPT = ",".join(sorted(SUPPORTED_EXTENSIONS))
SAMPLE_BYTES = 256 * 1024
DELIMITERS = (",", ";", "\t", "|")
HEADER_SCAN_ROWS = 30
NUMERIC_SHARE = 0.98

_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_PARQUET_MAGIC = b"PAR1"

# Unigram letter frequencies (percent). Only the ranking matters: a decoding
# whose letters are typical for its own script scores higher than mojibake.
_RUSSIAN = (
    "о10.97 е8.45 а8.01 и7.35 н6.70 т6.26 с5.47 р4.73 в4.54 л4.40 к3.49 м3.21 "
    "д2.98 п2.81 у2.62 я2.01 ы1.90 ь1.74 г1.70 з1.65 б1.59 ч1.44 й1.21 х0.97 "
    "ж0.94 ш0.73 ю0.64 ц0.48 щ0.36 э0.32 ф0.26 ъ0.04 ё0.04"
)
_HEBREW = (
    "י11.06 ו10.38 ה8.68 ל7.15 א6.34 מ6.17 ר5.59 ת5.27 ב4.74 נ4.25 ש4.17 ע3.13 "
    "כ2.71 ם2.61 ד2.58 ח2.33 ק1.88 פ1.45 ן1.43 ס1.33 ג1.26 צ1.24 ט1.17 ז0.92 "
    "ך0.77 ף0.36 ץ0.32"
)
_LATIN = (
    "e12.0 a8.5 t7.8 o7.3 i7.1 n7.0 s6.6 r6.4 h4.2 l4.3 d3.9 c3.3 u3.4 m2.8 "
    "p2.3 f2.0 g2.1 w1.6 y1.6 b1.6 v1.2 k1.0 z0.6 j0.3 x0.3 q0.2 é0.6 ä0.25 "
    "ö0.2 ü0.25 ß0.1 è0.15 à0.15 ç0.06 ê0.05 ô0.03 â0.03 î0.02 û0.02 ñ0.06 "
    "á0.08 í0.06 ó0.08 ú0.03 å0.08 ø0.06 æ0.03 ë0.01 ï0.01"
)
_FLOOR_LOG_PROBABILITY = math.log(1e-5)
_CANDIDATE_ENCODINGS = ("cp1251", "cp1255", "cp1252")


def _model(table: str) -> Dict[str, float]:
    pairs = [(item[0], float(item[1:])) for item in table.split()]
    total = sum(weight for _, weight in pairs)
    return {letter: math.log(weight / total) for letter, weight in pairs}


_LETTER_MODELS = (_model(_RUSSIAN), _model(_HEBREW), _model(_LATIN))
_WORD = re.compile(r"[^\W\d_]+")


_CSV_FIELD_LIMIT_LOCK = threading.Lock()


class TableReadError(ValueError):
    """The upload is not a table this reader can interpret safely."""


@dataclass
class TableRead:
    frame: pd.DataFrame
    report: Dict[str, Any] = field(default_factory=dict)


Source = Union[str, Path, bytes, BinaryIO]


def is_supported_upload(filename: Optional[str]) -> bool:
    return bool(filename) and Path(str(filename)).suffix.lower() in SUPPORTED_EXTENSIONS


def read_table(
    source: Source,
    *,
    nrows: Optional[int] = None,
    max_columns: Optional[int] = None,
) -> TableRead:
    """Return the table and a report of format, encoding, delimiter and fixes."""
    data = _read_bytes(source)
    if not data.strip():
        raise TableReadError("The file is empty.")
    if data.startswith(_ZIP_MAGIC) or data.startswith(_OLE_MAGIC):
        result = _read_workbook(data, nrows=nrows)
    elif data.startswith(_PARQUET_MAGIC):
        result = _read_parquet(data, nrows=nrows)
    else:
        result = _read_delimited(data, nrows=nrows, max_columns=max_columns)
    frame = result.frame
    if max_columns is not None and frame.shape[1] > max_columns:
        raise TableReadError("The table exceeds the column limit.")
    frame.columns = [str(column).strip() for column in frame.columns]
    result.report["rows"] = int(len(frame))
    result.report["columns"] = int(frame.shape[1])
    result.report["warnings"] = result.report.get("warnings", []) + _suspicious_read_warnings(frame)
    return result


def _read_bytes(source: Source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "seek"):
        source.seek(0)
    data = source.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


# ---------------------------------------------------------------- text files


def detect_encoding(data: bytes) -> Tuple[str, str]:
    """Return ``(encoding, how)`` for delimited text."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "byte-order mark"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", "byte-order mark"
    sample = data[:SAMPLE_BYTES]
    if len(sample) >= 4:
        even_zero = sample[0::2].count(0) / max(1, len(sample[0::2]))
        odd_zero = sample[1::2].count(0) / max(1, len(sample[1::2]))
        if odd_zero > 0.3 and even_zero < 0.05:
            return "utf-16-le", "zero-byte pattern"
        if even_zero > 0.3 and odd_zero < 0.05:
            return "utf-16-be", "zero-byte pattern"
    try:
        data.decode("utf-8")
        return "utf-8", "valid UTF-8"
    except UnicodeDecodeError:
        pass
    best = max(_CANDIDATE_ENCODINGS, key=lambda encoding: _letter_score(sample, encoding))
    return best, "letter-frequency match"


def _script(character: str) -> int:
    code = ord(character)
    if 0x0400 <= code <= 0x04FF:
        return 0
    if 0x0590 <= code <= 0x05FF:
        return 1
    return 2


def _letter_score(sample: bytes, encoding: str) -> float:
    """Mean letter log-probability of the words that contain non-ASCII bytes.

    Each letter is scored by the model of its own script. A word that mixes
    scripts (``GrцЯe`` from German read as Cyrillic, for example) is what a
    wrong code page produces, so all of its letters get the floor value.
    """
    text = sample.decode(encoding, errors="replace")
    total, letters = 0.0, 0
    for word in _WORD.findall(text):
        if word.isascii():
            continue
        word = word.lower()
        scripts = {_script(character) for character in word}
        letters += len(word)
        if len(scripts) > 1:
            total += _FLOOR_LOG_PROBABILITY * len(word)
            continue
        model = _LETTER_MODELS[scripts.pop()]
        total += sum(model.get(character, _FLOOR_LOG_PROBABILITY) for character in word)
    return total / letters if letters else _FLOOR_LOG_PROBABILITY


def detect_delimiter(text: str) -> str:
    """Pick the delimiter that splits the first lines into the most consistent columns."""
    lines = [line for line in text.splitlines()[:60] if line.strip()]
    if len(lines) > 1:
        lines = lines[:-1]  # the sample may cut the last line
    best, best_score = ",", (0.0, 0)
    for delimiter in DELIMITERS:
        try:
            counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        except csv.Error:
            continue
        if not counts:
            continue
        mode, frequency = Counter(counts).most_common(1)[0]
        if mode < 2:
            continue
        score = (frequency / len(counts), mode)
        if score > best_score:
            best, best_score = delimiter, score
    return best


def first_record(data: bytes, encoding: str, delimiter: str) -> List[str]:
    """Parse the complete first CSV record, however wide its fields are.

    A fixed-size sample can cut a very wide header and hide columns from the
    column-limit check. ``csv.field_size_limit`` is process-global, so the
    enlarged limit is held under a lock and restored afterwards.
    """
    with _CSV_FIELD_LIMIT_LOCK:
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, len(data)))
        try:
            text = io.TextIOWrapper(
                io.BytesIO(data), encoding=encoding, errors="replace", newline=""
            )
            return next(csv.reader(text, delimiter=delimiter), [])
        finally:
            csv.field_size_limit(previous_limit)


def _read_delimited(
    data: bytes, *, nrows: Optional[int], max_columns: Optional[int]
) -> TableRead:
    encoding, encoding_reason = detect_encoding(data)
    sample = data[:SAMPLE_BYTES].decode(encoding, errors="replace")
    delimiter = detect_delimiter(sample)
    header = [str(name).strip() for name in first_record(data, encoding, delimiter)]
    if max_columns is not None and len(header) > max_columns:
        raise TableReadError("The table exceeds the column limit.")
    options: Dict[str, Any] = {
        "sep": delimiter,
        "encoding": encoding,
        "encoding_errors": "replace",
    }
    if nrows is not None:
        options["nrows"] = nrows
    try:
        frame = pd.read_csv(io.BytesIO(data), **options)
    except (pd.errors.ParserError, csv.Error, UnicodeError) as error:
        raise TableReadError("The table rows could not be split into columns.") from error
    if len(header) == frame.shape[1] and len(set(header)) != len(header):
        # pandas renames duplicates to ``name.1``; keep the real names so the
        # readiness check can report the duplicate columns.
        frame.columns = header
    frame, converted = convert_text_numbers(frame, comma_delimited=delimiter == ",")
    warnings: List[str] = []
    if "\ufffd" in sample:
        warnings.append(
            f"Some characters could not be decoded as {encoding}; text values may be damaged."
        )
    return TableRead(
        frame,
        {
            "format": "delimited text",
            "encoding": encoding,
            "encoding_detection": encoding_reason,
            "delimiter": _delimiter_name(delimiter),
            "number_formats": converted,
            "warnings": warnings,
        },
    )


def _delimiter_name(delimiter: str) -> str:
    return {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delimiter]


_DECIMAL_COMMA = re.compile(r"^[+-]?(\d{1,3}([ .\u00a0\u202f']\d{3})+|\d+)(,\d+)?$")
_THOUSANDS_COMMA = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")


def convert_text_numbers(
    frame: pd.DataFrame, *, comma_delimited: bool
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Convert text columns such as ``1 200,5`` or ``"1,200.5"`` to numbers.

    A column is converted only when at least 98% of its filled values fit one
    format. ``1,234`` alone is ambiguous: in a comma-separated file the comma
    can only appear inside quotes, which is how US-style exports write
    thousands, so it is read as 1234 there and as 1.234 everywhere else.
    """
    converted: Dict[str, str] = {}
    for column in frame.columns:
        series = frame[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if len(values) < 2 or not values.str.contains(",", regex=False).any():
            continue
        decimal_share = values.str.fullmatch(_DECIMAL_COMMA).mean()
        thousands_share = values.str.fullmatch(_THOUSANDS_COMMA).mean()
        decimal = decimal_share >= NUMERIC_SHARE
        thousands = thousands_share >= NUMERIC_SHARE
        if not (decimal or thousands):
            continue
        text = series.astype("string").str.strip()
        if thousands and (comma_delimited or not decimal):
            cleaned = text.str.replace(",", "", regex=False)
            converted[str(column)] = "thousands comma"
        else:
            cleaned = (
                text.str.replace(r"[ .\u00a0\u202f']", "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            converted[str(column)] = "decimal comma"
        frame[column] = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    return frame, converted


# ----------------------------------------------------------------- workbooks


def _read_workbook(data: bytes, *, nrows: Optional[int]) -> TableRead:
    engine = "openpyxl" if data.startswith(_ZIP_MAGIC) else "xlrd"
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine=engine)
    except Exception as error:  # corrupt or password-protected workbooks
        raise TableReadError("The workbook could not be opened.") from error
    filled = {
        name: sheet.dropna(how="all").dropna(axis=1, how="all")
        for name, sheet in sheets.items()
    }
    filled = {name: sheet for name, sheet in filled.items() if not sheet.empty}
    if not filled:
        raise TableReadError("The workbook does not contain any filled cells.")
    sheet_name = max(filled, key=lambda name: filled[name].size)
    raw = filled[sheet_name]
    header_row = _header_row(raw)
    sheet_header_row = int(raw.index[header_row]) + 1
    header = [
        value if not (isinstance(value, float) and math.isnan(value)) else f"column_{index + 1}"
        for index, value in enumerate(raw.iloc[header_row].tolist())
    ]
    frame = raw.iloc[header_row + 1:].reset_index(drop=True)
    frame.columns = header
    if nrows is not None:
        frame = frame.head(nrows)
    frame = frame.infer_objects()
    frame, converted = convert_text_numbers(frame, comma_delimited=False)
    warnings = []
    others = [name for name in sheets if name != sheet_name]
    if others:
        warnings.append(
            f"The workbook has {len(sheets)} sheets; the largest one, “{sheet_name}”, was analysed. "
            f"Other sheets: {', '.join(map(str, others[:5]))}{'…' if len(others) > 5 else ''}."
        )
    if sheet_header_row > 1:
        warnings.append(
            f"Column names were taken from row {sheet_header_row}; "
            f"the {sheet_header_row - 1} row(s) above it were treated as titles."
        )
    return TableRead(
        frame,
        {
            "format": "Excel workbook",
            "sheet": str(sheet_name),
            "sheet_count": len(sheets),
            "header_row": sheet_header_row,
            "number_formats": converted,
            "warnings": warnings,
        },
    )


def _header_row(raw: pd.DataFrame) -> int:
    """First row that is as wide as the table body: titles above it are skipped."""
    counts = raw.notna().sum(axis=1).to_numpy()
    width = int(np.median(counts[: max(HEADER_SCAN_ROWS * 3, 1)]))
    for index in range(min(HEADER_SCAN_ROWS, len(raw))):
        if counts[index] >= max(2, width) or (width < 2 and counts[index] >= 1):
            return index
    return 0


# ------------------------------------------------------------------- parquet


def _read_parquet(data: bytes, *, nrows: Optional[int]) -> TableRead:
    try:
        frame = pd.read_parquet(io.BytesIO(data))
    except Exception as error:
        raise TableReadError("The Parquet file could not be read.") from error
    if nrows is not None:
        frame = frame.head(nrows)
    return TableRead(frame, {"format": "Parquet", "warnings": []})


# ------------------------------------------------------------------ checks


# Text decoded with the wrong code page: UTF-8 read as cp1252 ("Ã©"), or
# Cyrillic read as Latin-1 ("Äàòà"): whole words of accented Latin letters,
# which no Western language writes without a plain ASCII letter.
_MOJIBAKE = re.compile(r"Ã.|Â[\u00a0-\u00bf]|\ufffd|\b[\u00c0-\u00ff]{3,}\b")


def _suspicious_read_warnings(frame: pd.DataFrame) -> List[str]:
    warnings = []
    header = " ".join(map(str, frame.columns))
    if frame.shape[1] == 1 and any(mark in header for mark in DELIMITERS):
        warnings.append(
            "Only one column was found, but its name contains separator characters; "
            "check that the file is a table with one header row."
        )
    if _MOJIBAKE.search(header):
        warnings.append("Column names contain garbled characters; the text encoding may be wrong.")
    unnamed = sum(1 for column in frame.columns if str(column).startswith("Unnamed:"))
    if unnamed:
        warnings.append(f"{unnamed} column(s) have no name in the header row.")
    return warnings


def describe_read(report: Dict[str, Any]) -> str:
    """One plain sentence on how the file was interpreted."""
    kind = report.get("format")
    if kind == "Excel workbook":
        text = f"Read as an Excel workbook, sheet “{report.get('sheet')}”, column names from row {report.get('header_row')}."
    elif kind == "Parquet":
        text = "Read as a Parquet file."
    else:
        text = (
            f"Read as {report.get('delimiter')}-separated text in {report.get('encoding')} "
            f"({report.get('encoding_detection')})."
        )
    formats = report.get("number_formats") or {}
    for style in ("decimal comma", "thousands comma"):
        columns = [name for name, value in formats.items() if value == style]
        if columns:
            noun = "decimal commas" if style == "decimal comma" else "thousands separators"
            text += f" Numbers with {noun} were converted in {len(columns)} column(s): {', '.join(columns[:5])}."
    return text
