# src/data_loader.py
import os

import pandas as pd


SUPPORTED_DATA_EXTENSIONS = frozenset({'.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet'})


def is_supported_data_file(filename):
    """Возвращает True, если расширение файла поддерживается загрузчиком."""
    if not filename:
        return False
    return os.path.splitext(filename)[1].lower() in SUPPORTED_DATA_EXTENSIONS


def load_data(file_path, max_rows=100000):
    """
    Загружает данные из различных форматов с ограничением на количество строк.
    :param file_path: путь к файлу
    :param max_rows: максимум строк для загрузки
    :return: DataFrame, truncated (флаг обрезки)
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    df = None
    truncated = False

    try:
        if not isinstance(max_rows, int) or max_rows <= 0:
            raise ValueError("max_rows должен быть положительным целым числом.")

        read_limit = max_rows + 1
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=read_limit)
        elif ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t', nrows=read_limit)
        elif ext == '.json':
            try:
                df = pd.read_json(file_path)
            except ValueError:
                df = pd.read_json(file_path, lines=True)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif ext == '.csv':
            for enc in ['utf-8', 'ISO-8859-1', 'windows-1252']:
                try:
                    df = pd.read_csv(file_path, encoding=enc, nrows=read_limit)
                    break
                except Exception:
                    continue
            else:
                raise ValueError("❌ Ни одна из кодировок не подошла для CSV.")
        else:
            raise ValueError(f"❌ Формат файла '{ext}' не поддерживается.")

        if df is None or df.shape[1] == 0:
            raise ValueError("❌ Файл не содержит доступных для анализа столбцов.")

        if len(df) > max_rows:
            df = df.iloc[:max_rows].copy()
            truncated = True

        print(f"✅ Файл '{filename}' успешно загружен. Строк: {df.shape[0]}, Колонок: {df.shape[1]}")
        if truncated:
            print(f"⚠️ Данные были обрезаны до {max_rows:,} строк для оптимальной загрузки.")

        return df, truncated

    except Exception as e:
        print(f"❌ Ошибка при загрузке файла '{filename}': {e}")
        return None, False
