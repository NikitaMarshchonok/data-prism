import pandas as pd

def generate_dynamic_kpis(df):
    kpis = {}  # ✅ Инициализация пустого словаря

    # 📦 Общая заполненность
    total_cells = df.shape[0] * df.shape[1]
    missing_total = df.isnull().sum().sum()
    if total_cells > 0:
        percent_filled = 100 * (1 - missing_total / total_cells)
        kpis["📦 Заполненность данных"] = f"{percent_filled:.1f}%"

    # 🧬 Первая колонка и 🔁 Уникальность
    if not df.empty:
        first_col = df.columns[0]
        kpis["🧬 Первая колонка"] = first_col

        non_null_count = df[first_col].notnull().sum()
        unique_count = df[first_col].nunique()
        uniqueness_percent = 100 * unique_count / non_null_count if non_null_count > 0 else 0
        kpis["🔁 Уникальность"] = f"{uniqueness_percent:.1f}"
    else:
        kpis["🧬 Первая колонка"] = "N/A"
        kpis["🔁 Уникальность"] = "0.0"

    # 🔎 Расширенный анализ по названиям колонок
    for col in df.columns:
        col_lower = col.lower()
        col_data = df[col]

        # 💸 Средняя цена/сумма/стоимость
        if any(word in col_lower for word in ['price', 'amount', 'sum', 'value', 'cost']):
            if pd.api.types.is_numeric_dtype(col_data):
                kpis["💸 Среднее значение"] = f"{col_data.mean():,.2f}"

        # ⭐ Рейтинг или оценка
        if any(word in col_lower for word in ['score', 'rating']):
            if pd.api.types.is_numeric_dtype(col_data):
                kpis["⭐ Средний рейтинг"] = f"{col_data.mean():.2f}"

        # 👤 Уникальные пользователи / ID
        if any(word in col_lower for word in ['user', 'id', 'uid']):
            kpis["🧠 Уникальных ID"] = str(col_data.nunique())

        # 📆 Проверим, есть ли временные данные
        if any(word in col_lower for word in ['date', 'time', 'timestamp']):
            try:
                converted = pd.to_datetime(col_data, errors='coerce')
                if converted.notnull().sum() > 0:
                    first = converted.min().strftime("%Y-%m-%d")
                    last = converted.max().strftime("%Y-%m-%d")
                    kpis["⏱ Диапазон дат"] = f"{first} → {last}"
            except Exception:
                pass

    return kpis
