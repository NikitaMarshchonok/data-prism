import io
import unittest

import numpy as np
import pandas as pd

from src.row_selection import select_rows
from src.table_reader import (
    TableReadError,
    convert_text_numbers,
    describe_read,
    detect_delimiter,
    detect_encoding,
    is_supported_upload,
    read_table,
)


VOCABULARY = {
    "cp1251": (
        "дата выручка регион игрок команда голы город сезон музей посетители проба "
        "температура выставка зритель билет стоимость количество средний доход клиент "
        "заказ товар магазин счёт матч турнир время площадь население район школа"
    ).split(),
    "cp1255": (
        "תאריך הכנסות אזור שחקן קבוצה שערים עיר עונה מוזיאון מבקרים דגימה טמפרטורה "
        "תערוכה צופה כרטיס מחיר כמות ממוצע הכנסה לקוח הזמנה מוצר חנות משחק טורניר "
        "זמן שטח אוכלוסייה מחוז בית ספר"
    ).split(),
    "cp1252": (
        "größe umsatz städte gemüse prüfung qualität straße müller jährlich "
        "région équipe été société café élève année données mesure température "
        "año niño señal región población campeón estación sesión "
        "säsong målvakt höjd åtgärd ålder"
    ).split(),
}


def _sample(rng, encoding, rows):
    words = VOCABULARY[encoding]
    delimiter = rng.choice([",", ";", "\t"])
    header = delimiter.join(rng.choice(words, size=4, replace=False))
    lines = [header]
    for _ in range(rows):
        lines.append(delimiter.join([
            str(rng.choice(words)), str(rng.integers(0, 1000)), str(rng.choice(words)), f"{rng.random():.2f}",
        ]))
    return "\r\n".join(lines).encode(encoding), delimiter


class EncodingDetectionQualityTests(unittest.TestCase):
    def test_code_pages_are_recognised_from_short_and_longer_files(self):
        rng = np.random.default_rng(2026)
        misses = {encoding: 0 for encoding in VOCABULARY}
        trials = 150
        for encoding in VOCABULARY:
            for trial in range(trials):
                data, _ = _sample(rng, encoding, rows=2 if trial % 3 == 0 else 20)
                if detect_encoding(data)[0] != encoding:
                    misses[encoding] += 1
        for encoding, missed in misses.items():
            self.assertLessEqual(missed / trials, 0.02, f"{encoding}: {missed} of {trials} misread")

    def test_delimiter_is_recovered_for_every_sample(self):
        rng = np.random.default_rng(7)
        for encoding in VOCABULARY:
            for _ in range(50):
                data, delimiter = _sample(rng, encoding, rows=10)
                self.assertEqual(detect_delimiter(data.decode(encoding)), delimiter)


class TableReaderTests(unittest.TestCase):
    def test_russian_excel_csv_is_read_as_columns_with_numbers(self):
        text = "Дата;Игрок;Голы;Рейтинг\r\n2025-01-01;Иванов;2;7,5\r\n2025-01-02;Петров;0;6,25\r\n"
        result = read_table(text.encode("cp1251"))
        self.assertEqual(list(result.frame.columns), ["Дата", "Игрок", "Голы", "Рейтинг"])
        self.assertEqual(result.frame["Рейтинг"].tolist(), [7.5, 6.25])
        self.assertEqual(result.frame["Рейтинг"].dtype, np.float64)
        self.assertEqual(result.report["encoding"], "cp1251")
        self.assertEqual(result.report["delimiter"], "semicolon")
        self.assertEqual(result.report["number_formats"], {"Рейтинг": "decimal comma"})
        self.assertIn("semicolon-separated text in cp1251", describe_read(result.report))

    def test_hebrew_code_page_and_quoted_thousands(self):
        text = 'תאריך,סניף,הכנסות\r\n2025-01-01,חיפה,"1,250.50"\r\n2025-01-02,עכו,"12,000"\r\n'
        result = read_table(text.encode("cp1255"))
        self.assertEqual(result.report["encoding"], "cp1255")
        self.assertEqual(result.frame["הכנסות"].tolist(), [1250.5, 12000.0])
        self.assertEqual(result.report["number_formats"], {"הכנסות": "thousands comma"})

    def test_european_thousands_with_spaces_and_dots(self):
        frame = pd.DataFrame({"value": ["1 200,5", "3.400,25", "12,0"], "label": ["a, b", "c", "d"]})
        converted, formats = convert_text_numbers(frame, comma_delimited=False)
        self.assertEqual(converted["value"].tolist(), [1200.5, 3400.25, 12.0])
        self.assertEqual(converted["label"].tolist(), ["a, b", "c", "d"])
        self.assertEqual(formats, {"value": "decimal comma"})

    def test_mostly_text_column_is_never_converted(self):
        frame = pd.DataFrame({"code": ["1,5", "2,5", "n/a", "unknown", "3,5"]})
        converted, formats = convert_text_numbers(frame, comma_delimited=False)
        self.assertEqual(formats, {})
        self.assertEqual(converted["code"].tolist(), ["1,5", "2,5", "n/a", "unknown", "3,5"])

    def test_utf16_tab_separated_unicode_text(self):
        text = "Region\tYear\tValue\r\nNorth\t2024\t10\r\nSouth\t2025\t12\r\n"
        result = read_table(text.encode("utf-16"))
        self.assertEqual(result.report["encoding"], "utf-16")
        self.assertEqual(result.report["delimiter"], "tab")
        self.assertEqual(result.frame["Value"].tolist(), [10, 12])

    def test_plain_utf8_csv_is_unchanged(self):
        data = b"a,b,c\n1,2,x\n3,4,y\n"
        result = read_table(data)
        pd.testing.assert_frame_equal(result.frame, pd.read_csv(io.BytesIO(data)))
        self.assertEqual(result.report["encoding"], "utf-8")
        self.assertEqual(result.report["warnings"], [])

    def test_duplicate_header_names_are_kept_for_the_readiness_check(self):
        result = read_table(b"x,x,y\n1,2,3\n4,5,6\n")
        self.assertEqual(list(result.frame.columns), ["x", "x", "y"])

    def test_column_limit_is_checked_on_the_complete_header(self):
        header = ",".join(["x" * (300 * 1024)] + [f"c{i}" for i in range(10)])
        with self.assertRaises(TableReadError):
            read_table((header + "\n" + ",".join(["1"] * 11) + "\n").encode(), max_columns=5)

    def test_excel_workbook_with_title_rows_and_several_sheets(self):
        buffer = io.BytesIO()
        body = pd.DataFrame({
            "Museum": ["A", "B", "A", "C"],
            "Date": pd.date_range("2025-01-01", periods=4),
            "Visitors": [120, 340, 150, 90],
        })
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame([["Visitors 2025"]]).to_excel(writer, sheet_name="Data", header=False, index=False)
            body.to_excel(writer, sheet_name="Data", startrow=2, index=False)
            pd.DataFrame({"note": ["x"]}).to_excel(writer, sheet_name="Notes", index=False)
        result = read_table(buffer.getvalue())
        self.assertEqual(list(result.frame.columns), ["Museum", "Date", "Visitors"])
        self.assertEqual(result.frame["Visitors"].tolist(), [120, 340, 150, 90])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.frame["Date"]))
        self.assertEqual(result.report["sheet"], "Data")
        self.assertEqual(result.report["header_row"], 3)
        self.assertEqual(len(result.report["warnings"]), 2)

    def test_parquet_is_read(self):
        buffer = io.BytesIO()
        pd.DataFrame({"t": pd.date_range("2025-01-01", periods=3, freq="s"), "v": [1.0, 2.0, 3.0]}).to_parquet(buffer)
        result = read_table(buffer.getvalue())
        self.assertEqual(result.report["format"], "Parquet")
        self.assertEqual(result.frame["v"].tolist(), [1.0, 2.0, 3.0])

    def test_unreadable_inputs_raise_a_table_error(self):
        with self.assertRaises(TableReadError):
            read_table(b"   \n")
        with self.assertRaises(TableReadError):
            read_table(b"PK\x03\x04not really a workbook")

    def test_garbled_single_column_reads_are_flagged(self):
        result = read_table("Äàòà;Âûðó÷êà\n".encode("utf-8") + b"1\n")
        self.assertTrue(result.report["warnings"])

    def test_supported_extensions(self):
        for name in ("a.csv", "b.TSV", "c.txt", "d.xlsx", "e.xls", "f.parquet"):
            self.assertTrue(is_supported_upload(name))
        for name in ("a.pdf", "b", "", None):
            self.assertFalse(is_supported_upload(name))


class RowSelectionTests(unittest.TestCase):
    def test_dated_tables_keep_their_most_recent_rows_in_original_order(self):
        dates = pd.date_range("2020-01-01", periods=1000).astype(str)
        frame = pd.DataFrame({"Date": dates, "value": np.arange(1000)}).sample(frac=1, random_state=3)
        kept, info = select_rows(frame.reset_index(drop=True), 100)
        self.assertEqual(info["method"], "latest_rows_by_time")
        self.assertEqual(info["time_column"], "Date")
        self.assertEqual(sorted(kept["value"].tolist()), list(range(900, 1000)))
        original_order = [value for value in frame["value"] if value >= 900]
        self.assertEqual(kept["value"].tolist(), original_order)

    def test_undated_tables_keep_a_reproducible_random_sample(self):
        frame = pd.DataFrame({"group": np.repeat(list("abcd"), 250), "value": np.arange(1000)})
        first, info = select_rows(frame, 200)
        second, _ = select_rows(frame, 200)
        self.assertEqual(info["method"], "random_sample")
        self.assertEqual(first["value"].tolist(), second["value"].tolist())
        self.assertEqual(set(first["group"]), set("abcd"))
        self.assertTrue(first["value"].is_monotonic_increasing)

    def test_small_tables_are_untouched(self):
        frame = pd.DataFrame({"value": [1, 2, 3]})
        kept, info = select_rows(frame, 10)
        self.assertIs(kept, frame)
        self.assertEqual(info["method"], "all_rows")


if __name__ == "__main__":
    unittest.main()
