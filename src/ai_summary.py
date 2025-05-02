'''import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def generate_ai_summary(df: pd.DataFrame) -> str:
    try:
        sample = df.head(100).to_csv(index=False)

        prompt = f"""
"Ты профессиональный дата-аналитик. Сделай краткое резюме по следующим данным таблицы. "
    "Ответ предоставь в виде структурированного отчета с пунктами. Не пиши ничего лишнего — "
    "не повторяй саму инструкцию и не добавляй вводных слов. Укажи:\n"
    "1. Основные статистические наблюдения по первым 5 строкам таблицы\n"
    "2. Интересные закономерности и аномалии\n"
    "3. Возможные выводы и рекомендации\n\n"
    f"Вот таблица:\n\n{preview}"

Вот фрагмент данных:
{sample}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты — эксперт по анализу данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Ошибка при генерации AI-резюме: {e}")
        return "⚠️ Не удалось сгенерировать AI-анализ."
        '''

# src/ai_summary.py


import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_summary_openai(df: pd.DataFrame, max_rows=4) -> str:
    preview = df.head(max_rows).iloc[:, :6].to_string(index=False)
    num_rows, num_cols = df.shape

    prompt = (
        f"Проанализируй таблицу данных. Кол-во строк: {num_rows}, колонок: {num_cols}.\n"
        "Сделай краткий и чёткий отчёт в формате:\n\n"
        "**Общие данные:**\n"
        "- Кол-во строк: ...\n"
        "- Колонок: ...\n\n"
        "**Ключевые наблюдения:**\n"
        "- ...\n\n"
        "**Аномалии:**\n"
        "- ...\n\n"
        "**Вывод:**\n"
        "...\n\n"
        f"Вот первые строки таблицы:\n{preview}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # или gpt-4, если доступен
            messages=[
                {"role": "system", "content": "Ты — профессиональный дата-аналитик."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=700
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Ошибка при генерации AI Summary с ChatGPT: {e}"
