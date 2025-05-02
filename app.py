# app.py
from src.data_loader import load_data
from src.data_analyzer import analyze_data
from src.visualizer import plot_histogram
from src.report_generator import generate_report


def main():
    # Путь к тестовому файлу
    file_path = 'data/test_data.csv'

    # Загрузка данных
    data = load_data(file_path)
    if data is None:
        return

    # Анализ данных
    stats, numeric_cols = analyze_data(data)

    # Создание графиков
    images = {}
    for col in numeric_cols:
        image_path = f'images/{col}_hist.png'
        plot_histogram(data, col, image_path)
        images[col] = image_path

    # Генерация отчёта
    generate_report(stats, images)
    print("Отчёт сгенерирован успешно!")


if __name__ == '__main__':
    main()
