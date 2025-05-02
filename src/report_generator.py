from jinja2 import Environment, FileSystemLoader

def generate_report(stats, interactive_charts, missing_data=None, corr_chart=None,
                    categorical_data=None, column_overview=None, column_descriptions=None,
                    time_trends=None,# ✅ добавлено
                    template_path='templates',
                    template_file='report_template.html',
                    output_file='final_report.html'):

    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(template_file)

    report_html = template.render(
        stats=stats,
        interactive_charts=interactive_charts,
        missing_data=missing_data,
        corr_chart=corr_chart,
        categorical_data=categorical_data,
        column_overview=column_overview,
        time_trends=time_trends,
        column_descriptions=column_descriptions
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_html)

    print(f"✅ Отчёт успешно создан: {output_file}")
