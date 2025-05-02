import os

def export_report_to_pdf(html_path, pdf_path):
    try:
        from weasyprint import HTML
        HTML(html_path).write_pdf(pdf_path)
        print(f"✅ PDF-отчёт успешно создан: {pdf_path}")
    except Exception as e:
        print(f"❌ Не удалось создать PDF: {e}")
