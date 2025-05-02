# 📊 Data Prism

**Data Prism** is a universal tool for automated data analysis and visualization. 
Upload a dataset (CSV, Excel, JSON, or Parquet) and get an interactive BI dashboard with key metrics,
advanced visualizations, AI-generated insights, and a downloadable PDF report.

---

## 🚀 Features

- 📁 Supports: `.csv`, `.xlsx`, `.tsv`, `.json`, `.parquet`
- 📊 Interactive dashboard: KPI cards, histograms, frequency charts, time trends
- 🧠 AI Summary: automatic insights using LLMs
- 📉 Missing values, correlation matrix, outlier detection (IQR)
- 🧾 PDF export of full analytics report
- ⚡ Handles large files (up to 100,000 rows)
- 🔍 Filtering, sorting, and search-enabled data tables

---

## 📷 Screenshots

### 🔹 General Dashboard Overview

https://github.com/NikitaMarshchonok/data-prism/blob/c5583592b4699a1f7363331a420ba2cad5e9e622/5.png

---

### 🔹 Graphs: Distribution & Category Frequencies

https://github.com/NikitaMarshchonok/data-prism/blob/c5583592b4699a1f7363331a420ba2cad5e9e622/6.png

---

### 🔹 Correlation Matrix and Raw Data Table

https://github.com/NikitaMarshchonok/data-prism/blob/c5583592b4699a1f7363331a420ba2cad5e9e622/3.png
---

### 🔹 Gauge Indicators
https://github.com/NikitaMarshchonok/data-prism/blob/c5583592b4699a1f7363331a420ba2cad5e9e622/4.png

---

## 🛠️ Tech Stack

- **Python**, **Flask**, **Pandas**
- **Plotly**, **Matplotlib**, **DataTables.js**, **Bootstrap 5**
- **Jinja2** for templating
- **OpenAI API / HuggingFace API** for AI summaries
- **WeasyPrint** for generating styled PDF reports
- **HTML5 / CSS3 / JavaScript (vanilla)**
- **dotenv** for secure environment variables
- **os, io, base64, json** — internal data logic
- **Git & GitHub** for version control


---

## 🧪 Installation & Run

```bash
# Clone the repository
git clone https://github.com/NikitaMarshchonok/data-prism.git
cd data-prism

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

```

---

## 🔄 Project Status

This project is actively maintained and will be continuously improved.  
New features, performance optimizations, and visual enhancements will be added over time.

### 🛠 Planned Features

- [ ] Export filtered data to Excel
- [ ] Add AutoML prediction block
- [ ] Deploy on cloud (e.g. Render, AWS, or Railway)
- [ ] Add dynamic drill-down graphs
- [ ] Improve mobile layout and responsiveness

Stay tuned for updates! 🚀
