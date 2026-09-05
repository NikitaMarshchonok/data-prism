#  Data Prism

**Data Prism** is a universal tool for automated data analysis and visualization. 
Upload a dataset (CSV, Excel, JSON, or Parquet) and get an interactive BI dashboard with key metrics,
advanced visualizations, AI-generated insights, and a downloadable PDF report.

---

##  Features

- 📁 Supports: `.csv`, `.xlsx`, `.tsv`, `.json`, `.parquet`
- 📊 Interactive dashboard: KPI cards, histograms, frequency charts, time trends
- 🧠 AI Summary: automatic insights using LLMs
- 🔎 Evidence-based insights with metrics, support levels, and recommended next steps
- 🧪 Statistical validation with effect sizes, confidence intervals, and FDR correction
- 🧭 Multivariate anomaly scoring and quality-validated exploratory segmentation
- 🧪 Leakage-safe ML evaluation with holdout metrics and naive baseline comparison
- 📉 Missing values, correlation matrix, outlier detection (IQR)
- 🛡️ Data quality score with duplicate, constant-column, and outlier recommendations
- 🧾 PDF export of full analytics report
- ⚡ Handles large files (up to 100,000 rows)
- 🔍 Filtering, sorting, and search-enabled data tables


---

##  Screenshots

###  General Dashboard Overview



---

###  Graphs: Distribution & Category Frequencies



---

###  Correlation Matrix and Raw Data Table


---

###  Gauge Indicators


---

##  Tech Stack

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

##  Installation & Run

```bash
# Clone the repository
git clone https://github.com/NikitaMarshchonok/data-prism.git
cd data-prism

# Install dependencies
pip install -r requirements.txt

# Configure a persistent session key (required outside local development)
export FLASK_SECRET_KEY="replace-with-a-long-random-value"

# Optional: upload limit in megabytes (default: 100)
export MAX_UPLOAD_MB=100

# Optional for local development only
export FLASK_DEBUG=true

# Run the Flask web app
python web_app.py

```

---



##  Project Status

This project is actively maintained and will be continuously improved.  
New features, performance optimizations, and visual enhancements will be added over time.


###  Planned Features

- [ ] Export filtered data to Excel
- [x] Add leakage-safe baseline ML evaluation block
- [x] Compare multiple model families with cross-validation
- [x] Add auditable statistical hypothesis validation
- [x] Add anomaly detection and validated segmentation
- [ ] Deploy on cloud (e.g. Render, AWS, or Railway)
- [ ] Add dynamic drill-down graphs
- [ ] Improve mobile layout and responsiveness

Stay tuned for updates! 









