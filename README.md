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
- 🔬 Holdout permutation importance, per-class metrics, calibration, and residual diagnostics
- 🧯 Train/holdout stability checks and sufficiently supported subgroup performance diagnostics
- 🛰️ Persistent aggregate baselines with PSI, categorical drift, missingness, and schema-change monitoring
- 🚨 Session-isolated drift history with deduplicated in-app alerts and configurable retention
- 🔌 API-key-protected drift baselines, checks, history, full reports, and alert acknowledgement
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

# Optional: number of drift runs retained per browser session (default: 100)
export DRIFT_HISTORY_RETENTION=100

# Run the Flask web app
python web_app.py

```

---

## Monitoring API

The monitoring API is disabled until a key is configured:

```bash
export DATA_PRISM_API_KEY="replace-with-a-long-random-value"
python web_app.py
```

Create an aggregate baseline profile without retaining the uploaded raw file:

```bash
curl -X POST http://localhost:5001/api/v1/drift/baselines \
  -H "X-API-Key: $DATA_PRISM_API_KEY" \
  -F "datafile=@baseline.csv"
```

Run a drift check using the returned `baseline_id`:

```bash
curl -X POST http://localhost:5001/api/v1/drift/checks \
  -H "X-API-Key: $DATA_PRISM_API_KEY" \
  -H "Idempotency-Key: batch-2026-09-06" \
  -F "baseline_id=$BASELINE_ID" \
  -F "datafile=@current.csv"
```

Read recent runs and active alerts:

```bash
curl http://localhost:5001/api/v1/drift/runs -H "X-API-Key: $DATA_PRISM_API_KEY"
curl http://localhost:5001/api/v1/drift/alerts -H "X-API-Key: $DATA_PRISM_API_KEY"
```

Bearer authentication is also supported. API-key rotation intentionally creates a new isolated
monitoring scope. Without an `Idempotency-Key`, identical uploads against the same baseline are
deduplicated by their SHA-256 content hash.

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
- [x] Add model diagnostics and holdout explainability
- [x] Add split-stability and subgroup reliability diagnostics
- [x] Add persistent baseline-to-current data drift monitoring
- [x] Add persistent drift history and in-app alert events
- [x] Add authenticated machine-readable drift monitoring API
- [ ] Add scheduled drift runs and external alert delivery
- [ ] Deploy on cloud (e.g. Render, AWS, or Railway)
- [ ] Add dynamic drill-down graphs
- [ ] Improve mobile layout and responsiveness

Stay tuned for updates! 









