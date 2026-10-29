# Unknown Attack Behaviour Detector

> A college-level machine-learning cybersecurity project that learns normal network traffic patterns and identifies unusual traffic that may indicate unknown or zero-day attacks.

---

## Abstract

This project implements an **unsupervised anomaly detection system** for network traffic analysis using the **Isolation Forest** algorithm. Unlike supervised classifiers that require labelled attack examples, Isolation Forest learns what "normal" traffic looks like and flags statistical outliers as potential threats. The web-based dashboard provides real-time visualisations, downloadable reports, and transparent explanations for each flagged record.

---

## Problem Statement

Traditional signature-based intrusion detection systems (IDS) fail against **zero-day attacks** — threats that have never been seen before and for which no known signature exists. A system that can identify *abnormal behaviour* without prior knowledge of specific attack types would provide an additional layer of defence.

---

## Objectives

1. Build an unsupervised ML model that detects anomalous network flows.
2. Provide a professional, browser-accessible dashboard for non-experts.
3. Support real-world CIC-IDS2017-style CSV datasets as well as synthetic demo data.
4. Present results with transparent explanations and appropriate caveats.
5. Maintain academic honesty: never overstate model capabilities or fabricate metrics.

---

## Existing vs Proposed System

| Aspect | Existing Systems | Proposed System |
|---|---|---|
| Detection Method | Signature / rule-based | Unsupervised anomaly detection |
| Zero-Day Coverage | ❌ Misses novel attacks | ✅ Flags statistical outliers |
| Labelled Data Required | ✅ Yes (large datasets) | ❌ No labels needed for training |
| Interpretability | Low | Heuristic feature-deviation explanations |
| Deployment | Complex SIEM integration | Lightweight Flask web app |

---

## Architecture & Workflow

```
                        ┌──────────────────┐
                        │  CSV Network Flow│
                        │  Data (upload or │
                        │  demo dataset)   │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Data Cleaning   │
                        │  (inf, NaN,      │
                        │   constant cols) │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Feature         │
                        │  Selection &     │
                        │  StandardScaler  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Isolation       │
                        │  Forest          │
                        │  (unsupervised)  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴──────────┐
                    │                       │
           ┌────────▼──────┐      ┌─────────▼──────┐
           │  Anomaly Score │      │  Heuristic      │
           │  & Prediction  │      │  Explanation    │
           └────────┬───────┘      └─────────┬───────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                        ┌────────▼──────────┐
                        │  Flask REST API   │
                        └────────┬──────────┘
                                 │
                        ┌────────▼──────────┐
                        │  Chart.js Dashboard│
                        │  (Dark Navy / Cyan)│
                        └───────────────────┘
```

---

## Algorithm Used

### Isolation Forest (Liu et al., 2008)

Isolation Forest isolates observations by randomly selecting a feature and then randomly selecting a split value between the minimum and maximum values of that feature. Because anomalous data points are rare and different, they require fewer random splits to isolate — they have **shorter path lengths** in the isolation trees.

- **Anomaly Score** = `score_samples()` — more negative = more anomalous.
- **Contamination** = 0.10 (assumed 10% anomaly rate in training data).
- **n_estimators** = 100 isolation trees.
- **Threshold**: Records at or below the 10th percentile of scores are classified as anomalies.

> **Important**: Isolation Forest identifies statistical outliers. It does **not** prove that a record is a specific attack type, nor does it recognise named attack families unless a separately trained and evaluated classifier supports that claim.

---

## Dataset Description

### Built-in Synthetic Demo Data (`data/demo_dataset.csv`)

- **500 records**: 400 normal + 100 anomaly-like synthetic flows.
- **16 network-flow features**: duration, packet counts, byte counts, IAT statistics, TCP flags.
- **3 synthetic anomaly archetypes**: port-scan-like, flood-like, exfiltration-like.
- **CLEARLY LABELLED**: This is synthetic data generated by `data/generate_demo_data.py`. It is NOT real network traffic.

### Real Dataset Support (CIC-IDS2017)

The system supports the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset out of the box. Download individual CSV files from the UNB website and upload them through the dashboard. The preprocessing pipeline handles:

- Column name whitespace stripping.
- Missing and infinite values.
- Constant / near-zero-variance feature removal.
- Automatic feature detection from ~70 known CIC-IDS2017 column names.

---

## Project Structure

```
Unknown-Attack-Detector/
├── app.py                      # Flask web application + REST API
├── train_model.py              # Standalone Isolation Forest training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── demo_dataset.csv        # Synthetic demo data (500 records)
│   └── generate_demo_data.py   # Script to regenerate demo data
├── models/
│   ├── isolation_forest.pkl    # Trained model (created by train_model.py)
│   ├── scaler.pkl              # StandardScaler (created by train_model.py)
│   ├── features.json           # List of features used (created by train_model.py)
│   └── metrics.json            # Training metrics (created by train_model.py)
├── templates/
│   └── index.html              # Dashboard SPA (Jinja2 template)
├── static/
│   ├── style.css               # Dark navy + cyan theme
│   └── script.js               # Chart.js + Fetch API dashboard logic
├── tests/
│   └── test_pipeline.py        # Unit tests (pytest)
└── uploads/                    # Temporary upload directory (auto-created)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or navigate to the project directory
cd Unknown-Attack-Detector

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Generate Demo Data and Train the Model

```bash
# Generate the synthetic demo dataset (if not already present)
python data/generate_demo_data.py

# Train the Isolation Forest on the demo data
python train_model.py
```

The training script prints metrics (precision, recall, F1 if labels are available) and saves model artefacts to the `models/` directory.

To train on your own CIC-IDS2017 CSV:

```bash
python train_model.py --csv path/to/your/dataset.csv
```

### Step 2 — Launch the Flask Application

```bash
python app.py
```

Open your browser and navigate to: **http://localhost:5000**

---

## Dashboard Features

| Feature | Description |
|---|---|
| Run Demo Analysis | Analyse the 500-record synthetic demo dataset instantly |
| Upload & Analyse | Upload your own CSV (up to 10 MB) for anomaly detection |
| Retrain Model | Upload a CSV to retrain the Isolation Forest |
| Stats Cards | Total records, normal count, anomaly count, anomaly rate |
| Traffic Distribution | Doughnut chart (normal vs anomalous) |
| Score Histogram | Bar chart of anomaly score distribution |
| Score Timeline | Line chart of scores for first 200 records |
| Suspicious Table | Top anomalies with severity badges and explanations |
| Download Report | Download predictions + scores as CSV |
| Model Info | Feature list, metrics, confusion matrix |

---

## Testing and Evaluation

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

The test suite covers:

- Synthetic data generation and label distribution.
- `clean_df()` — handles inf values and whitespace column names.
- `select_features()` — drops constant columns, raises on all-NaN.
- `impute_and_scale()` — no NaN in output, handles missing columns.
- `train()` — creates model files, produces evaluation metrics when labels present.
- `assign_severity()` — correct threshold-based labelling.
- `extract_labels()` — correct binary mapping.
- `evaluate()` — correct metric key output.
- Edge cases: empty DataFrame, single-row DataFrame, all-NaN columns.

### Manual Evaluation with Labelled Data

If you have a CIC-IDS2017 CSV with a `Label` column:

```bash
python train_model.py --csv data/your_labelled_data.csv
```

Precision, recall, F1-score, and confusion matrix will be printed and saved to `models/metrics.json`.

> **Caveat**: Metrics computed on training data are optimistic. For unbiased evaluation, use a separate held-out test set.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve dashboard |
| `/api/demo` | GET | Analyse synthetic demo data |
| `/api/upload` | POST | Upload CSV and run analysis |
| `/api/train` | POST | Retrain model on uploaded CSV |
| `/api/model-info` | GET | Return model metadata and metrics |
| `/api/download` | GET | Download last predictions as CSV |
| `/api/sample-csv` | GET | Download sample CSV template |

---

## Severity Labels

| Label | Anomaly Score Range | Meaning |
|---|---|---|
| **High** | < −0.25 | Strong statistical deviation. Requires immediate review. |
| **Medium** | −0.25 to −0.10 | Moderate deviation. Should be investigated. |
| **Low** | ≥ −0.10 | Mild deviation. Likely benign but slightly unusual. |

> ⚠ Severity labels are **heuristic risk indicators** based on the Isolation Forest anomaly score. They are NOT proven attack severity levels and must not be used as evidence of malicious activity without expert analysis.

---

## Limitations

1. **Unsupervised only**: Without labelled real-world attack data, the model cannot verify what it detects.
2. **Synthetic training data**: The demo model is trained on generated data, not real traffic. Do not use demo metrics in production contexts.
3. **Training-data evaluation**: Metrics shown in the dashboard are computed on training data and are optimistic.
4. **No packet payload inspection**: The system analyses flow-level statistics only.
5. **Static threshold**: The 10% contamination rate may not suit all environments.
6. **No real-time stream processing**: Batch CSV analysis only.
7. **No authentication**: The Flask app has no login system — do not expose publicly without adding authentication.

---

## Future Enhancements

- [ ] Real-time packet capture integration (e.g., via Scapy or libpcap).
- [ ] Autoencoder-based anomaly detection for richer representations.
- [ ] SHAP values for more rigorous feature attribution.
- [ ] Role-based access control and user authentication.
- [ ] Alert email / webhook notifications for high-severity anomalies.
- [ ] Time-series anomaly detection with sliding window analysis.
- [ ] Docker containerisation for one-command deployment.
- [ ] Multi-model ensemble (Isolation Forest + LOF + Autoencoder).
- [ ] Streaming API integration with real network monitoring tools.

---

## Acknowledgements & References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM 2008*.
- Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset. *ICISSP 2018*. (CIC-IDS2017 dataset)
- Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

---

*This project was built for educational purposes as a college-level ML demonstration. Results are for research use only.*
