# Machine Learning Based Intrusion Detection System (IDS)

This project implements the **Machine Learning based Intrusion Detection System** described in the provided report:
- Data preprocessing (cleaning, encoding, scaling)
- XGBoost model training and evaluation
- Intrusion detection on uploaded datasets
- Visualization dashboard with basic authentication
- Optional live monitoring (packet sniffing) with a safe fallback simulation

## Quickstart

### 1) Setup

```bash
cd /home/dell/project-endsem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) One-command demo setup (NSL-KDD)

This will download NSL-KDD CSVs, create `data/train.csv` + `data/test.csv` with a `label` column, and train the model.

```bash
python scripts/setup_demo.py
```

### 2) Train a model (using your dataset CSV)

Put a CSV dataset at `data/train.csv` (or provide a path). The CSV must include a label column:
- default label column name: `label`
- label values: `normal` / `attack` (case-insensitive), or `0/1`

```bash
python -m ids.train --data data/train.csv --label-col label --out models/ids_model.joblib
```

### 3) Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Login defaults (change via env vars):
- `IDS_USER` (default `admin`)
- `IDS_PASS` (default `admin123`)

Example:

```bash
IDS_USER=admin IDS_PASS=admin123 streamlit run app/streamlit_app.py
```

## Data notes (NSL-KDD / CICIDS2017)

The report references NSL-KDD and CICIDS2017. Those datasets are large and not bundled here.
You can export them to CSV and point `ids.train` to the CSV.

## Project structure

- `ids/`: core package (preprocess, model, predict, live monitoring helpers)
- `app/`: Streamlit dashboard (auth, upload, charts, live view)
- `models/`: saved model artifacts (created after training)
- `data/`: your datasets (not committed by default)

## Commands

- Train: `python -m ids.train --data data/train.csv --label-col label`
- Predict CSV: `python -m ids.predict --model models/ids_model.joblib --data data/test.csv`

