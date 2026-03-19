# Screen Time, Sleep & Stress Analysis

A Streamlit-based ML + LLM application that:

- trains a stress-level classifier from CSV data,
- compares multiple ML models and selects the best one,
- serves interactive predictions in a UI,
- and generates AI responses using the Hugging Face Router API.

---

## Features

- Dynamic CSV loading with schema inspection
- Automatic preprocessing pipeline generation
- Multi-model training:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Best-model selection by accuracy
- Saved artifacts for reuse:
  - `models/model.pkl`
  - `models/preprocessor.pkl`
  - `models/metrics.json`
- Streamlit pages:
  - Dashboard
  - Model Prediction
  - AI Response Generator
  - Model Analytics
  - Admin Panel (upload + train/retrain)

---

## Project Structure

```text
project/
├── app.py
├── train.py
├── predict.py
├── hf_api.py
├── utils.py
├── config.py
├── requirements.txt
├── assets/
│   └── style.css
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── metrics.json
└── data/
    └── dataset.csv
```

---

## Quick Start (Windows / macOS / Linux)

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd project
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

- **macOS/Linux**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add dataset

Put your CSV at:

- `data/dataset.csv` (recommended), or
- update `RAW_DATA_PATH` in `config.py` to your absolute dataset path.

The dataset must include the target column:

- `Stress_Level`

### 5) Configure Hugging Face token

Set `HF_TOKEN` as an environment variable (recommended):

- **Windows (PowerShell)**

```powershell
$env:HF_TOKEN="your_hf_token_here"
```

- **macOS/Linux**

```bash
export HF_TOKEN="your_hf_token_here"
```

### 6) Train model (optional manual step)

```bash
python train.py
```

If you skip this step, the app auto-trains when model artifacts are missing.

### 7) Run app

```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

---

## How to Use the App

### Dashboard

- View row/feature counts
- View best model accuracy
- View class distribution chart

### Model Prediction

- Fill feature values from generated form
- Click **Run Prediction**
- See predicted stress class and confidence

### AI Response Generator

- Enter user/ticket text
- Generate response from Hugging Face Router model

### Model Analytics

- Compare model accuracies
- Inspect confusion matrix (best model)
- View feature importance when available

### Admin Panel

- Upload a new CSV
- Click **Train / Retrain Model**
- Refresh analytics and prediction with updated artifacts

---

## Configuration

Main config file: `config.py`

Important values:

- `PROJECT_TITLE`
- `RAW_DATA_PATH`
- `TARGET_COLUMN` (`Stress_Level`)
- `MODEL_PATH`, `PREPROCESSOR_PATH`, `METRICS_PATH`

---

## Tech Stack

- Streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- plotly
- requests
- joblib

---

## Troubleshooting

### `Model not available` / `No metrics available`

- Go to **Admin Panel**
- Click **Train / Retrain Model**
- Ensure training completes without errors
- Confirm files exist:
  - `models/model.pkl`
  - `models/metrics.json`

### Dataset not found

- Ensure file exists at `data/dataset.csv`, or
- update dataset path in `config.py`

### LLM errors

- Verify `HF_TOKEN` is set in your environment
- Check internet connectivity
- Re-run after token validation

### Streamlit not opening

- Ensure virtual env is active
- Run:
  - `python -m streamlit run app.py`

---

## Security Notes

- Do not commit secrets or API tokens to GitHub.
- Prefer environment variables for credentials.
- Rotate tokens immediately if exposed.

---

## Roadmap (Suggested)

- Add per-model fault tolerance (continue if one model fails)
- Improve high-cardinality categorical encoding path
- Add test suite (unit + integration)
- Add experiment tracking/versioning
- Add containerized deployment (Docker)

---

## License

Add your license here (for example: MIT).

