# Indian Stock Buy/Sell Recommendation System
AIMLCZG549 — API-Driven Cloud Native Solutions — Assignment I

## ⚠️ Before you run this
This was built and syntax/logic-tested in a sandbox **without internet
access to Yahoo Finance**. Every module except `ingester.py`'s live
network call has been validated end-to-end using synthetic data
(`test_with_synthetic_data.py`). You must run this on a machine/VM
with normal internet access.

## Setup

```bash
# 1. Clone/copy this folder, then:
cd stock_recommender
pip install -r requirements.txt

# 2. Start the full stack (Postgres, MLflow, Airflow)
docker compose up -d --build

# 3. Wait ~30s for airflow-init to finish, then open:
#    Airflow UI:  http://localhost:8080  (user: admin / pass: admin)
#    MLflow UI:   http://localhost:5000

# 4. Run ingestion once manually (populates Postgres + local CSVs)
python ingester.py

# 5. Run the ML pipeline (trains RF + XGBoost, logs to MLflow)
python train_ml.py

# 6. Retrieve application details via APIs
python api_access.py
```

The Airflow DAG (`dags/stock_pipeline_dag.py`) will pick up automatically
inside the `airflow-scheduler` container and run preprocessing + EDA
every 2 minutes against the latest ingested data (see the docstring in
that file for why ingestion itself runs on a separate, slower schedule).

## Before submitting
1. Run `python test_with_synthetic_data.py` locally first to sanity-check
   the environment, then run the real pipeline with live data once
   internet access + Docker are confirmed working.
2. Take screenshots of: Airflow DAG graph view, DAG run history (showing
   the 2-minute cadence), MLflow experiment runs list, MLflow model
   registry, and the `eda_plots/` outputs — these go in the Word report.
3. Fill in the Group No + contribution table in the report.
4. Record the demo video required by the submission guidelines.
5. Resolve the "public repository vs live API" data source question
   with your professor before finalizing — see prior conversation.

## Project Structure
| File | Purpose |
|---|---|
| `config.py` | All configurable parameters (stocks, dates, DB, MLflow, Airflow) |
| `ingester.py` | 1.2 Data ingestion via yfinance → PostgreSQL |
| `preprocessor.py` | 1.3 Pre-processing: stats, missing values, imputation, normalization, encoding |
| `feature_engineer.py` | Technical indicators (MA, RSI, MACD, Bollinger Bands) + Buy/Sell label |
| `eda_analysis.py` | 1.4 EDA: correlation, binning, feature importance, uni/bivariate charts |
| `dags/stock_pipeline_dag.py` | 1.5 DataOps: Airflow DAG, scheduled every 2 minutes |
| `train_ml.py` | 2.1–2.4 ML pipeline: RF + XGBoost, time-ordered split, evaluation, MLflow logging |
| `api_access.py` | 3.1–3.2 Retrieves + displays 4 app details via Airflow/MLflow REST APIs |
| `docker-compose.yml` / `Dockerfile` | Full stack: Airflow, PostgreSQL, MLflow |
| `test_with_synthetic_data.py` | Local validation harness (NOT part of the graded pipeline) |
