"""
dags/stock_pipeline_dag.py — Activity 1.5: DataOps

Full daily pipeline: ingest → preprocess → feature_engineer → eda → train.
Runs once per day (see config.DAG_SCHEDULE), after NSE market close, with
every step logged and visible on the Airflow dashboard.

Ingestion and retraining are INSIDE the DAG (unlike an earlier every-2-minute
design, where ingestion had to be kept external to avoid rate-limiting
Yahoo Finance with overly-frequent polling). At a daily cadence that
constraint no longer applies — one ingest call a day is completely normal
API usage, and it means every stage, including "does the model reflect
today's data", is genuinely automated end to end, not just the middle
preprocessing steps.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ingester
import preprocessor
import feature_engineer
import eda_analysis
import train_ml
import pandas as pd


default_args = {
    "owner": "group114",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_ingest(**kwargs):
    price_df, fund_df, quarterly_df = ingester.run_ingestion()
    print(f"[DataOps] Ingestion complete — {len(price_df)} price rows, {len(fund_df)} fundamental rows")


def task_preprocess(**kwargs):
    price_df = pd.read_csv(f"{config.DATA_DIR}/raw_prices.csv")
    fund_df = pd.read_csv(f"{config.DATA_DIR}/raw_fundamentals.csv")
    try:
        quarterly_df = pd.read_csv(f"{config.DATA_DIR}/raw_quarterly_fundamentals.csv")
    except FileNotFoundError:
        quarterly_df = None
    processed = preprocessor.run_preprocessing(price_df, fund_df, quarterly_df, save_scalers_to=config.MODEL_DIR)
    processed.to_csv(f"{config.DATA_DIR}/processed.csv", index=False)
    print(f"[DataOps] Preprocessing complete — {len(processed)} rows")


def task_feature_engineer(**kwargs):
    processed = pd.read_csv(f"{config.DATA_DIR}/processed.csv", parse_dates=["Date"])
    featured = feature_engineer.run_feature_engineering(processed)
    featured.to_csv(f"{config.DATA_DIR}/featured.csv", index=False)
    print(f"[DataOps] Feature engineering complete — {len(featured)} rows")


def task_eda(**kwargs):
    featured = pd.read_csv(f"{config.DATA_DIR}/featured.csv", parse_dates=["Date"])
    tech_cols = [
        "daily_return", "MA7", "MA21", "MA_crossover", "RSI_14",
        "MACD", "MACD_signal", "MACD_hist", "BB_width",
        "volume_change", "price_range",
    ]
    fund_cols = [c for c in ["pe_ratio", "eps", "market_cap", "debt_to_equity", "current_ratio"] if c in featured.columns]
    eda_analysis.run_eda(featured, tech_cols + fund_cols)
    print("[DataOps] EDA complete — plots saved to eda_plots/")


def task_train(**kwargs):
    featured = pd.read_csv(f"{config.DATA_DIR}/featured.csv", parse_dates=["Date"])
    results = train_ml.run_training(featured)
    print(f"[DataOps] Training complete — {results}")


with DAG(
    dag_id=config.DAG_ID,
    default_args=default_args,
    description="Daily pipeline: ingest, preprocess, feature-engineer, EDA, and retrain for the stock Buy/Sell system",
    schedule_interval=config.DAG_SCHEDULE,  # "30 10 * * *" — once daily, 30 min after NSE close
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["stock-recommendation", "dataops", "mlops"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    feature_engineer_task = PythonOperator(
        task_id="feature_engineer",
        python_callable=task_feature_engineer,
    )

    eda_task = PythonOperator(
        task_id="run_eda",
        python_callable=task_eda,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    ingest_task >> preprocess_task >> feature_engineer_task >> eda_task >> train_task
