"""
api_access.py — Sub-Objective 3: API Access

3.1 Retrieve Key Application Details via the built-in REST APIs of
    Airflow and MLflow, plus a custom-built REST API (data_api.py)
    wrapping PostgreSQL.
3.2 Display at least 4 application details.

Retrieves:
  1. DAG ID and schedule interval          (Airflow REST API)
  2. Last DAG run status and timestamp     (Airflow REST API)
  3. MLflow experiment name and latest run ID  (MLflow REST API)
  4. Registered model name and latest version  (MLflow REST API)
  5. Row counts per pipeline table         (Data API — FastAPI over Postgres)
  6. Ticker list (count + sample)          (Data API — FastAPI over Postgres)
  7. Latest sample price (RELIANCE.NS)     (Data API — FastAPI over Postgres)
"""
import logging

import requests
from requests.auth import HTTPBasicAuth

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_access")


def get_dag_info():
    """Detail 1: DAG ID and schedule interval."""
    url = f"{config.AIRFLOW_BASE_URL}/api/v1/dags/{config.DAG_ID}"
    resp = requests.get(url, auth=HTTPBasicAuth(config.AIRFLOW_API_USER, config.AIRFLOW_API_PASSWORD))
    resp.raise_for_status()
    data = resp.json()
    return {
        "dag_id": data.get("dag_id"),
        "schedule_interval": data.get("schedule_interval"),
        "is_paused": data.get("is_paused"),
    }


def get_last_dag_run():
    """Detail 2: last DAG run status and timestamp."""
    url = f"{config.AIRFLOW_BASE_URL}/api/v1/dags/{config.DAG_ID}/dagRuns"
    params = {"order_by": "-execution_date", "limit": 1}
    resp = requests.get(url, auth=HTTPBasicAuth(config.AIRFLOW_API_USER, config.AIRFLOW_API_PASSWORD), params=params)
    resp.raise_for_status()
    runs = resp.json().get("dag_runs", [])
    if not runs:
        return {"status": "no runs yet"}
    latest = runs[0]
    return {
        "run_id": latest.get("dag_run_id"),
        "state": latest.get("state"),
        "execution_date": latest.get("execution_date"),
    }


def get_mlflow_experiment_info():
    """Detail 3: MLflow experiment name and latest run ID."""
    url = f"{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/get-by-name"
    resp = requests.get(url, params={"experiment_name": config.MLFLOW_EXPERIMENT_NAME})
    resp.raise_for_status()
    experiment = resp.json()["experiment"]

    search_url = f"{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search"
    search_resp = requests.post(search_url, json={
        "experiment_ids": [experiment["experiment_id"]],
        "order_by": ["attribute.start_time DESC"],
        "max_results": 1,
    })
    search_resp.raise_for_status()
    runs = search_resp.json().get("runs", [])
    latest_run_id = runs[0]["info"]["run_id"] if runs else None

    return {
        "experiment_name": experiment["name"],
        "experiment_id": experiment["experiment_id"],
        "latest_run_id": latest_run_id,
    }


def get_registered_model_info(model_name: str = "stock_xgb_classifier"):
    """Detail 4: registered model name and latest version."""
    url = f"{config.MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get-latest-versions"
    resp = requests.post(url, json={"name": model_name})
    resp.raise_for_status()
    versions = resp.json().get("model_versions", [])
    if not versions:
        return {"model_name": model_name, "latest_version": "none registered yet"}
    latest = versions[0]
    return {
        "model_name": model_name,
        "latest_version": latest.get("version"),
        "current_stage": latest.get("current_stage"),
    }


def get_data_api_stats():
    """Detail 5: row counts per pipeline table, via the custom Data API
    (FastAPI service wrapping PostgreSQL — see data_api.py)."""
    url = f"{config.DATA_API_BASE_URL}/stats/row-counts"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_data_api_tickers():
    """Detail 6: list of tickers available, via the custom Data API."""
    url = f"{config.DATA_API_BASE_URL}/tickers"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {"ticker_count": data.get("count"), "sample": data.get("tickers", [])[:5]}


def get_data_api_sample_price(ticker: str = "RELIANCE.NS"):
    """Detail 7: latest price row for a sample ticker, via the custom Data API."""
    url = f"{config.DATA_API_BASE_URL}/prices/{ticker}"
    resp = requests.get(url, params={"limit": 1}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    latest = data.get("data", [{}])[0]
    return {
        "ticker": ticker,
        "date": latest.get("Date"),
        "close": latest.get("Close"),
        "volume": latest.get("Volume"),
    }


def display_application_details():
    """3.2: Display at least 4 application details retrieved via APIs."""
    details = {}

    try:
        details["1. DAG Info"] = get_dag_info()
    except Exception as e:
        details["1. DAG Info"] = f"unavailable ({e})"

    try:
        details["2. Last DAG Run"] = get_last_dag_run()
    except Exception as e:
        details["2. Last DAG Run"] = f"unavailable ({e})"

    try:
        details["3. MLflow Experiment"] = get_mlflow_experiment_info()
    except Exception as e:
        details["3. MLflow Experiment"] = f"unavailable ({e})"

    try:
        details["4. Registered Model"] = get_registered_model_info()
    except Exception as e:
        details["4. Registered Model"] = f"unavailable ({e})"

    try:
        details["5. Data API Row Counts"] = get_data_api_stats()
    except Exception as e:
        details["5. Data API Row Counts"] = f"unavailable ({e})"

    try:
        details["6. Data API Tickers"] = get_data_api_tickers()
    except Exception as e:
        details["6. Data API Tickers"] = f"unavailable ({e})"

    try:
        details["7. Data API Sample Price"] = get_data_api_sample_price()
    except Exception as e:
        details["7. Data API Sample Price"] = f"unavailable ({e})"

    print("=" * 60)
    print("APPLICATION DETAILS — retrieved via Airflow + MLflow + Data API REST APIs")
    print("=" * 60)
    for label, value in details.items():
        print(f"\n{label}:")
        print(f"  {value}")

    return details


if __name__ == "__main__":
    display_application_details()
