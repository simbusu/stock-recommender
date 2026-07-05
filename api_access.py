"""
api_access.py — Sub-Objective 3: API Access

3.1 Retrieve Key Application Details via the built-in REST APIs of
    Airflow and MLflow.
3.2 Display at least 4 application details.

Retrieves:
  1. DAG ID and schedule interval          (Airflow REST API)
  2. Last DAG run status and timestamp     (Airflow REST API)
  3. MLflow experiment name and latest run ID  (MLflow REST API)
  4. Registered model name and latest version  (MLflow REST API)
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

    print("=" * 60)
    print("APPLICATION DETAILS — retrieved via Airflow + MLflow REST APIs")
    print("=" * 60)
    for label, value in details.items():
        print(f"\n{label}:")
        print(f"  {value}")

    return details


if __name__ == "__main__":
    display_application_details()
