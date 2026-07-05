"""
dashboard/data_access.py — Pure data-loading/computation functions, kept
separate from Streamlit UI code so they can be unit-tested independently.

Falls back gracefully: tries PostgreSQL first, falls back to local CSVs
if the DB isn't reachable (e.g. dashboard run outside Docker network).
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_ml import FEATURE_COLS  # single source of truth — was previously duplicated here
import explain


def load_featured_data() -> pd.DataFrame:
    """Load the fully feature-engineered dataset (technical + fundamental + label)."""
    path = os.path.join(config.DATA_DIR, "featured.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run ingester.py, preprocessor.py, and "
            f"feature_engineer.py first (or wait for the Airflow DAG to complete a cycle)."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def get_available_tickers(df: pd.DataFrame) -> list:
    return sorted(df["ticker"].unique().tolist())


def get_ticker_history(df: pd.DataFrame, ticker: str, last_n_days: int = 180) -> pd.DataFrame:
    """Latest N days of price + indicator history for one ticker, sorted by date."""
    t_df = df[df["ticker"] == ticker].sort_values("Date")
    return t_df.tail(last_n_days).reset_index(drop=True)


def load_models() -> dict:
    """Load trained models from disk. Returns dict of {name: model}, skipping any missing."""
    models = {}
    for name, fname in [("Random Forest", "random_forest.pkl"), ("XGBoost", "xgboost.pkl")]:
        path = os.path.join(config.MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models


def get_latest_recommendation(df: pd.DataFrame, ticker: str, models: dict, with_explanation: bool = False) -> dict:
    """
    Run each loaded model on the most recent feature row for a ticker.
    Returns {model_name: {"signal": "BUY"/"SELL", "confidence": float, [explanation, narrative]}}.
    Explanation (SHAP-based reasoning) is opt-in via with_explanation since
    it's noticeably slower than a bare prediction — skip it for the bulk
    all-tickers summary table, request it only when a user asks "why?" for one.
    """
    t_df = df[df["ticker"] == ticker].sort_values("Date")
    if t_df.empty:
        return {}

    latest_row = t_df.iloc[[-1]][FEATURE_COLS].fillna(0)
    results = {}
    for name, model in models.items():
        try:
            pred = model.predict(latest_row)[0]
            proba = model.predict_proba(latest_row)[0]
            signal = "BUY" if pred == 1 else "SELL"
            entry = {"signal": signal, "confidence": float(max(proba))}
            if with_explanation:
                exp = explain.explain_prediction(model, latest_row, FEATURE_COLS, top_n=5)
                entry["explanation"] = exp
                entry["narrative"] = explain.build_narrative(exp, signal)
            results[name] = entry
        except Exception as e:
            results[name] = {"signal": "ERROR", "confidence": 0.0, "error": str(e)}
    return results


def get_recommendations_for_all_tickers(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Build a summary table: one row per ticker, one column per model's signal."""
    rows = []
    for ticker in get_available_tickers(df):
        rec = get_latest_recommendation(df, ticker, models)
        row = {"ticker": ticker}
        for model_name, result in rec.items():
            row[f"{model_name} Signal"] = result.get("signal", "N/A")
            row[f"{model_name} Confidence"] = round(result.get("confidence", 0.0), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def get_mlflow_experiment_overview(tracking_uri: str = None, experiment_name: str = None) -> dict:
    """Experiment-level details — mirrors the top of MLflow's Experiment page."""
    import mlflow

    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
        "tracking_uri": tracking_uri,
    }


def get_run_full_details(run_id: str, tracking_uri: str = None) -> dict:
    """
    Everything MLflow's single-Run page shows: params, metrics, tags,
    and the artifact file tree (names + sizes, not contents).
    """
    import mlflow

    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    def _list_artifacts(path=""):
        entries = []
        for f in client.list_artifacts(run_id, path):
            if f.is_dir:
                entries.extend(_list_artifacts(f.path))
            else:
                entries.append({"path": f.path, "size_bytes": f.file_size})
        return entries

    return {
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
        "status": run.info.status,
        "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
        "end_time": pd.to_datetime(run.info.end_time, unit="ms") if run.info.end_time else None,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "artifacts": _list_artifacts(),
    }


def get_registered_models_overview(tracking_uri: str = None) -> pd.DataFrame:
    """All registered models + their versions — mirrors MLflow's Model Registry page."""
    import mlflow

    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    rows = []
    for rm in client.search_registered_models():
        for v in client.search_model_versions(f"name='{rm.name}'"):
            rows.append({
                "model_name": rm.name,
                "version": v.version,
                "stage": getattr(v, "current_stage", "n/a"),
                "run_id": v.run_id,
                "status": v.status,
                "created": pd.to_datetime(v.creation_timestamp, unit="ms"),
            })
    return pd.DataFrame(rows).sort_values(["model_name", "version"], ascending=[True, False]) if rows else pd.DataFrame()


def get_mlflow_runs_summary(tracking_uri: str = None, experiment_name: str = None) -> pd.DataFrame:
    """Fetch recent MLflow runs with their logged metrics as a flat DataFrame."""
    import mlflow

    tracking_uri = tracking_uri or config.MLFLOW_TRACKING_URI
    experiment_name = experiment_name or config.MLFLOW_EXPERIMENT_NAME

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=20,
    )
    rows = []
    for run in runs:
        row = {
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
        }
        row.update(run.data.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def trigger_training_run(df: pd.DataFrame) -> dict:
    """
    Manually trigger a fresh RF + XGBoost training run on the current
    featured dataset, logging metrics/params/models to MLflow exactly
    like train_ml.py's scheduled/manual CLI run does.
    Returns {"random_forest": {...metrics...}, "xgboost": {...metrics...}}.
    """
    import train_ml
    train, test = train_ml.time_ordered_split(df)
    results = train_ml.train_and_log(train, test)
    return results


def project_price_5y(hist_df: pd.DataFrame, years: int = 5) -> dict:
    """
    Simple historical-CAGR price projection — an educational illustration
    of trend extrapolation, NOT a forecast or investment advice.

    Computes the ticker's historical CAGR (Compound Annual Growth Rate)
    and annualized volatility from its full available Close-price history,
    then extrapolates a base / optimistic / pessimistic price path forward
    by `years` years. Returns {} if there isn't enough history.
    """
    d = hist_df.dropna(subset=["Close"]).sort_values("Date")
    if len(d) < 60:
        return {}

    first_close, last_close = float(d["Close"].iloc[0]), float(d["Close"].iloc[-1])
    n_years = (d["Date"].iloc[-1] - d["Date"].iloc[0]).days / 365.25
    if n_years <= 0 or first_close <= 0:
        return {}

    cagr = (last_close / first_close) ** (1 / n_years) - 1
    daily_ret = d["Close"].pct_change().dropna()
    annual_vol = float(daily_ret.std() * np.sqrt(252))

    future_dates = pd.date_range(d["Date"].iloc[-1], periods=years + 1, freq="YE")[1:]
    t = np.arange(1, years + 1)
    base = last_close * (1 + cagr) ** t
    optimistic = last_close * (1 + cagr + annual_vol) ** t
    pessimistic = last_close * (1 + max(cagr - annual_vol, -0.95)) ** t

    return {
        "last_close": last_close,
        "cagr": float(cagr),
        "annual_vol": annual_vol,
        "history_years": round(n_years, 1),
        "future_dates": future_dates,
        "base": base,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
    }


def get_airflow_dag_status(base_url: str = None, user: str = None, password: str = None) -> dict:
    """Fetch DAG status + last run info from Airflow's REST API."""
    import requests
    from requests.auth import HTTPBasicAuth

    base_url = base_url or config.AIRFLOW_BASE_URL
    user = user or config.AIRFLOW_API_USER
    password = password or config.AIRFLOW_API_PASSWORD
    auth = HTTPBasicAuth(user, password)

    dag_resp = requests.get(f"{base_url}/api/v1/dags/{config.DAG_ID}", auth=auth, timeout=5)
    dag_resp.raise_for_status()
    dag_info = dag_resp.json()

    runs_resp = requests.get(
        f"{base_url}/api/v1/dags/{config.DAG_ID}/dagRuns",
        auth=auth, timeout=5,
        params={"order_by": "-execution_date", "limit": 5},
    )
    runs_resp.raise_for_status()
    recent_runs = runs_resp.json().get("dag_runs", [])

    return {
        "is_paused": dag_info.get("is_paused"),
        "schedule_interval": dag_info.get("schedule_interval"),
        "recent_runs": recent_runs,
    }
