"""
train_ml.py — Sub-Objective 2: Machine Learning Pipeline

2.1 Model Preparation: Random Forest + XGBoost
2.2 Train/Test Split: 70/30, time-ordered (no shuffling — avoids
    look-ahead / data leakage on time-series data)
2.3 Model Evaluation: accuracy, confusion matrix, classification report
2.4 MLOps: logs accuracy, precision, recall, F1 (>= 4 metrics) to MLflow,
    registers both models in the MLflow Model Registry

Hyperparameter tuning uses sklearn's TimeSeriesSplit (walk-forward CV)
rather than standard k-fold: k-fold shuffles rows across folds, which
for time-series data lets a fold "see" future dates while validating
on earlier ones — an easy, common way to accidentally leak information
and report an inflated CV score. TimeSeriesSplit only ever validates on
dates AFTER everything a given fold trained on.
"""
import logging

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, make_scorer,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier

import config
from quarterly_fundamentals import QUARTERLY_FEATURE_COLS

logger = logging.getLogger("train_ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CANDLESTICK_FEATURE_COLS = [
    "candle_doji", "candle_hammer", "candle_inverted_hammer", "candle_shooting_star",
    "candle_bullish_marubozu", "candle_bearish_marubozu",
    "candle_bullish_engulfing", "candle_bearish_engulfing",
    "candle_morning_star", "candle_evening_star",
    "candle_bullish_signal_count", "candle_bearish_signal_count", "candle_net_signal",
]

FEATURE_COLS = [
    "daily_return", "MA7", "MA21", "MA_crossover", "RSI_14",
    "MACD", "MACD_signal", "MACD_hist", "BB_width",
    "volume_change", "price_range",
    "Stoch_K", "Stoch_D", "ATR_14", "ADX_14", "OBV_roc_5",
    "daily_return_lag1", "daily_return_lag2", "daily_return_lag3",
    "RSI_14_lag1", "RSI_14_lag2",
    "pe_ratio", "eps", "market_cap", "debt_to_equity", "current_ratio",
] + CANDLESTICK_FEATURE_COLS + QUARTERLY_FEATURE_COLS
TARGET_COL = "label"

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.5],
}
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}
CV_N_SPLITS = 5
CV_N_ITER = 15  # RandomizedSearchCV candidates per model — keeps runtime reasonable


def time_ordered_split(df: pd.DataFrame, ratio: float = config.TRAIN_SPLIT_RATIO):
    """
    Split by date, NOT randomly. First `ratio` of dates -> train,
    remaining dates -> test. Prevents future data leaking into training.
    """
    df = df.sort_values("Date")
    cutoff_idx = int(len(df) * ratio)
    cutoff_date = df["Date"].iloc[cutoff_idx]
    train = df[df["Date"] < cutoff_date]
    test = df[df["Date"] >= cutoff_date]
    logger.info(
        "Time-ordered split at %s: train=%d rows, test=%d rows",
        cutoff_date.date(), len(train), len(test),
    )
    return train, test


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    logger.info("%s metrics: %s", model_name, metrics)
    logger.info("%s confusion matrix:\n%s", model_name, confusion_matrix(y_test, y_pred))
    logger.info("%s classification report:\n%s", model_name, classification_report(y_test, y_pred, zero_division=0))
    return metrics


def tune_model(estimator, param_grid, X_train, y_train, n_splits=CV_N_SPLITS, n_iter=CV_N_ITER):
    """
    Walk-forward hyperparameter search: TimeSeriesSplit ensures every CV
    fold only validates on dates strictly after what it trained on, so
    the reported CV score can't be inflated by folds peeking forward.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator, param_distributions=param_grid, n_iter=n_iter,
        cv=tscv, scoring=make_scorer(f1_score, zero_division=0),
        random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("Best params: %s (CV f1=%.4f)", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_


def train_and_log(train, test, feature_cols=FEATURE_COLS, target_col=TARGET_COL, tune=True):
    X_train, y_train = train[feature_cols].fillna(0), train[target_col]
    X_test, y_test = test[feature_cols].fillna(0), test[target_col]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    results = {}

    # ── Random Forest ────────────────────────────────────────
    with mlflow.start_run(run_name="random_forest") as run:
        base_rf = RandomForestClassifier(class_weight="balanced", random_state=42)
        if tune:
            rf, best_params = tune_model(base_rf, RF_PARAM_GRID, X_train, y_train)
        else:
            rf, best_params = base_rf.set_params(n_estimators=100), {"n_estimators": 100}
            rf.fit(X_train, y_train)
        metrics = evaluate(rf, X_test, y_test, "RandomForest")
        mlflow.log_params({**best_params, "class_weight": "balanced", "tuned": tune})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf, "model", registered_model_name="stock_rf_classifier")
        joblib.dump(rf, f"{config.MODEL_DIR}/random_forest.pkl")
        results["random_forest"] = {"run_id": run.info.run_id, **metrics}

    # ── XGBoost ───────────────────────────────────────────────
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    with mlflow.start_run(run_name="xgboost") as run:
        base_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42)
        if tune:
            xgb, best_params = tune_model(base_xgb, XGB_PARAM_GRID, X_train, y_train)
        else:
            xgb, best_params = base_xgb.set_params(n_estimators=200, max_depth=5, learning_rate=0.05), {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}
            xgb.fit(X_train, y_train)
        metrics = evaluate(xgb, X_test, y_test, "XGBoost")
        mlflow.log_params({**best_params, "scale_pos_weight": round(float(scale_pos_weight), 3), "tuned": tune})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgb, "model", registered_model_name="stock_xgb_classifier")
        joblib.dump(xgb, f"{config.MODEL_DIR}/xgboost.pkl")
        results["xgboost"] = {"run_id": run.info.run_id, **metrics}

    return results


def run_training(df: pd.DataFrame):
    train, test = time_ordered_split(df)
    results = train_and_log(train, test)
    logger.info("Training complete: %s", results)
    return results


if __name__ == "__main__":
    featured = pd.read_csv("data/stocks/featured.csv", parse_dates=["Date"])
    run_training(featured)
