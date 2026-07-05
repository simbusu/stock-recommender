"""
predict_new_ticker.py — On-demand Buy/Sell recommendation for ANY stock
ticker, not just the 8 the pipeline was trained on.

Critical correctness point: the ratio features (pe_ratio, debt_to_equity,
current_ratio) must be scaled the exact same way as during training. We
do NOT refit a new scaler on the new ticker's own values (that would be
meaningless relative to what the model learned) — instead we load the
scaler saved by preprocessor.py during the last real training run and
apply .transform() only. OHLC price data is kept raw throughout (see
feature_engineer.compute_technical_features / compute_candlestick_patterns),
matching how the training pipeline computes indicators and candlestick
patterns from real price relationships.

Usage:
    from predict_new_ticker import predict_ticker
    result = predict_ticker("SBIN.NS")
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

import config
import feature_engineer
import explain
import quarterly_fundamentals
from train_ml import FEATURE_COLS

logger = logging.getLogger("predict_new_ticker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MIN_HISTORY_DAYS = 60  # need enough bars for MA21/BB(20)/MACD(26) to warm up


class InsufficientHistoryError(Exception):
    pass


class ScalersNotFoundError(Exception):
    pass


def _load_saved_scalers():
    """
    Load the scaler objects saved by preprocessor.py during the last training run.

    Only the RATIO scaler is actually needed here — pe_ratio/debt_to_equity/
    current_ratio are fed to the model as scaled values. The price scaler is
    NOT applied: OHLC must stay raw so technical indicators and candlestick
    patterns (which depend on real Open/High/Low/Close relationships) are
    computed correctly, matching how the training pipeline now works.
    """
    paths = {
        "ratio_scaler": os.path.join(config.MODEL_DIR, "ratio_scaler.pkl"),
        "ratio_scaler_cols": os.path.join(config.MODEL_DIR, "ratio_scaler_cols.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise ScalersNotFoundError(
            f"Missing saved scaler files: {missing}. Run preprocessor.py (or the full "
            f"pipeline / Airflow DAG) at least once first so scalers are persisted to {config.MODEL_DIR}."
        )
    return {k: joblib.load(p) for k, p in paths.items()}


def _fetch_new_ticker_data(ticker: str, lookback_days: int = 150) -> pd.DataFrame:
    """Fetch enough recent OHLCV history for a brand-new ticker to compute indicators."""
    df = yf.download(ticker, period=f"{lookback_days}d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Check the symbol is correct.")
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["ticker"] = ticker

    if len(df) < MIN_HISTORY_DAYS:
        raise InsufficientHistoryError(
            f"'{ticker}' only has {len(df)} trading days of history available; "
            f"need at least {MIN_HISTORY_DAYS} for indicators to be meaningful (MA21/BB/MACD warm-up)."
        )
    return df


def _fetch_new_ticker_fundamentals(ticker: str) -> dict:
    """Fetch latest fundamentals; missing values are fine — the model handles NaNs via fillna(0)."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    fundamentals = {
        "ticker": ticker,
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "market_cap": info.get("marketCap"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": None,
    }
    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty and "Current Assets" in bs.index and "Current Liabilities" in bs.index:
            ca, cl = bs.loc["Current Assets"].iloc[0], bs.loc["Current Liabilities"].iloc[0]
            fundamentals["current_ratio"] = ca / cl if cl else None
    except Exception:
        pass
    return fundamentals


def _apply_saved_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    Transform-only (never refit) using the scalers saved during training.
    Raw OHLC is intentionally left untouched — see _load_saved_scalers().
    """
    df = df.copy()

    ratio_cols = scalers["ratio_scaler_cols"]
    present_ratio = [c for c in ratio_cols if c in df.columns]
    if present_ratio == ratio_cols:
        df[present_ratio] = scalers["ratio_scaler"].transform(df[present_ratio].fillna(0))
    else:
        logger.warning("Ratio columns mismatch (expected %s, got %s) — skipping ratio scaling", ratio_cols, present_ratio)

    return df


def predict_ticker(ticker: str, models: dict) -> dict:
    """
    Full on-demand pipeline for one new ticker: fetch -> scale (reusing
    saved scalers) -> engineer features -> predict with each provided model.

    `models` should be {"Random Forest": <fitted model>, "XGBoost": <fitted model>}
    (same dict shape as dashboard.data_access.load_models()).

    Returns {"ticker": ..., "history_days": N, "predictions": {model_name: {...}}}
    or raises InsufficientHistoryError / ScalersNotFoundError / ValueError with a
    human-readable message the caller (e.g. Streamlit) should just display.
    """
    scalers = _load_saved_scalers()
    price_df = _fetch_new_ticker_data(ticker)
    fund_row = _fetch_new_ticker_fundamentals(ticker)
    fund_df = pd.DataFrame([fund_row])

    df = price_df.merge(fund_df, on="ticker", how="left")
    df["Date"] = pd.to_datetime(df["Date"])

    # Same point-in-time-safe quarterly attachment as the training pipeline
    # (predict_ticker only ever needs the LATEST row, so leakage risk here
    # is moot for the single row we ultimately use — but keeping the same
    # code path avoids subtle train/inference skew in the growth features).
    quarterly_df = quarterly_fundamentals.fetch_quarterly_history(ticker)
    if not quarterly_df.empty:
        df = quarterly_fundamentals.attach_quarterly_features(df, quarterly_df)
    else:
        for col in quarterly_fundamentals.QUARTERLY_FEATURE_COLS:
            df[col] = np.nan

    df = _apply_saved_scalers(df, scalers)
    df = feature_engineer.compute_technical_features(df)  # ticker-agnostic, works on a single df directly
    df = df.replace([np.inf, -np.inf], np.nan)

    latest_row = df.tail(1)
    if latest_row[["MA21", "RSI_14", "MACD_signal", "BB_width"]].isnull().any(axis=1).iloc[0]:
        raise InsufficientHistoryError(
            f"Not enough warm-up history for '{ticker}' to compute all indicators reliably."
        )

    X = latest_row[FEATURE_COLS].fillna(0)

    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            signal = "BUY" if pred == 1 else "SELL"
            explanation = explain.explain_prediction(model, X, FEATURE_COLS, top_n=5)
            predictions[name] = {
                "signal": signal,
                "confidence": float(max(proba)),
                "explanation": explanation,
                "narrative": explain.build_narrative(explanation, signal),
            }
        except Exception as e:
            predictions[name] = {"signal": "ERROR", "confidence": 0.0, "error": str(e)}

    return {
        "ticker": ticker,
        "history_days": len(price_df),
        "as_of_date": str(latest_row["Date"].iloc[0].date()),
        "predictions": predictions,
    }


if __name__ == "__main__":
    import sys
    import joblib as _joblib

    ticker = sys.argv[1] if len(sys.argv) > 1 else "SBIN.NS"
    models = {
        "Random Forest": _joblib.load(os.path.join(config.MODEL_DIR, "random_forest.pkl")),
        "XGBoost": _joblib.load(os.path.join(config.MODEL_DIR, "xgboost.pkl")),
    }
    result = predict_ticker(ticker, models)
    print(result)
