"""
preprocessor.py — Activity 1.3: Data Pre-processing

- Displays summary statistics
- Checks for missing values across all columns/stocks
- Imputes missing numeric values (forward-fill for time series continuity,
  median for remaining gaps)
- Displays data types and converts where necessary
- Normalizes continuous features (MinMaxScaler for price, StandardScaler
  for returns/ratios)
- One-hot encodes the stock ticker
"""
import logging
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import quarterly_fundamentals

logger = logging.getLogger("preprocessor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
RATIO_COLS = ["pe_ratio", "debt_to_equity", "current_ratio"] + quarterly_fundamentals.QUARTERLY_FEATURE_COLS


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std/min/max/quartiles for all numeric columns."""
    stats = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    logger.info("Summary statistics computed for %d numeric columns", len(stats))
    return stats


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    logger.info("Columns with missing values:\n%s", missing.to_string() if not missing.empty else "None")
    return missing


def impute_missing(df: pd.DataFrame, group_col: str = "ticker") -> pd.DataFrame:
    """
    Forward-fill within each ticker group (preserves time-series continuity),
    then fall back to column median for any remaining gaps (e.g. leading NaNs).
    """
    df = df.sort_values(["ticker", "Date"]).copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(lambda s: s.ffill())
    remaining_na = df[numeric_cols].isnull().sum().sum()
    if remaining_na > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info("Filled %d remaining NaNs with column median", remaining_na)

    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column, ensure numeric columns are numeric, ticker is categorical."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("category")
    logger.info("Dtypes after conversion:\n%s", df.dtypes.to_string())
    return df


def normalize_features(df: pd.DataFrame, save_scalers_to: str = None) -> pd.DataFrame:
    """
    MinMaxScaler for raw price columns, StandardScaler for return/ratio-type
    columns (can be negative, roughly normal-ish, benefit from zero-mean/
    unit-variance scaling).

    IMPORTANT: price columns are scaled into NEW `<col>_norm` columns —
    the original raw Open/High/Low/Close/Volume are left untouched. Two
    reasons:
      1. Technical-indicator math (RSI, MACD, ATR, candlestick body/shadow
         ratios, etc.) needs REAL price relationships. MinMaxScaler fits
         each column independently, so overwriting OHLC in place would
         distort the Open-vs-Close/High-vs-Low relationships that those
         calculations — and any candlestick chart — depend on.
      2. The dashboard needs real rupee prices to plot and to compute
         things like a multi-year price projection.
    Ratio columns (pe_ratio, debt_to_equity, etc.) ARE scaled in place —
    they're fed to the model directly and don't have this constraint.

    If save_scalers_to is given, the FITTED scaler objects are persisted
    there (joblib) so the exact same transform can be reapplied later to
    a brand-new, unseen ticker at inference time — refitting a scaler on
    a single new stock's own price range would trivially rescale it to
    0-1 and be inconsistent with what the trained model actually learned.
    """
    df = df.copy()

    price_cols_present = [c for c in PRICE_COLS if c in df.columns]
    if price_cols_present:
        mm_scaler = MinMaxScaler()
        norm_cols = [f"{c}_norm" for c in price_cols_present]
        df[norm_cols] = mm_scaler.fit_transform(df[price_cols_present])
        if save_scalers_to:
            joblib.dump(mm_scaler, os.path.join(save_scalers_to, "price_scaler.pkl"))
            joblib.dump(price_cols_present, os.path.join(save_scalers_to, "price_scaler_cols.pkl"))

    ratio_cols_present = [c for c in RATIO_COLS if c in df.columns]
    if ratio_cols_present:
        std_scaler = StandardScaler()
        df[ratio_cols_present] = std_scaler.fit_transform(df[ratio_cols_present].fillna(0))
        if save_scalers_to:
            joblib.dump(std_scaler, os.path.join(save_scalers_to, "ratio_scaler.pkl"))
            joblib.dump(ratio_cols_present, os.path.join(save_scalers_to, "ratio_scaler_cols.pkl"))

    logger.info(
        "Normalized %d price columns (MinMax, stored as _norm columns; raw OHLC "
        "preserved) and %d ratio columns (Standard, in place)",
        len(price_cols_present), len(ratio_cols_present),
    )
    return df


def encode_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ticker column."""
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    return pd.concat([df, dummies], axis=1)


def run_preprocessing(price_df: pd.DataFrame, fund_df: pd.DataFrame, quarterly_df: pd.DataFrame = None, save_scalers_to: str = None) -> pd.DataFrame:
    """Full preprocessing pipeline: merge, clean, impute, normalize, encode."""
    df = price_df.merge(fund_df, on="ticker", how="left")
    df = fix_dtypes(df)  # Date must be real datetime before the point-in-time merge below

    if quarterly_df is not None and not quarterly_df.empty:
        quarterly_df = quarterly_df.copy()
        quarterly_df["quarter_end"] = pd.to_datetime(quarterly_df["quarter_end"])
        quarterly_df["available_date"] = pd.to_datetime(quarterly_df["available_date"])
        df = quarterly_fundamentals.attach_quarterly_features(df, quarterly_df)
    else:
        logger.info("No quarterly fundamentals provided — skipping quarterly feature attachment")
        for col in quarterly_fundamentals.QUARTERLY_FEATURE_COLS:
            df[col] = np.nan

    summary_statistics(df)
    check_missing(df)

    df = impute_missing(df)
    df = normalize_features(df, save_scalers_to=save_scalers_to)
    df = encode_ticker(df)

    logger.info("Preprocessing complete: final shape %s", df.shape)
    return df


if __name__ == "__main__":
    import config
    price_df = pd.read_csv("data/stocks/raw_prices.csv")
    fund_df = pd.read_csv("data/stocks/raw_fundamentals.csv")
    try:
        quarterly_df = pd.read_csv("data/stocks/raw_quarterly_fundamentals.csv")
    except FileNotFoundError:
        quarterly_df = None
    processed = run_preprocessing(price_df, fund_df, quarterly_df, save_scalers_to=config.MODEL_DIR)
    processed.to_csv("data/stocks/processed.csv", index=False)
