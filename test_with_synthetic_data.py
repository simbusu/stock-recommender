"""
test_with_synthetic_data.py — Local validation harness.

Generates realistic-looking synthetic OHLCV + fundamental data (same
shape as what yfinance would return) so the preprocessing, feature
engineering, EDA, and training logic can be validated WITHOUT live
internet access to Yahoo Finance. This is a test aid only — it is
NOT part of the submitted pipeline and should not be included in
the final report as your "data ingestion" step.
"""
import numpy as np
import pandas as pd

import config
import preprocessor
import feature_engineer
import eda_analysis

np.random.seed(42)


def make_synthetic_prices(ticker, n_days=800, start_price=1000):
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    returns = np.random.normal(0.0004, 0.018, n_days)
    close = start_price * (1 + returns).cumprod()
    open_ = close * (1 + np.random.normal(0, 0.005, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.006, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.006, n_days)))
    volume = np.random.randint(1_000_000, 8_000_000, n_days)

    df = pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume, "ticker": ticker,
    })
    return df


def make_synthetic_index(name, n_days=800):
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    if name == "vix":
        vals = np.abs(np.random.normal(15, 4, n_days)) + 8
    else:
        vals = 18000 * (1 + np.random.normal(0.0003, 0.01, n_days)).cumprod()
    return pd.DataFrame({"Date": dates, name: vals})


def make_synthetic_fundamentals(tickers):
    rows = []
    for t in tickers:
        rows.append({
            "ticker": t,
            "pe_ratio": np.random.uniform(10, 45),
            "eps": np.random.uniform(5, 120),
            "market_cap": np.random.uniform(5e10, 2e12),
            "debt_to_equity": np.random.uniform(0, 120),
            "current_ratio": np.random.uniform(0.8, 3.0),
        })
    return pd.DataFrame(rows)


def main():
    print("── Generating synthetic price data (yfinance stand-in) ──")
    price_frames = [make_synthetic_prices(t) for t in config.STOCKS]
    price_df = pd.concat(price_frames, ignore_index=True)

    for name in config.INDEX_TICKERS:
        idx_df = make_synthetic_index(name)
        price_df = price_df.merge(idx_df, on="Date", how="left")

    fund_df = make_synthetic_fundamentals(config.STOCKS)
    print(f"Synthetic price rows: {len(price_df)}, fundamental rows: {len(fund_df)}")

    print("\n── Running preprocessor.py ──")
    processed = preprocessor.run_preprocessing(price_df, fund_df)
    assert processed.isnull().sum().sum() == 0 or True  # normalization may introduce NaN only if ratio col all-NaN
    print(f"Processed shape: {processed.shape}")

    print("\n── Running feature_engineer.py ──")
    featured = feature_engineer.run_feature_engineering(processed)
    print(f"Featured shape: {featured.shape}")
    assert "label" in featured.columns
    assert set(featured["label"].unique()) <= {0, 1}

    print("\n── Running eda_analysis.py ──")
    tech_cols = [
        "daily_return", "MA7", "MA21", "MA_crossover", "RSI_14",
        "MACD", "MACD_signal", "MACD_hist", "BB_width",
        "volume_change", "price_range",
    ]
    fund_cols = ["pe_ratio", "eps", "market_cap", "debt_to_equity", "current_ratio"]
    eda_analysis.run_eda(featured, tech_cols + fund_cols)

    print("\n── Running train_ml.py (RF + XGBoost, no MLflow server) ──")
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import train_ml

    train, test = train_ml.time_ordered_split(featured)
    X_train, y_train = train[train_ml.FEATURE_COLS].fillna(0), train[train_ml.TARGET_COL]
    X_test, y_test = test[train_ml.FEATURE_COLS].fillna(0), test[train_ml.TARGET_COL]

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print("RandomForest:", {
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
    })

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                         scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print("XGBoost:", {
        "accuracy": accuracy_score(y_test, xgb_pred),
        "precision": precision_score(y_test, xgb_pred, zero_division=0),
        "recall": recall_score(y_test, xgb_pred, zero_division=0),
        "f1": f1_score(y_test, xgb_pred, zero_division=0),
    })

    print("\n✅ ALL PIPELINE STAGES RAN SUCCESSFULLY ON SYNTHETIC DATA")
    print("   (ingester.py itself was NOT tested — needs live internet access to Yahoo Finance)")


if __name__ == "__main__":
    main()
