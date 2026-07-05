"""
confidence_analysis.py — Does accuracy improve on high-confidence predictions?

Standard technique in systematic trading: a model doesn't have to be
right on every single day to be useful — if it's meaningfully more
accurate specifically on the days it's MOST confident, that subset of
calls can be actionable even when overall accuracy sits near 50%.

This reports BOTH accuracy AND coverage (% of days that fall in each
confidence bucket) — a high-confidence bucket that's 90% accurate but
only fires twice a year isn't very useful. Both numbers matter together.

Usage: python3 confidence_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import train_ml

CONFIDENCE_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]  # |proba - 0.5| * 2 scale, 0=coinflip 1=certain


def confidence_binned_accuracy(proba: np.ndarray, y_test: pd.Series, bins=CONFIDENCE_BINS) -> pd.DataFrame:
    """
    proba: predicted probability of class 1 (BUY), shape (n,)
    Returns a table of accuracy + coverage per confidence bucket.
    """
    pred = (proba >= 0.5).astype(int)
    correct = (pred == y_test.values).astype(int)
    confidence = np.abs(proba - 0.5) * 2  # rescale so 0=coin-flip, 1=fully certain

    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "confidence_range": f"{lo:.2f}-{hi:.2f}",
            "n_predictions": n,
            "coverage_pct": round(100 * n / len(y_test), 1),
            "accuracy": round(float(correct[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


def threshold_sweep(proba: np.ndarray, y_test: pd.Series, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70)) -> pd.DataFrame:
    """
    'Only act when the model says BUY with probability >= T (or SELL
    with probability <= 1-T)' — the standard selective-prediction framing
    used in systematic trading: skip the ambiguous, low-conviction days.
    """
    rows = []
    for t in thresholds:
        acted_mask = (proba >= t) | (proba <= (1 - t))
        n = int(acted_mask.sum())
        if n == 0:
            rows.append({"threshold": t, "n_acted": 0, "coverage_pct": 0.0, "accuracy_on_acted": None})
            continue
        pred = (proba[acted_mask] >= 0.5).astype(int)
        acc = accuracy_score(y_test.values[acted_mask], pred)
        rows.append({
            "threshold": t,
            "n_acted": n,
            "coverage_pct": round(100 * n / len(y_test), 1),
            "accuracy_on_acted": round(float(acc), 4),
        })
    return pd.DataFrame(rows)


def analyze_model(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    overall_acc = accuracy_score(y_test, (proba >= 0.5).astype(int))

    print(f"\n{'='*70}\n{name} — overall accuracy: {overall_acc:.4f}\n{'='*70}")
    print("\n-- Accuracy by confidence bucket --")
    print(confidence_binned_accuracy(proba, y_test).to_string(index=False))
    print("\n-- Accuracy when only acting above a probability threshold --")
    print(threshold_sweep(proba, y_test).to_string(index=False))


def main():
    featured = pd.read_csv("data/stocks/featured.csv", parse_dates=["Date"])
    train, test = train_ml.time_ordered_split(featured)
    X_train, y_train = train[train_ml.FEATURE_COLS].fillna(0), train[train_ml.TARGET_COL]
    X_test, y_test = test[train_ml.FEATURE_COLS].fillna(0), test[train_ml.TARGET_COL]

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    analyze_model("RandomForest", rf, X_test, y_test)

    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    analyze_model("XGBoost", xgb, X_test, y_test)

    print(f"\n{'='*70}")
    print("How to read this: if accuracy climbs meaningfully in the higher")
    print("confidence buckets (and coverage isn't tiny), that's a genuine,")
    print("actionable finding — 'don't trade every day, only the confident")
    print("ones.' If accuracy stays flat ~50% across all buckets regardless")
    print("of stated confidence, the model's confidence isn't well-calibrated")
    print("and shouldn't be trusted as a filter either.")


if __name__ == "__main__":
    main()
