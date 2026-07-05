"""
compare_models.py — Standalone experiment: does model choice matter?

Runs 5 structurally different model families on the SAME real featured
data and time-ordered split, to test whether a different algorithm can
extract meaningfully more signal than RandomForest/XGBoost already do.

This does NOT touch your production models/*.pkl or MLflow — it's a
side experiment. Run it, read the table, decide from there.

Usage: python3 compare_models.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import train_ml

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("lightgbm not installed — skipping (pip install lightgbm to include it)")


def report(name, y_test, pred):
    print(f"{name:28s} acc={accuracy_score(y_test, pred):.4f}  "
          f"prec={precision_score(y_test, pred, zero_division=0):.4f}  "
          f"rec={recall_score(y_test, pred, zero_division=0):.4f}  "
          f"f1={f1_score(y_test, pred, zero_division=0):.4f}")


def main():
    featured = pd.read_csv("data/stocks/featured.csv", parse_dates=["Date"])
    train, test = train_ml.time_ordered_split(featured)
    X_train, y_train = train[train_ml.FEATURE_COLS].fillna(0), train[train_ml.TARGET_COL]
    X_test, y_test = test[train_ml.FEATURE_COLS].fillna(0), test[train_ml.TARGET_COL]

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"Test label balance: {y_test.value_counts(normalize=True).to_dict()}")
    print()

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    report("RandomForest", y_test, rf.predict(X_test))

    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    report("XGBoost", y_test, xgb.predict(X_test))

    if HAS_LIGHTGBM:
        lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm.fit(X_train, y_train)
        report("LightGBM", y_test, lgbm.predict(X_test))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_s, y_train)
    report("LogisticRegression (linear)", y_test, logreg.predict(X_test_s))

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    report("MLP Neural Net", y_test, mlp.predict(X_test_s))

    print()
    print("If all models cluster within a few points of each other (and near 50%),")
    print("that's strong evidence the bottleneck is signal in the DATA at this label")
    print("definition, not model choice — matches the synthetic-data sanity check.")


if __name__ == "__main__":
    main()
