"""
eda_analysis.py — Activity 1.4: Exploratory Data Analysis

- Correlation matrix (all numeric features)
- Correlation between technical indicators / fundamentals and the label
- Binning (RSI zones, P/E value/growth/overvalued)
- Feature importance via a quick Random Forest fit
- Univariate charts (return distributions, RSI distribution)
- Bivariate charts (MACD vs label, PE vs returns, VIX vs sell frequency)

All plots are saved to disk (PNG) so they can be embedded as
screenshots in the assignment report.
"""
import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("eda")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PLOT_DIR = "eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def correlation_matrix(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Matrix — All Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/correlation_matrix.png", dpi=120)
    plt.close()
    logger.info("Saved correlation matrix heatmap")
    return corr


def label_correlations(df: pd.DataFrame, top_n: int = 15):
    numeric_df = df.select_dtypes(include="number")
    corr_with_label = numeric_df.corr()["label"].drop("label").sort_values(key=abs, ascending=False)
    logger.info("Top %d features correlated with label:\n%s", top_n, corr_with_label.head(top_n).to_string())
    return corr_with_label


def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bin RSI into zones and P/E into value/growth/overvalued buckets."""
    df = df.copy()
    df["RSI_zone"] = pd.cut(
        df["RSI_14"], bins=[-1, 30, 70, 101],
        labels=["Oversold", "Neutral", "Overbought"],
    )
    if "pe_ratio" in df.columns:
        df["PE_zone"] = pd.qcut(
            df["pe_ratio"].rank(method="first"), q=3,
            labels=["Value", "Growth", "Overvalued"],
        )
    logger.info("Binned RSI into zones:\n%s", df["RSI_zone"].value_counts().to_string())
    return df


def feature_importance(df: pd.DataFrame, feature_cols: list, target_col: str = "label"):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    importances.head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=120)
    plt.close()
    logger.info("Saved feature importance chart")
    return importances


def univariate_charts(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="daily_return", hue="ticker", bins=50, ax=axes[0], legend=False)
    axes[0].set_title("Distribution of Daily Returns (all tickers)")

    sns.histplot(df["RSI_14"], bins=40, ax=axes[1], color="steelblue")
    axes[1].set_title("Distribution of RSI(14)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/univariate_charts.png", dpi=120)
    plt.close()
    logger.info("Saved univariate charts")


def bivariate_charts(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=df, x="label", y="MACD_hist", ax=axes[0])
    axes[0].set_title("MACD Histogram vs Buy/Sell Label")

    if "pe_ratio" in df.columns:
        sns.scatterplot(data=df, x="pe_ratio", y="daily_return", hue="label", alpha=0.4, ax=axes[1])
        axes[1].set_title("PE Ratio vs Daily Return")

    if "vix" in df.columns:
        sell_rate_by_vix_bin = df.groupby(pd.qcut(df["vix"], 5))["label"].apply(lambda s: (s == 0).mean())
        sell_rate_by_vix_bin.plot(kind="bar", ax=axes[2], color="indianred")
        axes[2].set_title("Sell Frequency by VIX Quintile")
        axes[2].set_ylabel("Sell rate")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/bivariate_charts.png", dpi=120)
    plt.close()
    logger.info("Saved bivariate charts")


def run_eda(df: pd.DataFrame, feature_cols: list):
    correlation_matrix(df)
    label_correlations(df)
    df = bin_features(df)
    feature_importance(df, feature_cols)
    univariate_charts(df)
    bivariate_charts(df)
    logger.info("EDA complete. Plots saved to ./%s/", PLOT_DIR)
    return df


if __name__ == "__main__":
    featured = pd.read_csv("data/stocks/featured.csv", parse_dates=["Date"])
    tech_cols = [
        "daily_return", "MA7", "MA21", "MA_crossover", "RSI_14",
        "MACD", "MACD_signal", "MACD_hist", "BB_width",
        "volume_change", "price_range",
    ]
    fund_cols = [c for c in ["pe_ratio", "eps", "market_cap", "debt_to_equity", "current_ratio"] if c in featured.columns]
    run_eda(featured, tech_cols + fund_cols)
