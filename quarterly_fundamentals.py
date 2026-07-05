"""
quarterly_fundamentals.py — Fetches trailing quarterly revenue/net income
history per ticker and derives growth-trend features (YoY, QoQ, growth
streak), merged onto daily price data POINT-IN-TIME CORRECTLY.

Why this matters: the original single-snapshot fundamentals design
(pe_ratio, eps, etc. fetched once "today" and left-joined onto every
historical price row) meant a 2020 price row was implicitly paired
with 2026's P/E ratio — a real look-ahead leak, quieter than a label
leak but still a leak. This module fixes that by:
  1. Fetching each quarter's ACTUAL reported numbers
  2. Estimating when that quarter's numbers became publicly available
     (quarter-end + a conservative reporting lag — see config)
  3. Using pandas merge_asof (backward direction) so each daily row
     only ever sees quarters that had ALREADY been reported by that date
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger("quarterly_fundamentals")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

QUARTERLY_FEATURE_COLS = [
    "revenue_yoy_growth", "net_income_yoy_growth",
    "revenue_qoq_growth", "net_income_qoq_growth",
    "earnings_growth_streak",
]


def fetch_quarterly_history(ticker: str, lookback: int = None) -> pd.DataFrame:
    """
    Fetch up to `lookback` quarters of Revenue/Net Income for one ticker,
    with an estimated 'available_date' (quarter end + reporting lag) and
    YoY/QoQ growth + a growth-streak feature computed FROM PRIOR quarters
    only (never using a quarter to compute its own trailing growth).
    """
    lookback = lookback or config.QUARTERLY_LOOKBACK
    t = yf.Ticker(ticker)

    try:
        qf = t.quarterly_financials
    except Exception as e:
        logger.warning(f"Could not fetch quarterly financials for {ticker}: {e}")
        return pd.DataFrame()

    if qf is None or qf.empty or "Total Revenue" not in qf.index:
        logger.warning(f"No usable quarterly financials for {ticker}")
        return pd.DataFrame()

    revenue = qf.loc["Total Revenue"]
    net_income = qf.loc["Net Income"] if "Net Income" in qf.index else pd.Series(dtype=float)

    # yfinance returns columns as quarter-end Timestamps, most recent first
    quarter_ends = sorted(revenue.index)[-lookback:] if len(revenue) > lookback else sorted(revenue.index)

    rows = []
    for q_end in quarter_ends:
        rows.append({
            "ticker": ticker,
            "quarter_end": q_end,
            "available_date": q_end + pd.Timedelta(days=config.EARNINGS_REPORT_LAG_DAYS),
            "revenue": revenue.get(q_end, np.nan),
            "net_income": net_income.get(q_end, np.nan) if not net_income.empty else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("quarter_end").reset_index(drop=True)

    # QoQ growth: vs immediately preceding quarter
    df["revenue_qoq_growth"] = df["revenue"].pct_change(1)
    df["net_income_qoq_growth"] = df["net_income"].pct_change(1)

    # YoY growth: vs same quarter ~4 quarters prior (only valid once we have 5+ quarters)
    df["revenue_yoy_growth"] = df["revenue"].pct_change(4)
    df["net_income_yoy_growth"] = df["net_income"].pct_change(4)

    # Growth streak: running count of consecutive quarters of positive YoY
    # net income growth (negative count = consecutive declines). Purely
    # backward-looking — streak at row i uses only rows <= i.
    streak = []
    current = 0
    for val in df["net_income_yoy_growth"]:
        if pd.isna(val):
            current = 0
        elif val > 0:
            current = current + 1 if current >= 0 else 1
        else:
            current = current - 1 if current <= 0 else -1
        streak.append(current)
    df["earnings_growth_streak"] = streak

    return df


def attach_quarterly_features(price_df: pd.DataFrame, quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-time-correct merge: for each (ticker, Date) row in price_df,
    attach the most recent quarter whose available_date <= Date — i.e.
    only quarters that had actually been reported by that day. Uses
    pandas merge_asof (backward direction), per ticker, which is the
    standard leakage-safe way to join time-varying reference data onto
    a time series.
    """
    if quarterly_df.empty:
        for col in ["revenue", "net_income"] + QUARTERLY_FEATURE_COLS:
            price_df[col] = np.nan
        return price_df

    merged_parts = []
    for ticker, p_group in price_df.groupby("ticker", observed=True):
        q_group = quarterly_df[quarterly_df["ticker"] == ticker].sort_values("available_date")
        if q_group.empty:
            p_group = p_group.copy()
            for col in ["revenue", "net_income"] + QUARTERLY_FEATURE_COLS:
                p_group[col] = np.nan
            merged_parts.append(p_group)
            continue

        p_sorted = p_group.sort_values("Date").copy()
        merge_cols = ["available_date", "revenue", "net_income"] + QUARTERLY_FEATURE_COLS
        merged = pd.merge_asof(
            p_sorted, q_group[merge_cols],
            left_on="Date", right_on="available_date", direction="backward",
        )
        merged = merged.drop(columns=["available_date"])
        merged_parts.append(merged)

    result = pd.concat(merged_parts, ignore_index=True)
    logger.info(
        "Attached quarterly features point-in-time; %d/%d rows have a matched quarter",
        result["revenue"].notna().sum(), len(result),
    )
    return result


def fetch_all_quarterly_fundamentals(tickers: list) -> pd.DataFrame:
    frames = [fetch_quarterly_history(t) for t in tickers]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
