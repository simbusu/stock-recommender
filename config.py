"""
config.py — Single source of truth for all configurable parameters.

Edit this file to change stocks, date ranges, indicator windows,
or database connection details without touching pipeline code.
"""
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Target stocks — Nifty 50 constituents, NSE ticker format
# Sourced from live index weightage data (49 of 50 slots returned —
# Nifty occasionally sits at 49 mid-reconstitution; add one more
# ticker below if your instructor expects an exact round 50).
# NOTE: 2 tickers are flagged uncertain due to recent corporate
# actions (rename / demerger) — verify against yfinance before
# your first live ingestion run:
#   - ETERNAL.NS      (formerly ZOMATO.NS, renamed 2025)
#   - TMPV.NS   (Tata Motors demerged into passenger-vehicle
#                       and commercial-vehicle entities in 2025;
#                       this list is for the passenger-vehicle arm —
#                       confirm the exact live ticker on yfinance)
# ─────────────────────────────────────────────────────────────
STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "SBIN.NS",
    "TCS.NS", "BAJFINANCE.NS", "LT.NS", "HINDUNILVR.NS", "SUNPHARMA.NS",
    "MARUTI.NS", "ADANIPORTS.NS", "INFY.NS", "ADANIENT.NS", "AXISBANK.NS",
    "TITAN.NS", "KOTAKBANK.NS", "M&M.NS", "ITC.NS", "NTPC.NS",
    "ULTRACEMCO.NS", "HCLTECH.NS", "BEL.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS",
    "ONGC.NS", "BAJAJ-AUTO.NS", "ETERNAL.NS", "COALINDIA.NS", "POWERGRID.NS",
    "ASIANPAINT.NS", "SHRIRAMFIN.NS", "TATASTEEL.NS", "GRASIM.NS", "HINDALCO.NS",
    "INDIGO.NS", "EICHERMOT.NS", "SBILIFE.NS", "TRENT.NS", "WIPRO.NS",
    "JIOFIN.NS", "TECHM.NS", "APOLLOHOSP.NS", "TMPV.NS", "HDFCLIFE.NS",
    "CIPLA.NS", "DRREDDY.NS", "MAXHEALTH.NS", "TATACONSUM.NS",
]

INDEX_TICKERS = {
    "vix": "^INDIAVIX",
    "nifty": "^NSEI",
}

# ─────────────────────────────────────────────────────────────
# Date range
# ─────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")  # dynamic — was hardcoded to a fixed
                                                  # past date, which silently broke daily
                                                  # ingestion: every run re-fetched the
                                                  # exact same stale window forever

# ─────────────────────────────────────────────────────────────
# Feature engineering windows
# ─────────────────────────────────────────────────────────────
MA_SHORT_WINDOW = 7
MA_LONG_WINDOW = 21
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
STOCH_WINDOW = 14
STOCH_SMOOTH = 3
ATR_WINDOW = 14
ADX_WINDOW = 14
LAG_PERIODS = [1, 2, 3]  # for daily_return and RSI_14 lag features

# ─────────────────────────────────────────────────────────────
# Quarterly fundamentals (point-in-time correct)
# ─────────────────────────────────────────────────────────────
# Companies don't report a quarter's results the day the quarter ends —
# there's a real-world lag (typically 3-7 weeks in India). Using the
# quarter-END date as if it were the AVAILABLE date would leak future
# information into past price rows. EARNINGS_REPORT_LAG_DAYS is a
# conservative estimate of that reporting delay, used to compute a
# safe "available as of" date for each quarter's numbers.
QUARTERLY_LOOKBACK = 8
EARNINGS_REPORT_LAG_DAYS = 45

# ─────────────────────────────────────────────────────────────
# Label definition
# ─────────────────────────────────────────────────────────────
# Predicting a single day's direction is close to pure noise (see
# README's "Model Performance" discussion — this matches published
# findings on daily-frequency direction prediction). Widening the
# horizon and excluding near-flat moves are the two most legitimate
# levers to raise signal without introducing look-ahead bias: both
# only change WHICH rows are labeled and HOW FAR AHEAD the label
# looks, never what information is visible at prediction time.
LABEL_HORIZON_DAYS = 5      # predict N-day-ahead direction instead of next-day
NEUTRAL_ZONE_PCT = 0.01     # drop rows where |forward return| < 1% (ambiguous/noisy moves)

# ─────────────────────────────────────────────────────────────
# Train/test split
# ─────────────────────────────────────────────────────────────
TRAIN_SPLIT_RATIO = 0.70  # time-ordered, NOT random shuffle

# ─────────────────────────────────────────────────────────────
# Database (PostgreSQL)
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "stockdb"),
    "user": os.getenv("POSTGRES_USER", "stockuser"),
    "password": os.getenv("POSTGRES_PASSWORD", "stockpass"),
}

# ─────────────────────────────────────────────────────────────
# MLflow
# ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "stock_buy_sell_recommendation"

# ─────────────────────────────────────────────────────────────
# Airflow
# ─────────────────────────────────────────────────────────────
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_API_USER = os.getenv("AIRFLOW_API_USER", "admin")
AIRFLOW_API_PASSWORD = os.getenv("AIRFLOW_API_PASSWORD", "admin")
DAG_ID = "stock_pipeline_dag"
DAG_SCHEDULE = "30 10 * * *"  # 10:30 UTC = 16:00 IST — 30 min after NSE close (15:30 IST), daily

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "stocks")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
