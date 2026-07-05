"""
ingester.py — Activity 1.2: Data Ingestion

Downloads daily OHLCV price data, index/VIX data, and quarterly
fundamental data via the yfinance library, and persists everything
to PostgreSQL for downstream pipeline stages.

NOTE: yfinance calls require outbound internet access to
query1/query2.finance.yahoo.com. This script must be run in an
environment with normal internet access (not a sandboxed CI runner
with restricted egress).
"""
import logging
import time

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

import config
import quarterly_fundamentals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingester")


def get_engine():
    url = (
        f"postgresql+psycopg2://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}"
        f"@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['dbname']}"
    )
    return create_engine(url)


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV for one ticker."""
    logger.info(f"Fetching OHLCV for {ticker} ({start} -> {end})")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
        return df
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["ticker"] = ticker
    return df


def fetch_index_data(name: str, ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close for an index/VIX ticker."""
    logger.info(f"Fetching index data for {name} ({ticker})")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
        return df
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Date", "Close"]].rename(columns={"Close": name})
    return df


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch latest-snapshot fundamental data for one ticker."""
    logger.info(f"Fetching fundamentals for {ticker}")
    t = yf.Ticker(ticker)
    info = t.info or {}

    fundamentals = {
        "ticker": ticker,
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "market_cap": info.get("marketCap"),
        "debt_to_equity": info.get("debtToEquity"),
    }

    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            if "Total Revenue" in qf.index:
                fundamentals["latest_revenue"] = qf.loc["Total Revenue"].iloc[0]
            if "Net Income" in qf.index:
                fundamentals["latest_net_income"] = qf.loc["Net Income"].iloc[0]
    except Exception as e:
        logger.warning(f"Quarterly financials unavailable for {ticker}: {e}")

    try:
        cf = t.cashflow
        if cf is not None and not cf.empty and "Operating Cash Flow" in cf.index:
            fundamentals["operating_cash_flow"] = cf.loc["Operating Cash Flow"].iloc[0]
        if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
            fundamentals["free_cash_flow"] = cf.loc["Free Cash Flow"].iloc[0]
    except Exception as e:
        logger.warning(f"Cashflow unavailable for {ticker}: {e}")

    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            if "Current Assets" in bs.index and "Current Liabilities" in bs.index:
                ca = bs.loc["Current Assets"].iloc[0]
                cl = bs.loc["Current Liabilities"].iloc[0]
                fundamentals["current_ratio"] = ca / cl if cl else None
    except Exception as e:
        logger.warning(f"Balance sheet unavailable for {ticker}: {e}")

    return fundamentals


def run_ingestion():
    """Full ingestion run: price data for all stocks + indices + fundamentals."""
    all_prices = []
    for ticker in config.STOCKS:
        df = fetch_price_data(ticker, config.START_DATE, config.END_DATE)
        if not df.empty:
            all_prices.append(df)
        time.sleep(0.5)  # be polite to Yahoo Finance

    if not all_prices:
        raise RuntimeError("No price data was fetched for any ticker.")

    price_df = pd.concat(all_prices, ignore_index=True)

    # Index / VIX data, merged on Date
    index_frames = []
    for name, ticker in config.INDEX_TICKERS.items():
        idx_df = fetch_index_data(name, ticker, config.START_DATE, config.END_DATE)
        if not idx_df.empty:
            index_frames.append(idx_df)
    for idx_df in index_frames:
        price_df = price_df.merge(idx_df, on="Date", how="left")

    # Fundamentals (latest snapshot per ticker, forward-filled to daily in preprocessor)
    fundamentals = [fetch_fundamentals(t) for t in config.STOCKS]
    fund_df = pd.DataFrame(fundamentals)

    # Quarterly history (point-in-time correct — see quarterly_fundamentals.py)
    logger.info("Fetching quarterly fundamentals history (last %d quarters per ticker)", config.QUARTERLY_LOOKBACK)
    quarterly_df = quarterly_fundamentals.fetch_all_quarterly_fundamentals(config.STOCKS)

    engine = get_engine()
    price_df.to_sql("raw_prices", engine, if_exists="replace", index=False)
    fund_df.to_sql("raw_fundamentals", engine, if_exists="replace", index=False)
    if not quarterly_df.empty:
        quarterly_df.to_sql("raw_quarterly_fundamentals", engine, if_exists="replace", index=False)

    # Also cache to disk for offline reruns / debugging
    price_df.to_csv(f"{config.DATA_DIR}/raw_prices.csv", index=False)
    fund_df.to_csv(f"{config.DATA_DIR}/raw_fundamentals.csv", index=False)
    quarterly_df.to_csv(f"{config.DATA_DIR}/raw_quarterly_fundamentals.csv", index=False)

    logger.info(
        "Ingestion complete: %d price rows, %d fundamental rows, %d quarterly-history rows",
        len(price_df), len(fund_df), len(quarterly_df),
    )
    return price_df, fund_df, quarterly_df


if __name__ == "__main__":
    run_ingestion()
