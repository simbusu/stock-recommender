"""
feature_engineer.py — Technical indicator computation + label generation.

Computes MA, RSI, MACD, Bollinger Bands, candlestick patterns, and
return-based features per ticker, then derives the binary Buy/Sell label
from forward price movement (no look-ahead bias: the label uses future
closes relative to t, and the row for t is dropped only after the label
is attached — nothing at time t uses information beyond time t to build
features).
"""
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("feature_engineer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stochastic(high, low, close, window, smooth):
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(smooth).mean()
    return pct_k, pct_d


def _atr(high, low, close, window):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx(high, low, close, window):
    """Average Directional Index (trend strength, 0-100), Wilder's smoothing."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx


def _obv_momentum(close: pd.Series, volume: pd.Series, window: int = 5) -> pd.Series:
    """
    On-Balance Volume, expressed as a bounded RATE OF CHANGE rather than
    the raw cumulative level. Raw OBV grows monotonically with however
    much history is in the series, so its absolute scale is NOT
    comparable between a ticker with years of training history and a
    ticker fetched with only ~100 days for a new/unseen-stock prediction
    (predict_new_ticker.py). Normalizing by the trailing volume sum keeps
    this feature stationary and consistent across both use cases.
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    vol_sum = volume.rolling(window).sum().replace(0, np.nan)
    return obv.diff(window) / vol_sum


def compute_candlestick_patterns(g: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based Japanese candlestick pattern detection on RAW OHLC.

    Each pattern is a 0/1 flag for the CURRENT row. Multi-candle patterns
    (engulfing, morning/evening star) reference only the CURRENT and PAST
    rows via .shift(), so there's no look-ahead leakage — consistent with
    every other indicator in this file.

    Definitions used (standard technical-analysis rules of thumb):
      - Doji: open ~= close (tiny body relative to the day's range)
      - Hammer / Inverted Hammer: small body with a long shadow on one
        side, appearing after a down-move (bullish reversal signal)
      - Shooting Star: same shape as inverted hammer but after an
        up-move (bearish reversal signal)
      - Bullish/Bearish Marubozu: strong body, almost no shadows
      - Bullish/Bearish Engulfing: today's body fully engulfs yesterday's
        opposite-colored body
      - Morning/Evening Star: 3-candle reversal (long body, small body,
        long opposite body closing past the midpoint of candle 1)
    """
    o, h, l, c = g["Open"], g["High"], g["Low"], g["Close"]
    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    prior_trend = c.pct_change(5)  # 5-day momentum, gives reversal patterns context

    # ── Single-candle patterns ──────────────────────────────────────
    g["candle_doji"] = (body <= 0.1 * candle_range).astype(int)
    g["candle_bullish_marubozu"] = ((body >= 0.9 * candle_range) & (c > o)).astype(int)
    g["candle_bearish_marubozu"] = ((body >= 0.9 * candle_range) & (c < o)).astype(int)

    hammer_shape = (lower_shadow >= 2 * body) & (upper_shadow <= 0.3 * body) & (body > 0)
    star_shape = (upper_shadow >= 2 * body) & (lower_shadow <= 0.3 * body) & (body > 0)
    g["candle_hammer"] = (hammer_shape & (prior_trend < 0)).astype(int)
    g["candle_inverted_hammer"] = (star_shape & (prior_trend < 0)).astype(int)
    g["candle_shooting_star"] = (star_shape & (prior_trend > 0)).astype(int)

    # ── Two-candle patterns ──────────────────────────────────────────
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bearish = prev_c < prev_o
    prev_bullish = prev_c > prev_o
    g["candle_bullish_engulfing"] = (
        prev_bearish & (c > o) & (o <= prev_c) & (c >= prev_o)
    ).astype(int)
    g["candle_bearish_engulfing"] = (
        prev_bullish & (c < o) & (o >= prev_c) & (c <= prev_o)
    ).astype(int)

    # ── Three-candle patterns ─────────────────────────────────────────
    o1, c1 = o.shift(2), c.shift(2)   # candle 1 (oldest of the three)
    o2, c2 = o.shift(1), c.shift(1)   # candle 2 (middle, small body)
    body1 = (c1 - o1).abs()
    body2 = (c2 - o2).abs()
    mid1 = (o1 + c1) / 2
    g["candle_morning_star"] = (
        (c1 < o1) & (body1 > 0) & (body2 <= 0.3 * body1) & (c > o) & (c >= mid1)
    ).astype(int)
    g["candle_evening_star"] = (
        (c1 > o1) & (body1 > 0) & (body2 <= 0.3 * body1) & (c < o) & (c <= mid1)
    ).astype(int)

    bullish_cols = [
        "candle_bullish_marubozu", "candle_hammer", "candle_inverted_hammer",
        "candle_bullish_engulfing", "candle_morning_star",
    ]
    bearish_cols = [
        "candle_bearish_marubozu", "candle_shooting_star",
        "candle_bearish_engulfing", "candle_evening_star",
    ]
    g["candle_bullish_signal_count"] = g[bullish_cols].sum(axis=1)
    g["candle_bearish_signal_count"] = g[bearish_cols].sum(axis=1)
    g["candle_net_signal"] = g["candle_bullish_signal_count"] - g["candle_bearish_signal_count"]

    return g


def compute_technical_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for a single ticker's time series (sorted by date)."""
    g = g.sort_values("Date").copy()

    g["daily_return"] = g["Close"].pct_change()
    g["MA7"] = g["Close"].rolling(config.MA_SHORT_WINDOW).mean()
    g["MA21"] = g["Close"].rolling(config.MA_LONG_WINDOW).mean()
    g["MA_crossover"] = (g["MA7"] > g["MA21"]).astype(int)

    g["RSI_14"] = _rsi(g["Close"], config.RSI_WINDOW)

    ema_fast = g["Close"].ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = g["Close"].ewm(span=config.MACD_SLOW, adjust=False).mean()
    g["MACD"] = ema_fast - ema_slow
    g["MACD_signal"] = g["MACD"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    g["MACD_hist"] = g["MACD"] - g["MACD_signal"]

    bb_mid = g["Close"].rolling(config.BB_WINDOW).mean()
    bb_std = g["Close"].rolling(config.BB_WINDOW).std()
    g["BB_upper"] = bb_mid + config.BB_STD * bb_std
    g["BB_lower"] = bb_mid - config.BB_STD * bb_std
    g["BB_width"] = (g["BB_upper"] - g["BB_lower"]) / bb_mid

    g["volume_change"] = g["Volume"].pct_change()
    g["price_range"] = (g["High"] - g["Low"]) / g["Low"]

    # ── New indicators (all trailing/rolling-window — no look-ahead) ──
    g["Stoch_K"], g["Stoch_D"] = _stochastic(g["High"], g["Low"], g["Close"], config.STOCH_WINDOW, config.STOCH_SMOOTH)
    g["ATR_14"] = _atr(g["High"], g["Low"], g["Close"], config.ATR_WINDOW)
    g["ADX_14"] = _adx(g["High"], g["Low"], g["Close"], config.ADX_WINDOW)
    g["OBV_roc_5"] = _obv_momentum(g["Close"], g["Volume"], window=5)

    # ── Lag features (shift forward = look at the PAST, safe) ─────────
    for lag in config.LAG_PERIODS:
        g[f"daily_return_lag{lag}"] = g["daily_return"].shift(lag)
        g[f"RSI_14_lag{lag}"] = g["RSI_14"].shift(lag)

    # ── Candlestick pattern flags (raw OHLC — see function docstring) ──
    g = compute_candlestick_patterns(g)

    return g


def add_label(g: pd.DataFrame, horizon: int = None, neutral_zone_pct: float = None) -> pd.DataFrame:
    """
    Buy (1) if the close `horizon` trading days ahead is up by more than
    `neutral_zone_pct`, Sell (0) if it's down by more than that threshold.
    Rows where the forward move is smaller than the threshold (ambiguous,
    noisy moves) are dropped entirely rather than forced into a class —
    this is what changes the model's job from "call every coin-flip day"
    to "call the days that actually moved."

    Still leakage-safe: g["Close"].shift(-horizon) only ever looks
    STRICTLY FORWARD from row t, and every row without a full forward
    window is dropped (g = g.iloc[:-horizon]) — nothing at prediction
    time t uses information from beyond t.
    """
    horizon = horizon if horizon is not None else config.LABEL_HORIZON_DAYS
    neutral_zone_pct = neutral_zone_pct if neutral_zone_pct is not None else config.NEUTRAL_ZONE_PCT

    g = g.sort_values("Date").copy()
    g["forward_close"] = g["Close"].shift(-horizon)
    g["forward_return"] = (g["forward_close"] - g["Close"]) / g["Close"]

    g["label"] = np.where(
        g["forward_return"] > neutral_zone_pct, 1,
        np.where(g["forward_return"] < -neutral_zone_pct, 0, np.nan),
    )
    g = g.iloc[:-horizon]  # drop rows with no full forward window available
    g = g.dropna(subset=["label"])  # drop neutral-zone (ambiguous) rows
    g["label"] = g["label"].astype(int)
    return g


def run_feature_engineering(df: pd.DataFrame, horizon: int = None, neutral_zone_pct: float = None) -> pd.DataFrame:
    # Note: using an explicit per-ticker loop (rather than groupby().apply())
    # because recent pandas versions drop the grouping column from the frame
    # passed into apply(), which would silently lose 'ticker' downstream.
    processed_groups = []
    for ticker, g in df.groupby("ticker", observed=True):
        g = compute_technical_features(g)
        rows_before_labeling = len(g)
        g = add_label(g, horizon=horizon, neutral_zone_pct=neutral_zone_pct)
        g["ticker"] = ticker
        processed_groups.append(g)

    df = pd.concat(processed_groups, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)  # guard against div-by-zero in BB_width/pct_change/OBV_roc

    warmup_cols = [
        "MA21", "RSI_14", "MACD_signal", "BB_width",
        "Stoch_D", "ATR_14", "ADX_14", "OBV_roc_5",
        f"daily_return_lag{max(config.LAG_PERIODS)}",
    ]
    before_warmup_drop = len(df)
    df = df.dropna(subset=warmup_cols)  # drop indicator warm-up rows

    logger.info(
        "Feature engineering complete: shape %s (dropped %d rows for indicator "
        "warm-up + neutral-zone exclusion), label balance:\n%s",
        df.shape, before_warmup_drop - len(df) if before_warmup_drop >= len(df) else 0,
        df["label"].value_counts(normalize=True).to_string(),
    )
    return df


if __name__ == "__main__":
    processed = pd.read_csv("data/stocks/processed.csv", parse_dates=["Date"])
    featured = run_feature_engineering(processed)
    featured.to_csv("data/stocks/featured.csv", index=False)
