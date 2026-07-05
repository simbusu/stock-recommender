"""
explain.py — SHAP-based "why this recommendation" reasoning.

Handles a real, version-specific quirk confirmed by direct testing
(shap==0.52.0) rather than assumed from docs: TreeExplainer.shap_values()
returns a 3D array (rows, features, classes) for RandomForestClassifier —
additive directly in PROBABILITY space — but a 2D array (rows, features)
for XGBClassifier — additive in LOG-ODDS (margin) space. Both use the
same sign convention (positive SHAP value pushes toward class 1 / BUY),
so ranking by |shap value| within a single model is valid either way,
but the two models' magnitudes are NOT directly comparable to each other.
"""
import numpy as np
import pandas as pd


def _rsi_hint(v):
    if v > 70: return "overbought — often precedes a pullback"
    if v < 30: return "oversold — often precedes a rebound"
    return None

def _stoch_hint(v):
    if v > 80: return "overbought zone"
    if v < 20: return "oversold zone"
    return None

def _adx_hint(v):
    if v > 25: return "strong trend in force"
    return "weak/no clear trend"

def _macd_hist_hint(v):
    return "bullish momentum building" if v > 0 else "bearish momentum building"

def _ma_crossover_hint(v):
    return "short-term trend above long-term trend (bullish)" if v >= 1 else "short-term trend below long-term trend (bearish)"

def _candle_flag_hint(v):
    return "pattern present on this candle" if v >= 1 else None

def _candle_net_signal_hint(v):
    if v > 0: return "more bullish than bearish patterns detected recently"
    if v < 0: return "more bearish than bullish patterns detected recently"
    return None


# display name + optional hint function (returns None if no clear signal)
FEATURE_INFO = {
    "RSI_14": ("RSI (14-day)", _rsi_hint),
    "MACD_hist": ("MACD histogram", _macd_hist_hint),
    "MA_crossover": ("MA7 vs MA21 crossover", _ma_crossover_hint),
    "Stoch_K": ("Stochastic %K", _stoch_hint),
    "ADX_14": ("ADX (trend strength)", _adx_hint),
    "candle_doji": ("Doji pattern", _candle_flag_hint),
    "candle_hammer": ("Hammer pattern", _candle_flag_hint),
    "candle_inverted_hammer": ("Inverted Hammer pattern", _candle_flag_hint),
    "candle_shooting_star": ("Shooting Star pattern", _candle_flag_hint),
    "candle_bullish_marubozu": ("Bullish Marubozu", _candle_flag_hint),
    "candle_bearish_marubozu": ("Bearish Marubozu", _candle_flag_hint),
    "candle_bullish_engulfing": ("Bullish Engulfing pattern", _candle_flag_hint),
    "candle_bearish_engulfing": ("Bearish Engulfing pattern", _candle_flag_hint),
    "candle_morning_star": ("Morning Star pattern", _candle_flag_hint),
    "candle_evening_star": ("Evening Star pattern", _candle_flag_hint),
    "candle_bullish_signal_count": ("Bullish candlestick signal count", None),
    "candle_bearish_signal_count": ("Bearish candlestick signal count", None),
    "candle_net_signal": ("Net candlestick signal (bullish − bearish)", _candle_net_signal_hint),
}


def _to_positive_class_shap(shap_values: np.ndarray) -> np.ndarray:
    """Normalize both the 3D (RandomForest) and 2D (XGBoost) shapes down to
    a single 2D (rows, features) array of contributions toward class 1 (BUY)."""
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _format_value(feat: str, val: float) -> str:
    """Human-readable formatting — large monetary figures get suffixes, ratios stay precise."""
    if "market_cap" in feat or "revenue" in feat or "cash_flow" in feat:
        for threshold, suffix in [(1e12, "T"), (1e9, "B"), (1e7, "Cr"), (1e5, "L")]:
            if abs(val) >= threshold:
                return f"{val / threshold:.2f}{suffix}"
        return f"{val:,.0f}"
    if "RSI" in feat or "Stoch" in feat or "ADX" in feat:
        return f"{val:.1f}"
    return f"{val:.3f}" if abs(val) < 10 else f"{val:.2f}"


def _base_feature(feat: str) -> str:
    """Strip _lag1/_lag2/_lag3 suffix so lagged variants reuse the same hint logic as the base indicator."""
    for lag in (1, 2, 3):
        suffix = f"_lag{lag}"
        if feat.endswith(suffix):
            return feat[: -len(suffix)]
    return feat


def explain_prediction(model, X_row: pd.DataFrame, feature_names: list, top_n: int = 5) -> list:
    """
    Return the top_n features (by |SHAP value|) driving this single
    prediction, each as {feature, display_name, value, shap_value, pushes_toward, hint}.
    """
    import shap  # lazy import — SHAP is only needed when reasoning is requested

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_row)
    shap_row = _to_positive_class_shap(np.asarray(raw_shap))[0]  # single-row input

    rows = []
    for i, feat in enumerate(feature_names):
        base_feat = _base_feature(feat)
        display_name, hint_fn = FEATURE_INFO.get(base_feat, (feat.replace("_", " "), None))
        if base_feat != feat:  # e.g. "RSI_14_lag1" -> use RSI's hint fn but label it as lag1
            display_name = f"{display_name} ({feat.split('_lag')[1]}d ago)"
        sv = float(shap_row[i])
        val = float(X_row.iloc[0][feat])
        hint = hint_fn(val) if hint_fn else None
        rows.append({
            "feature": feat,
            "display_name": display_name,
            "value": val,
            "value_display": _format_value(feat, val),
            "shap_value": sv,
            "pushes_toward": "BUY" if sv > 0 else "SELL",
            "hint": hint,
        })

    rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    return rows[:top_n]


def build_narrative(explanation: list, signal: str) -> str:
    """Turn the ranked SHAP explanation into a short, plain-English narrative."""
    if not explanation:
        return "No explanation available."

    supporting = [r for r in explanation if r["pushes_toward"] == signal]
    opposing = [r for r in explanation if r["pushes_toward"] != signal]

    lines = [f"**Top factors behind this {signal} signal:**"]
    for r in supporting[:3]:
        hint_text = f" ({r['hint']})" if r["hint"] else ""
        lines.append(f"- ✅ **{r['display_name']}** = {r['value_display']}{hint_text} → pushed toward {signal}")

    if opposing:
        other = "SELL" if signal == "BUY" else "BUY"
        lines.append(f"\n**Working against this call:**")
        for r in opposing[:2]:
            hint_text = f" ({r['hint']})" if r["hint"] else ""
            lines.append(f"- ⚠️ **{r['display_name']}** = {r['value_display']}{hint_text} → pushed toward {other}")

    return "\n".join(lines)
