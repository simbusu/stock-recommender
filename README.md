# Stock Buy/Sell Recommendation System

**AIMLCZG549 — API-Driven Cloud Native Solutions — Assignment I**
Source: [github.com/simbusu/stock-recommender](https://github.com/simbusu/stock-recommender)

An end-to-end ML system that recommends Buy/Sell signals on **49 Nifty 50 stocks** — combining
technical indicators, candlestick pattern recognition, and point-in-time-correct quarterly
fundamentals, orchestrated with Airflow, tracked with MLflow, and served through a Streamlit
dashboard.

---

## Who's this for, and how it helps

- **Assignment grader** — verify each rubric activity (1.2–3.2) is genuinely implemented and
  running, via the dashboard's **Pipeline Health** and **Model Performance** tabs, without
  needing separate Airflow/MLflow UI access.
- **A teammate joining the project** — get oriented fast: what each file does, why the pipeline
  runs daily, why ~51% accuracy is the honest result rather than a bug (see [Trial & Error](#trial--error--improving-accuracy)).
- **A retail investor / learner** — check any ticker's technical read (candlestick patterns,
  RSI/MACD, a rough 5-year trend illustration) via **Try a New Stock**, not limited to the 49
  trained tickers.
- **Someone evaluating whether ML adds value to trading** — the Trial & Error section documents
  every attempt to push accuracy higher and reports, without spin, that the ceiling seems to be
  data, not modeling technique.

> **Boundary that applies to every audience above:** this is not financial advice, and its value
> is in the engineering (a genuinely automated, leak-free, evaluated ML pipeline), not in a
> proven trading edge.

---

## Architecture

```
Yahoo Finance
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Airflow DAG · once daily, 30 min after NSE close        │
│                                                           │
│  ingest (49 tickers + indices + fundamentals)             │
│      │                                                    │
│      ▼                                                    │
│  preprocess  →  feature_engineer  →  run_eda  →  train     │
└──────────────────────────────┬────────────────────────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │  MLflow           │
                     │  Tracking +       │
                     │  Model Registry   │
                     └─────────┬────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │  Streamlit Dashboard         │
                  │  Candlesticks · Recs ·       │
                  │  MLflow explorer · EDA ·     │
                  │  Airflow status · New Stock  │
                  └────────────────────────────┘
```

Everything runs on a single GCP VM via Docker Compose. Data and model files are bind-mounted
host folders, so both host-run scripts and in-container Airflow tasks share the same files.

> **This is a redesign, not the original layout.** The pipeline originally ran every 2 minutes,
> with ingestion and training kept *outside* the DAG (that cadence was too aggressive to safely
> call Yahoo Finance or retrain a model against). That 2-minute interval existed to let a grader
> see multiple successful runs within a minute or two, not because it reflected how stock data or
> training actually work. At a daily cadence both constraints disappear, so the full pipeline —
> ingest through train — is now one honest, complete daily cycle.

---

## Setup

```bash
git clone https://github.com/simbusu/stock-recommender
cd stock-recommender
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Fix the Docker build context and host/container UID mismatch *before* your first build

```bash
cat > .dockerignore << 'EOF'
venv/
.venv/
__pycache__/
*.pyc
.git/
mlruns/
mlflow_db/
mlflow.db
data/stocks/*.csv
*.log
EOF

echo "AIRFLOW_UID=$(id -u)" > .env
```

Without the `.dockerignore`, Docker bundles your entire project — including any local `venv/`
and cached data — into the build context on every build (can bloat a routine build to 1.8GB+).
Without the `.env` UID fix, Airflow's container writes to `data/`/`models/` as a different UID
than your host user, causing intermittent `PermissionError` on alternating scheduled runs.

### Bring up the stack

```bash
docker compose up -d --build
sleep 30
docker compose ps
```

Expect **5 running containers**: `postgres_airflow`, `postgres_app`, `mlflow`,
`airflow-webserver`, `airflow-scheduler`. A 6th, `airflow-init`, runs once and then correctly
shows `Exited (0)` — that's success, not a crash.

### Populate data and train — run manually once to validate, before trusting the DAG

```bash
python3 ingester.py           # pulls live prices + fundamentals for 49 tickers via yfinance
python3 preprocessor.py
python3 feature_engineer.py
python3 eda_analysis.py
python3 train_ml.py           # trains RF + XGBoost, logs to MLflow — several minutes at this scale
python3 api_access.py         # prints the 4 API details for your report
```

Then un-pause `stock_pipeline_dag` in the Airflow UI (`http://<EXTERNAL_IP>:8080`, `admin`/`admin`)
so it runs the same 5 steps automatically, once daily, from then on.

### Launch the dashboard as a persistent service

```bash
sudo tee /etc/systemd/system/stock-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=Stock Recommender Streamlit Dashboard
After=network.target docker.service

[Service]
Type=simple
User=<your-username>
WorkingDirectory=/home/<your-username>/stock-recommender
ExecStart=/home/<your-username>/stock-recommender/venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now stock-dashboard
```

| What | URL |
|---|---|
| Dashboard | `http://<EXTERNAL_IP>:8501` |
| Airflow UI | `http://<EXTERNAL_IP>:8080` (`admin`/`admin`) |
| MLflow | Not exposed by default — use the dashboard's **Model Performance** tab instead |

---

## Dashboard tour

| Tab | What it shows |
|---|---|
| 📊 **Price & Indicators** | Real candlestick chart (true OHLC), MA7/MA21 + Bollinger overlays, green ▲/red ▼ markers for detected candlestick patterns, the live model's BUY/SELL call annotated on the latest candle, RSI/MACD panels, and a **5-Year Tentative Price Outlook** (historical CAGR + volatility extrapolation — clearly labeled as illustrative, not a forecast) |
| 🎯 **Recommendations** | Latest Buy/Sell signal + confidence per stock, both models side by side |
| 🤖 **Model Performance** | A full MLflow mirror built into the dashboard: experiment overview, sortable runs table, per-run drill-down (params/metrics/tags/artifacts), Model Registry, and a "🚀 Train Now" button — works over the internal Docker network regardless of whether MLflow's own port is exposed |
| 🔍 **EDA Insights** | Correlation heatmap + feature importance, refreshed every DAG cycle |
| ⚙️ **Pipeline Health** | Airflow DAG status via REST API — schedule, recent runs, color-coded state |
| 🔮 **Try a New Stock** | Enter *any* Yahoo Finance ticker, not limited to the 49 trained stocks — see [What happens with a new stock](#what-happens-when-you-enter-a-new-stock) |

---

## What happens when you enter a new stock

The model was never trained on "TCS" or "SBIN" as entities — the ticker itself isn't a feature.
It learned purely the *shape* of technical + fundamental + candlestick patterns and what tended
to follow. A new ticker works because the model is really asking: *"have I seen this combination
of RSI/MACD/Bollinger-position/P-E-ratio/candlestick-pattern before, and what usually followed?"*

1. **Fetch ~150 days of raw OHLCV** via `yfinance` — enough for MA21/Bollinger/MACD to warm up;
   refuses outright below 60 days of history.
2. **Fetch current fundamentals** — trailing P/E, EPS, market cap, debt-to-equity, current ratio,
   recent quarterly growth — same point-in-time-safe logic as training.
3. **Scale only the ratio columns using the *training-time* scaler** (never refit on the new
   ticker's own range — that would be meaningless relative to what the model learned). Raw OHLC
   is left completely untouched.
4. **Run the exact same `compute_technical_features()`** used in training — all ~50 features,
   including all 13 candlestick pattern flags, computed from this ticker's own real prices.
5. **Take only the most recent row**, fill gaps with 0 — that becomes the model's input `X`.
6. **`model.predict(X)` / `model.predict_proba(X)`** — the same trained Random Forest/XGBoost,
   given one new row describing a different company's current state.

> **Real limitation:** if a new ticker's fundamentals sit far outside anything the 49 training
> stocks ever showed, the model is extrapolating into territory it never learned from — the
> prediction still comes back, but with no real basis for confidence in it.

---

## Feature engineering, in depth

`feature_engineer.py` computes ~50 features per row, entirely from **raw** OHLC (see note below)
plus point-in-time-correct fundamentals.

### Trend & momentum
| Feature | Detail |
|---|---|
| `MA7`, `MA21`, `MA_crossover` | 7/21-day SMAs of Close; crossover is a binary short-vs-long trend flag |
| `RSI_14` | 14-day Relative Strength Index (Wilder smoothing) |
| `MACD`, `MACD_signal`, `MACD_hist` | 12/26-day EMA difference, 9-day signal, and histogram |
| `ADX_14` | 14-day Average Directional Index — trend *strength*, independent of direction |
| `OBV_roc_5` | 5-day rate-of-change of On-Balance Volume |

### Volatility & range
| Feature | Detail |
|---|---|
| `BB_upper`, `BB_lower`, `BB_width` | 20-day Bollinger Bands; width = `(upper − lower) / mid`, a pure volatility read |
| `ATR_14` | 14-day Average True Range |
| `Stoch_K`, `Stoch_D` | Stochastic oscillator |
| `price_range`, `volume_change`, `daily_return` | Same-day range, day-over-day volume/close % change |
| `daily_return_lag1/2/3`, `RSI_14_lag1/2` | Lagged versions — recent history without a recurrent architecture |

### Candlestick pattern flags (13 features)
Purely rule-based on real body/shadow geometry — no black box:

| Pattern | Rule |
|---|---|
| Doji | Body ≤ 10% of the day's High-Low range |
| Bullish/Bearish Marubozu | Body ≥ 90% of range, minimal shadows |
| Hammer / Inverted Hammer | Small body, long one-sided shadow, after a 5-day **down**trend (context-aware) |
| Shooting Star | Same shape as Inverted Hammer, but after a 5-day **up**trend |
| Bullish/Bearish Engulfing | Today's body fully engulfs yesterday's opposite-colored body |
| Morning/Evening Star | 3-candle reversal: long body → small body → long opposite body past the first candle's midpoint |
| `candle_bullish_signal_count`, `candle_bearish_signal_count`, `candle_net_signal` | Aggregated "how many patterns agree" summary |

All patterns use `.shift()`-based lookbacks only — nothing can see the future.

### Point-in-time-correct fundamentals
| Feature | Detail |
|---|---|
| `pe_ratio`, `debt_to_equity`, `current_ratio` | StandardScaler'd (z-scores) — see the ratio-unscaling note below |
| `eps`, `market_cap` | Left raw/unscaled |
| `revenue_yoy_growth`, `net_income_yoy_growth`, `revenue_qoq_growth`, `net_income_qoq_growth`, `earnings_growth_streak` | From `quarterly_fundamentals.py`, merged via `merge_asof` (backward) using an estimated public-reporting date — a row never sees a quarter's numbers before they'd actually have been published |

### Label generation

```python
forward_return = (Close[t+5] - Close[t]) / Close[t]

label = 1 (Buy)   if forward_return > +1%
label = 0 (Sell)  if forward_return < -1%
label = dropped   if -1% <= forward_return <= +1%   # neutral zone excluded
```

A 5-day horizon (next-day alone is closer to pure noise) with the ±1% neutral zone dropped
(forces the model to learn only from days that moved meaningfully, not guess on near-flat days).
This is why label balance sits around 55/45 rather than a forced 50/50.

> **Raw OHLC is preserved through the whole pipeline.** `preprocessor.py` writes MinMax-scaled
> prices into separate `_norm` columns rather than overwriting Open/High/Low/Close in place —
> keeps candlestick math correct and lets the dashboard plot genuine rupee prices.

> **Scaled fundamentals can look wrong if displayed raw.** `current_ratio = -0.14` reads as
> nonsensical (a real current ratio can't be negative) — it's actually "0.14 std-devs below the
> peer-group average." `explain.py`'s SHAP narratives now invert this transform before display;
> the model's actual input and SHAP ranking are untouched either way.

---

## Exploratory Data Analysis, in depth

`eda_analysis.py` runs six analyses every DAG cycle, saved as PNGs to `eda_plots/` for the report.

| Analysis | What it does | Output |
|---|---|---|
| Correlation matrix | Pearson correlation across every numeric column | `correlation_matrix.png` |
| Label correlations | Ranks every feature by absolute correlation with the label | Log output (top 15) |
| Binning | RSI → Oversold/Neutral/Overbought (30/70); P/E → Value/Growth/Overvalued terciles | `RSI_zone`/`PE_zone` columns |
| Feature importance | Standalone `RandomForestClassifier` fit purely for ranking | `feature_importance.png` |
| Univariate charts | Daily-return distribution (all tickers); RSI(14) distribution | `univariate_charts.png` |
| Bivariate charts | MACD histogram vs label; P/E vs return by label; sell-rate by VIX quintile | `bivariate_charts.png` |

> `forward_return` itself is consistently the single strongest correlate with the label
> (expected — the label is derived from it) with every other feature an order of magnitude
> weaker — further evidence for the Trial & Error conclusion below.

---

## Models & evaluation

Random Forest and XGBoost, tuned via `RandomizedSearchCV` with `TimeSeriesSplit` (walk-forward,
never shuffled), 70/30 time-ordered split, evaluated on accuracy/precision/recall/F1, both
registered in the MLflow Model Registry.

> **~51% accuracy is the expected result, not a bug.** Predicting next-period stock direction
> from public indicators sits close to a coin flip in the literature — if it were reliably
> easier, the edge would already be arbitraged away. Report this as a legitimate finding, not a
> modeling failure to chase away with more tuning.

---

## Trial & Error — improving accuracy

Several distinct things were tried to move accuracy past ~51%. None produced a large lift —
which is itself the finding.

1. **Fixed a look-ahead leak in fundamentals** — the original design left-joined a single
   fundamentals snapshot onto every historical row. Fixed via point-in-time `merge_asof`. Doesn't
   raise accuracy (closing a leak often lowers a score, since the leaked version was cheating) —
   the point is trustworthiness, not a higher number.
2. **Fixed OHLC being scaled in place before indicator calculation** — distorted RSI/MACD/ATR/
   candlestick math. Fixed by scaling into separate `_norm` columns. A correctness fix, not a
   guaranteed accuracy lift.
3. **Added 13 candlestick pattern features** — F1 moved from 0.5533→0.5579 (RF) and
   0.5015→0.4922 (XGB): small, inconsistent. Kept for genuine visual/explanatory value even
   without a reliable accuracy lift.
4. **Compared 4 model families** (`compare_models.py`) — Random Forest, XGBoost, Logistic
   Regression, and an MLP all clustered within a few points of 50% accuracy, including
   structurally different algorithms. Evidence the ceiling is data-driven, not model-choice-driven.
5. **Tested confidence-based filtering** (`confidence_analysis.py`) — accuracy stayed flat
   (~45-53%) across every confidence bucket for both models. The models' predicted probabilities
   aren't well-calibrated indicators of correctness, so "only trade high-confidence days" isn't a
   valid workaround here.

> **Overall conclusion:** across a leakage fix, an indicator-correctness fix, a new feature
> family, four model architectures, and a confidence-filtering strategy — accuracy stayed
> anchored in the low-to-mid 50s throughout. That consistency is the actual evidence: the
> bottleneck is the amount of predictable signal in short-horizon price direction from public
> data, not something fixable by more tuning.

---

## Project structure

| File | Purpose | Run how |
|---|---|---|
| `config.py` | Single source of truth — 49-stock Nifty list, date ranges, indicator windows, DB/API settings | Imported everywhere; edit, don't run |
| `ingester.py` | 1.2 Data ingestion — OHLCV + fundamentals for 49 tickers via `yfinance` | Manual, or the DAG's `ingest` task |
| `preprocessor.py` | 1.3 Pre-processing — stats, imputation, dtype fixes, normalization (raw OHLC preserved) | DAG `preprocess` task, or manual |
| `feature_engineer.py` | Technical indicators + 13 candlestick patterns + label generation | DAG `feature_engineer` task, or manual |
| `quarterly_fundamentals.py` | Point-in-time-correct quarterly fundamentals merge (fixes a look-ahead leak) | Called by the pipeline |
| `eda_analysis.py` | 1.4 EDA — correlation, binning, importance, uni/bivariate charts | DAG `run_eda` task, or manual |
| `train_ml.py` | 2.1–2.4 ML pipeline — RF + XGBoost, time-ordered split, evaluation, MLflow logging | DAG `train` task, or manual |
| `explain.py` | SHAP-based per-prediction explanations, with ratio-unscaling for readable output | Imported by the dashboard |
| `api_access.py` | 3.1–3.2 Retrieves 4 app details via Airflow + MLflow REST APIs | Manual, for verification/report |
| `predict_new_ticker.py` | Runs any new ticker through the trained models (never refits scalers) | Imported by dashboard Tab 6 |
| `compare_models.py` | Side experiment — 4 model families compared on the same data | Manual, exploratory only |
| `confidence_analysis.py` | Side experiment — accuracy vs. confidence bucket | Manual, exploratory only |
| `dags/stock_pipeline_dag.py` | 1.5 DataOps — full daily DAG: ingest → preprocess → feature_engineer → run_eda → train | Runs inside `airflow-scheduler` |
| `dashboard/app.py` | Streamlit UI — all 6 tabs above | `streamlit run dashboard/app.py ...` |
| `dashboard/data_access.py` | All dashboard data-loading/query helpers, incl. MLflow/Airflow REST calls | Imported by `app.py` |
| `docker-compose.yml` / `Dockerfile` | Full stack: Airflow, PostgreSQL, MLflow (with persistent volume mounts — see Troubleshooting) | `docker compose up -d --build` |
| `test_with_synthetic_data.py` | Local validation harness using fabricated data — **not** part of the graded pipeline | Manual, sanity-check only |

---

## Troubleshooting — issues actually hit building this

**MLflow: `403 Invalid Host header - possible DNS rebinding attack detected`**
A newer MLflow version (3.x) added a Host-header allowlist. The documented `--allowed-hosts` CLI
flag does **not** actually wire through (confirmed by reading `mlflow.server.security_utils`
source — it only reads an environment variable). Fix:
```yaml
environment:
  MLFLOW_SERVER_ALLOWED_HOSTS: "mlflow:*,mlflow,localhost:*,localhost,127.0.0.1:*,127.0.0.1,<VM_EXTERNAL_IP>:*,<VM_EXTERNAL_IP>"
```

**MLflow UI browser-only 403s that `curl` never catches**
A separate CORS `Origin` check exists — different env var, full origin URLs:
```yaml
environment:
  MLFLOW_SERVER_CORS_ALLOWED_ORIGINS: "http://<VM_EXTERNAL_IP>:5000,http://localhost:5000"
```

**The big one — MLflow's entire run/registry history silently wiped on container recreate**
The original `mlflow` service ran `--backend-store-uri sqlite:///mlflow.db` with **no volume
mount for that file** — only `./mlruns:/mlruns` (artifacts) was persisted. Every
`docker compose up -d --force-recreate mlflow` silently created a brand-new, empty database,
deleting every logged run and registered model version. Fix — mount the database itself:
```yaml
volumes:
  - ./mlruns:/mlruns
  - ./mlflow_db:/mlflow_db
command: >
  ... --backend-store-uri sqlite:////mlflow_db/mlflow.db ...   # note: 4 slashes
```

**`PermissionError` on `models/price_scaler.pkl` during a scheduled DAG run**
Host/container UID mismatch on bind-mounted `data/`/`models/`. Fix: `AIRFLOW_UID=$(id -u)` in
`.env`, plus `user: "${AIRFLOW_UID:-50000}:0"` on the Airflow services in `docker-compose.yml`.

**A ticker 404s: `Quote not found for symbol: TATAMOTORS.NS`**
Real corporate actions break hardcoded ticker lists over time. Tata Motors demerged in Oct 2025;
the Nifty 50 constituent (passenger vehicles + JLR) now trades as `TMPV.NS`. Run `ingester.py`
manually first and watch for `[ERROR]`/`[WARNING]` lines before trusting the DAG unattended.

**Only ~20% of rows have real quarterly fundamentals**
Not a bug — `QUARTERLY_LOOKBACK` in `config.py` only fetches ~2 years of quarterly history per
ticker, against ~6.5 years of price history. Rows older than that legitimately have no quarter to
match yet (point-in-time correctness working as intended) and get median-imputed. Increase
`QUARTERLY_LOOKBACK` for deeper real coverage if needed.

**`docker compose ps` shows only long-running containers**
`airflow-init` is a one-shot container — use `docker compose ps -a` to see it; `Exited (0)` is
success, not a crash.
