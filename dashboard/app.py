"""
dashboard/app.py — Streamlit UI for the Stock Buy/Sell Recommendation System.

Run with:
    streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0

Five tabs:
  1. Price & Indicators   — candlestick + MA/RSI/MACD/Bollinger overlays
  2. Recommendations       — latest Buy/Sell signal per stock, per model
  3. Model Performance     — MLflow run metrics (accuracy/precision/recall/F1)
  4. EDA Insights          — correlation heatmap + feature importance
  5. Pipeline Health       — Airflow DAG status via REST API
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_access as da

st.set_page_config(page_title="Stock Buy/Sell Recommender", layout="wide", page_icon="📈")

st.title("📈 Nifty Stock Buy/Sell Recommendation Dashboard")
st.caption("AIMLCZG549 — API-Driven Cloud Native Solutions — Assignment I — Group 42")


@st.cache_data(ttl=1800)  # 30 min — pipeline now runs once/day, no need to poll every 2 min
def cached_load_featured_data():
    return da.load_featured_data()


@st.cache_resource(ttl=3600)  # 1 hr — picks up the DAG's daily retrain without a manual "Train Now" click
def cached_load_models():
    return da.load_models()


# ── Load data (with a clear error if the pipeline hasn't run yet) ──────────
try:
    df = cached_load_featured_data()
    data_load_error = None
except FileNotFoundError as e:
    df = None
    data_load_error = str(e)

if data_load_error:
    st.error(f"⚠️ {data_load_error}")
    st.stop()

models = cached_load_models()
tickers = da.get_available_tickers(df)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Price & Indicators", "🎯 Recommendations", "🤖 Model Performance",
    "🔍 EDA Insights", "⚙️ Pipeline Health", "🔮 Try a New Stock",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: Price & Indicators
# ═══════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        selected_ticker = st.selectbox("Select stock", tickers, key="price_ticker")
        lookback = st.slider("Days to show", 30, 500, 180)

    hist = da.get_ticker_history(df, selected_ticker, lookback)

    if hist.empty:
        st.warning("No data available for this ticker.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist["Date"], open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ))
        if "MA7" in hist.columns:
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MA7"], name="MA7", line=dict(color="orange", dash="dot")))
        if "MA21" in hist.columns:
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MA21"], name="MA21", line=dict(color="blue", dash="dot")))
        if "BB_upper" in hist.columns and "BB_lower" in hist.columns:
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["BB_upper"], name="BB Upper", line=dict(color="gray", width=1), showlegend=False))
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["BB_lower"], name="BB Lower", line=dict(color="gray", width=1), fill="tonexty", fillcolor="rgba(128,128,128,0.1)", showlegend=False))

        # ── Candlestick pattern markers ──────────────────────────────
        bullish_pattern_cols = [c for c in [
            "candle_hammer", "candle_inverted_hammer", "candle_bullish_marubozu",
            "candle_bullish_engulfing", "candle_morning_star",
        ] if c in hist.columns]
        bearish_pattern_cols = [c for c in [
            "candle_shooting_star", "candle_bearish_marubozu",
            "candle_bearish_engulfing", "candle_evening_star",
        ] if c in hist.columns]

        def _pattern_hover(row, cols):
            names = [c.replace("candle_", "").replace("_", " ").title() for c in cols if row[c] == 1]
            return ", ".join(names)

        if bullish_pattern_cols:
            hits = hist[hist[bullish_pattern_cols].sum(axis=1) > 0]
            if not hits.empty:
                fig.add_trace(go.Scatter(
                    x=hits["Date"], y=hits["Low"] * 0.985, mode="markers",
                    marker=dict(symbol="triangle-up", color="#00c853", size=10, line=dict(width=1, color="black")),
                    name="Bullish pattern",
                    text=hits.apply(lambda r: _pattern_hover(r, bullish_pattern_cols), axis=1),
                    hovertemplate="%{text}<extra></extra>",
                ))
        if bearish_pattern_cols:
            hits = hist[hist[bearish_pattern_cols].sum(axis=1) > 0]
            if not hits.empty:
                fig.add_trace(go.Scatter(
                    x=hits["Date"], y=hits["High"] * 1.015, mode="markers",
                    marker=dict(symbol="triangle-down", color="#d50000", size=10, line=dict(width=1, color="black")),
                    name="Bearish pattern",
                    text=hits.apply(lambda r: _pattern_hover(r, bearish_pattern_cols), axis=1),
                    hovertemplate="%{text}<extra></extra>",
                ))

        # ── Latest model signal, annotated directly on the chart ──────
        if models:
            latest_rec = da.get_latest_recommendation(df, selected_ticker, models)
            offset = 0
            for model_name, result in latest_rec.items():
                sig = result.get("signal")
                if sig in ("BUY", "SELL"):
                    fig.add_annotation(
                        x=hist["Date"].iloc[-1], y=hist["High"].iloc[-1],
                        text=f"{model_name}: {sig} ({result.get('confidence', 0):.0%})",
                        showarrow=True, arrowhead=2, ay=-40 - offset,
                        bgcolor="#d4edda" if sig == "BUY" else "#f8d7da",
                        font=dict(size=11),
                    )
                    offset += 28

        fig.update_layout(
            title=f"{selected_ticker} — Candlestick + Moving Averages + Bollinger Bands",
            height=500, xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("▲ green = bullish candlestick pattern · ▼ red = bearish candlestick pattern (hover for pattern name)")

        col1, col2 = st.columns(2)
        with col1:
            if "RSI_14" in hist.columns:
                rsi_fig = px.line(hist, x="Date", y="RSI_14", title="RSI (14-day)")
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                st.plotly_chart(rsi_fig, use_container_width=True)
        with col2:
            if "MACD" in hist.columns and "MACD_signal" in hist.columns:
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MACD"], name="MACD"))
                macd_fig.add_trace(go.Scatter(x=hist["Date"], y=hist["MACD_signal"], name="Signal"))
                macd_fig.add_trace(go.Bar(x=hist["Date"], y=hist["MACD_hist"], name="Histogram", opacity=0.4))
                macd_fig.update_layout(title="MACD")
                st.plotly_chart(macd_fig, use_container_width=True)

        # ── 5-Year Tentative Price Outlook ─────────────────────────────
        st.divider()
        st.subheader("🔮 Tentative Price — 5 Year Outlook")
        st.caption(
            "Simple historical-trend (CAGR) extrapolation of past prices. This is an "
            "**educational illustration, not a forecast or investment advice** — real "
            "future prices depend on countless factors this simple projection ignores."
        )
        full_hist = da.get_ticker_history(df, selected_ticker, last_n_days=100_000)
        proj = da.project_price_5y(full_hist, years=5)
        if not proj:
            st.info("Not enough price history for this ticker to compute a projection.")
        else:
            proj_fig = go.Figure()
            proj_fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], name="Historical Close", line=dict(color="#1f77b4")))
            proj_fig.add_trace(go.Scatter(x=proj["future_dates"], y=proj["optimistic"], name="Optimistic", line=dict(color="#00c853", dash="dot")))
            proj_fig.add_trace(go.Scatter(x=proj["future_dates"], y=proj["base"], name="Tentative (base CAGR)", line=dict(color="orange", dash="dash")))
            proj_fig.add_trace(go.Scatter(x=proj["future_dates"], y=proj["pessimistic"], name="Pessimistic", line=dict(color="#d50000", dash="dot")))
            proj_fig.update_layout(title=f"{selected_ticker} — Tentative Price, Next 5 Years (illustrative only)", height=400)
            st.plotly_chart(proj_fig, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Historical CAGR", f"{proj['cagr'] * 100:.1f}%/yr", help=f"Computed from {proj['history_years']} years of history")
            m2.metric("Tentative price in 5y (base)", f"₹{proj['base'][-1]:,.0f}")
            m3.metric("Optimistic (5y)", f"₹{proj['optimistic'][-1]:,.0f}")
            m4.metric("Pessimistic (5y)", f"₹{proj['pessimistic'][-1]:,.0f}")

# ═══════════════════════════════════════════════════════════════
# TAB 2: Recommendations
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Latest Buy/Sell Signal — All Stocks")
    if not models:
        st.warning(
            "No trained models found in `models/`. Run `python train_ml.py` first, "
            "then refresh this page."
        )
    else:
        rec_df = da.get_recommendations_for_all_tickers(df, models)

        def highlight_signal(val):
            if val == "BUY":
                return "background-color: #d4edda; color: #155724; font-weight: bold"
            elif val == "SELL":
                return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            return ""

        signal_cols = [c for c in rec_df.columns if "Signal" in c]
        st.dataframe(
            rec_df.style.applymap(highlight_signal, subset=signal_cols),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "⚠️ Educational project output only — not investment advice. "
            "Model accuracy is close to random baseline (see Model Performance tab)."
        )

        st.divider()
        st.subheader("🧠 Why? (SHAP-based reasoning)")
        why_ticker = st.selectbox("Explain the recommendation for", tickers, key="why_ticker")
        if st.button("Explain this recommendation"):
            with st.spinner("Computing SHAP explanation..."):
                rec_with_reason = da.get_latest_recommendation(df, why_ticker, models, with_explanation=True)
                cols = st.columns(len(rec_with_reason))
                for col, (model_name, result) in zip(cols, rec_with_reason.items()):
                    with col:
                        st.markdown(f"**{model_name}**")
                        if "narrative" in result:
                            st.markdown(result["narrative"])
                        else:
                            st.error(result.get("error", "explanation unavailable"))

# ═══════════════════════════════════════════════════════════════
# TAB 3: Model Performance
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("MLflow Experiment Runs")
    st.caption(
        "This tab talks to MLflow over the internal Docker network, so it works "
        "even if port 5000 isn't reachable from your browser — no separate MLflow "
        "UI access needed."
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        train_clicked = st.button("🚀 Train Now", type="primary", use_container_width=True)

    if train_clicked:
        with st.spinner("Training Random Forest + XGBoost and logging to MLflow (~10-30s)..."):
            try:
                results = da.trigger_training_run(df)
                st.success("✅ New run logged to MLflow.")
                for model_name, metrics in results.items():
                    shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if k != "run_id"}
                    st.write(f"**{model_name}**: {shown}")
                cached_load_models.clear()  # force reload of the freshly-saved .pkl files
                st.cache_data.clear()  # force MLflow runs table + featured-data cache to refresh
            except Exception as e:
                st.error(f"Training run failed: {e}")
                st.exception(e)

    try:
        # ── Experiment overview (mirrors the header of MLflow's Experiment page) ──
        overview = da.get_mlflow_experiment_overview()
        if overview:
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Experiment", overview["name"])
            oc2.metric("Experiment ID", overview["experiment_id"])
            oc3.metric("Lifecycle Stage", overview["lifecycle_stage"])
            st.caption(f"Artifact location: `{overview['artifact_location']}`  ·  Tracking URI: `{overview['tracking_uri']}`")

        runs_df = da.get_mlflow_runs_summary()
        if runs_df.empty:
            st.info("No MLflow runs found yet. Run `python train_ml.py` first.")
        else:
            metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score"] if c in runs_df.columns]
            display_cols = ["run_name", "status", "start_time"] + metric_cols
            st.dataframe(runs_df[display_cols].round(4), use_container_width=True, hide_index=True)

            if metric_cols:
                latest_per_model = runs_df.drop_duplicates(subset="run_name", keep="first")
                melted = latest_per_model.melt(id_vars="run_name", value_vars=metric_cols, var_name="metric", value_name="value")
                bar_fig = px.bar(melted, x="metric", y="value", color="run_name", barmode="group", title="Latest Metrics by Model")
                bar_fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(bar_fig, use_container_width=True)

            st.caption(
                "Both models perform close to the ~50% random baseline for next-day "
                "direction prediction — a well-documented characteristic of this problem "
                "class, not a pipeline defect."
            )

            # ── Run detail drill-down (mirrors MLflow's single-Run page) ──────
            st.divider()
            st.subheader("🔍 Run Details")
            run_options = {f"{r.run_name} — {r.run_id[:8]}": r.run_id for r in runs_df.itertuples()}
            selected_run_label = st.selectbox("Select a run to inspect", list(run_options.keys()))
            if selected_run_label:
                run_detail = da.get_run_full_details(run_options[selected_run_label])
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**Parameters**")
                    st.json(run_detail["params"])
                with rc2:
                    st.markdown("**Metrics**")
                    st.json(run_detail["metrics"])
                if run_detail["tags"]:
                    st.markdown("**Tags**")
                    st.json(run_detail["tags"])
                st.markdown("**Artifacts**")
                if run_detail["artifacts"]:
                    st.dataframe(pd.DataFrame(run_detail["artifacts"]), use_container_width=True, hide_index=True)
                else:
                    st.caption("No artifacts logged (or artifact store unreachable from this run).")

            # ── Model Registry (mirrors MLflow's Model Registry page) ─────────
            st.divider()
            st.subheader("📦 Registered Models")
            registry_df = da.get_registered_models_overview()
            if registry_df.empty:
                st.info("No registered models found yet.")
            else:
                st.dataframe(registry_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not reach MLflow tracking server: {e}")
        st.info(f"Check that MLflow is running at `{da.config.MLFLOW_TRACKING_URI}`.")

# ═══════════════════════════════════════════════════════════════
# TAB 4: EDA Insights
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    corr_fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature Correlation Matrix")
    corr_fig.update_layout(height=600)
    st.plotly_chart(corr_fig, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    if "Random Forest" in models:
        try:
            importances = pd.Series(
                models["Random Forest"].feature_importances_, index=da.FEATURE_COLS
            ).sort_values(ascending=False)
            imp_fig = px.bar(importances.head(15), orientation="h", title="Top 15 Feature Importances")
            imp_fig.update_layout(yaxis_title="Feature", xaxis_title="Importance", showlegend=False)
            imp_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(imp_fig, use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute feature importance: {e}")
    else:
        st.info("Random Forest model not loaded — run `train_ml.py` first.")

# ═══════════════════════════════════════════════════════════════
# TAB 5: Pipeline Health
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Airflow DataOps Pipeline Status")
    try:
        status = da.get_airflow_dag_status()
        col1, col2 = st.columns(2)
        col1.metric("DAG Paused?", "Yes ⏸️" if status["is_paused"] else "No ▶️ (Active)")
        col2.metric("Schedule", str(status["schedule_interval"]))

        st.write("**Recent Runs:**")
        runs = status["recent_runs"]
        if runs:
            runs_table = pd.DataFrame([{
                "Run ID": r.get("dag_run_id"),
                "State": r.get("state"),
                "Execution Date": r.get("execution_date"),
            } for r in runs])

            def highlight_state(val):
                if val == "success":
                    return "background-color: #d4edda"
                elif val == "failed":
                    return "background-color: #f8d7da"
                elif val == "running":
                    return "background-color: #fff3cd"
                return ""

            st.dataframe(
                runs_table.style.applymap(highlight_state, subset=["State"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No DAG runs yet.")
    except Exception as e:
        st.error(f"Could not reach Airflow API: {e}")
        st.info(f"Check that Airflow webserver is running at `{da.config.AIRFLOW_BASE_URL}`.")


# ═══════════════════════════════════════════════════════════════
# TAB 6: Try a New Stock (any ticker, not just the trained set)
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Get a Recommendation for Any Stock")
    st.caption(
        "Enter any Yahoo Finance ticker symbol — not limited to the Nifty 50 stocks the "
        "models were trained on. We fetch its live history, apply the *same* "
        "feature scaling learned during training (not a fresh refit), and run it "
        "through both trained models."
    )

    new_ticker = st.text_input(
        "Ticker symbol (e.g. SBIN.NS, AXISBANK.NS, AAPL, TSLA)",
        placeholder="SBIN.NS",
    ).strip().upper()

    lookup_clicked = st.button("🔎 Get Recommendation", type="primary")

    if lookup_clicked and new_ticker:
        with st.spinner(f"Fetching live data for {new_ticker} and running models..."):
            try:
                import predict_new_ticker as pnt
                result = pnt.predict_ticker(new_ticker, models)

                st.success(f"**{result['ticker']}** — as of {result['as_of_date']} ({result['history_days']} days of history fetched)")
                cols = st.columns(len(result["predictions"]))
                for col, (model_name, pred) in zip(cols, result["predictions"].items()):
                    with col:
                        if pred["signal"] == "BUY":
                            st.metric(model_name, "🟢 BUY", f"{pred['confidence']:.1%} confidence")
                        elif pred["signal"] == "SELL":
                            st.metric(model_name, "🔴 SELL", f"{pred['confidence']:.1%} confidence")
                        else:
                            st.error(f"{model_name}: {pred.get('error', 'prediction failed')}")

                st.divider()
                st.subheader("🧠 Why?")
                reason_cols = st.columns(len(result["predictions"]))
                for col, (model_name, pred) in zip(reason_cols, result["predictions"].items()):
                    with col:
                        if "narrative" in pred:
                            st.markdown(f"**{model_name}**")
                            st.markdown(pred["narrative"])

                st.caption(
                    "⚠️ Educational project output only — not investment advice. "
                    "This ticker was NOT part of model training; treat this as a "
                    "generalization demo, not a validated signal."
                )
            except pnt.InsufficientHistoryError as e:
                st.warning(f"⚠️ {e}")
            except pnt.ScalersNotFoundError as e:
                st.error(f"⚠️ {e}")
            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.exception(e)
    elif lookup_clicked:
        st.warning("Enter a ticker symbol first.")

st.divider()
st.caption("Data refreshes every 30 min, models every hour — matching the DAG's once-daily ingest→retrain cycle. Use \"🚀 Train Now\" in Model Performance to force an immediate refresh.")
