from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.errors import EmptyDataError


st.set_page_config(page_title="Polymarket Edge Research V2", layout="wide")
st.title("Polymarket Edge Research Dashboard (V2)")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()

base = Path("data/research_v2")
strategy_path = base / "strategy_report.csv"
score_path = base / "edge_scores_v2.csv"
portfolio_path = base / "portfolio_summary.csv"
run_summary_path = base / "run_summary.json"
orderbook_missingness_path = base / "orderbook_missingness.csv"
orderbook_distribution_path = base / "orderbook_distribution.csv"
panel_sanity_path = base / "panel_sanity.csv"
readiness_report_path = base / "readiness_report.csv"
label_diag_path = base / "label_diagnostics.csv"
insufficient_report_path = base / "insufficient_data_report.csv"
integrity_pre_path = base / "integrity_report_pre.csv"
integrity_post_path = base / "integrity_report_post.csv"
data_quality_summary_path = base / "data_quality_summary.csv"
data_quality_missingness_path = base / "data_quality_missingness.csv"
data_quality_pinned_path = base / "data_quality_pinned_markets.csv"
arb_data_quality_path = base / "arb_data_quality.csv"
live_opps_path = base / "live_opportunities.csv"
portfolio_risk_report_path = base / "portfolio_risk_report.csv"
strategy_correlation_path = base / "strategy_correlation.csv"
pbo_debug_path = base / "pbo_debug.csv"

strategy = _read_csv_safe(strategy_path)
scores = _read_csv_safe(score_path)
portfolio = _read_csv_safe(portfolio_path)
orderbook_missingness = _read_csv_safe(orderbook_missingness_path)
orderbook_distribution = _read_csv_safe(orderbook_distribution_path)
panel_sanity = _read_csv_safe(panel_sanity_path)
readiness_report = _read_csv_safe(readiness_report_path)
label_diag = _read_csv_safe(label_diag_path)
insufficient_report = _read_csv_safe(insufficient_report_path)
integrity_pre = _read_csv_safe(integrity_pre_path)
integrity_post = _read_csv_safe(integrity_post_path)
data_quality_summary = _read_csv_safe(data_quality_summary_path)
data_quality_missingness = _read_csv_safe(data_quality_missingness_path)
data_quality_pinned = _read_csv_safe(data_quality_pinned_path)
arb_data_quality = _read_csv_safe(arb_data_quality_path)
live_opps = _read_csv_safe(live_opps_path)
portfolio_risk_report = _read_csv_safe(portfolio_risk_report_path)
strategy_corr_default = _read_csv_safe(strategy_correlation_path)
pbo_debug = _read_csv_safe(pbo_debug_path)
run_summary = {}
if run_summary_path.exists():
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
deployment_summary_path = base / "deployment_readiness_summary.csv"
deployment_summary = _read_csv_safe(deployment_summary_path)

if strategy.empty:
    st.warning("No strategy report found. Research may have exited early (e.g., INSUFFICIENT_DATA).")
    if run_summary.get("message"):
        st.error(str(run_summary.get("message")))
    if not insufficient_report.empty:
        st.subheader("Insufficient Data Report")
        st.dataframe(insufficient_report, use_container_width=True)
    if not integrity_pre.empty:
        st.subheader("Integrity Checks (Pre)")
        st.dataframe(integrity_pre, use_container_width=True)
    if not panel_sanity.empty:
        st.subheader("Panel Sanity")
        st.dataframe(panel_sanity, use_container_width=True)
    if not readiness_report.empty:
        st.subheader("Readiness Report")
        st.dataframe(readiness_report, use_container_width=True)
    st.stop()
methods = portfolio["method"].tolist() if (not portfolio.empty and "method" in portfolio.columns) else []
selected_method = st.selectbox("Portfolio Method", methods or ["risk_parity"])
weights_path = base / f"portfolio_weights_{selected_method}.csv"
corr_path = base / f"portfolio_correlation_{selected_method}.csv"
ts_path = base / f"portfolio_timeseries_{selected_method}.csv"
weights = _read_csv_safe(weights_path)
corr = _read_csv_safe(corr_path)
if corr.empty and not strategy_corr_default.empty:
    corr = strategy_corr_default.copy()
portfolio_ts = _read_csv_safe(ts_path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Strategies", int(strategy["strategy"].nunique()))
col2.metric("Total Test Trades", int(strategy["test_trade_count"].sum()))
if "deployable" in strategy.columns:
    col3.metric("Deployable Strategies", int(strategy["deployable"].sum()))
else:
    col3.metric("Median Capacity (USD)", f"{float(strategy['capacity_usd'].median()):,.0f}")
n_unique_ts = int(run_summary.get("n_unique_ts", 0)) if run_summary else 0
col4.metric("Panel Unique Timestamps", n_unique_ts)

if not orderbook_missingness.empty:
    ob_cols = st.columns(3)
    ob_cols[0].metric("Orderbook Markets", int(len(orderbook_missingness)))
    if "two_sided_rate" in orderbook_missingness.columns:
        ob_cols[1].metric("Avg Two-Sided Rate", f"{float(orderbook_missingness['two_sided_rate'].mean()):.2%}")
    if "bid_missing_rate" in orderbook_missingness.columns:
        ob_cols[2].metric("Avg Bid Missing", f"{float(orderbook_missingness['bid_missing_rate'].mean()):.2%}")
if not panel_sanity.empty:
    st.subheader("Panel Sanity")
    st.dataframe(panel_sanity, use_container_width=True)
if not data_quality_summary.empty:
    st.subheader("Data Quality Summary")
    st.dataframe(data_quality_summary, use_container_width=True)
if not arb_data_quality.empty:
    st.subheader("Arbitrage Data Quality (YES/NO Both-Leg Coverage)")
    agg = arb_data_quality[arb_data_quality.get("level", "").astype(str).eq("aggregate")] if "level" in arb_data_quality.columns else arb_data_quality
    if not agg.empty:
        st.dataframe(agg, use_container_width=True)
    bad_mkts = arb_data_quality[(arb_data_quality.get("level", "").astype(str) == "market") & (pd.to_numeric(arb_data_quality.get("passed"), errors="coerce").fillna(0.0) < 0.5)] if {"level", "passed"}.issubset(arb_data_quality.columns) else pd.DataFrame()
    if not bad_mkts.empty:
        st.caption("Low-coverage markets (top 50)")
        st.dataframe(bad_mkts.head(50), use_container_width=True)
if not integrity_pre.empty or not integrity_post.empty:
    st.subheader("Research Integrity")
    if not integrity_pre.empty:
        st.caption("Pre-run integrity checks")
        st.dataframe(integrity_pre, use_container_width=True)
    if not integrity_post.empty:
        st.caption("Post-run integrity checks")
        st.dataframe(integrity_post, use_container_width=True)
if not insufficient_report.empty:
    st.error("Research integrity failed. See insufficient data report below.")
    st.dataframe(insufficient_report, use_container_width=True)
if not readiness_report.empty:
    st.subheader("Research Readiness")
    st.dataframe(readiness_report, use_container_width=True)
if run_summary.get("insufficient_trade_depth"):
    st.warning(
        f"INSUFFICIENT_DATA: max strategy test trades below target "
        f"{int(run_summary.get('min_trades_test', 200))}."
    )

st.subheader("Strategy Report")
st.dataframe(strategy.sort_values("test_expectancy", ascending=False), use_container_width=True)
if not label_diag.empty:
    st.subheader("Label Diagnostics")
    st.dataframe(label_diag, use_container_width=True)
if not live_opps.empty:
    st.subheader("Top 20 Live Opportunities")
    live_cols = [c for c in ["strategy", "ts", "market_id", "expected_net_ev", "direction", "traded_side", "required_notional", "estimated_fill_probability", "rationale"] if c in live_opps.columns]
    show_live = live_opps[live_cols].copy() if live_cols else live_opps.copy()
    if "expected_net_ev" in show_live.columns:
        show_live = show_live.sort_values("expected_net_ev", ascending=False)
    st.dataframe(show_live.head(20), use_container_width=True)

if not deployment_summary.empty:
    dep_cols = [
        c
        for c in [
            "strategy",
            "pass_fail",
            "test_mean_return",
            "ci_low",
            "ci_high",
            "walkforward_valid_folds",
            "n_test_trades",
            "walkforward_mean_expectancy",
            "ev_monotonic_pass",
            "ev_spearman",
            "ev_monotonic_spearman",
            "pessimistic_avg_expected_ev",
            "notes",
        ]
        if c in deployment_summary.columns
    ]
    if dep_cols:
        st.subheader("Deployment Gate")
        st.dataframe(deployment_summary[dep_cols], use_container_width=True)
if not pbo_debug.empty:
    st.subheader("PBO Debug")
    st.dataframe(pbo_debug.head(200), use_container_width=True)
    if {"strategy", "lambda", "status"}.issubset(pbo_debug.columns):
        ok = pbo_debug[pbo_debug["status"].astype(str).eq("ok")].copy()
        if not ok.empty:
            fig_pbo = px.box(ok, x="strategy", y="lambda", title="PBO Lambda Distribution by Strategy")
            st.plotly_chart(fig_pbo, use_container_width=True)

if {"strategy", "model", "walkforward_mean_expectancy"}.issubset(strategy.columns):
    fig_wf = px.bar(
        strategy.sort_values("walkforward_mean_expectancy", ascending=False),
        x="strategy",
        y="walkforward_mean_expectancy",
        color="model",
        title="Walk-Forward Mean Expectancy by Strategy",
    )
    st.plotly_chart(fig_wf, use_container_width=True)

selected_strategy = st.selectbox("Strategy Diagnostics", strategy["strategy"].tolist())
strat_dir = base / selected_strategy
calib_path = strat_dir / "calibration_curve.csv"
decile_path = strat_dir / "signal_deciles.csv"
attrib_path = strat_dir / "edge_attribution.csv"
deploy_path = strat_dir / "deployment_readiness.csv"
ev_diag_path = strat_dir / "ev_diagnostics.csv"
cost_path = strat_dir / "cost_decomposition.csv"
test_trades_path = strat_dir / "test_trades.csv"

calib = _read_csv_safe(calib_path)
deciles = _read_csv_safe(decile_path)
attrib = _read_csv_safe(attrib_path)
deploy = _read_csv_safe(deploy_path)
ev_diag = _read_csv_safe(ev_diag_path)
cost = _read_csv_safe(cost_path)
test_trades = _read_csv_safe(test_trades_path)

profit_cols = [c for c in ["strategy", "return_per_day", "expectancy_per_trade", "bootstrap_mean_ci_low", "bootstrap_mean_ci_high", "deployable"] if c in strategy.columns]
if profit_cols:
    st.subheader("Profitability Snapshot")
    st.dataframe(strategy[profit_cols].sort_values("return_per_day", ascending=False) if "return_per_day" in strategy.columns else strategy[profit_cols], use_container_width=True)

if not calib.empty and {"source", "avg_pred", "observed_win_rate", "bin"}.issubset(calib.columns):
    st.subheader(f"Calibration: {selected_strategy}")
    fig_cal = px.line(
        calib.sort_values(["source", "bin"]),
        x="avg_pred",
        y="observed_win_rate",
        color="source",
        markers=True,
        title="Predicted Probability vs Observed Win Rate",
    )
    fig_cal.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration")
    st.plotly_chart(fig_cal, use_container_width=True)

if not deciles.empty and {"signal_decile", "avg_return"}.issubset(deciles.columns):
    fig_dec = px.bar(
        deciles.sort_values("signal_decile"),
        x="signal_decile",
        y="avg_return",
        title="Return by Signal Strength Decile",
    )
    st.plotly_chart(fig_dec, use_container_width=True)

if not ev_diag.empty and {"ev_bin", "avg_expected_ev", "avg_realized_return"}.issubset(ev_diag.columns):
    st.subheader("EV Decile Calibration")
    ev_bins = ev_diag.copy()
    if not ev_bins.empty:
        fig_ev = px.line(
            ev_bins.sort_values("avg_expected_ev"),
            x="avg_expected_ev",
            y="avg_realized_return",
            markers=True,
            title="Realized Return vs Expected Net EV (Deciles)",
        )
        fig_ev.add_scatter(x=[ev_bins["avg_expected_ev"].min(), ev_bins["avg_expected_ev"].max()], y=[ev_bins["avg_expected_ev"].min(), ev_bins["avg_expected_ev"].max()], mode="lines", name="y=x")
        st.plotly_chart(fig_ev, use_container_width=True)
    st.dataframe(ev_diag.sort_values("ev_bin"), use_container_width=True)
    overall = ev_diag.iloc[[0]]
    if not overall.empty and "ev_monotonic_pass" in overall.columns:
        pass_flag = bool(float(overall["ev_monotonic_pass"].iloc[0]) >= 0.5)
        spearman = float(overall.get("ev_spearman", overall.get("ev_monotonic_spearman", pd.Series([0.0]))).iloc[0])
        st.metric("EV Monotonicity", "PASS" if pass_flag else "FAIL")
        st.caption(f"Spearman={spearman:.3f}")
        if "hit_rate_pos_ev" in overall.columns:
            st.caption(f"Hit rate (EV>0)={float(overall['hit_rate_pos_ev'].iloc[0]):.2%}")
        if "ev_model_valid" in overall.columns and float(overall["ev_model_valid"].iloc[0]) < 0.5:
            st.error("EV model flagged invalid (negative rank correlation with sufficient trades).")

if not attrib.empty:
    st.subheader("Edge Attribution")
    st.dataframe(attrib.sort_values("brier_improvement", ascending=False), use_container_width=True)

if not deploy.empty:
    st.subheader("Deployment Readiness")
    st.dataframe(deploy, use_container_width=True)

if not cost.empty:
    st.subheader("Cost Decomposition")
    st.dataframe(cost, use_container_width=True)
    cost_cols_present = [c for c in ["gross_pnl", "fees", "spread_cost", "vol_slippage", "impact_cost", "unwind_cost", "net_pnl"] if c in cost.columns]
    if set(["gross_pnl", "fees", "spread_cost", "vol_slippage", "impact_cost", "net_pnl"]).issubset(cost.columns):
        cost_long = cost[cost_cols_present].T.reset_index()
        cost_long.columns = ["component", "value"]
        fig_cost = px.bar(cost_long, x="component", y="value", title="PnL Cost Decomposition")
        st.plotly_chart(fig_cost, use_container_width=True)

if selected_strategy == "consistency_arb_v2" and not test_trades.empty:
    st.subheader("Consistency Arb Diagnostics")
    if "pair_direction" in test_trades.columns:
        dir_stats = (
            test_trades.groupby("pair_direction", observed=True)["trade_return"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "avg_return"})
        )
        st.dataframe(dir_stats, use_container_width=True)
    if {"edge_raw", "expected_net_ev"}.issubset(test_trades.columns):
        fig_edge = px.scatter(
            test_trades.sample(min(len(test_trades), 3000), random_state=7),
            x="edge_raw",
            y="expected_net_ev",
            color="pair_direction" if "pair_direction" in test_trades.columns else None,
            title="Raw Edge vs Expected Net EV",
        )
        st.plotly_chart(fig_edge, use_container_width=True)
    if {"ts", "net_pnl"}.issubset(test_trades.columns):
        pts = test_trades.copy()
        pts["ts"] = pd.to_datetime(pts["ts"], errors="coerce")
        pnl_ts = pts.groupby("ts", observed=True)["net_pnl"].sum().sort_index().cumsum().reset_index()
        pnl_ts["equity"] = 100000.0 + pnl_ts["net_pnl"]
        fig_pnl = px.line(pnl_ts, x="ts", y="equity", title="Consistency Arb Test Equity Curve")
        st.plotly_chart(fig_pnl, use_container_width=True)
    if {"spread_yes_bps", "trade_return"}.issubset(test_trades.columns):
        tt = test_trades.copy()
        for col in ["spread_yes_bps", "min_pair_depth"]:
            if col not in tt.columns:
                continue
            x = pd.to_numeric(tt[col], errors="coerce")
            r = pd.to_numeric(tt["trade_return"], errors="coerce")
            valid = x.notna() & r.notna()
            if int(valid.sum()) < 20:
                continue
            x = x.loc[valid]
            r = r.loc[valid]
            bins = pd.qcut(x.rank(method="first"), q=min(10, x.nunique()), labels=False, duplicates="drop") if x.nunique() > 1 else pd.Series(0, index=x.index)
            g = pd.DataFrame({"bin": bins, "trade_return": r}).groupby("bin", observed=True)["trade_return"].agg(["count", "mean"]).reset_index()
            g.columns = ["decile", "n", "avg_return"]
            st.caption(f"Performance by {col} decile")
            st.dataframe(g, use_container_width=True)

if not test_trades.empty and "trade_return" in test_trades.columns:
    st.subheader("Performance by Regime")
    tt = test_trades.copy()
    regime_rows = []
    for col, group_name in [("prob_vol_6", "volatility"), ("liquidity_log", "liquidity"), ("time_to_resolution_h", "time_to_resolution")]:
        if col not in tt.columns:
            continue
        x = pd.to_numeric(tt[col], errors="coerce")
        r = pd.to_numeric(tt["trade_return"], errors="coerce")
        valid = x.notna() & r.notna()
        if int(valid.sum()) < 10:
            continue
        x = x.loc[valid]
        r = r.loc[valid]
        q1 = float(x.quantile(0.33))
        q2 = float(x.quantile(0.66))
        bucket = pd.Series(np.where(x <= q1, "low", np.where(x <= q2, "mid", "high")), index=x.index)
        tmp = pd.DataFrame({"regime_group": group_name, "regime_bucket": bucket, "trade_return": r})
        grp = tmp.groupby(["regime_group", "regime_bucket"], observed=True)["trade_return"].agg(["count", "mean"]).reset_index()
        grp.columns = ["regime_group", "regime_bucket", "n", "avg_return"]
        regime_rows.append(grp)
    if regime_rows:
        regime_df = pd.concat(regime_rows, ignore_index=True)
        st.dataframe(regime_df, use_container_width=True)

if not orderbook_missingness.empty:
    st.subheader("Orderbook Sanity")
    show_cols = [c for c in ["market_id", "n_snapshots", "bid_missing_rate", "ask_missing_rate", "two_sided_rate", "mid_extreme_rate", "spread_zero_rate"] if c in orderbook_missingness.columns]
    st.dataframe(orderbook_missingness[show_cols].head(200), use_container_width=True)

if not orderbook_distribution.empty and {"distribution_type", "bucket", "pct"}.issubset(orderbook_distribution.columns):
    fig_ob = px.bar(orderbook_distribution, x="bucket", y="pct", color="distribution_type", barmode="group", title="Orderbook Mid/Spread/Quote Presence Distribution")
    st.plotly_chart(fig_ob, use_container_width=True)
if not data_quality_missingness.empty:
    st.subheader("Data Quality Missingness")
    st.dataframe(data_quality_missingness, use_container_width=True)
if not data_quality_pinned.empty:
    st.subheader("Pinned Markets / Exclusion Audit")
    st.dataframe(data_quality_pinned.head(200), use_container_width=True)

if not scores.empty:
    st.subheader("Edge Scores")
    st.dataframe(scores, use_container_width=True)
    fig_score = px.bar(scores, x="edge_name", y="edge_score", title="Edge Score Ranking")
    st.plotly_chart(fig_score, use_container_width=True)

if not portfolio.empty:
    st.subheader("Portfolio Summary")
    st.dataframe(portfolio.sort_values("portfolio_sharpe", ascending=False), use_container_width=True)
    chosen = portfolio[portfolio["method"] == selected_method]
    if not chosen.empty and "portfolio_total_pnl" in chosen.columns:
        pnl = float(chosen["portfolio_total_pnl"].iloc[0])
        st.metric(f"{selected_method} PnL vs 100k", f"{pnl:,.2f}")
        if "n_periods" in chosen.columns:
            st.caption(f"n_periods={int(chosen['n_periods'].iloc[0])}")
        if "warning" in chosen.columns:
            warn_val = chosen["warning"].iloc[0]
            if pd.notna(warn_val) and str(warn_val).strip() and str(warn_val).strip().lower() != "nan":
                st.warning(str(warn_val))
if not portfolio_risk_report.empty:
    st.subheader("Portfolio Risk Report")
    st.dataframe(portfolio_risk_report, use_container_width=True)

if not portfolio_ts.empty and {"ts", "portfolio_equity"}.issubset(portfolio_ts.columns):
    fig_eq = px.line(portfolio_ts.sort_values("ts"), x="ts", y="portfolio_equity", title=f"{selected_method} Portfolio Equity")
    st.plotly_chart(fig_eq, use_container_width=True)

if not weights.empty:
    fig_w = px.pie(weights, names="strategy", values="weight", title=f"{selected_method} Strategy Weights")
    st.plotly_chart(fig_w, use_container_width=True)

if not corr.empty:
    cmat = corr.set_index("strategy")
    fig_corr = px.imshow(cmat, text_auto=".2f", title="Strategy Return Correlation")
    st.plotly_chart(fig_corr, use_container_width=True)
