from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.config import load_runtime_config
from polymarket_edge.db import get_connection, init_db
from polymarket_edge.edges.consistency_arb import run as run_consistency_arb
from polymarket_edge.edges.microstructure_mm import run as run_microstructure_mm
from polymarket_edge.edges.resolution_mechanics import run as run_resolution_mechanics
from polymarket_edge.research.arb_data_quality import ArbDataQualityThresholds, write_arb_data_quality_report
from polymarket_edge.research.data import build_orderbook_sanity_reports, build_yes_feature_panel
from polymarket_edge.research.data_quality import audit_feature_panel
from polymarket_edge.research.integrity import IntegrityThresholds, run_research_integrity
from polymarket_edge.research.linking import build_market_links_summary, refresh_market_links, refresh_market_sets, refresh_outcome_links
from polymarket_edge.research.portfolio import build_portfolio
from polymarket_edge.research.advanced_validation import build_pbo_debug
from polymarket_edge.scoring import score_edges
from polymarket_edge.strategies import (
    ConsistencyArbStrategy,
    CrossMarketStrategy,
    LiquidityPremiumStrategy,
    MomentumReversionStrategy,
    ResolutionRuleStrategy,
    WhaleBehaviorStrategy,
)
from polymarket_edge.strategies.consistency_arb import ConsistencyArbStrategyConfig
from polymarket_edge.strategies.cross_market import CrossMarketStrategyConfig
from polymarket_edge.strategies.liquidity_premium import LiquidityPremiumStrategyConfig
from polymarket_edge.strategies.momentum import MomentumStrategyConfig
from polymarket_edge.strategies.resolution_rules import ResolutionRuleStrategyConfig
from polymarket_edge.strategies.whale import WhaleStrategyConfig

MIN_TRADES_TEST = 300
MIN_SNAPSHOTS = 500


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_insufficient_data_report(out_dir: Path, report: pd.DataFrame, *, message: str) -> None:
    _ensure_dir(out_dir)
    out = report.copy() if isinstance(report, pd.DataFrame) else pd.DataFrame()
    if out.empty:
        out = pd.DataFrame(
            [
                {
                    "check_group": "integrity",
                    "metric": "unknown",
                    "value": 0.0,
                    "required_min": 0.0,
                    "passed": 0,
                    "missing_qty": 0.0,
                    "reason": message,
                }
            ]
        )
    out.to_csv(out_dir / "insufficient_data_report.csv", index=False)


def _best_row(summary: pd.DataFrame) -> pd.Series | None:
    if summary.empty:
        return None
    score = summary["expectancy"].fillna(0.0) * np.sqrt(summary["trade_count"].clip(lower=0.0))
    return summary.loc[int(score.idxmax())]


def _assert_required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def _save_strategy_outputs(base_dir: Path, name: str, result) -> None:
    d = base_dir / name
    _ensure_dir(d)
    result.signal_frame.to_csv(d / "signals.csv", index=False)
    result.signal_frame.to_csv(d / "opportunities.csv", index=False)
    if str(name).startswith("consistency_arb"):
        alias = base_dir / "consistency_arb"
        _ensure_dir(alias)
        result.signal_frame.to_csv(alias / "opportunities.csv", index=False)
    result.train_summary.to_csv(d / "train_summary.csv", index=False)
    result.test_summary.to_csv(d / "test_summary.csv", index=False)
    result.train_trades.to_csv(d / "train_trades.csv", index=False)
    result.test_trades.to_csv(d / "test_trades.csv", index=False)
    if str(name).startswith("consistency_arb"):
        result.test_trades.to_csv(d / "paired_trades.csv", index=False)
    result.validation_summary.to_csv(d / "validation.csv", index=False)
    result.explain_summary.to_csv(d / "feature_importance.csv", index=False)
    result.strong_conditions.to_csv(d / "strong_conditions.csv", index=False)
    result.weak_conditions.to_csv(d / "weak_conditions.csv", index=False)
    result.capacity_summary.to_csv(d / "capacity.csv", index=False)
    result.model_comparison.to_csv(d / "model_comparison.csv", index=False)
    result.model_feature_importance.to_csv(d / "model_feature_importance.csv", index=False)
    result.label_diagnostics.to_csv(d / "label_diagnostics.csv", index=False)
    result.walkforward_folds.to_csv(d / "walkforward_folds.csv", index=False)
    result.walkforward_summary.to_csv(d / "walkforward_summary.csv", index=False)
    result.stability_time.to_csv(d / "stability_time.csv", index=False)
    result.stability_category.to_csv(d / "stability_category.csv", index=False)
    result.parameter_sensitivity.to_csv(d / "parameter_sensitivity.csv", index=False)
    result.filter_impact.to_csv(d / "filter_impact.csv", index=False)
    result.ev_threshold_grid.to_csv(d / "filter_impact_ev_threshold.csv", index=False)
    result.edge_attribution.to_csv(d / "edge_attribution.csv", index=False)
    result.calibration_curve.to_csv(d / "calibration_curve.csv", index=False)
    result.signal_deciles.to_csv(d / "signal_deciles.csv", index=False)
    result.feature_drift.to_csv(d / "feature_drift.csv", index=False)
    result.model_vs_market.to_csv(d / "model_vs_market.csv", index=False)
    result.ev_diagnostics.to_csv(d / "ev_diagnostics.csv", index=False)
    result.ev_diagnostics.to_csv(d / "realized_vs_expected_ev_by_decile.csv", index=False)
    result.advanced_validation.to_csv(d / "advanced_validation.csv", index=False)
    result.deployment_readiness.to_csv(d / "deployment_readiness.csv", index=False)
    result.execution_sensitivity.to_csv(d / "execution_sensitivity.csv", index=False)
    result.execution_calibration.to_csv(d / "execution_calibration.csv", index=False)
    result.cost_decomposition.to_csv(d / "cost_decomposition.csv", index=False)
    (d / "tuned_params.json").write_text(json.dumps(result.tuned_params, indent=2), encoding="utf-8")


def _strategy_report_row(name: str, result) -> dict[str, Any]:
    hp = int(result.tuned_params.get("holding_period", 1))
    threshold = float(result.tuned_params.get("threshold", 0.2))
    chosen_trades = (
        result.test_trades[
            (result.test_trades["holding_period"] == hp)
            & (result.test_trades["threshold"] == threshold)
        ].copy()
        if not result.test_trades.empty
        else pd.DataFrame()
    )
    chosen_summary = (
        result.test_summary[
            (result.test_summary["holding_period"] == float(hp))
            & (result.test_summary["threshold"] == float(threshold))
        ].copy()
        if not result.test_summary.empty
        else pd.DataFrame()
    )
    test_best = chosen_summary.iloc[0] if not chosen_summary.empty else _best_row(result.test_summary)
    if not chosen_trades.empty:
        trade_count = float(len(chosen_trades))
        expectancy = float(chosen_trades["trade_return"].mean())
    else:
        trade_count = float(test_best.get("trade_count", 0.0)) if test_best is not None else 0.0
        expectancy = float(test_best.get("expectancy", 0.0)) if test_best is not None else 0.0
    _assert_required_columns(
        result.validation_summary,
        ["test_mean_return", "mean_return_ci_low", "mean_return_ci_high", "ttest_pvalue"],
        f"{name}.validation_summary",
    )
    _assert_required_columns(result.walkforward_summary, ["mean_expectancy", "folds", "valid_folds"], f"{name}.walkforward_summary")
    _assert_required_columns(result.advanced_validation, ["white_pvalue", "pbo"], f"{name}.advanced_validation")
    _assert_required_columns(
        result.deployment_readiness,
        ["criterion", "test_mean_return", "ci_low", "ci_high", "n_test_trades", "n_walkforward_trades", "walkforward_mean_expectancy"],
        f"{name}.deployment_readiness",
    )
    valid = result.validation_summary.iloc[0].to_dict()
    cap = result.capacity_summary.iloc[0].to_dict() if not result.capacity_summary.empty else {}
    wf = result.walkforward_summary.iloc[0].to_dict()
    adv = result.advanced_validation.iloc[0].to_dict()
    attr = {}
    if not result.edge_attribution.empty and "segment" in result.edge_attribution.columns:
        overall = result.edge_attribution[result.edge_attribution["segment"] == "overall"]
        if not overall.empty:
            attr = overall.iloc[0].to_dict()
    ev = {}
    if not result.ev_diagnostics.empty:
        ev = result.ev_diagnostics.iloc[0].to_dict()
    label = result.label_diagnostics.iloc[-1].to_dict() if not result.label_diagnostics.empty else {}
    selected_model = "heuristic"
    model_reason = ""
    if not result.model_comparison.empty and "selected" in result.model_comparison.columns:
        sel = result.model_comparison[result.model_comparison["selected"] == True]  # noqa: E712
        if not sel.empty:
            selected_model = ",".join(sel["model"].astype(str).tolist())
    if selected_model == "heuristic" and not result.model_comparison.empty and "reason" in result.model_comparison.columns:
        reasons = [r for r in result.model_comparison["reason"].astype(str).tolist() if r]
        if reasons:
            model_reason = reasons[0]
    deployable = False
    deployment_summary = {}
    dec = result.deployment_readiness[result.deployment_readiness["criterion"] == "deployment_decision"]
    if not dec.empty and "passed" in dec.columns:
        deployable = bool(dec["passed"].iloc[0])
    summary_row = result.deployment_readiness[result.deployment_readiness["criterion"] == "summary"]
    if summary_row.empty:
        raise RuntimeError(f"{name}.deployment_readiness has no 'summary' row")
    deployment_summary = summary_row.iloc[0].to_dict()
    required_dep = [
        "test_mean_return",
        "ci_low",
        "ci_high",
        "n_test_trades",
        "n_walkforward_trades",
        "walkforward_valid_folds",
        "walkforward_mean_expectancy",
        "stability_time_positive_ratio",
        "stability_score",
        "pessimistic_avg_expected_ev",
        "advanced_white_pvalue",
        "advanced_pbo",
        "ev_monotonic_pass",
        "ev_monotonic_spearman",
        "ev_spearman",
        "ev_model_valid",
    ]
    for col in required_dep:
        if col not in deployment_summary:
            raise RuntimeError(f"{name}.deployment_summary missing required field: {col}")
    pess_ev = 0.0
    if not result.execution_sensitivity.empty:
        pess = result.execution_sensitivity[
            result.execution_sensitivity["execution_regime"].astype(str).str.lower() == "pessimistic"
        ]
        if not pess.empty and "avg_expected_ev" in pess.columns:
            pess_ev = float(pess["avg_expected_ev"].iloc[0])
    cost = result.cost_decomposition.iloc[0].to_dict() if not result.cost_decomposition.empty else {}
    expectancy_per_trade = (
        float(test_best.get("expectancy_per_trade", expectancy))
        if test_best is not None
        else expectancy
    )
    return_per_day = float(test_best.get("return_per_day", 0.0)) if test_best is not None else 0.0
    return_per_event = float(test_best.get("return_per_event", 0.0)) if test_best is not None else 0.0
    hedged_fill_rate = float(test_best.get("hedged_fill_rate", 0.0)) if test_best is not None else 0.0
    gross_abs = abs(float(cost.get("gross_pnl", 0.0)))
    total_cost_abs = sum(
        abs(float(cost.get(k, 0.0)))
        for k in ["fees", "spread_cost", "vol_slippage", "impact_cost"]
    )
    cost_dominance_ratio = float(total_cost_abs / max(gross_abs, 1e-9)) if gross_abs > 0 else (1.0 if total_cost_abs > 0 else 0.0)
    return {
        "strategy": name,
        "model": selected_model,
        "model_reason": model_reason,
        "model_target_type": str(chosen_trades.get("model_target_type", pd.Series(dtype=str)).astype(str).mode().iloc[0]) if (not chosen_trades.empty and "model_target_type" in chosen_trades.columns and not chosen_trades["model_target_type"].dropna().empty) else "",
        "model_horizon": float(chosen_trades.get("model_horizon", pd.Series(dtype=float)).dropna().iloc[0]) if (not chosen_trades.empty and "model_horizon" in chosen_trades.columns and not chosen_trades["model_horizon"].dropna().empty) else float(result.tuned_params.get("holding_period", 1)),
        "label_mode": str(label.get("label_mode", "")),
        "label_source": str(label.get("label_source", "")),
        "label_warning": str(label.get("label_warning", "")),
        "label_pos_rate": float(label.get("pos_rate", 0.0)),
        "label_n_samples": float(label.get("n_samples", 0.0)),
        "label_y_unique": float(label.get("y_unique", 0.0)),
        "threshold": threshold,
        "holding_period": hp,
        "test_trade_count": trade_count,
        "test_expectancy": expectancy,
        "expectancy_per_trade": expectancy_per_trade,
        "return_per_day": return_per_day,
        "return_per_event": return_per_event,
        "test_sharpe": float(test_best.get("sharpe", 0.0)) if test_best is not None else 0.0,
        "test_drawdown": float(test_best.get("max_drawdown", 0.0)) if test_best is not None else 0.0,
        "hedged_fill_rate": hedged_fill_rate,
        "cost_dominance_ratio": cost_dominance_ratio,
        "ttest_stat": float(valid.get("ttest_stat", 0.0)),
        "ttest_pvalue": float(valid.get("ttest_pvalue", 1.0)),
        "ttest_pvalue_pos": float(valid.get("ttest_pvalue_pos", 1.0)),
        "stability_score": float(valid.get("stability_score", 0.0)),
        "overfit_gap": float(valid.get("overfit_gap", 0.0)),
        "capacity_usd": float(cap.get("estimated_daily_capacity_usd", 0.0)),
        "bootstrap_mean_ci_low": float(valid.get("mean_return_ci_low", 0.0)),
        "bootstrap_mean_ci_high": float(valid.get("mean_return_ci_high", 0.0)),
        "walkforward_mean_expectancy": float(wf.get("mean_expectancy", 0.0)),
        "walkforward_folds": float(wf.get("folds", 0.0)),
        "walkforward_valid_folds": float(wf.get("valid_folds", 0.0)),
        "perm_pvalue": float(adv.get("perm_pvalue", 1.0)),
        "validation_permutation_pvalue": float(valid.get("permutation_pvalue", 1.0)),
        "white_pvalue": float(adv.get("white_pvalue", 1.0)),
        "pbo": float(adv.get("pbo", 1.0)),
        "av_n_candidates": float(adv.get("n_candidates", 0.0)),
        "av_n_obs": float(adv.get("white_n_obs", 0.0)),
        "brier_improvement": float(attr.get("brier_improvement", 0.0)),
        "ev_calibration_beta": float(ev.get("ev_calibration_beta", 0.0)),
        "ev_realized_corr": float(ev.get("ev_realized_corr", 0.0)),
        "gross_pnl": float(cost.get("gross_pnl", 0.0)),
        "fees": float(cost.get("fees", 0.0)),
        "spread_cost": float(cost.get("spread_cost", 0.0)),
        "vol_slippage": float(cost.get("vol_slippage", 0.0)),
        "impact_cost": float(cost.get("impact_cost", 0.0)),
        "net_pnl": float(cost.get("net_pnl", 0.0)),
        "pessimistic_avg_expected_ev": pess_ev,
        "deployment_test_mean_return": float(deployment_summary.get("test_mean_return", 0.0)),
        "deployment_n_test_trades": float(deployment_summary.get("n_test_trades", trade_count)),
        "deployment_n_walkforward_trades": float(deployment_summary.get("n_walkforward_trades", 0.0)),
        "deployment_walkforward_valid_folds": float(deployment_summary.get("walkforward_valid_folds", 0.0)),
        "deployment_ev_monotonic_pass": float(deployment_summary.get("ev_monotonic_pass", 0.0)),
        "deployment_ev_monotonic_spearman": float(deployment_summary.get("ev_monotonic_spearman", 0.0)),
        "deployment_ev_spearman": float(deployment_summary.get("ev_spearman", deployment_summary.get("ev_monotonic_spearman", 0.0))),
        "deployment_ev_model_valid": float(deployment_summary.get("ev_model_valid", 0.0)),
        "deployable": int(deployable),
    }


def _returns_series_for_portfolio(result) -> pd.DataFrame:
    if result.test_trades.empty:
        return pd.DataFrame(columns=["ts", "ret"])
    hp = int(result.tuned_params.get("holding_period", 1))
    threshold = float(result.tuned_params.get("threshold", 0.2))
    trades = result.test_trades[
        (result.test_trades["holding_period"] == hp)
        & (result.test_trades["threshold"] == threshold)
    ].copy()
    if trades.empty:
        return pd.DataFrame(columns=["ts", "ret"])
    ret = trades.groupby("ts", observed=True)["net_pnl"].sum().reset_index()
    ret["ret"] = ret["net_pnl"] / 100_000.0
    return ret[["ts", "ret"]]


def _build_edge_score_input(report_df: pd.DataFrame) -> pd.DataFrame:
    if report_df.empty:
        return report_df
    frame = report_df.copy()
    frame["edge_name"] = frame["strategy"]
    frame["p_value"] = frame["ttest_pvalue"].clip(lower=1e-12, upper=1.0)
    frame["p_value_pos"] = frame["ttest_pvalue_pos"].clip(lower=1e-12, upper=1.0)
    frame["stability"] = frame["stability_score"].clip(lower=0.0, upper=1.0)
    frame["mean_return"] = frame["return_per_day"]
    frame["ci_low"] = frame["bootstrap_mean_ci_low"]
    frame["t_stat"] = frame["ttest_stat"]
    frame["n_obs"] = frame["deployment_n_test_trades"]
    frame["expectancy_per_trade"] = frame["expectancy_per_trade"]
    frame["return_per_day"] = frame["return_per_day"]
    frame["return_per_event"] = frame["return_per_event"]
    frame["test_trade_count"] = frame["test_trade_count"]
    frame["walkforward_valid_folds"] = frame["walkforward_valid_folds"]
    frame["label_degenerate"] = (
        frame["model_reason"].astype(str).str.contains("degenerate_train_labels", case=False, na=False)
        | frame["label_warning"].astype(str).str.contains("degenerate_train_labels", case=False, na=False)
    ).astype(float)
    cols = [
        "edge_name",
        "p_value",
        "p_value_pos",
        "t_stat",
        "n_obs",
        "ci_low",
        "capacity_usd",
        "stability",
        "mean_return",
        "expectancy_per_trade",
        "return_per_day",
        "return_per_event",
        "test_trade_count",
        "walkforward_valid_folds",
        "label_degenerate",
    ]
    for extra in ["hedged_fill_rate", "cost_dominance_ratio"]:
        if extra in frame.columns:
            cols.append(extra)
    return frame[cols]


def _panel_sanity_frame(panel: pd.DataFrame, orderbook_missingness: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            [
                {
                    "n_unique_ts": 0.0,
                    "ts_min": "",
                    "ts_max": "",
                    "markets_covered": 0.0,
                    "tokens_covered": 0.0,
                    "rows": 0.0,
                    "missing_best_bid_rate": 0.0,
                    "missing_best_ask_rate": 0.0,
                    "two_sided_rate": 0.0,
                }
            ]
        )
    p = panel.copy()
    p["ts"] = pd.to_datetime(p["ts"], errors="coerce")
    miss_bid = float(p["best_bid"].isna().mean()) if "best_bid" in p.columns else 0.0
    miss_ask = float(p["best_ask"].isna().mean()) if "best_ask" in p.columns else 0.0
    two_sided = 0.0
    if {"best_bid", "best_ask"}.issubset(p.columns):
        two_sided = float((p["best_bid"].notna() & p["best_ask"].notna()).mean())
    if not orderbook_missingness.empty and "two_sided_rate" in orderbook_missingness.columns:
        two_sided = float(orderbook_missingness["two_sided_rate"].mean())
        miss_bid = float(orderbook_missingness["bid_missing_rate"].mean()) if "bid_missing_rate" in orderbook_missingness.columns else miss_bid
        miss_ask = float(orderbook_missingness["ask_missing_rate"].mean()) if "ask_missing_rate" in orderbook_missingness.columns else miss_ask
    return pd.DataFrame(
        [
            {
                "n_unique_ts": float(p["ts"].dropna().nunique()),
                "ts_min": str(p["ts"].min()),
                "ts_max": str(p["ts"].max()),
                "markets_covered": float(p["market_id"].nunique()) if "market_id" in p.columns else 0.0,
                "tokens_covered": float(p["token_id"].nunique()) if "token_id" in p.columns else 0.0,
                "rows": float(len(p)),
                "missing_best_bid_rate": miss_bid,
                "missing_best_ask_rate": miss_ask,
                "two_sided_rate": two_sided,
            }
        ]
    )


def _build_readiness_report(report: pd.DataFrame, n_unique_ts: int) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame(columns=["strategy", "DATA_READY", "MODEL_READY", "ALPHA_READY", "DEPLOYABLE", "readiness_state", "reason"])
    rows: list[dict[str, object]] = []
    for row in report.to_dict(orient="records"):
        data_ready = bool((n_unique_ts >= MIN_SNAPSHOTS) and (float(row.get("test_trade_count", 0.0)) >= float(MIN_TRADES_TEST)))
        model_ready = bool(
            float(row.get("label_y_unique", 0.0)) >= 2.0
            and float(row.get("label_n_samples", 0.0)) >= float(MIN_TRADES_TEST)
            and not str(row.get("model_reason", "")).lower().startswith("degenerate_train_labels")
            and str(row.get("model", "")).strip().lower() != "heuristic"
        )
        alpha_ready = bool(
            float(row.get("bootstrap_mean_ci_low", 0.0)) > 0.0
            and float(row.get("walkforward_mean_expectancy", 0.0)) > 0.0
        )
        deployable = bool(
            alpha_ready
            and float(row.get("pessimistic_avg_expected_ev", -1.0)) >= 0.0
            and int(row.get("deployable", 0)) == 1
        )
        if deployable:
            state = "DEPLOYABLE"
        elif alpha_ready:
            state = "ALPHA_READY"
        elif model_ready:
            state = "MODEL_READY"
        elif data_ready:
            state = "DATA_READY"
        else:
            state = "INSUFFICIENT_DATA"
        reasons = []
        if n_unique_ts < MIN_SNAPSHOTS:
            reasons.append(f"n_unique_ts<{MIN_SNAPSHOTS}")
        if float(row.get("test_trade_count", 0.0)) < float(MIN_TRADES_TEST):
            reasons.append(f"test_trade_count<{MIN_TRADES_TEST}")
        if float(row.get("label_y_unique", 0.0)) < 2.0:
            reasons.append("degenerate_labels")
        if str(row.get("model", "")).strip().lower() == "heuristic":
            reasons.append("model_not_trained")
        if float(row.get("bootstrap_mean_ci_low", 0.0)) <= 0.0:
            reasons.append("ci_low<=0")
        if float(row.get("walkforward_mean_expectancy", 0.0)) <= 0.0:
            reasons.append("wf_mean<=0")
        rows.append(
            {
                "strategy": row.get("strategy", ""),
                "DATA_READY": int(data_ready),
                "MODEL_READY": int(model_ready),
                "ALPHA_READY": int(alpha_ready),
                "DEPLOYABLE": int(deployable),
                "readiness_state": state,
                "reason": ";".join(reasons) if reasons else "ok",
            }
        )
    return pd.DataFrame(rows)


def _edge_validation_report_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "",
                "model": "",
                "walkforward_mean_expectancy": 0.0,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.0,
                "ttest_pvalue": 1.0,
                "perm_pvalue": 1.0,
                "white_pvalue": 1.0,
                "pbo": 1.0,
                "brier_improvement": 0.0,
                "ev_calibration_beta": 0.0,
                "ev_realized_corr": 0.0,
                "capacity_usd": 0.0,
                "deployment_decision": 0,
                "notes": "",
            }
        ]
    )


def main() -> None:
    runtime_cfg = load_runtime_config()
    min_cfg = (((runtime_cfg.get("research") or {}).get("min_data")) or {})
    integrity_thresholds = IntegrityThresholds(
        min_ts=int(min_cfg.get("min_ts", MIN_SNAPSHOTS)),
        min_mkts=int(min_cfg.get("min_mkts", 200)),
        min_snapshots=int(min_cfg.get("min_snapshots", 50_000)),
        min_test_trades=int(min_cfg.get("min_test_trades", MIN_TRADES_TEST)),
        min_fold_trades=int(min_cfg.get("min_fold_trades", 50)),
    )
    conn = get_connection()
    init_db(conn)
    run_ts = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    out_dir = Path(((runtime_cfg.get("research") or {}).get("output_dir")) or "data/research_v2")
    _ensure_dir(out_dir)

    ob_reports = build_orderbook_sanity_reports(conn)
    ob_missing = ob_reports.get("orderbook_missingness", pd.DataFrame())
    ob_dist = ob_reports.get("orderbook_distribution", pd.DataFrame())
    ob_missing.to_csv(out_dir / "orderbook_missingness.csv", index=False)
    ob_dist.to_csv(out_dir / "orderbook_distribution.csv", index=False)

    panel = build_yes_feature_panel(conn)
    panel_sanity = _panel_sanity_frame(panel, ob_missing)
    panel_sanity.to_csv(out_dir / "panel_sanity.csv", index=False)
    dq = audit_feature_panel(panel)
    dq.get("summary", pd.DataFrame()).to_csv(out_dir / "data_quality_summary.csv", index=False)
    dq.get("field_missingness", pd.DataFrame()).to_csv(out_dir / "data_quality_missingness.csv", index=False)
    dq.get("pinned_markets", pd.DataFrame()).to_csv(out_dir / "data_quality_pinned_markets.csv", index=False)
    if panel.empty:
        msg = "INSUFFICIENT_DATA: no feature panel rows available. Ingest snapshots first."
        _write_insufficient_data_report(
            out_dir,
            pd.DataFrame(
                [
                    {
                        "check_group": "data_depth",
                        "metric": "panel_rows",
                        "value": 0.0,
                        "required_min": float(integrity_thresholds.min_snapshots),
                        "passed": 0,
                        "missing_qty": float(integrity_thresholds.min_snapshots),
                        "reason": "no_feature_panel_rows",
                    }
                ]
            ),
            message=msg,
        )
        print(msg)
        return
    panel["ts"] = pd.to_datetime(panel["ts"], errors="coerce")
    n_unique_ts = int(panel["ts"].dropna().nunique())
    ts_min = panel["ts"].min()
    ts_max = panel["ts"].max()
    print(f"Panel diagnostics: rows={len(panel)} n_unique_ts={n_unique_ts} ts_min={ts_min} ts_max={ts_max}")
    # Build reusable market links + structural edge scanners before the main strategy run.
    try:
        links = refresh_market_links(conn)
    except Exception:
        links = pd.DataFrame()
    try:
        outcome_links = refresh_outcome_links(conn)
    except Exception:
        outcome_links = pd.DataFrame()
    try:
        market_sets = refresh_market_sets(conn)
    except Exception:
        market_sets = pd.DataFrame()
    links.to_csv(out_dir / "market_links.csv", index=False)
    if not outcome_links.empty:
        outcome_links.to_csv(out_dir / "outcome_links.csv", index=False)
    if not market_sets.empty:
        market_sets.to_csv(out_dir / "market_sets.csv", index=False)
    try:
        build_market_links_summary(conn).to_csv(out_dir / "market_links_summary.csv", index=False)
    except Exception:
        pd.DataFrame().to_csv(out_dir / "market_links_summary.csv", index=False)
    arb_q = write_arb_data_quality_report(
        conn,
        out_dir / "arb_data_quality.csv",
        thresholds=ArbDataQualityThresholds(min_both_legs_coverage=0.80, recent_hours=72.0),
    )
    arb_coverage = float(arb_q.get("both_legs_coverage", 0.0)) if isinstance(arb_q, dict) else 0.0
    if not bool(arb_q.get("passed", False)):
        msg = (
            "INSUFFICIENT_DATA: complement YES/NO two-leg coverage is too low for consistency arbitrage "
            f"(both_legs_coverage={arb_coverage:.2%}, required>=80%). See arb_data_quality.csv."
        )
        _write_insufficient_data_report(
            out_dir,
            pd.DataFrame(
                [
                    {
                        "check_group": "arb_data_quality",
                        "metric": "both_legs_coverage",
                        "value": arb_coverage,
                        "required_min": 0.80,
                        "passed": 0,
                        "missing_qty": max(0.0, 0.80 - arb_coverage),
                        "reason": "both_legs_coverage_below_threshold",
                    }
                ]
            ),
            message=msg,
        )
        summary_payload = {
            "run_ts": str(run_ts),
            "panel_rows": int(len(panel)),
            "n_unique_ts": n_unique_ts,
            "ts_min": str(ts_min) if ts_min is not None else None,
            "ts_max": str(ts_max) if ts_max is not None else None,
            "arb_both_legs_coverage": arb_coverage,
            "insufficient_data": True,
            "message": msg,
            "outputs_dir": str(out_dir),
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        raise SystemExit(msg)
    consistency = run_consistency_arb(conn)
    micro_mm = run_microstructure_mm(conn, panel=panel)
    resolution_mech = run_resolution_mechanics(conn)
    for name, payload in [
        ("consistency_arb", consistency),
        ("microstructure_mm", micro_mm),
        ("resolution_mechanics", resolution_mech),
    ]:
        if isinstance(payload, dict):
            for key, frame in payload.items():
                if isinstance(frame, pd.DataFrame):
                    frame.to_csv(out_dir / f"{name}_{key}.csv", index=False)
    module_live_candidates: list[pd.DataFrame] = []
    cp0 = consistency.get("complement_parity", pd.DataFrame()) if isinstance(consistency, dict) else pd.DataFrame()
    if isinstance(cp0, pd.DataFrame) and not cp0.empty:
        x = cp0.copy()
        x["strategy"] = "consistency_arb"
        x["direction"] = "PAIR_HEDGE"
        x["traded_side"] = "BASKET"
        x["required_notional"] = 250.0
        x["estimated_fill_probability"] = 0.5
        x["rationale"] = "complement_parity_violation"
        x["expected_net_ev"] = pd.to_numeric(x.get("expected_net_return"), errors="coerce").fillna(0.0)
        module_live_candidates.append(x)
    mm0 = micro_mm.get("live_quotes", pd.DataFrame()) if isinstance(micro_mm, dict) else pd.DataFrame()
    if isinstance(mm0, pd.DataFrame) and not mm0.empty:
        x = mm0.copy()
        x["strategy"] = "microstructure_mm"
        x["direction"] = "MARKET_MAKE"
        x["traded_side"] = "BOTH"
        x["required_notional"] = pd.to_numeric(x.get("quoted_notional", 100.0), errors="coerce").fillna(100.0)
        x["estimated_fill_probability"] = pd.to_numeric(x.get("expected_fill_probability", 0.5), errors="coerce").fillna(0.5)
        x["rationale"] = x.get("strategy_rationale", "passive_spread_capture")
        x["expected_net_ev"] = pd.to_numeric(x.get("expected_net_return", 0.0), errors="coerce").fillna(0.0)
        module_live_candidates.append(x)
    rm0 = resolution_mech.get("human_review", pd.DataFrame()) if isinstance(resolution_mech, dict) else pd.DataFrame()
    if isinstance(rm0, pd.DataFrame) and not rm0.empty:
        x = rm0.copy()
        x["strategy"] = "resolution_mechanics"
        x["direction"] = "HUMAN_REVIEW"
        x["traded_side"] = "YES/NO"
        x["required_notional"] = 100.0
        x["estimated_fill_probability"] = 0.25
        x["expected_net_ev"] = pd.to_numeric(x.get("expected_net_return", 0.0), errors="coerce").fillna(0.0)
        module_live_candidates.append(x)
    if module_live_candidates:
        module_live = pd.concat(module_live_candidates, ignore_index=True, sort=False)
        if "ts" in module_live.columns:
            module_live["ts"] = pd.to_datetime(module_live["ts"], errors="coerce")
        dedupe_cols = [c for c in ["strategy", "market_id", "token_id"] if c in module_live.columns]
        module_live = module_live.sort_values(
            [c for c in ["expected_net_ev", "ts"] if c in module_live.columns],
            ascending=[False, False][: len([c for c in ["expected_net_ev", "ts"] if c in module_live.columns])],
        )
        if dedupe_cols:
            module_live = module_live.drop_duplicates(subset=dedupe_cols, keep="first")
        module_live.head(200).to_csv(out_dir / "live_opportunities.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "live_opportunities.csv", index=False)

    integrity_pre = run_research_integrity(
        panel=panel,
        thresholds=integrity_thresholds,
        output_path=out_dir / "integrity_report_pre.csv",
    )
    if not bool(integrity_pre.get("passed", False)):
        msg = (
            "INSUFFICIENT_DATA: research integrity checks failed before strategy evaluation. "
            "Collect more snapshots and market coverage (see insufficient_data_report.csv). "
            "Suggested: `python scripts/collect_snapshots.py --minutes 5 --hours 24 --with-trades`."
        )
        summary_payload = {
            "run_ts": str(run_ts),
            "panel_rows": int(len(panel)),
            "n_unique_ts": n_unique_ts,
            "ts_min": str(ts_min) if ts_min is not None else None,
            "ts_max": str(ts_max) if ts_max is not None else None,
            "insufficient_data": True,
            "message": msg,
            "outputs_dir": str(out_dir),
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        _write_insufficient_data_report(out_dir, integrity_pre.get("report", pd.DataFrame()), message=msg)
        for stale in [
            out_dir / "strategy_report.csv",
            out_dir / "edge_scores_v2.csv",
            out_dir / "deployment_readiness.csv",
            out_dir / "deployment_readiness_summary.csv",
            out_dir / "label_diagnostics.csv",
        ]:
            if stale.exists():
                stale.unlink()
        pd.DataFrame(
            [
                {
                    "strategy": "all",
                    "DATA_READY": 0,
                    "MODEL_READY": 0,
                    "ALPHA_READY": 0,
                    "DEPLOYABLE": 0,
                    "readiness_state": "INSUFFICIENT_DATA",
                    "reason": msg,
                }
            ]
        ).to_csv(out_dir / "readiness_report.csv", index=False)
        raise SystemExit(msg)

    strategies = [
        ConsistencyArbStrategy(ConsistencyArbStrategyConfig()),
        CrossMarketStrategy(CrossMarketStrategyConfig(model_type="auto")),
        ResolutionRuleStrategy(ResolutionRuleStrategyConfig(model_type="auto")),
        LiquidityPremiumStrategy(LiquidityPremiumStrategyConfig(model_type="auto")),
        MomentumReversionStrategy(MomentumStrategyConfig(model_type="auto")),
        WhaleBehaviorStrategy(WhaleStrategyConfig(model_type="ensemble")),
    ]

    report_rows: list[dict[str, Any]] = []
    returns_by_strategy: dict[str, pd.DataFrame] = {}
    trade_details_by_strategy: dict[str, pd.DataFrame] = {}
    param_rows: list[dict[str, Any]] = []
    deployment_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []
    walkforward_fold_rows: list[pd.DataFrame] = []
    live_opportunity_rows: list[pd.DataFrame] = []
    pbo_debug_rows: list[pd.DataFrame] = []

    for strat in strategies:
        result = strat.run(conn, panel, context={})
        _save_strategy_outputs(out_dir, strat.name, result)
        report_rows.append(_strategy_report_row(strat.name, result))
        returns_by_strategy[strat.name] = _returns_series_for_portfolio(result)
        chosen_hp = int(result.tuned_params.get("holding_period", 1))
        chosen_thr = float(result.tuned_params.get("threshold", 0.2))
        chosen_trades = (
            result.test_trades[
                (pd.to_numeric(result.test_trades.get("holding_period"), errors="coerce") == float(chosen_hp))
                & (pd.to_numeric(result.test_trades.get("threshold"), errors="coerce") == float(chosen_thr))
            ].copy()
            if not result.test_trades.empty
            else pd.DataFrame()
        )
        trade_details_by_strategy[strat.name] = chosen_trades
        cfg = asdict(strat.config)
        cfg["strategy"] = strat.name
        param_rows.append(cfg)
        if not result.deployment_readiness.empty:
            dep = result.deployment_readiness.copy()
            dep["strategy"] = strat.name
            deployment_rows.append(dep)
        if not result.label_diagnostics.empty:
            lab = result.label_diagnostics.copy()
            lab["strategy"] = strat.name
            label_rows.append(lab)
        if not result.walkforward_folds.empty:
            wf_folds = result.walkforward_folds.copy()
            wf_folds["strategy"] = strat.name
            walkforward_fold_rows.append(wf_folds)
        try:
            pbo_debug = build_pbo_debug(
                result.test_trades if isinstance(result.test_trades, pd.DataFrame) else pd.DataFrame(),
                strategy_name=strat.name,
                chosen_threshold=float(result.tuned_params.get("threshold", 0.0)),
                chosen_holding_period=int(result.tuned_params.get("holding_period", 1)),
            )
        except Exception as exc:
            pbo_debug = pd.DataFrame(
                [
                    {
                        "strategy": strat.name,
                        "fold_id": np.nan,
                        "n_configs": 0.0,
                        "chosen_config": "",
                        "chosen_final_config": "",
                        "chosen_is_rank": np.nan,
                        "chosen_oos_rank": np.nan,
                        "lambda": np.nan,
                        "status": "invalid",
                        "reason": f"pbo_debug_error:{type(exc).__name__}",
                    }
                ]
            )
        pbo_debug_rows.append(pbo_debug)
        if not result.signal_frame.empty:
            sig = result.signal_frame.copy()
            if "ts" in sig.columns:
                sig["ts"] = pd.to_datetime(sig["ts"], errors="coerce")
                latest_ts = sig["ts"].max()
                live = sig[sig["ts"] == latest_ts].copy()
            else:
                live = sig.copy()
            if not live.empty:
                if "expected_net_ev" not in live.columns:
                    try:
                        live, _ = strat._apply_signal_filters(live, split="live", ev_floor=float(result.tuned_params.get("min_expected_net_ev", 0.0)))  # type: ignore[attr-defined]
                    except Exception:
                        live = live.copy()
                if "expected_net_ev" in live.columns:
                    cols_keep = [
                        c for c in [
                            "ts", "market_id", "token_id", "category", "expected_net_ev", "direction",
                            "traded_side", "confidence", "signal", "depth_total", "spread_bps", "time_to_resolution_h"
                        ] if c in live.columns
                    ]
                    live = live[cols_keep].copy()
                    live["strategy"] = strat.name
                    live["required_notional"] = float(getattr(strat.config, "base_trade_notional", 250.0))
                    depth_series = (
                        live["depth_total"]
                        if "depth_total" in live.columns
                        else (live["min_pair_depth"] if "min_pair_depth" in live.columns else pd.Series(0.0, index=live.index))
                    )
                    live["estimated_fill_probability"] = np.clip(
                        pd.to_numeric(depth_series, errors="coerce").fillna(0.0) / 1000.0,
                        0.01,
                        1.0,
                    )
                    live["rationale"] = "model_signal_expected_net_ev"
                    live_opportunity_rows.append(live.sort_values("expected_net_ev", ascending=False).head(50))
        print(f"{strat.name}: tuned={result.tuned_params}")

    report = pd.DataFrame(report_rows)
    params = pd.DataFrame(param_rows)
    report.to_csv(out_dir / "strategy_report.csv", index=False)
    params.to_csv(out_dir / "parameter_ranges.csv", index=False)
    _edge_validation_report_template().to_csv(out_dir / "edge_validation_report_template.csv", index=False)
    label_diag = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    label_diag.to_csv(out_dir / "label_diagnostics.csv", index=False)
    walkforward_folds_all = pd.concat(walkforward_fold_rows, ignore_index=True) if walkforward_fold_rows else pd.DataFrame()
    if not walkforward_folds_all.empty:
        walkforward_folds_all.to_csv(out_dir / "walkforward_folds_all.csv", index=False)
    pbo_debug_all = pd.concat(pbo_debug_rows, ignore_index=True) if pbo_debug_rows else pd.DataFrame()
    pbo_debug_all.to_csv(out_dir / "pbo_debug.csv", index=False)
    deployment_df = pd.concat(deployment_rows, ignore_index=True) if deployment_rows else pd.DataFrame()
    deployment_df.to_csv(out_dir / "deployment_readiness.csv", index=False)
    if not deployment_df.empty and "criterion" in deployment_df.columns:
        dep_summary = deployment_df[deployment_df["criterion"] == "summary"].copy()
    else:
        dep_summary = pd.DataFrame()
    if not dep_summary.empty:
        if "passed" in dep_summary.columns:
            dep_summary["pass"] = dep_summary["passed"].astype(bool).astype(int)
            dep_summary["pass_fail"] = dep_summary["passed"].map({True: "PASS", False: "FAIL"})
        elif "decision" in dep_summary.columns:
            dep_summary["pass_fail"] = dep_summary["decision"].astype(str).str.upper()
            dep_summary["pass"] = dep_summary["pass_fail"].eq("PASS").astype(int)
        else:
            dep_summary["pass"] = 0
            dep_summary["pass_fail"] = "FAIL"
    dep_summary.to_csv(out_dir / "deployment_readiness_summary.csv", index=False)

    integrity_post = run_research_integrity(
        panel=panel,
        strategy_report=report,
        walkforward_folds=walkforward_folds_all,
        thresholds=integrity_thresholds,
        output_path=out_dir / "integrity_report_post.csv",
    )
    if not bool(integrity_post.get("passed", False)):
        msg = (
            "INSUFFICIENT_DATA: research integrity checks failed after strategy evaluation. "
            "Results are not statistically meaningful yet (see insufficient_data_report.csv)."
        )
        _write_insufficient_data_report(out_dir, integrity_post.get("report", pd.DataFrame()), message=msg)
        print(msg)
        summary_payload = {
            "run_ts": str(run_ts),
            "panel_rows": int(len(panel)),
            "n_unique_ts": n_unique_ts,
            "ts_min": str(ts_min) if ts_min is not None else None,
            "ts_max": str(ts_max) if ts_max is not None else None,
            "insufficient_data": True,
            "message": msg,
            "outputs_dir": str(out_dir),
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        raise SystemExit(msg)
    else:
        stale_insufficient = out_dir / "insufficient_data_report.csv"
        if stale_insufficient.exists():
            stale_insufficient.unlink()

    # Consolidated live opportunities (strategy + structural modules).
    extra_live: list[pd.DataFrame] = []
    if isinstance(consistency, dict):
        cp = consistency.get("complement_parity", pd.DataFrame())
        if isinstance(cp, pd.DataFrame) and not cp.empty:
            live_cp = cp.copy()
            live_cp["strategy"] = "consistency_arb"
            live_cp["direction"] = "PAIR_HEDGE"
            live_cp["traded_side"] = "BASKET"
            live_cp["required_notional"] = 250.0
            live_cp["estimated_fill_probability"] = 0.5
            live_cp["rationale"] = "complement_parity_violation"
            extra_live.append(live_cp.rename(columns={"expected_net_return": "expected_net_ev"}))
    if isinstance(micro_mm, dict):
        mm_live = micro_mm.get("live_quotes", pd.DataFrame())
        if isinstance(mm_live, pd.DataFrame) and not mm_live.empty:
            live_mm = mm_live.copy()
            live_mm["strategy"] = "microstructure_mm"
            live_mm["direction"] = "MARKET_MAKE"
            live_mm["traded_side"] = "BOTH"
            live_mm["required_notional"] = pd.to_numeric(live_mm.get("quoted_notional", 100.0), errors="coerce").fillna(100.0)
            live_mm["estimated_fill_probability"] = pd.to_numeric(live_mm.get("expected_fill_probability", 0.5), errors="coerce").fillna(0.5)
            live_mm["rationale"] = live_mm.get("strategy_rationale", "passive_spread_capture")
            live_mm["expected_net_ev"] = pd.to_numeric(live_mm.get("expected_net_return", 0.0), errors="coerce").fillna(0.0)
            extra_live.append(live_mm)
    if isinstance(resolution_mech, dict):
        rm = resolution_mech.get("human_review", pd.DataFrame())
        if isinstance(rm, pd.DataFrame) and not rm.empty:
            live_rm = rm.copy()
            live_rm["strategy"] = "resolution_mechanics"
            live_rm["direction"] = "HUMAN_REVIEW"
            live_rm["traded_side"] = "YES/NO"
            live_rm["required_notional"] = 100.0
            live_rm["estimated_fill_probability"] = 0.25
            live_rm["expected_net_ev"] = pd.to_numeric(live_rm.get("expected_net_return", 0.0), errors="coerce").fillna(0.0)
            extra_live.append(live_rm)
    if live_opportunity_rows or extra_live:
        live_all = pd.concat(live_opportunity_rows + extra_live, ignore_index=True, sort=False)
        if "expected_net_ev" in live_all.columns:
            live_all = live_all.sort_values("expected_net_ev", ascending=False)
        if "ts" in live_all.columns:
            live_all["ts"] = pd.to_datetime(live_all["ts"], errors="coerce")
        dedupe_cols = [c for c in ["strategy", "market_id", "token_id"] if c in live_all.columns]
        if dedupe_cols:
            sort_cols = [c for c in ["expected_net_ev", "ts"] if c in live_all.columns]
            if sort_cols:
                asc = [False] * len(sort_cols)
                live_all = live_all.sort_values(sort_cols, ascending=asc)
            live_all = live_all.drop_duplicates(subset=dedupe_cols, keep="first")
        live_all.head(200).to_csv(out_dir / "live_opportunities.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "live_opportunities.csv", index=False)

    score_input = _build_edge_score_input(report)
    scored = score_edges(score_input) if not score_input.empty else pd.DataFrame()
    scored.to_csv(out_dir / "edge_scores_v2.csv", index=False)
    readiness = _build_readiness_report(report, n_unique_ts=n_unique_ts)
    readiness.to_csv(out_dir / "readiness_report.csv", index=False)
    insufficient_trade_depth = False
    if not report.empty and float(report["test_trade_count"].max()) < float(MIN_TRADES_TEST):
        insufficient_trade_depth = True
        print(
            "INSUFFICIENT_DATA: need more snapshots/markets "
            f"(max_test_trade_count={float(report['test_trade_count'].max()):.0f} < {MIN_TRADES_TEST})."
        )

    methods = ["risk_parity", "mean_variance", "max_sharpe", "kelly"]
    capacity_by_strategy = (
        {str(r["strategy"]): float(r.get("capacity_usd", 0.0)) for r in report.to_dict(orient="records")}
        if not report.empty
        else {}
    )
    portfolio_summaries: list[pd.DataFrame] = []
    for method in methods:
        portfolio = build_portfolio(
            returns_by_strategy,
            method=method,
            capacity_by_strategy=capacity_by_strategy,
            trade_details_by_strategy=trade_details_by_strategy,
        )
        portfolio["correlation"].to_csv(out_dir / f"portfolio_correlation_{method}.csv", index=False)
        if "strategy_correlation" in portfolio:
            portfolio["strategy_correlation"].to_csv(out_dir / f"strategy_correlation_{method}.csv", index=False)
        portfolio["weights"].to_csv(out_dir / f"portfolio_weights_{method}.csv", index=False)
        portfolio["timeseries"].to_csv(out_dir / f"portfolio_timeseries_{method}.csv", index=False)
        portfolio["summary"].to_csv(out_dir / f"portfolio_summary_{method}.csv", index=False)
        if "risk_report" in portfolio:
            portfolio["risk_report"].to_csv(out_dir / f"portfolio_risk_report_{method}.csv", index=False)
        if not portfolio["summary"].empty:
            portfolio_summaries.append(portfolio["summary"])

    portfolio_summary_all = pd.concat(portfolio_summaries, ignore_index=True) if portfolio_summaries else pd.DataFrame()
    portfolio_summary_all.to_csv(out_dir / "portfolio_summary.csv", index=False)
    if (out_dir / "strategy_correlation_risk_parity.csv").exists():
        pd.read_csv(out_dir / "strategy_correlation_risk_parity.csv").to_csv(out_dir / "strategy_correlation.csv", index=False)
    if (out_dir / "portfolio_risk_report_risk_parity.csv").exists():
        pd.read_csv(out_dir / "portfolio_risk_report_risk_parity.csv").to_csv(out_dir / "portfolio_risk_report.csv", index=False)

    summary_payload = {
        "run_ts": str(run_ts),
        "panel_rows": int(len(panel)),
        "n_unique_ts": n_unique_ts,
        "ts_min": str(ts_min) if ts_min is not None else None,
        "ts_max": str(ts_max) if ts_max is not None else None,
        "orderbook_markets": int(len(ob_missing)) if not ob_missing.empty else 0,
        "orderbook_bid_missing_rate_mean": float(ob_missing["bid_missing_rate"].mean()) if (not ob_missing.empty and "bid_missing_rate" in ob_missing.columns) else 0.0,
        "orderbook_ask_missing_rate_mean": float(ob_missing["ask_missing_rate"].mean()) if (not ob_missing.empty and "ask_missing_rate" in ob_missing.columns) else 0.0,
        "orderbook_two_sided_rate_mean": float(ob_missing["two_sided_rate"].mean()) if (not ob_missing.empty and "two_sided_rate" in ob_missing.columns) else 0.0,
        "insufficient_trade_depth": bool(insufficient_trade_depth),
        "min_trades_test": int(MIN_TRADES_TEST),
        "strategies": int(len(strategies)),
        "outputs_dir": str(out_dir),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Research v2 complete.")
    print(
        report[
            [
                "strategy",
                "model",
                "test_trade_count",
                "test_expectancy",
                "test_sharpe",
                "ttest_pvalue",
                "perm_pvalue",
                "white_pvalue",
                "pbo",
                "capacity_usd",
                "deployable",
            ]
        ]
    )
    if not portfolio_summary_all.empty:
        print("Portfolio summary:")
        print(portfolio_summary_all)


if __name__ == "__main__":
    main()
