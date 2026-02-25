from __future__ import annotations

import numpy as np
import pandas as pd


def _bool_row(criterion: str, value: float, threshold: str, passed: bool, notes: str = "") -> dict[str, object]:
    return {
        "criterion": criterion,
        "value": float(value),
        "threshold": threshold,
        "passed": bool(passed),
        "notes": notes,
    }


def evaluate_deployment_readiness(
    *,
    strategy_name: str,
    walkforward_summary: pd.DataFrame,
    walkforward_folds: pd.DataFrame,
    validation_summary: pd.DataFrame,
    stability_time: pd.DataFrame,
    test_trades: pd.DataFrame,
    execution_sensitivity: pd.DataFrame,
    advanced_validation: pd.DataFrame,
    ev_diagnostics: pd.DataFrame | None = None,
    pbo_threshold: float = 0.5,
    min_valid_folds: int = 3,
    min_test_trades: int = 200,
    concentration_market_max_pnl_share: float = 0.40,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    wf_mean = float(walkforward_summary["mean_expectancy"].iloc[0]) if not walkforward_summary.empty else 0.0
    wf_valid_folds = (
        float(walkforward_summary["valid_folds"].iloc[0])
        if (not walkforward_summary.empty and "valid_folds" in walkforward_summary.columns)
        else 0.0
    )
    wf_trade_count = 0.0
    if not walkforward_folds.empty and "trade_count" in walkforward_folds.columns:
        if "valid_fold" in walkforward_folds.columns:
            wf_trade_count = float(
                pd.to_numeric(
                    walkforward_folds[walkforward_folds["valid_fold"] == True]["trade_count"],  # noqa: E712
                    errors="coerce",
                ).sum()
            )
        else:
            wf_trade_count = float(pd.to_numeric(walkforward_folds["trade_count"], errors="coerce").sum())

    test_mean = float(validation_summary["test_mean_return"].iloc[0]) if not validation_summary.empty else 0.0
    ci_low = float(validation_summary["mean_return_ci_low"].iloc[0]) if not validation_summary.empty else -1.0
    ci_high = float(validation_summary["mean_return_ci_high"].iloc[0]) if not validation_summary.empty else 0.0
    ttest_p = float(validation_summary["ttest_pvalue"].iloc[0]) if not validation_summary.empty else 1.0
    perm_p = float(validation_summary["permutation_pvalue"].iloc[0]) if not validation_summary.empty else 1.0
    n_test_trades = float(len(test_trades))

    if stability_time.empty or "avg_return" not in stability_time.columns:
        stable_ratio = 0.0
        stable_std_ratio = np.inf
    else:
        avg = pd.to_numeric(stability_time["avg_return"], errors="coerce").fillna(0.0)
        stable_ratio = float(np.mean(avg > 0.0)) if len(avg) else 0.0
        denom = max(abs(float(avg.mean())), 1e-8)
        stable_std_ratio = float(avg.std(ddof=0) / denom) if len(avg) else np.inf
    stable_score = stable_ratio / (1.0 + stable_std_ratio if np.isfinite(stable_std_ratio) else np.inf)

    pess_ev = -np.inf
    if not execution_sensitivity.empty:
        pess = execution_sensitivity[
            execution_sensitivity["execution_regime"].astype(str).str.lower() == "pessimistic"
        ]
        if not pess.empty and "avg_expected_ev" in pess.columns:
            pess_ev = float(pd.to_numeric(pess["avg_expected_ev"], errors="coerce").iloc[0])

    ci_pass = ci_low > 0.0
    wf_pass = wf_mean > 0.0
    wf_folds_pass = wf_valid_folds >= float(min_valid_folds)
    test_mean_pass = test_mean > 0.0
    trade_pass = n_test_trades >= float(min_test_trades)
    pess_pass = pess_ev >= 0.0
    pbo_val = float(advanced_validation["pbo"].iloc[0]) if (not advanced_validation.empty and "pbo" in advanced_validation.columns) else float("nan")
    pbo_computable = bool(np.isfinite(pbo_val))
    pbo_pass = True if not pbo_computable else (pbo_val < float(pbo_threshold))
    ev_monotonic_pass = False
    ev_spearman = 0.0
    hit_rate_pos_ev = 0.0
    ev_model_valid = True
    if ev_diagnostics is not None and not ev_diagnostics.empty:
        if "ev_monotonic_pass" in ev_diagnostics.columns:
            ev_monotonic_pass = bool(float(pd.to_numeric(ev_diagnostics["ev_monotonic_pass"], errors="coerce").fillna(0.0).iloc[0]) >= 0.5)
        if "ev_spearman" in ev_diagnostics.columns:
            ev_spearman = float(pd.to_numeric(ev_diagnostics["ev_spearman"], errors="coerce").fillna(0.0).iloc[0])
        elif "ev_monotonic_spearman" in ev_diagnostics.columns:
            ev_spearman = float(pd.to_numeric(ev_diagnostics["ev_monotonic_spearman"], errors="coerce").fillna(0.0).iloc[0])
        if "hit_rate_pos_ev" in ev_diagnostics.columns:
            hit_rate_pos_ev = float(pd.to_numeric(ev_diagnostics["hit_rate_pos_ev"], errors="coerce").fillna(0.0).iloc[0])
        if "ev_model_valid" in ev_diagnostics.columns:
            ev_model_valid = bool(float(pd.to_numeric(ev_diagnostics["ev_model_valid"], errors="coerce").fillna(0.0).iloc[0]) >= 0.5)
        elif len(test_trades) > 100 and ev_spearman < 0.0:
            ev_model_valid = False
    ev_quality_pass = bool(ev_monotonic_pass and (ev_spearman >= 0.2) and ev_model_valid)
    market_conc_share = 0.0
    concentration_pass = True
    if not test_trades.empty and {"market_id", "net_pnl"}.issubset(test_trades.columns):
        by_mkt = pd.to_numeric(test_trades["net_pnl"], errors="coerce").fillna(0.0)
        tmp = pd.DataFrame({"market_id": test_trades["market_id"].astype(str), "net_pnl": by_mkt})
        pnl_by_market = tmp.groupby("market_id", observed=True)["net_pnl"].sum()
        total_pos = float(pnl_by_market.clip(lower=0.0).sum())
        if total_pos > 0:
            market_conc_share = float((pnl_by_market.clip(lower=0.0) / total_pos).max())
        else:
            market_conc_share = 1.0 if float(tmp["net_pnl"].sum()) > 0 else 0.0
        concentration_pass = market_conc_share <= float(concentration_market_max_pnl_share)
    passed_all = bool(
        ci_pass
        and wf_pass
        and wf_folds_pass
        and test_mean_pass
        and trade_pass
        and pess_pass
        and pbo_pass
        and ev_quality_pass
        and concentration_pass
    )

    rows.append(_bool_row("ci_lower_bound_positive", ci_low, "> 0", ci_pass, "Conservative CI lower bound"))
    rows.append(_bool_row("walkforward_mean_positive", wf_mean, "> 0", wf_pass, "Mean OOS expectancy"))
    rows.append(_bool_row("walkforward_valid_folds_min", wf_valid_folds, f">= {int(min_valid_folds)}", wf_folds_pass, "Minimum valid walk-forward folds"))
    rows.append(_bool_row("test_mean_return_positive", test_mean, "> 0", test_mean_pass, "Mean return on test trades"))
    rows.append(_bool_row("test_trade_count_minimum", n_test_trades, f">= {int(min_test_trades)}", trade_pass, "Minimum OOS trades"))
    rows.append(
        _bool_row(
            "pessimistic_execution_ev_nonnegative",
            pess_ev if np.isfinite(pess_ev) else -1.0,
            ">= 0",
            pess_pass,
            "Average expected EV under pessimistic execution regime",
        )
    )
    rows.append(
        _bool_row(
            "pbo_below_threshold",
            pbo_val if np.isfinite(pbo_val) else np.nan,
            f"< {float(pbo_threshold):.3f}" if pbo_computable else "not_computable_allowed",
            pbo_pass,
            "Probability of backtest overfitting" if pbo_computable else "PBO unavailable; rely on stronger CI + walk-forward",
        )
    )
    rows.append(
        _bool_row(
            "market_profit_concentration",
            market_conc_share,
            f"<= {float(concentration_market_max_pnl_share):.2f}",
            concentration_pass,
            "No single market should dominate positive profits",
        )
    )
    rows.append(
        _bool_row(
            "ev_deciles_monotonic",
            ev_spearman,
            "spearman>=0.20 and monotonic_pass and ev_model_valid",
            ev_quality_pass,
            "Expected EV should align with realized return and remain valid",
        )
    )

    fail_reasons = []
    if not ci_pass:
        fail_reasons.append("ci_low<=0")
    if not wf_pass:
        fail_reasons.append("wf_mean<=0")
    if not wf_folds_pass:
        fail_reasons.append(f"valid_folds<{int(min_valid_folds)}")
    if not test_mean_pass:
        fail_reasons.append("test_mean<=0")
    if not trade_pass:
        fail_reasons.append(f"n_test<{int(min_test_trades)}")
    if not pess_pass:
        fail_reasons.append("pessimistic_ev<0")
    if pbo_computable and not pbo_pass:
        fail_reasons.append("pbo>=threshold")
    if not concentration_pass:
        fail_reasons.append("market_profit_concentration")
    if not ev_monotonic_pass:
        fail_reasons.append("ev_deciles_nonmonotonic")
    if ev_spearman < 0.2:
        fail_reasons.append("ev_spearman<0.2")
    if not ev_model_valid:
        fail_reasons.append("ev_model_invalid")
    reason_text = ";".join(fail_reasons) if fail_reasons else "all_checks_passed"

    # Summary row with key diagnostics requested for deployment decisions.
    rows.append(
        {
            "criterion": "summary",
            "strategy": strategy_name,
            "test_mean_return": test_mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ttest_pvalue": ttest_p,
            "permutation_pvalue": perm_p,
            "n_test_trades": n_test_trades,
            "n_walkforward_trades": wf_trade_count,
            "walkforward_mean_expectancy": wf_mean,
            "walkforward_valid_folds": wf_valid_folds,
            "stability_time_positive_ratio": stable_ratio,
            "stability_score": stable_score if np.isfinite(stable_score) else 0.0,
            "pessimistic_avg_expected_ev": pess_ev if np.isfinite(pess_ev) else -1.0,
            "advanced_white_pvalue": float(advanced_validation["white_pvalue"].iloc[0]) if not advanced_validation.empty else 1.0,
            "advanced_pbo": pbo_val if pbo_computable else np.nan,
            "advanced_pbo_computable": float(pbo_computable),
            "top_market_profit_share": market_conc_share,
            "ev_monotonic_pass": float(ev_monotonic_pass),
            "ev_monotonic_spearman": ev_spearman,
            "ev_spearman": ev_spearman,
            "hit_rate_pos_ev": hit_rate_pos_ev,
            "ev_model_valid": float(ev_model_valid),
            "passed": passed_all,
            "decision": "PASS" if passed_all else "FAIL",
            "notes": ("GO: " + reason_text) if passed_all else ("NO_GO: " + reason_text),
            "threshold": f"test_mean>0, CI>0, WF>0, valid_folds>={int(min_valid_folds)}, n_test>={int(min_test_trades)}, pessimistic_EV>=0, pbo<{float(pbo_threshold):.3f} (if computable), ev_spearman>=0.20, ev_monotonic, ev_model_valid, top_market_profit_share<={float(concentration_market_max_pnl_share):.2f}",
            "value": float(int(passed_all)),
        }
    )
    rows.append(
        {
            "criterion": "deployment_decision",
            "value": float(int(passed_all)),
            "threshold": "CI>0 and WF>0 and valid_folds>=min and n_test>=min and pessimistic_EV>=0 and ev_spearman>=0.20 and ev_monotonic and ev_model_valid and market_profit_concentration",
            "passed": passed_all,
            "decision": "PASS" if passed_all else "FAIL",
            "notes": "GO" if passed_all else "NO_GO",
            "strategy": strategy_name,
        }
    )
    out = pd.DataFrame(rows)
    required_numeric = [
        "value",
        "test_mean_return",
        "ci_low",
        "ci_high",
        "ttest_pvalue",
        "permutation_pvalue",
        "n_test_trades",
        "n_walkforward_trades",
        "walkforward_valid_folds",
        "walkforward_mean_expectancy",
        "stability_time_positive_ratio",
        "stability_score",
        "pessimistic_avg_expected_ev",
        "advanced_white_pvalue",
        "advanced_pbo",
        "advanced_pbo_computable",
        "top_market_profit_share",
        "ev_monotonic_pass",
        "ev_monotonic_spearman",
        "ev_spearman",
        "hit_rate_pos_ev",
        "ev_model_valid",
    ]
    for col in required_numeric:
        if col not in out.columns:
            out[col] = 0.0
        ser = pd.to_numeric(out[col], errors="coerce")
        if col == "advanced_pbo":
            out[col] = ser
        else:
            out[col] = ser.fillna(0.0)
    for col in ["strategy", "notes", "threshold", "criterion", "decision"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    if "passed" not in out.columns:
        out["passed"] = False
    out["passed"] = out["passed"].fillna(False).astype(bool)
    return out
