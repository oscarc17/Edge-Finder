from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntegrityThresholds:
    min_ts: int = 500
    min_mkts: int = 200
    min_snapshots: int = 50_000
    min_test_trades: int = 300
    min_fold_trades: int = 50


def _to_num(series: pd.Series | Any) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.Series(dtype=float)


def _stationarity_drift(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "ts" not in panel.columns:
        return pd.DataFrame(columns=["check_group", "metric", "early_mean", "late_mean", "drift_abs", "drift_ratio", "status"])
    p = panel.copy()
    p["ts"] = pd.to_datetime(p["ts"], errors="coerce")
    p = p.dropna(subset=["ts"]).sort_values("ts")
    if p.empty:
        return pd.DataFrame(columns=["check_group", "metric", "early_mean", "late_mean", "drift_abs", "drift_ratio", "status"])
    n = len(p)
    head = p.iloc[: max(1, n // 3)]
    tail = p.iloc[-max(1, n // 3) :]
    metrics = {
        "spread_bps": "spread_bps",
        "liquidity": "liquidity_log" if "liquidity_log" in p.columns else "liquidity_score",
        "volatility": "prob_vol_6",
    }
    rows: list[dict[str, object]] = []
    for label, col in metrics.items():
        if col not in p.columns:
            continue
        e = _to_num(head[col]).dropna()
        l = _to_num(tail[col]).dropna()
        em = float(e.mean()) if len(e) else 0.0
        lm = float(l.mean()) if len(l) else 0.0
        drift_abs = float(lm - em)
        denom = abs(em) + 1e-9
        drift_ratio = float(abs(drift_abs) / denom)
        status = "warning_regime_drift" if drift_ratio > 1.0 else "ok"
        rows.append(
            {
                "check_group": "stationarity",
                "metric": label,
                "early_mean": em,
                "late_mean": lm,
                "drift_abs": drift_abs,
                "drift_ratio": drift_ratio,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def run_research_integrity(
    *,
    panel: pd.DataFrame,
    strategy_report: pd.DataFrame | None = None,
    walkforward_folds: pd.DataFrame | None = None,
    thresholds: IntegrityThresholds = IntegrityThresholds(),
    output_path: str | Path | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    p = panel.copy()
    if not p.empty and "ts" in p.columns:
        p["ts"] = pd.to_datetime(p["ts"], errors="coerce")
    n_unique_ts = int(p["ts"].dropna().nunique()) if (not p.empty and "ts" in p.columns) else 0
    n_unique_mkts = int(p["market_id"].nunique()) if (not p.empty and "market_id" in p.columns) else 0
    n_snapshots = int(len(p))
    rows.extend(
        [
            {
                "check_group": "data_depth",
                "metric": "unique_timestamps",
                "value": float(n_unique_ts),
                "required_min": float(thresholds.min_ts),
                "passed": int(n_unique_ts >= thresholds.min_ts),
                "missing_qty": float(max(0, thresholds.min_ts - n_unique_ts)),
                "reason": "" if n_unique_ts >= thresholds.min_ts else "need_more_time_slices",
            },
            {
                "check_group": "data_depth",
                "metric": "unique_markets",
                "value": float(n_unique_mkts),
                "required_min": float(thresholds.min_mkts),
                "passed": int(n_unique_mkts >= thresholds.min_mkts),
                "missing_qty": float(max(0, thresholds.min_mkts - n_unique_mkts)),
                "reason": "" if n_unique_mkts >= thresholds.min_mkts else "need_more_market_coverage",
            },
            {
                "check_group": "data_depth",
                "metric": "total_snapshots",
                "value": float(n_snapshots),
                "required_min": float(thresholds.min_snapshots),
                "passed": int(n_snapshots >= thresholds.min_snapshots),
                "missing_qty": float(max(0, thresholds.min_snapshots - n_snapshots)),
                "reason": "" if n_snapshots >= thresholds.min_snapshots else "need_more_snapshots",
            },
        ]
    )
    drift = _stationarity_drift(p)
    if not drift.empty:
        for rec in drift.to_dict(orient="records"):
            rows.append(
                {
                    "check_group": rec.get("check_group"),
                    "metric": rec.get("metric"),
                    "value": float(rec.get("late_mean", 0.0)),
                    "required_min": np.nan,
                    "passed": 1,
                    "missing_qty": 0.0,
                    "reason": str(rec.get("status", "ok")),
                    "early_mean": float(rec.get("early_mean", 0.0)),
                    "late_mean": float(rec.get("late_mean", 0.0)),
                    "drift_abs": float(rec.get("drift_abs", 0.0)),
                    "drift_ratio": float(rec.get("drift_ratio", 0.0)),
                }
            )

    if strategy_report is not None and not strategy_report.empty:
        test_counts = _to_num(strategy_report.get("test_trade_count")).dropna()
        max_test = float(test_counts.max()) if len(test_counts) else 0.0
        rows.append(
            {
                "check_group": "trade_counts",
                "metric": "max_strategy_test_trades",
                "value": max_test,
                "required_min": float(thresholds.min_test_trades),
                "passed": int(max_test >= thresholds.min_test_trades),
                "missing_qty": float(max(0.0, thresholds.min_test_trades - max_test)),
                "reason": "" if max_test >= thresholds.min_test_trades else "need_more_trades_or_looser_filters",
            }
        )
    if walkforward_folds is not None and not walkforward_folds.empty:
        wf = walkforward_folds.copy()
        if "strategy" in wf.columns:
            status_ser = wf["status"].astype(str) if "status" in wf.columns else pd.Series("ok", index=wf.index)
            fold_min = (
                wf[status_ser.eq("ok")]
                .groupby("strategy", observed=True)["trade_count"]
                .min()
                if "trade_count" in wf.columns
                else pd.Series(dtype=float)
            )
            min_valid_fold_trades = float(fold_min.min()) if len(fold_min) else 0.0
        else:
            min_valid_fold_trades = float(pd.to_numeric(wf.get("trade_count"), errors="coerce").min()) if "trade_count" in wf.columns else 0.0
        rows.append(
            {
                "check_group": "trade_counts",
                "metric": "min_valid_fold_trades",
                "value": min_valid_fold_trades,
                "required_min": float(thresholds.min_fold_trades),
                "passed": int(min_valid_fold_trades >= thresholds.min_fold_trades),
                "missing_qty": float(max(0.0, thresholds.min_fold_trades - min_valid_fold_trades)),
                "reason": "" if min_valid_fold_trades >= thresholds.min_fold_trades else "walkforward_fold_too_sparse",
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        report = pd.DataFrame(
            [
                {
                    "check_group": "integrity",
                    "metric": "empty",
                    "value": 0.0,
                    "required_min": 0.0,
                    "passed": 0,
                    "missing_qty": 0.0,
                    "reason": "no_inputs",
                }
            ]
        )
    fail_mask = report["passed"].fillna(1).astype(float) < 0.5
    passed = not bool(fail_mask.any())
    if output_path is not None:
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(op, index=False)
    return {"passed": passed, "report": report}
