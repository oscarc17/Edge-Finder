from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.db import get_connection, init_db
from polymarket_edge.research.data import build_yes_feature_panel
from polymarket_edge.strategies import (
    CrossMarketStrategy,
    LiquidityPremiumStrategy,
    MomentumReversionStrategy,
    ResolutionRuleStrategy,
    WhaleBehaviorStrategy,
)

MIN_SNAPSHOTS = 200


def _strategy_by_name(name: str):
    lookup = {
        "cross_market": CrossMarketStrategy,
        "resolution_rules": ResolutionRuleStrategy,
        "liquidity_premium": LiquidityPremiumStrategy,
        "momentum": MomentumReversionStrategy,
        "whale": WhaleBehaviorStrategy,
    }
    if name not in lookup:
        raise ValueError(f"Unknown strategy: {name}")
    return lookup[name]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one upgraded strategy module independently.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["cross_market", "resolution_rules", "liquidity_premium", "momentum", "whale"],
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["heuristic", "logistic", "gradient_boosting", "bayesian", "auto", "ensemble"],
    )
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)
    panel = build_yes_feature_panel(conn)
    if panel.empty:
        raise SystemExit("No feature panel rows available. Ingest snapshots first.")
    panel["ts"] = pd.to_datetime(panel["ts"], errors="coerce")
    n_unique_ts = int(panel["ts"].dropna().nunique())
    ts_min = panel["ts"].min()
    ts_max = panel["ts"].max()
    print(f"Panel diagnostics: rows={len(panel)} n_unique_ts={n_unique_ts} ts_min={ts_min} ts_max={ts_max}")
    if n_unique_ts < MIN_SNAPSHOTS:
        raise SystemExit(
            f"INSUFFICIENT_DATA: need more snapshots/markets. n_unique_ts={n_unique_ts} < {MIN_SNAPSHOTS}. "
            "Run `python scripts/collect_snapshots.py --minutes 5 --hours 24`."
        )
    strat = _strategy_by_name(args.strategy)
    strat.config.model_type = args.model_type
    result = strat.run(conn, panel, context={})

    out_dir = Path("data/research_v2_single") / strat.name
    out_dir.mkdir(parents=True, exist_ok=True)
    result.signal_frame.to_csv(out_dir / "signals.csv", index=False)
    result.train_summary.to_csv(out_dir / "train_summary.csv", index=False)
    result.test_summary.to_csv(out_dir / "test_summary.csv", index=False)
    result.validation_summary.to_csv(out_dir / "validation.csv", index=False)
    result.explain_summary.to_csv(out_dir / "feature_importance.csv", index=False)
    result.capacity_summary.to_csv(out_dir / "capacity.csv", index=False)
    result.model_comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    result.model_feature_importance.to_csv(out_dir / "model_feature_importance.csv", index=False)
    result.label_diagnostics.to_csv(out_dir / "label_diagnostics.csv", index=False)
    result.walkforward_folds.to_csv(out_dir / "walkforward_folds.csv", index=False)
    result.walkforward_summary.to_csv(out_dir / "walkforward_summary.csv", index=False)
    result.stability_time.to_csv(out_dir / "stability_time.csv", index=False)
    result.stability_category.to_csv(out_dir / "stability_category.csv", index=False)
    result.parameter_sensitivity.to_csv(out_dir / "parameter_sensitivity.csv", index=False)
    result.filter_impact.to_csv(out_dir / "filter_impact.csv", index=False)
    result.ev_threshold_grid.to_csv(out_dir / "filter_impact_ev_threshold.csv", index=False)
    result.edge_attribution.to_csv(out_dir / "edge_attribution.csv", index=False)
    result.calibration_curve.to_csv(out_dir / "calibration_curve.csv", index=False)
    result.signal_deciles.to_csv(out_dir / "signal_deciles.csv", index=False)
    result.feature_drift.to_csv(out_dir / "feature_drift.csv", index=False)
    result.model_vs_market.to_csv(out_dir / "model_vs_market.csv", index=False)
    result.ev_diagnostics.to_csv(out_dir / "ev_diagnostics.csv", index=False)
    result.ev_diagnostics.to_csv(out_dir / "realized_vs_expected_ev_by_decile.csv", index=False)
    result.advanced_validation.to_csv(out_dir / "advanced_validation.csv", index=False)
    result.deployment_readiness.to_csv(out_dir / "deployment_readiness.csv", index=False)
    result.execution_sensitivity.to_csv(out_dir / "execution_sensitivity.csv", index=False)
    result.execution_calibration.to_csv(out_dir / "execution_calibration.csv", index=False)
    result.cost_decomposition.to_csv(out_dir / "cost_decomposition.csv", index=False)
    (out_dir / "tuned_params.json").write_text(json.dumps(result.tuned_params, indent=2), encoding="utf-8")

    print(f"Strategy complete: {strat.name}")
    print(f"Tuned params: {result.tuned_params}")
    if not result.deployment_readiness.empty:
        decision = result.deployment_readiness[result.deployment_readiness["criterion"] == "deployment_decision"]
        if not decision.empty:
            print(f"Deployment decision: {decision['notes'].iloc[0]}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
