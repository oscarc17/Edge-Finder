from __future__ import annotations

from dataclasses import asdict

import duckdb
import pandas as pd

from polymarket_edge.research.data import build_yes_feature_panel
from polymarket_edge.strategies.consistency_arb import ConsistencyArbStrategy, ConsistencyArbStrategyConfig


def run(
    conn: duckdb.DuckDBPyConnection,
    *,
    panel: pd.DataFrame | None = None,
    config: ConsistencyArbStrategyConfig | None = None,
) -> dict[str, pd.DataFrame]:
    strat = ConsistencyArbStrategy(config or ConsistencyArbStrategyConfig())
    if panel is None:
        try:
            panel = build_yes_feature_panel(conn)
        except Exception:
            panel = pd.DataFrame()
    signal_frame = strat.build_signal_frame(conn, panel if isinstance(panel, pd.DataFrame) else pd.DataFrame())
    latest = pd.DataFrame()
    if not signal_frame.empty and "ts" in signal_frame.columns:
        sf = signal_frame.copy()
        sf["ts"] = pd.to_datetime(sf["ts"], errors="coerce")
        latest_ts = sf["ts"].max()
        latest = sf[sf["ts"] == latest_ts].copy()
    elif isinstance(signal_frame, pd.DataFrame):
        latest = signal_frame.copy()
    summary = pd.DataFrame(
        [
            {
                "module": "consistency_arb_v2",
                "n_rows": float(len(signal_frame)),
                "n_positive_expected_net": float(
                    (pd.to_numeric(signal_frame.get("expected_net_ev"), errors="coerce").fillna(0.0) > 0.0).sum()
                )
                if not signal_frame.empty
                else 0.0,
            }
        ]
    )
    params = pd.DataFrame([asdict(strat.config)])
    return {
        "summary": summary,
        "opportunities": signal_frame,
        "live_opportunities": latest.sort_values("expected_net_ev", ascending=False) if (not latest.empty and "expected_net_ev" in latest.columns) else latest,
        "params": params,
    }

