from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArbDataQualityThresholds:
    min_both_legs_coverage: float = 0.80
    recent_hours: float = 72.0


def _raw_outcome_books(conn: duckdb.DuckDBPyConnection, *, recent_hours: float | None = None) -> pd.DataFrame:
    where_recent = ""
    params: list[Any] = []
    if recent_hours is not None and float(recent_hours) > 0:
        where_recent = "AND o.snapshot_ts >= (SELECT MAX(snapshot_ts) - (? * INTERVAL '1 hour') FROM orderbook_snapshots)"
        params.append(float(recent_hours))
    return conn.execute(
        f"""
        SELECT
            o.snapshot_ts AS ts,
            o.market_id,
            o.token_id,
            lower(trim(coalesce(mo.outcome_label, ''))) AS outcome_label,
            o.best_bid,
            o.best_ask,
            o.mid,
            o.spread,
            o.spread_bps,
            o.bid_depth,
            o.ask_depth
        FROM orderbook_snapshots o
        LEFT JOIN market_outcomes mo
          ON o.market_id = mo.market_id
         AND o.token_id = mo.token_id
        WHERE o.market_id IS NOT NULL
          {where_recent}
        """,
        params,
    ).df()


def _pair_snapshot_panel(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    t = raw.copy()
    t["outcome_label"] = t["outcome_label"].astype(str).str.lower().str.strip()
    t = t[t["outcome_label"].isin(["yes", "no"])].copy()
    if t.empty:
        return pd.DataFrame()
    base_cols = ["ts", "market_id"]
    val_cols = ["token_id", "best_bid", "best_ask", "mid", "spread", "spread_bps", "bid_depth", "ask_depth"]
    pieces: list[pd.DataFrame] = []
    for lab in ["yes", "no"]:
        cur = t[t["outcome_label"] == lab][base_cols + val_cols].copy()
        cur = cur.drop_duplicates(subset=base_cols, keep="last")
        cur = cur.rename(columns={c: f"{c}_{lab}" for c in val_cols})
        pieces.append(cur)
    if len(pieces) != 2:
        return pd.DataFrame()
    panel = pieces[0].merge(pieces[1], on=base_cols, how="outer")
    panel["has_yes"] = panel["token_id_yes"].notna()
    panel["has_no"] = panel["token_id_no"].notna()
    panel["both_legs_present"] = panel["has_yes"] & panel["has_no"]
    panel["yes_two_sided"] = panel["best_bid_yes"].notna() & panel["best_ask_yes"].notna()
    panel["no_two_sided"] = panel["best_bid_no"].notna() & panel["best_ask_no"].notna()
    panel["both_legs_two_sided"] = panel["yes_two_sided"] & panel["no_two_sided"]
    panel["depth_yes"] = pd.to_numeric(panel["bid_depth_yes"], errors="coerce").fillna(0.0) + pd.to_numeric(panel["ask_depth_yes"], errors="coerce").fillna(0.0)
    panel["depth_no"] = pd.to_numeric(panel["bid_depth_no"], errors="coerce").fillna(0.0) + pd.to_numeric(panel["ask_depth_no"], errors="coerce").fillna(0.0)
    panel["min_pair_depth"] = np.minimum(panel["depth_yes"], panel["depth_no"])
    return panel


def build_arb_data_quality(
    conn: duckdb.DuckDBPyConnection,
    *,
    thresholds: ArbDataQualityThresholds = ArbDataQualityThresholds(),
) -> dict[str, pd.DataFrame | bool | float]:
    raw = _raw_outcome_books(conn, recent_hours=thresholds.recent_hours)
    pair = _pair_snapshot_panel(raw)
    if pair.empty:
        report = pd.DataFrame(
            [
                {
                    "level": "aggregate",
                    "metric": "both_legs_coverage",
                    "value": 0.0,
                    "required_min": float(thresholds.min_both_legs_coverage),
                    "passed": 0,
                    "reason": "no_yes_no_pairs_found",
                }
            ]
        )
        return {"report": report, "pair_panel": pair, "passed": False, "both_legs_coverage": 0.0}

    agg_rows: list[dict[str, object]] = []
    both_legs_coverage = float(pair["both_legs_present"].mean()) if len(pair) else 0.0
    agg_rows.append(
        {
            "level": "aggregate",
            "metric": "both_legs_coverage",
            "value": both_legs_coverage,
            "required_min": float(thresholds.min_both_legs_coverage),
            "passed": int(both_legs_coverage >= float(thresholds.min_both_legs_coverage)),
            "reason": "" if both_legs_coverage >= float(thresholds.min_both_legs_coverage) else "both_legs_coverage_below_threshold",
        }
    )
    metrics = {
        "both_legs_two_sided_rate": float(pair["both_legs_two_sided"].mean()),
        "yes_missing_bid_rate": float(pair["best_bid_yes"].isna().mean()),
        "yes_missing_ask_rate": float(pair["best_ask_yes"].isna().mean()),
        "no_missing_bid_rate": float(pair["best_bid_no"].isna().mean()),
        "no_missing_ask_rate": float(pair["best_ask_no"].isna().mean()),
        "spread_yes_median": float(pd.to_numeric(pair["spread_yes"], errors="coerce").dropna().median()) if pair["spread_yes"].notna().any() else 0.0,
        "spread_no_median": float(pd.to_numeric(pair["spread_no"], errors="coerce").dropna().median()) if pair["spread_no"].notna().any() else 0.0,
        "depth_yes_median": float(pd.to_numeric(pair["depth_yes"], errors="coerce").dropna().median()) if pair["depth_yes"].notna().any() else 0.0,
        "depth_no_median": float(pd.to_numeric(pair["depth_no"], errors="coerce").dropna().median()) if pair["depth_no"].notna().any() else 0.0,
        "min_pair_depth_median": float(pd.to_numeric(pair["min_pair_depth"], errors="coerce").dropna().median()) if pair["min_pair_depth"].notna().any() else 0.0,
        "snapshot_pairs": float(len(pair)),
        "markets_with_pairs": float(pair["market_id"].nunique()),
    }
    for k, v in metrics.items():
        agg_rows.append(
            {
                "level": "aggregate",
                "metric": k,
                "value": float(v),
                "required_min": np.nan,
                "passed": 1,
                "reason": "",
            }
        )

    per_market = (
        pair.groupby("market_id", observed=True)
        .agg(
            n_snapshot_pairs=("ts", "count"),
            both_legs_coverage=("both_legs_present", "mean"),
            both_legs_two_sided_rate=("both_legs_two_sided", "mean"),
            yes_missing_bid_rate=("best_bid_yes", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())),
            yes_missing_ask_rate=("best_ask_yes", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())),
            no_missing_bid_rate=("best_bid_no", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())),
            no_missing_ask_rate=("best_ask_no", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())),
            spread_yes_median=("spread_yes", "median"),
            spread_no_median=("spread_no", "median"),
            min_pair_depth_median=("min_pair_depth", "median"),
        )
        .reset_index()
    )
    if not per_market.empty:
        per_market["level"] = "market"
        per_market["metric"] = "per_market"
        per_market["value"] = pd.to_numeric(per_market["both_legs_coverage"], errors="coerce").fillna(0.0)
        per_market["required_min"] = float(thresholds.min_both_legs_coverage)
        per_market["passed"] = (per_market["both_legs_coverage"] >= float(thresholds.min_both_legs_coverage)).astype(int)
        per_market["reason"] = np.where(per_market["passed"] > 0, "", "low_both_legs_coverage")
        cols = [
            "level",
            "metric",
            "market_id",
            "value",
            "required_min",
            "passed",
            "reason",
            "n_snapshot_pairs",
            "both_legs_coverage",
            "both_legs_two_sided_rate",
            "yes_missing_bid_rate",
            "yes_missing_ask_rate",
            "no_missing_bid_rate",
            "no_missing_ask_rate",
            "spread_yes_median",
            "spread_no_median",
            "min_pair_depth_median",
        ]
        per_market = per_market[cols].sort_values(["passed", "both_legs_coverage", "n_snapshot_pairs"], ascending=[True, True, False])

    report = pd.concat([pd.DataFrame(agg_rows), per_market], ignore_index=True, sort=False) if not per_market.empty else pd.DataFrame(agg_rows)
    passed = both_legs_coverage >= float(thresholds.min_both_legs_coverage)
    return {
        "report": report,
        "pair_panel": pair,
        "passed": bool(passed),
        "both_legs_coverage": float(both_legs_coverage),
    }


def write_arb_data_quality_report(
    conn: duckdb.DuckDBPyConnection,
    output_path: str | Path,
    *,
    thresholds: ArbDataQualityThresholds = ArbDataQualityThresholds(),
) -> dict[str, pd.DataFrame | bool | float]:
    result = build_arb_data_quality(conn, thresholds=thresholds)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = result["report"] if isinstance(result.get("report"), pd.DataFrame) else pd.DataFrame()
    report.to_csv(out, index=False)
    return result
