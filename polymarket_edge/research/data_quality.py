from __future__ import annotations

import numpy as np
import pandas as pd


def audit_feature_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if panel.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "field_missingness": empty, "pinned_markets": empty}

    p = panel.copy()
    p["ts"] = pd.to_datetime(p.get("ts"), errors="coerce")
    mid = pd.to_numeric(p.get("mid"), errors="coerce")
    bid = pd.to_numeric(p.get("best_bid"), errors="coerce") if "best_bid" in p.columns else pd.Series(np.nan, index=p.index)
    ask = pd.to_numeric(p.get("best_ask"), errors="coerce") if "best_ask" in p.columns else pd.Series(np.nan, index=p.index)
    depth = pd.to_numeric(p.get("depth_total"), errors="coerce").fillna(
        pd.to_numeric(p.get("bid_depth"), errors="coerce").fillna(0.0)
        + pd.to_numeric(p.get("ask_depth"), errors="coerce").fillna(0.0)
    )

    summary = pd.DataFrame(
        [
            {
                "rows": float(len(p)),
                "n_unique_ts": float(p["ts"].dropna().nunique()),
                "n_unique_markets": float(p["market_id"].nunique()) if "market_id" in p.columns else 0.0,
                "missing_mid_rate": float(mid.isna().mean()),
                "missing_bid_rate": float(bid.isna().mean()),
                "missing_ask_rate": float(ask.isna().mean()),
                "bid_gt_ask_rate": float(((bid > ask) & bid.notna() & ask.notna()).mean()) if len(p) else 0.0,
                "mid_out_of_bounds_rate": float(((mid < 0.0) | (mid > 1.0)).fillna(False).mean()) if len(p) else 0.0,
                "pinned_mid_rate": float((((mid <= 0.01) | (mid >= 0.99)).fillna(False)).mean()) if len(p) else 0.0,
            }
        ]
    )

    miss_rows: list[dict[str, float | str]] = []
    for col in ["best_bid", "best_ask", "mid", "spread", "spread_bps", "bid_depth", "ask_depth", "liquidity_score", "depth_total", "prob_vol_6"]:
        if col not in p.columns:
            continue
        s = p[col]
        miss_rows.append(
            {
                "field": col,
                "missing_rate": float(pd.to_numeric(s, errors="coerce").isna().mean()) if s.dtype != object else float(s.isna().mean()),
                "non_null_count": float(s.notna().sum()),
            }
        )
    field_missingness = pd.DataFrame(miss_rows).sort_values("missing_rate", ascending=False) if miss_rows else pd.DataFrame()

    if "market_id" in p.columns:
        pinned = (
            pd.DataFrame(
                {
                    "market_id": p["market_id"],
                    "mid": mid,
                    "depth_total": depth,
                    "two_sided": bid.notna() & ask.notna(),
                    "pinned": ((mid <= 0.01) | (mid >= 0.99)).fillna(False),
                }
            )
            .groupby("market_id", observed=True)
            .agg(
                n=("pinned", "size"),
                pinned_rate=("pinned", "mean"),
                avg_mid=("mid", "mean"),
                avg_depth_total=("depth_total", "mean"),
                two_sided_rate=("two_sided", "mean"),
            )
            .reset_index()
        )
        pinned["exclude_default"] = (
            (pinned["pinned_rate"] >= 0.8)
            & ((pinned["two_sided_rate"] < 0.5) | (pinned["avg_depth_total"] < 50.0))
        ).astype(int)
        pinned_markets = pinned.sort_values(["exclude_default", "pinned_rate"], ascending=[False, False])
    else:
        pinned_markets = pd.DataFrame()

    return {"summary": summary, "field_missingness": field_missingness, "pinned_markets": pinned_markets}

