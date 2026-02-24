from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.api import PolymarketClient
from polymarket_edge.db import get_connection, init_db, upsert_dataframe
from polymarket_edge.pipeline import extract_resolutions_from_markets, normalize_markets


def _resolved_markets_frame(markets: list[dict], *, snapshot_ts: datetime) -> pd.DataFrame:
    if not markets:
        return pd.DataFrame()
    market_df, _ = normalize_markets(markets, snapshot_ts)
    res_df, _ = extract_resolutions_from_markets(markets)
    if market_df.empty:
        return pd.DataFrame()
    out = market_df.merge(
        res_df[
            [
                "market_id",
                "resolved_ts",
                "winner_token_id",
                "winner_label",
                "resolution_source",
                "raw_json",
            ]
        ]
        if not res_df.empty
        else pd.DataFrame(columns=["market_id", "resolved_ts", "winner_token_id", "winner_label", "resolution_source", "raw_json"]),
        on="market_id",
        how="left",
        suffixes=("", "_res"),
    )
    out["last_backfill_ts"] = snapshot_ts
    out["resolved_ts"] = pd.to_datetime(out["resolved_ts"], errors="coerce")
    out = out[
        [
            "market_id",
            "event_id",
            "question",
            "category",
            "resolved_ts",
            "winner_token_id",
            "winner_label",
            "resolution_source",
            "close_ts",
            "last_backfill_ts",
            "raw_json",
        ]
    ].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill resolved markets + outcomes into DuckDB.")
    parser.add_argument("--max-rows", type=int, default=5000, help="Max closed markets to fetch.")
    parser.add_argument("--batch-size", type=int, default=2500, help="Fetch size per API pagination batch.")
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)
    client = PolymarketClient()

    snapshot_ts = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    closed = client.fetch_markets(closed=True, archived=None, page_size=int(args.batch_size), max_rows=int(args.max_rows))
    market_df, outcome_df = normalize_markets(closed, snapshot_ts)
    res_df, winners_df = extract_resolutions_from_markets(closed)
    resolved_market_df = _resolved_markets_frame(closed, snapshot_ts=snapshot_ts)

    n_m = upsert_dataframe(conn, "markets", market_df)
    n_o = upsert_dataframe(conn, "market_outcomes", outcome_df)
    n_r = upsert_dataframe(conn, "resolutions", res_df)
    n_w = upsert_dataframe(conn, "market_outcomes", winners_df) if not winners_df.empty else 0
    n_rm = upsert_dataframe(conn, "resolved_markets", resolved_market_df) if not resolved_market_df.empty else 0

    print(
        "Resolved backfill complete: "
        f"markets={n_m} outcomes={n_o} winners={n_w} resolutions={n_r} resolved_markets={n_rm}"
    )


if __name__ == "__main__":
    main()

