from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.api import PolymarketClient
from polymarket_edge.db import get_connection, init_db
from polymarket_edge.pipeline import DataPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Polymarket ingestion cycle.")
    parser.add_argument("--max-markets", type=int, default=800, help="Max active markets to ingest.")
    parser.add_argument("--trade-markets", type=int, default=100, help="How many markets to fetch trades for.")
    parser.add_argument(
        "--resolved-trade-markets",
        type=int,
        default=100,
        help="How many recently resolved markets to fetch trades for.",
    )
    args = parser.parse_args()

    conn = get_connection()
    init_db(conn)
    client = PolymarketClient()
    pipeline = DataPipeline(conn, client)

    report = pipeline.run_market_snapshot(max_markets=args.max_markets)
    print(f"Snapshot: markets={report.markets} outcomes={report.outcomes} books={report.books}")

    res_report = pipeline.run_resolution_refresh()
    print(f"Resolution refresh: resolved={res_report.resolutions}")

    active_market_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT market_id
            FROM markets
            WHERE active = TRUE
              AND condition_id IS NOT NULL
            ORDER BY updated_ts DESC NULLS LAST
            LIMIT ?
            """,
            [args.trade_markets],
        ).fetchall()
    ]
    resolved_market_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT r.market_id
            FROM resolutions r
            JOIN markets m
                ON r.market_id = m.market_id
            WHERE m.condition_id IS NOT NULL
            ORDER BY r.resolved_ts DESC NULLS LAST
            LIMIT ?
            """,
            [args.resolved_trade_markets],
        ).fetchall()
    ]
    trade_market_ids = list(dict.fromkeys(active_market_ids + resolved_market_ids))
    trade_count = pipeline.ingest_trades(trade_market_ids)
    holder_count = pipeline.ingest_holders(active_market_ids)
    print(f"Trades ingested={trade_count}, holders ingested={holder_count}")

    ts_row = conn.execute(
        """
        SELECT
            COUNT(DISTINCT snapshot_ts) AS n_unique_ts,
            MIN(snapshot_ts) AS ts_min,
            MAX(snapshot_ts) AS ts_max
        FROM orderbook_snapshots
        """
    ).fetchone()
    n_unique_ts = int(ts_row[0]) if ts_row and ts_row[0] is not None else 0
    ts_min = ts_row[1] if ts_row else None
    ts_max = ts_row[2] if ts_row else None
    print(f"Panel timestamp depth: n_unique_ts={n_unique_ts} ts_min={ts_min} ts_max={ts_max}")


if __name__ == "__main__":
    main()
