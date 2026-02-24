from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.api import PolymarketClient
from polymarket_edge.db import get_connection, init_db
from polymarket_edge.pipeline import DataPipeline


def _active_market_ids(conn, limit: int) -> list[str]:
    return [
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
            [limit],
        ).fetchall()
    ]


def _resolved_market_ids(conn, limit: int) -> list[str]:
    return [
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
            [limit],
        ).fetchall()
    ]


def _panel_ts_stats(conn) -> tuple[int, str | None, str | None]:
    row = conn.execute(
        """
        SELECT
            COUNT(DISTINCT snapshot_ts) AS n_unique_ts,
            MIN(snapshot_ts) AS ts_min,
            MAX(snapshot_ts) AS ts_max
        FROM orderbook_snapshots
        """
    ).fetchone()
    n_unique = int(row[0]) if row and row[0] is not None else 0
    ts_min = str(row[1]) if row and row[1] is not None else None
    ts_max = str(row[2]) if row and row[2] is not None else None
    return n_unique, ts_min, ts_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated Polymarket ingestion cycles to build time-series depth.")
    parser.add_argument("--minutes", type=float, default=5.0, help="Minutes between ingestion cycles.")
    parser.add_argument("--hours", type=float, default=12.0, help="Total runtime duration in hours.")
    parser.add_argument("--max-markets", type=int, default=800, help="Max active markets per snapshot.")
    parser.add_argument("--trade-markets", type=int, default=100, help="Active markets to ingest trades for.")
    parser.add_argument(
        "--resolved-trade-markets",
        type=int,
        default=100,
        help="Recently resolved markets to ingest trades for.",
    )
    parser.add_argument("--max-cycles", type=int, default=0, help="Optional hard cap on cycles; 0 means unlimited within hours.")
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Only ingest market + orderbook snapshots (fast path for building timestamp depth).",
    )
    args = parser.parse_args()

    if args.minutes <= 0:
        raise SystemExit("--minutes must be > 0")
    if args.hours <= 0:
        raise SystemExit("--hours must be > 0")

    conn = get_connection()
    init_db(conn)
    client = PolymarketClient()
    pipeline = DataPipeline(conn, client)

    interval_s = int(args.minutes * 60.0)
    end_ts = time.time() + args.hours * 3600.0
    total_cycles = int(math.floor((args.hours * 60.0) / args.minutes))
    if args.max_cycles > 0:
        total_cycles = min(total_cycles, int(args.max_cycles))

    print(
        f"Ingestion loop start: interval={args.minutes:.2f}m "
        f"hours={args.hours:.2f} planned_cycles={total_cycles}"
    )

    cycle = 0
    while time.time() < end_ts and cycle < total_cycles:
        cycle += 1
        cycle_start = time.time()
        cycle_dt = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        print(f"[Cycle {cycle}/{total_cycles}] start_utc={cycle_dt}")

        snap = pipeline.run_market_snapshot(max_markets=args.max_markets)
        if args.snapshot_only:
            res_count = 0
            trades = 0
            holders = 0
        else:
            res = pipeline.run_resolution_refresh()
            res_count = int(res.resolutions)
            active_ids = _active_market_ids(conn, args.trade_markets)
            resolved_ids = _resolved_market_ids(conn, args.resolved_trade_markets)
            trade_market_ids = list(dict.fromkeys(active_ids + resolved_ids))
            trades = pipeline.ingest_trades(trade_market_ids)
            holders = pipeline.ingest_holders(active_ids)

        n_unique_ts, ts_min, ts_max = _panel_ts_stats(conn)
        print(
            f"[Cycle {cycle}] markets={snap.markets} outcomes={snap.outcomes} books={snap.books} "
            f"resolutions={res_count} trades={trades} holders={holders} "
            f"panel_n_unique_ts={n_unique_ts} ts_min={ts_min} ts_max={ts_max}"
        )

        elapsed = time.time() - cycle_start
        sleep_s = max(0.0, float(interval_s - elapsed))
        if cycle < total_cycles and sleep_s > 0:
            time.sleep(sleep_s)

    n_unique_ts, ts_min, ts_max = _panel_ts_stats(conn)
    print(f"Ingestion loop complete. panel_n_unique_ts={n_unique_ts} ts_min={ts_min} ts_max={ts_max}")


if __name__ == "__main__":
    main()
