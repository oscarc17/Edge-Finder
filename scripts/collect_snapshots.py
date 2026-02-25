from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from polymarket_edge.api import PolymarketClient
from polymarket_edge.config import load_runtime_config
from polymarket_edge.db import get_connection, init_db
from polymarket_edge.pipeline import DataPipeline


def _panel_depth(conn) -> tuple[int, str | None, str | None]:
    row = conn.execute(
        """
        SELECT
            COUNT(DISTINCT snapshot_ts) AS n_unique_ts,
            MIN(snapshot_ts) AS ts_min,
            MAX(snapshot_ts) AS ts_max
        FROM orderbook_snapshots
        """
    ).fetchone()
    n_unique_ts = int(row[0]) if row and row[0] is not None else 0
    ts_min = str(row[1]) if row and row[1] is not None else None
    ts_max = str(row[2]) if row and row[2] is not None else None
    return n_unique_ts, ts_min, ts_max


def _both_legs_coverage(conn) -> float:
    try:
        row = conn.execute(
            """
            WITH x AS (
                SELECT
                    o.snapshot_ts,
                    o.market_id,
                    MAX(CASE WHEN lower(trim(coalesce(mo.outcome_label, '')))='yes' THEN 1 ELSE 0 END) AS has_yes,
                    MAX(CASE WHEN lower(trim(coalesce(mo.outcome_label, '')))='no' THEN 1 ELSE 0 END) AS has_no
                FROM orderbook_snapshots o
                LEFT JOIN market_outcomes mo
                  ON o.market_id = mo.market_id
                 AND o.token_id = mo.token_id
                WHERE o.snapshot_ts >= (SELECT MAX(snapshot_ts) - INTERVAL '24 hours' FROM orderbook_snapshots)
                GROUP BY 1,2
            )
            SELECT AVG(CASE WHEN has_yes=1 AND has_no=1 THEN 1.0 ELSE 0.0 END) FROM x
            """
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0
    except Exception:
        return 0.0


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


def main() -> None:
    runtime_cfg = load_runtime_config()
    coll_cfg = (runtime_cfg.get("collection") or {}) if isinstance(runtime_cfg, dict) else {}
    parser = argparse.ArgumentParser(description="Collect Polymarket market + orderbook snapshots on a schedule.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path (uses collection.* keys).")
    parser.add_argument("--minutes", type=float, default=float(coll_cfg.get("minutes", 5.0)), help="Polling interval in minutes.")
    parser.add_argument("--hours", type=float, default=float(coll_cfg.get("hours", 24.0)), help="Collection window in hours.")
    parser.add_argument("--max-markets", type=int, default=int(coll_cfg.get("max_markets", 800)), help="Max active markets per polling cycle.")
    parser.add_argument("--with-trades", action="store_true", help="Also ingest trades/holders each cycle.")
    parser.add_argument("--trade-markets", type=int, default=int(coll_cfg.get("trade_markets", 150)), help="Active markets to pull trades for when --with-trades is set.")
    parser.add_argument("--resolved-trade-markets", type=int, default=int(coll_cfg.get("resolved_trade_markets", 150)), help="Resolved markets to pull trades for when --with-trades is set.")
    parser.add_argument("--max-cycles", type=int, default=0, help="Optional hard cap on cycles. 0 means derived from hours/minutes.")
    args = parser.parse_args()

    if args.config:
        runtime_cfg = load_runtime_config(args.config)
        coll_cfg = (runtime_cfg.get("collection") or {}) if isinstance(runtime_cfg, dict) else {}
        # CLI still wins if explicitly set; argparse has no easy "was provided", so apply only when default values remain.
        if args.minutes == 5.0:
            args.minutes = float(coll_cfg.get("minutes", args.minutes))
        if args.hours == 24.0:
            args.hours = float(coll_cfg.get("hours", args.hours))
        if args.max_markets == 800:
            args.max_markets = int(coll_cfg.get("max_markets", args.max_markets))
        if args.trade_markets == 150:
            args.trade_markets = int(coll_cfg.get("trade_markets", args.trade_markets))
        if args.resolved_trade_markets == 150:
            args.resolved_trade_markets = int(coll_cfg.get("resolved_trade_markets", args.resolved_trade_markets))
        if not args.with_trades and bool(coll_cfg.get("with_trades", False)):
            args.with_trades = True

    if args.minutes <= 0:
        raise SystemExit("--minutes must be > 0")
    if args.hours <= 0:
        raise SystemExit("--hours must be > 0")

    conn = get_connection()
    init_db(conn)
    pipeline = DataPipeline(conn, PolymarketClient())

    total_cycles = int(math.floor((args.hours * 60.0) / args.minutes))
    if args.max_cycles > 0:
        total_cycles = min(total_cycles, int(args.max_cycles))
    if total_cycles <= 0:
        total_cycles = 1
    interval_s = args.minutes * 60.0
    end_ts = time.time() + args.hours * 3600.0

    print(
        f"collect_snapshots start: interval={args.minutes:.2f}m "
        f"hours={args.hours:.2f} planned_cycles={total_cycles} with_trades={args.with_trades}"
    )

    for i in range(total_cycles):
        if time.time() >= end_ts:
            break
        cycle = i + 1
        start = time.time()
        cycle_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        snap = pipeline.run_market_snapshot(max_markets=args.max_markets)

        res_count = 0
        trade_count = 0
        holder_count = 0
        if args.with_trades:
            try:
                res = pipeline.run_resolution_refresh()
                res_count = int(res.resolutions)
                active_ids = _active_market_ids(conn, args.trade_markets)
                resolved_ids = _resolved_market_ids(conn, args.resolved_trade_markets)
                trade_market_ids = list(dict.fromkeys(active_ids + resolved_ids))
                trade_count = int(pipeline.ingest_trades(trade_market_ids))
                try:
                    holder_count = int(pipeline.ingest_holders(active_ids))
                except Exception as exc:
                    holder_count = 0
                    print(f"warning: ingest_holders failed and was skipped: {type(exc).__name__}: {exc}")
            except Exception as exc:
                print(f"warning: trades/holders cycle failed and was skipped: {type(exc).__name__}: {exc}")

        n_unique_ts, ts_min, ts_max = _panel_depth(conn)
        both_legs_cov = _both_legs_coverage(conn)
        print(
            f"[Cycle {cycle}/{total_cycles}] utc={cycle_ts} "
            f"markets={snap.markets} outcomes={snap.outcomes} books={snap.books} "
            f"resolutions={res_count} trades={trade_count} holders={holder_count} "
            f"n_unique_ts={n_unique_ts} both_legs_cov_24h={both_legs_cov:.2%} ts_min={ts_min} ts_max={ts_max}"
        )

        elapsed = time.time() - start
        sleep_s = max(0.0, interval_s - elapsed)
        if cycle < total_cycles and sleep_s > 0:
            time.sleep(sleep_s)

    n_unique_ts, ts_min, ts_max = _panel_depth(conn)
    print(f"collect_snapshots complete: n_unique_ts={n_unique_ts} ts_min={ts_min} ts_max={ts_max}")


if __name__ == "__main__":
    main()
