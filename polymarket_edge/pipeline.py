from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.api import PolymarketClient
from polymarket_edge.db import upsert_dataframe


def _safe_load_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def _to_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if val < 1e11:
            ts = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
        elif val < 1e14:
            ts = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(value, unit="ns", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    ts = ts.floor("us")
    return ts.to_pydatetime().replace(tzinfo=None)


def _first_event_id(row: dict[str, Any]) -> str | None:
    events = row.get("events") or []
    if isinstance(events, list) and events:
        event = events[0]
        if isinstance(event, dict):
            return str(event.get("id")) if event.get("id") is not None else None
    return None


def normalize_markets(markets: list[dict[str, Any]], snapshot_ts: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    for row in markets:
        market_id = str(row.get("id"))
        tags = row.get("tags") or []
        if isinstance(tags, list):
            tags = [str(t.get("label", t)) if isinstance(t, dict) else str(t) for t in tags]
        event_id = _first_event_id(row)

        market_rows.append(
            {
                "market_id": market_id,
                "event_id": event_id,
                "slug": row.get("slug"),
                "question": row.get("question") or row.get("title"),
                "description": row.get("description"),
                "category": row.get("category"),
                "tags": json.dumps(tags),
                "condition_id": row.get("conditionId"),
                "start_ts": _to_ts(row.get("startDate")),
                "end_ts": _to_ts(row.get("endDate")),
                "close_ts": _to_ts(row.get("closeDate")),
                "active": bool(row.get("active", False)),
                "closed": bool(row.get("closed", False)),
                "archived": bool(row.get("archived", False)),
                "accepting_orders": bool(row.get("acceptingOrders", False)),
                "created_ts": _to_ts(row.get("createdAt")),
                "updated_ts": _to_ts(row.get("updatedAt")),
                "last_seen_ts": snapshot_ts,
                "raw_json": json.dumps(row),
            }
        )

        outcomes = _safe_load_list(row.get("outcomes"))
        token_ids = _safe_load_list(row.get("clobTokenIds"))
        token_info = row.get("tokens") if isinstance(row.get("tokens"), list) else []
        winner_by_token: dict[str, bool] = {}
        for tok in token_info:
            tok_id = str(tok.get("token_id") or tok.get("tokenId") or tok.get("id") or "")
            if tok_id:
                winner_by_token[tok_id] = bool(tok.get("winner", False))

        max_len = max(len(outcomes), len(token_ids))
        for idx in range(max_len):
            label = str(outcomes[idx]) if idx < len(outcomes) else None
            token_id = str(token_ids[idx]) if idx < len(token_ids) else None
            if token_id is None:
                continue
            outcome_rows.append(
                {
                    "market_id": market_id,
                    "token_id": token_id,
                    "outcome_index": idx,
                    "outcome_label": label,
                    "winner": winner_by_token.get(token_id, False),
                }
            )

    market_df = pd.DataFrame(market_rows)
    outcome_df = pd.DataFrame(outcome_rows)
    return market_df, outcome_df


def _extract_book_levels(levels: list[dict[str, Any]], side: str) -> tuple[float | None, float]:
    if not levels:
        return None, 0.0
    parsed: list[tuple[float, float]] = []
    for level in levels:
        try:
            p = float(level.get("price"))
            s = float(level.get("size"))
        except (TypeError, ValueError):
            continue
        parsed.append((p, s))
    if not parsed:
        return None, 0.0
    side_key = str(side).lower().strip()
    if side_key == "bid":
        parsed.sort(key=lambda x: x[0], reverse=True)
    else:
        parsed.sort(key=lambda x: x[0], reverse=False)
    best = float(parsed[0][0])
    depth = float(np.sum([s for _, s in parsed[:5]]))
    return best, depth


def normalize_books(
    books: list[dict[str, Any]],
    snapshot_ts: datetime,
    token_to_market: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for book in books:
        token_id = str(book.get("asset_id") or book.get("token_id") or book.get("market") or "")
        if not token_id:
            continue
        bid_px, bid_depth = _extract_book_levels(book.get("bids") or [], side="bid")
        ask_px, ask_depth = _extract_book_levels(book.get("asks") or [], side="ask")
        if bid_px is None and ask_px is None:
            continue
        if bid_px is not None and ask_px is not None:
            mid = 0.5 * (bid_px + ask_px)
            spread = ask_px - bid_px
        else:
            mid = bid_px if bid_px is not None else ask_px
            spread = 0.0
        spread_bps = (spread / mid) * 10000.0 if mid else None
        liquidity_score = min((bid_px or mid or 0.0) * bid_depth, (ask_px or mid or 0.0) * ask_depth)
        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "market_id": token_to_market.get(token_id),
                "token_id": token_id,
                "best_bid": bid_px,
                "best_ask": ask_px,
                "mid": mid,
                "spread": spread,
                "spread_bps": spread_bps,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "liquidity_score": liquidity_score,
            }
        )
    return pd.DataFrame(rows)


def normalize_trades(trades: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for t in trades:
        trade_id = str(t.get("id") or t.get("tradeID") or t.get("transactionHash") or "")
        market_id = t.get("market") or t.get("conditionId")
        token_id = t.get("asset") or t.get("tokenId") or t.get("token_id")
        side = str(t.get("side") or "").upper()
        try:
            price = float(t.get("price"))
            size = float(t.get("size"))
        except (TypeError, ValueError):
            continue
        notional = price * size
        trader = t.get("proxyWallet") or t.get("maker") or t.get("user")
        ts = _to_ts(t.get("timestamp") or t.get("lastUpdate"))
        rows.append(
            {
                "trade_id": trade_id,
                "market_id": str(market_id) if market_id is not None else None,
                "token_id": str(token_id) if token_id is not None else None,
                "trader": str(trader) if trader is not None else None,
                "side": side,
                "price": price,
                "size": size,
                "notional": notional,
                "trade_ts": ts,
                "tx_hash": t.get("transactionHash"),
                "raw_json": json.dumps(t),
            }
        )
    return pd.DataFrame(rows)


def normalize_holders(
    holders: list[dict[str, Any]],
    snapshot_ts: datetime,
    token_to_market: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for token_block in holders:
        token_id = str(token_block.get("token") or token_block.get("asset") or "")
        market_id = token_to_market.get(token_id)
        for row in token_block.get("holders", []) or []:
            wallet = row.get("proxyWallet") or row.get("wallet")
            if not wallet:
                continue
            shares = row.get("shares") or row.get("balance")
            pct = row.get("percent")
            try:
                shares_val = float(shares)
            except (TypeError, ValueError):
                shares_val = None
            try:
                pct_val = float(pct)
            except (TypeError, ValueError):
                pct_val = None
            rows.append(
                {
                    "snapshot_ts": snapshot_ts,
                    "market_id": market_id,
                    "wallet": str(wallet),
                    "outcome": str(row.get("outcome") or token_id),
                    "shares": shares_val,
                    "pct": pct_val,
                }
            )
    return pd.DataFrame(rows)


def extract_resolutions(simplified_markets: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolution_rows: list[dict[str, Any]] = []
    outcome_winner_rows: list[dict[str, Any]] = []
    for market in simplified_markets:
        market_id = str(market.get("id"))
        tokens = market.get("tokens") or []
        winner_token = None
        winner_label = None
        winner_idx = None

        for idx, token in enumerate(tokens):
            token_id = str(token.get("token_id") or token.get("tokenId") or token.get("id") or "")
            is_winner = bool(token.get("winner", False))
            if token_id:
                outcome_winner_rows.append(
                    {
                        "market_id": market_id,
                        "token_id": token_id,
                        "outcome_index": idx,
                        "outcome_label": token.get("outcome"),
                        "winner": is_winner,
                    }
                )
            if is_winner and winner_token is None:
                winner_token = token_id
                winner_label = token.get("outcome")
                winner_idx = idx

        if winner_token is None:
            continue

        resolved_ts = _to_ts(market.get("end_date_iso") or market.get("closeDate"))
        resolution_rows.append(
            {
                "market_id": market_id,
                "resolved_ts": resolved_ts,
                "winner_token_id": winner_token,
                "winner_outcome_index": winner_idx,
                "winner_label": winner_label,
                "resolution_source": market.get("resolve_source"),
                "raw_json": json.dumps(market),
            }
        )

    return pd.DataFrame(resolution_rows), pd.DataFrame(outcome_winner_rows)


def extract_resolutions_from_markets(markets: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolution_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    for market in markets:
        market_id = str(market.get("id"))
        outcomes = _safe_load_list(market.get("outcomes"))
        token_ids = _safe_load_list(market.get("clobTokenIds"))
        prices_raw = _safe_load_list(market.get("outcomePrices"))
        if not outcomes or not token_ids or not prices_raw:
            continue
        prices: list[float] = []
        for val in prices_raw:
            try:
                prices.append(float(val))
            except (TypeError, ValueError):
                prices.append(np.nan)
        if not prices or all(np.isnan(prices)):
            continue

        winner_idx = int(np.nanargmax(prices))
        winner_prob = prices[winner_idx]
        if not np.isfinite(winner_prob) or winner_prob < 0.95:
            continue

        winner_token_id = str(token_ids[winner_idx]) if winner_idx < len(token_ids) else None
        winner_label = str(outcomes[winner_idx]) if winner_idx < len(outcomes) else None
        if winner_token_id is None:
            continue

        for idx, token_id in enumerate(token_ids):
            token_str = str(token_id)
            outcome_rows.append(
                {
                    "market_id": market_id,
                    "token_id": token_str,
                    "outcome_index": idx,
                    "outcome_label": str(outcomes[idx]) if idx < len(outcomes) else None,
                    "winner": idx == winner_idx,
                }
            )

        resolution_rows.append(
            {
                "market_id": market_id,
                "resolved_ts": _to_ts(market.get("closedTime") or market.get("updatedAt") or market.get("endDate")),
                "winner_token_id": winner_token_id,
                "winner_outcome_index": winner_idx,
                "winner_label": winner_label,
                "resolution_source": market.get("resolutionSource"),
                "raw_json": json.dumps(market),
            }
        )

    return pd.DataFrame(resolution_rows), pd.DataFrame(outcome_rows)


@dataclass
class IngestionReport:
    markets: int = 0
    outcomes: int = 0
    books: int = 0
    trades: int = 0
    holders: int = 0
    resolutions: int = 0


class DataPipeline:
    def __init__(self, conn: duckdb.DuckDBPyConnection, client: PolymarketClient) -> None:
        self.conn = conn
        self.client = client

    def run_market_snapshot(self, *, max_markets: int | None = None) -> IngestionReport:
        snapshot_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        report = IngestionReport()

        markets = self.client.fetch_active_markets(max_rows=max_markets)
        market_df, outcome_df = normalize_markets(markets, snapshot_ts)
        report.markets = upsert_dataframe(self.conn, "markets", market_df)
        report.outcomes = upsert_dataframe(self.conn, "market_outcomes", outcome_df)

        if outcome_df.empty:
            return report
        token_to_market = dict(zip(outcome_df["token_id"], outcome_df["market_id"]))
        token_ids = sorted(token_to_market.keys())
        books = self.client.fetch_books(token_ids)
        book_df = normalize_books(books, snapshot_ts, token_to_market)
        report.books = upsert_dataframe(self.conn, "orderbook_snapshots", book_df)
        self.compute_returns()
        return report

    def run_resolution_refresh(self, *, max_rows: int = 2500) -> IngestionReport:
        report = IngestionReport()
        snapshot_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        closed_markets = self.client.fetch_markets(closed=True, archived=None, max_rows=max_rows)
        market_df, outcome_df = normalize_markets(closed_markets, snapshot_ts)
        report.markets = upsert_dataframe(self.conn, "markets", market_df)
        report.outcomes = upsert_dataframe(self.conn, "market_outcomes", outcome_df)
        res_df, winners_df = extract_resolutions_from_markets(closed_markets)
        report.resolutions = upsert_dataframe(self.conn, "resolutions", res_df)
        if not winners_df.empty:
            upsert_dataframe(self.conn, "market_outcomes", winners_df)
        return report

    def ingest_trades(self, market_ids: list[str], *, pages_per_market: int = 4, page_size: int = 500) -> int:
        if not market_ids:
            return 0
        market_values = ",".join([f"'{m}'" for m in market_ids])
        meta = self.conn.execute(
            f"""
            SELECT market_id, condition_id
            FROM markets
            WHERE market_id IN ({market_values})
              AND condition_id IS NOT NULL
            """
        ).df()
        if meta.empty:
            return 0
        cond_to_market = dict(zip(meta["condition_id"], meta["market_id"]))
        total = 0
        for row in meta.itertuples(index=False):
            market_id = row.market_id
            condition_id = row.condition_id
            for page in range(pages_per_market):
                offset = page * page_size
                trades = self.client.fetch_market_trades(condition_id, limit=page_size, offset=offset)
                if not trades:
                    break
                trade_df = normalize_trades(trades)
                if not trade_df.empty:
                    trade_df["market_id"] = trade_df["market_id"].map(cond_to_market).fillna(market_id)
                inserted = upsert_dataframe(self.conn, "trades", trade_df)
                total += inserted
                if len(trades) < page_size:
                    break
        return total

    def ingest_holders(self, market_ids: list[str], *, limit_per_market: int = 200) -> int:
        snapshot_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        if not market_ids:
            return 0
        market_values = ",".join([f"'{m}'" for m in market_ids])
        meta = self.conn.execute(
            f"""
            SELECT market_id, condition_id
            FROM markets
            WHERE market_id IN ({market_values})
              AND condition_id IS NOT NULL
            """
        ).df()
        token_map_df = self.conn.execute(
            f"""
            SELECT token_id, market_id
            FROM market_outcomes
            WHERE market_id IN ({market_values})
            """
        ).df()
        token_to_market = dict(zip(token_map_df["token_id"], token_map_df["market_id"]))

        total = 0
        for condition_id in meta["condition_id"].dropna().astype(str).unique():
            holders = self.client.fetch_market_holders([condition_id], limit=limit_per_market)
            holder_df = normalize_holders(holders, snapshot_ts, token_to_market)
            total += upsert_dataframe(self.conn, "holders", holder_df)
        return total

    def compute_returns(self) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO returns
            WITH ordered AS (
                SELECT
                    token_id,
                    snapshot_ts AS ts,
                    mid,
                    LAG(mid, 1) OVER (PARTITION BY token_id ORDER BY snapshot_ts) AS lag_1h,
                    LAG(mid, 6) OVER (PARTITION BY token_id ORDER BY snapshot_ts) AS lag_6h,
                    LAG(mid, 24) OVER (PARTITION BY token_id ORDER BY snapshot_ts) AS lag_24h
                FROM orderbook_snapshots
                WHERE mid IS NOT NULL
            )
            SELECT
                token_id,
                ts,
                CASE WHEN lag_1h > 0 THEN (mid / lag_1h) - 1 ELSE NULL END AS ret_1h,
                CASE WHEN lag_6h > 0 THEN (mid / lag_6h) - 1 ELSE NULL END AS ret_6h,
                CASE WHEN lag_24h > 0 THEN (mid / lag_24h) - 1 ELSE NULL END AS ret_24h
            FROM ordered
            """
        )
