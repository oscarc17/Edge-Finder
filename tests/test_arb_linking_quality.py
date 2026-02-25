from __future__ import annotations

import unittest
from datetime import datetime

import duckdb
import pandas as pd

from polymarket_edge.db import init_db, upsert_dataframe
from polymarket_edge.research.arb_data_quality import ArbDataQualityThresholds, build_arb_data_quality
from polymarket_edge.research.linking import refresh_outcome_links


class ArbLinkingQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = duckdb.connect(":memory:")
        init_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def test_outcome_links_and_both_leg_coverage(self) -> None:
        ts = datetime(2026, 2, 24, 12, 0, 0)
        markets = pd.DataFrame(
            [
                {"market_id": "m1", "event_id": "e1", "slug": "a", "question": "A?", "description": "", "category": "politics", "tags": "", "condition_id": "c1", "start_ts": None, "end_ts": None, "close_ts": None, "active": True, "closed": False, "archived": False, "accepting_orders": True, "created_ts": ts, "updated_ts": ts, "last_seen_ts": ts, "raw_json": "{}"},
                {"market_id": "m2", "event_id": "e2", "slug": "b", "question": "B?", "description": "", "category": "sports", "tags": "", "condition_id": "c2", "start_ts": None, "end_ts": None, "close_ts": None, "active": True, "closed": False, "archived": False, "accepting_orders": True, "created_ts": ts, "updated_ts": ts, "last_seen_ts": ts, "raw_json": "{}"},
            ]
        )
        outcomes = pd.DataFrame(
            [
                {"market_id": "m1", "token_id": "y1", "outcome_index": 0, "outcome_label": "Yes", "winner": False},
                {"market_id": "m1", "token_id": "n1", "outcome_index": 1, "outcome_label": "No", "winner": False},
                {"market_id": "m2", "token_id": "y2", "outcome_index": 0, "outcome_label": "Yes", "winner": False},
            ]
        )
        snaps = pd.DataFrame(
            [
                {"snapshot_ts": ts, "market_id": "m1", "token_id": "y1", "best_bid": 0.48, "best_ask": 0.49, "mid": 0.485, "spread": 0.01, "spread_bps": 206.0, "bid_depth": 100.0, "ask_depth": 100.0, "liquidity_score": 1000.0},
                {"snapshot_ts": ts, "market_id": "m1", "token_id": "n1", "best_bid": 0.48, "best_ask": 0.49, "mid": 0.485, "spread": 0.01, "spread_bps": 206.0, "bid_depth": 100.0, "ask_depth": 100.0, "liquidity_score": 1000.0},
                {"snapshot_ts": ts, "market_id": "m2", "token_id": "y2", "best_bid": 0.55, "best_ask": 0.56, "mid": 0.555, "spread": 0.01, "spread_bps": 180.0, "bid_depth": 80.0, "ask_depth": 90.0, "liquidity_score": 500.0},
            ]
        )
        upsert_dataframe(self.conn, "markets", markets)
        upsert_dataframe(self.conn, "market_outcomes", outcomes)
        upsert_dataframe(self.conn, "orderbook_snapshots", snaps)

        ol = refresh_outcome_links(self.conn)
        self.assertEqual(len(ol), 1)
        self.assertEqual(str(ol.iloc[0]["market_id"]), "m1")

        q = build_arb_data_quality(self.conn, thresholds=ArbDataQualityThresholds(min_both_legs_coverage=0.8, recent_hours=72.0))
        self.assertFalse(bool(q["passed"]))
        self.assertAlmostEqual(float(q["both_legs_coverage"]), 0.5, places=3)
        report = q["report"]
        self.assertTrue(isinstance(report, pd.DataFrame))
        self.assertIn("metric", report.columns)


if __name__ == "__main__":
    unittest.main()

