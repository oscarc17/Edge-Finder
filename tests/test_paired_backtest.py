from __future__ import annotations

import unittest

import pandas as pd

from polymarket_edge.research.paired_backtest import PairedBacktestConfig, run_paired_backtest_grid


class PairedBacktestTests(unittest.TestCase):
    def test_complement_parity_long_set_positive_edge(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ts": "2026-02-24T12:00:00",
                    "market_id": "m1",
                    "event_id": "e1",
                    "category": "politics",
                    "signal": 0.04,
                    "confidence": 0.95,
                    "ask_yes": 0.48,
                    "ask_no": 0.48,
                    "bid_yes": 0.47,
                    "bid_no": 0.47,
                    "spread_yes": 0.01,
                    "spread_no": 0.01,
                    "depth_yes": 5000.0,
                    "depth_no": 5000.0,
                    "pair_staleness_ms": 0.0,
                    "pair_volatility": 0.0,
                    "prob_vol_pair": 0.0,
                    "trade_frequency_pair_h": 0.0,
                    "edge_raw": 0.04,
                    "long_set_edge_raw": 0.04,
                    "short_set_edge_raw": -0.06,
                    "future_bid_yes_1": 0.49,
                    "future_ask_yes_1": 0.50,
                    "future_bid_no_1": 0.49,
                    "future_ask_no_1": 0.50,
                }
            ]
        )
        cfg = PairedBacktestConfig(
            holding_periods=(1,),
            fee_bps=0.0,
            impact_coeff=0.0,
            max_impact=0.0,
            vol_slippage_coeff=0.0,
            fill_risk_buffer_bps=0.0,
            fill_alpha_liquidity=5.0,
            fill_beta_spread=0.0,
            fill_gamma_flow=0.0,
            base_trade_notional=100.0,
            max_participation=0.5,
        )
        summary, trades = run_paired_backtest_grid(frame, thresholds=[0.01], cfg=cfg)
        self.assertFalse(trades.empty)
        t = trades.iloc[0]
        self.assertEqual(str(t["pair_direction"]), "LONG_SET")
        self.assertGreater(float(t["expected_ev"]), 0.0)
        self.assertGreater(float(t["trade_return"]), 0.0)
        self.assertGreaterEqual(float(t["realized_pair_fill_pct"]), 0.0)

        s = summary.iloc[0]
        self.assertGreater(float(s["expectancy"]), 0.0)
        self.assertGreaterEqual(float(s["hedged_fill_rate"]), 0.0)


if __name__ == "__main__":
    unittest.main()

