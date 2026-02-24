from __future__ import annotations

import unittest

import pandas as pd

from polymarket_edge.scoring import score_edges


class ScoringTests(unittest.TestCase):
    def test_negative_mean_not_rewarded_by_significance(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "edge_name": "loser",
                    "p_value": 1e-8,
                    "p_value_pos": 0.999,
                    "t_stat": -10.0,
                    "n_obs": 400,
                    "ci_low": -0.01,
                    "capacity_usd": 100_000.0,
                    "stability": 0.9,
                    "mean_return": -0.01,
                },
                {
                    "edge_name": "winner",
                    "p_value": 0.02,
                    "p_value_pos": 0.01,
                    "t_stat": 2.5,
                    "n_obs": 400,
                    "ci_low": 0.0002,
                    "capacity_usd": 10_000.0,
                    "stability": 0.5,
                    "mean_return": 0.0008,
                },
            ]
        )
        scored = score_edges(frame)
        self.assertIn("p_value_pos", scored.columns)
        self.assertIn("profitability_score", scored.columns)
        self.assertIn("edge_score_pro", scored.columns)

        loser = scored[scored["edge_name"] == "loser"].iloc[0]
        winner = scored[scored["edge_name"] == "winner"].iloc[0]
        self.assertEqual(float(loser["significance_score"]), 0.0)
        self.assertGreater(float(winner["edge_score_pro"]), float(loser["edge_score_pro"]))


if __name__ == "__main__":
    unittest.main()

