from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency handling
    yaml = None


@dataclass(frozen=True)
class Settings:
    gamma_base_url: str = os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com")
    clob_base_url: str = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")
    data_base_url: str = os.getenv("POLYMARKET_DATA_URL", "https://data-api.polymarket.com")
    db_path: str = os.getenv("POLYMARKET_DB_PATH", "data/polymarket.duckdb")
    request_timeout_s: int = int(os.getenv("POLYMARKET_TIMEOUT_S", "20"))
    books_chunk_size: int = int(os.getenv("POLYMARKET_BOOKS_CHUNK", "100"))
    fee_bps: float = float(os.getenv("POLYMARKET_FEE_BPS", "20"))


settings = Settings()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def default_runtime_config() -> dict[str, Any]:
    return {
        "collection": {
            "minutes": 5,
            "hours": 24,
            "max_markets": 800,
            "with_trades": True,
            "trade_markets": 150,
            "resolved_trade_markets": 150,
        },
        "research": {
            "output_dir": "data/research_v2",
            "min_data": {
                "min_ts": 500,
                "min_mkts": 200,
                "min_snapshots": 50000,
                "min_test_trades": 300,
                "min_fold_trades": 50,
            },
            "ev_threshold": 0.0,
            "execution_regime": "base",
            "portfolio_method": "risk_parity",
            "strategies": {
                "cross_market_inefficiency_v2": True,
                "resolution_rule_mispricing_v2": True,
                "liquidity_premium_v2": True,
                "momentum_vs_mean_reversion_v2": True,
                "whale_behavior_v2": True,
                "consistency_arb": True,
                "microstructure_mm": True,
                "resolution_mechanics": True,
            },
        },
        "deployment_gate": {
            "min_ev_spearman": 0.2,
            "min_valid_folds": 3,
            "pbo_max": 0.5,
            "concentration_market_max_pnl_share": 0.4,
        },
    }


def load_runtime_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    cfg = default_runtime_config()
    if path is None:
        path = os.getenv("POLYMARKET_RUNTIME_CONFIG")
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files. Install `pyyaml`.")
    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    return _deep_merge(cfg, loaded)
