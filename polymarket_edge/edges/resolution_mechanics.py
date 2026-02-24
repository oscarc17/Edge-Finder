from __future__ import annotations

from dataclasses import dataclass
import re

import duckdb
import numpy as np
import pandas as pd


AMBIGUOUS_TERMS = [
    "official",
    "reported",
    "declared",
    "according to",
    "confirmed",
    "if announced",
    "if approved",
    "unless",
    "deemed",
]
SOURCE_TERMS = [
    "ap",
    "reuters",
    "associated press",
    "official website",
    "sec",
    "federal reserve",
    "court",
    "cdc",
    "who",
    "election board",
]


@dataclass(frozen=True)
class ResolutionMechanicsConfig:
    min_rule_risk_score: float = 0.50
    min_clarity_edge: float = 0.05
    max_candidates: int = 200


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def _score_text(question: str, description: str) -> tuple[float, float, float, int]:
    q = _norm(question)
    d = _norm(description)
    text = f"{q} {d}".strip()
    ambiguous_hits = sum(1 for term in AMBIGUOUS_TERMS if term in text)
    source_hits = sum(1 for term in SOURCE_TERMS if term in text)
    has_source_anchor = 1.0 if source_hits > 0 else 0.0
    has_date = 1.0 if re.search(r"\b\d{4}\b|\bjan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec\b", text) else 0.0
    rule_risk = float(np.clip(0.15 * ambiguous_hits + 0.10 * (1.0 - has_source_anchor) + 0.05 * (1.0 - has_date), 0.0, 1.0))
    clarity = float(np.clip(0.30 * has_source_anchor + 0.20 * has_date + 0.05 * max(0, 6 - ambiguous_hits), 0.0, 1.0))
    lag_risk = float(np.clip(0.15 * ambiguous_hits + 0.10 * (source_hits > 1), 0.0, 1.0))
    return rule_risk, clarity, lag_risk, ambiguous_hits


def _precedent_features(conn: duckdb.DuckDBPyConnection, active: pd.DataFrame) -> pd.DataFrame:
    if active.empty:
        return pd.DataFrame(columns=["market_id", "precedent_count", "precedent_rule_risk_mean"])
    resolved = conn.execute(
        """
        SELECT m.market_id, m.category, m.question, m.description, r.winner_label
        FROM markets m
        JOIN resolutions r ON m.market_id = r.market_id
        WHERE m.question IS NOT NULL
        """
    ).df()
    if resolved.empty:
        out = active[["market_id"]].copy()
        out["precedent_count"] = 0.0
        out["precedent_rule_risk_mean"] = 0.0
        return out

    res = resolved.copy()
    res["q_norm"] = res["question"].map(_norm)
    res["category"] = res["category"].fillna("unknown")
    rows = []
    for r in active.itertuples(index=False):
        q = _norm(getattr(r, "question", ""))
        cat = str(getattr(r, "category", "unknown") or "unknown")
        cand = res[res["category"].astype(str) == cat].copy()
        if cand.empty:
            rows.append({"market_id": r.market_id, "precedent_count": 0.0, "precedent_rule_risk_mean": 0.0})
            continue
        toks = set(q.split())
        if not toks:
            rows.append({"market_id": r.market_id, "precedent_count": 0.0, "precedent_rule_risk_mean": 0.0})
            continue
        scores = []
        for rr in cand.itertuples(index=False):
            rtoks = set(str(rr.q_norm).split())
            sim = float(len(toks & rtoks) / max(1, len(toks | rtoks)))
            if sim < 0.25:
                continue
            rule_risk, _clarity, _lag, _hits = _score_text(rr.question, rr.description)
            scores.append((sim, rule_risk))
        if not scores:
            rows.append({"market_id": r.market_id, "precedent_count": 0.0, "precedent_rule_risk_mean": 0.0})
            continue
        rows.append(
            {
                "market_id": r.market_id,
                "precedent_count": float(len(scores)),
                "precedent_rule_risk_mean": float(np.mean([x[1] for x in scores])),
            }
        )
    return pd.DataFrame(rows)


def run(conn: duckdb.DuckDBPyConnection, config: ResolutionMechanicsConfig = ResolutionMechanicsConfig()) -> dict[str, pd.DataFrame]:
    active = conn.execute(
        """
        SELECT
            m.market_id,
            m.question,
            m.description,
            m.category,
            m.close_ts,
            m.created_ts,
            m.active,
            COALESCE(avg_yes.mid_yes, latest_any.mid_any, 0.5) AS market_prob_yes,
            m.raw_json
        FROM markets m
        LEFT JOIN (
            SELECT market_id, AVG(mid) AS mid_yes
            FROM (
                SELECT o.market_id, o.mid
                FROM orderbook_snapshots o
                JOIN market_outcomes mo
                  ON o.market_id = mo.market_id AND o.token_id = mo.token_id
                WHERE lower(coalesce(mo.outcome_label, '')) = 'yes'
            ) y
            GROUP BY market_id
        ) avg_yes ON m.market_id = avg_yes.market_id
        LEFT JOIN (
            SELECT market_id, AVG(mid) AS mid_any
            FROM orderbook_snapshots
            GROUP BY market_id
        ) latest_any ON m.market_id = latest_any.market_id
        WHERE m.active = TRUE
          AND m.question IS NOT NULL
        """
    ).df()
    if active.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "candidates": empty, "human_review": empty}

    rows = []
    for r in active.itertuples(index=False):
        rule_risk, clarity, lag_risk, ambiguous_hits = _score_text(r.question, r.description)
        rows.append(
            {
                "market_id": r.market_id,
                "question": r.question,
                "category": r.category,
                "market_prob_yes": float(pd.to_numeric(r.market_prob_yes, errors="coerce")),
                "rule_risk_score": rule_risk,
                "rule_clarity_score": clarity,
                "resolution_lag_risk": lag_risk,
                "ambiguous_term_hits": float(ambiguous_hits),
                "human_review_required": 1 if rule_risk >= config.min_rule_risk_score else 0,
            }
        )
    scored = pd.DataFrame(rows)
    preced = _precedent_features(conn, active[["market_id", "question", "category"]])
    scored = scored.merge(preced, on="market_id", how="left")
    scored["precedent_count"] = scored["precedent_count"].fillna(0.0)
    scored["precedent_rule_risk_mean"] = scored["precedent_rule_risk_mean"].fillna(0.0)
    # Conservative "edge" estimate: clarity minus risk, amplified when market is near certainty on ambiguous wording.
    extremeness = (scored["market_prob_yes"] - 0.5).abs() * 2.0
    scored["rule_edge_score"] = (
        scored["rule_clarity_score"]
        - scored["rule_risk_score"]
        + 0.25 * extremeness
        - 0.10 * scored["precedent_rule_risk_mean"]
    ).clip(lower=-1.0, upper=1.0)
    scored["expected_net_return"] = (0.15 * scored["rule_edge_score"] - 0.02).clip(lower=-0.50, upper=0.50)
    scored["rationale"] = np.where(
        scored["rule_edge_score"] > 0,
        "price_extreme_vs_rule_clarity_or_precedent",
        "no_clear_rule_edge",
    )

    candidates = scored[
        (scored["human_review_required"] == 1)
        & (scored["expected_net_return"] >= float(config.min_clarity_edge))
    ].copy()
    candidates = candidates.sort_values("expected_net_return", ascending=False).head(int(config.max_candidates))
    human_review = candidates.copy()
    if not human_review.empty:
        human_review["review_status"] = "PENDING"
        human_review["approval_required"] = 1

    summary = pd.DataFrame(
        [
            {
                "module": "resolution_mechanics",
                "n_active_markets_scored": float(len(scored)),
                "n_human_review_candidates": float(len(human_review)),
                "avg_rule_risk_score": float(scored["rule_risk_score"].mean()) if len(scored) else 0.0,
                "avg_rule_clarity_score": float(scored["rule_clarity_score"].mean()) if len(scored) else 0.0,
                "avg_expected_net_return_candidates": float(candidates["expected_net_return"].mean()) if len(candidates) else 0.0,
            }
        ]
    )
    return {"summary": summary, "candidates": candidates, "human_review": human_review}

