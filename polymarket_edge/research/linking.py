from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

import duckdb
import numpy as np
import pandas as pd

from polymarket_edge.db import upsert_dataframe


TOKEN_RE = re.compile(r"[a-z0-9]+")
DATE_HINT_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?\b|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LinkingConfig:
    max_markets: int = 1500
    min_title_similarity: float = 0.70
    min_token_jaccard: float = 0.55
    max_pairs_per_market: int = 12


def _norm_text(s: object) -> str:
    txt = str(s or "").lower().strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def _tokens(s: str) -> set[str]:
    return set(TOKEN_RE.findall(s))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _seq(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _relation_type(a: pd.Series, b: pd.Series, *, sim: float, jac: float) -> str:
    if str(a.get("event_id") or "") and str(a.get("event_id")) == str(b.get("event_id")):
        return "same_event"
    qa = _norm_text(a.get("question"))
    qb = _norm_text(b.get("question"))
    da = DATE_HINT_RE.findall(qa)
    db = DATE_HINT_RE.findall(qb)
    if da and db and (sim >= 0.7 or jac >= 0.5):
        return "time_window_consistency_candidate"
    return "fuzzy_related" if (sim >= 0.8 or jac >= 0.65) else "weak_related"


def build_market_links_frame(conn: duckdb.DuckDBPyConnection, cfg: LinkingConfig = LinkingConfig()) -> pd.DataFrame:
    markets = conn.execute(
        """
        SELECT
            market_id,
            event_id,
            question,
            description,
            category,
            close_ts,
            created_ts,
            active
        FROM markets
        WHERE question IS NOT NULL
        ORDER BY COALESCE(updated_ts, created_ts) DESC NULLS LAST
        LIMIT ?
        """,
        [int(cfg.max_markets)],
    ).df()
    if markets.empty:
        return pd.DataFrame()

    m = markets.copy()
    m["question_norm"] = m["question"].map(_norm_text)
    m["desc_norm"] = m["description"].map(_norm_text)
    m["tok"] = m["question_norm"].map(_tokens)
    m["cat"] = m["category"].fillna("unknown").astype(str)
    m["event_id"] = m["event_id"].astype(str)

    rows: list[dict[str, object]] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Candidate pools: same category or same event to avoid O(n^2) blowup.
    for idx, a in m.iterrows():
        if str(a.get("market_id") or "") == "":
            continue
        if str(a.get("event_id") or "") and str(a.get("event_id")).lower() != "none":
            pool = m[(m["event_id"] == a["event_id"]) & (m["market_id"] != a["market_id"])].copy()
        else:
            pool = m[(m["cat"] == a["cat"]) & (m["market_id"] != a["market_id"])].copy()
        if pool.empty:
            continue
        scored: list[tuple[float, int]] = []
        for jdx, b in pool.iterrows():
            sim = _seq(str(a["question_norm"]), str(b["question_norm"]))
            jac = _jaccard(a["tok"], b["tok"])
            if max(sim, jac) < min(cfg.min_title_similarity, cfg.min_token_jaccard):
                continue
            score = 0.6 * sim + 0.4 * jac
            scored.append((score, int(jdx)))
        if not scored:
            continue
        scored.sort(reverse=True)
        for score, jdx in scored[: int(cfg.max_pairs_per_market)]:
            b = m.loc[jdx]
            sim = _seq(str(a["question_norm"]), str(b["question_norm"]))
            jac = _jaccard(a["tok"], b["tok"])
            rel = _relation_type(a, b, sim=sim, jac=jac)
            a_id = str(a["market_id"])
            b_id = str(b["market_id"])
            lo, hi = sorted([a_id, b_id])
            link_id = hashlib.md5(f"{lo}|{hi}|{rel}".encode("utf-8")).hexdigest()
            rows.append(
                {
                    "link_id": link_id,
                    "market_id_a": lo,
                    "market_id_b": hi,
                    "relation_type": rel,
                    "link_confidence": float(np.clip(score, 0.0, 1.0)),
                    "heuristic_source": "event_id+fuzzy_title",
                    "created_ts": now,
                    "updated_ts": now,
                    "metadata_json": json.dumps(
                        {
                            "title_similarity": round(sim, 6),
                            "token_jaccard": round(jac, 6),
                            "category_a": str(a.get("category") or ""),
                            "category_b": str(b.get("category") or ""),
                            "event_id_a": str(a.get("event_id") or ""),
                            "event_id_b": str(b.get("event_id") or ""),
                        }
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates(subset=["link_id"]).sort_values("link_confidence", ascending=False)
    return out


def refresh_market_links(conn: duckdb.DuckDBPyConnection, cfg: LinkingConfig = LinkingConfig()) -> pd.DataFrame:
    links = build_market_links_frame(conn, cfg=cfg)
    if not links.empty:
        upsert_dataframe(conn, "market_links", links)
    return links

