"""Retrieval-side metric helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from .common import round_float


def doc_prefix_from_chunk_id(chunk_id: str) -> str:
    if "__" not in chunk_id:
        return chunk_id
    return chunk_id.rsplit("__", 1)[0]


def retrieval_evidence_coverage(
    case: Dict[str, Any],
    retrieved_ids: Sequence[str],
    retrieved_prefixes: Sequence[str],
) -> Dict[str, Any]:
    claim_evidence = case.get("claim_evidence", [])
    if not isinstance(claim_evidence, list) or not claim_evidence:
        return {
            "claim_evidence_total": None,
            "claim_evidence_covered": None,
            "claim_evidence_coverage": None,
        }

    retrieved_id_set = set(retrieved_ids)
    retrieved_prefix_set = set(retrieved_prefixes)
    covered = 0

    for item in claim_evidence:
        if not isinstance(item, dict):
            continue
        ev_ids = [str(v).strip() for v in item.get("evidence_chunk_ids", []) if str(v).strip()]
        ev_prefixes = [
            str(v).strip() for v in item.get("evidence_doc_prefixes", []) if str(v).strip()
        ]
        hit = any(ev_id in retrieved_id_set for ev_id in ev_ids) or any(
            ev_prefix in retrieved_prefix_set for ev_prefix in ev_prefixes
        )
        if hit:
            covered += 1

    total = len(claim_evidence)
    return {
        "claim_evidence_total": total,
        "claim_evidence_covered": covered,
        "claim_evidence_coverage": round_float(covered / total if total else None),
    }


def precision_at_k(gold_prefixes: Sequence[str], retrieved_prefixes: Sequence[str]) -> float | None:
    if not gold_prefixes or not retrieved_prefixes:
        return None
    k = len(retrieved_prefixes)
    hits = sum(1 for p in retrieved_prefixes if p in set(gold_prefixes))
    return hits / k if k else None


def top1_hit(gold_prefixes: Sequence[str], retrieved_prefixes: Sequence[str]) -> float | None:
    if not gold_prefixes or not retrieved_prefixes:
        return None
    return 1.0 if retrieved_prefixes[0] in set(gold_prefixes) else 0.0


def mrr_at_k(gold_prefixes: Sequence[str], retrieved_prefixes: Sequence[str]) -> float | None:
    if not gold_prefixes or not retrieved_prefixes:
        return None
    gold_set = set(gold_prefixes)
    for idx, prefix in enumerate(retrieved_prefixes):
        if prefix in gold_set:
            return 1.0 / float(idx + 1)
    return 0.0


def ndcg_at_k(gold_prefixes: Sequence[str], retrieved_prefixes: Sequence[str]) -> float | None:
    """Binary relevance nDCG@k using doc prefixes."""
    if not gold_prefixes or not retrieved_prefixes:
        return None
    gold_set = set(gold_prefixes)
    dcg = 0.0
    for idx, prefix in enumerate(retrieved_prefixes):
        rel = 1.0 if prefix in gold_set else 0.0
        dcg += rel / (math.log2(idx + 2))

    # Ideal DCG with all relevant items ranked first
    ideal_rels = [1.0] * min(len(gold_set), len(retrieved_prefixes))
    idcg = sum(rel / (math.log2(i + 2)) for i, rel in enumerate(ideal_rels))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def unique_prefix_fraction(retrieved_prefixes: Sequence[str]) -> float | None:
    if not retrieved_prefixes:
        return None
    return len(set(retrieved_prefixes)) / len(retrieved_prefixes)
