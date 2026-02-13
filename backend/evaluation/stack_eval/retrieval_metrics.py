"""Retrieval-side metric helpers."""

from __future__ import annotations

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
