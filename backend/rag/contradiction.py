"""Conflict detection/classification on retrieved policy chunks."""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import cohere

from .app_config import (
    CONTRADICTION_CLASSIFIER_MODEL,
    CONTRADICTION_MAX_PAIRS,
    CONTRADICTION_MIN_SIMILARITY,
    CONTRADICTION_MIN_TOKEN_OVERLAP,
    ENABLE_CONTRADICTION_LLM_CLASSIFIER,
)
from .embedding_client import embed_texts
from .rag_types import Chunk

LABELS = {
    "direct_contradiction",
    "partial_overlap",
    "amplification",
    "conditional_applicability",
    "no_conflict",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STRONG_OBLIGATION = {"must", "shall", "required", "require", "mandatory"}
STRONG_PROHIBITION = {
    "must not",
    "shall not",
    "prohibited",
    "not permitted",
    "not allowed",
    "cannot",
}
WEAK_MODAL = {"may", "can", "should"}
CONDITIONAL_MARKERS = {"if", "when", "unless", "except", "provided", "subject to"}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _content_tokens(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 2]


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(_content_tokens(a))
    b_tokens = set(_content_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / float(min(len(a_tokens), len(b_tokens)))


def _best_sentence(text: str) -> str:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return text.strip()[:320]
    return max(sentences, key=len)[:320]


def _contains_any_phrase(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _heuristic_label(text_a: str, text_b: str, sim: float, overlap: float) -> str:
    if sim < CONTRADICTION_MIN_SIMILARITY and overlap < CONTRADICTION_MIN_TOKEN_OVERLAP:
        return "no_conflict"

    a = _normalize_text(text_a)
    b = _normalize_text(text_b)
    a_prohibit = _contains_any_phrase(a, STRONG_PROHIBITION)
    b_prohibit = _contains_any_phrase(b, STRONG_PROHIBITION)
    a_strong_req = _contains_any_phrase(a, STRONG_OBLIGATION)
    b_strong_req = _contains_any_phrase(b, STRONG_OBLIGATION)
    a_weak = _contains_any_phrase(a, WEAK_MODAL)
    b_weak = _contains_any_phrase(b, WEAK_MODAL)
    conditional = _contains_any_phrase(a, CONDITIONAL_MARKERS) or _contains_any_phrase(
        b, CONDITIONAL_MARKERS
    )

    if overlap >= CONTRADICTION_MIN_TOKEN_OVERLAP and (
        (a_prohibit and (b_strong_req or b_weak))
        or (b_prohibit and (a_strong_req or a_weak))
    ):
        return "direct_contradiction"

    if conditional and (sim >= CONTRADICTION_MIN_SIMILARITY or overlap >= CONTRADICTION_MIN_TOKEN_OVERLAP):
        return "conditional_applicability"

    if overlap >= CONTRADICTION_MIN_TOKEN_OVERLAP and (
        (a_strong_req and b_weak) or (b_strong_req and a_weak)
    ):
        return "amplification"

    if sim >= CONTRADICTION_MIN_SIMILARITY or overlap >= CONTRADICTION_MIN_TOKEN_OVERLAP:
        return "partial_overlap"
    return "no_conflict"


def _llm_refine_labels(
    client: cohere.Client,
    pair_rows: List[Dict[str, Any]],
    model: str,
) -> Dict[str, str]:
    if not pair_rows:
        return {}

    lines = []
    for idx, row in enumerate(pair_rows, start=1):
        lines.append(
            f"PAIR_{idx}\nA[{row['a_chunk_id']}]: {row['a_excerpt']}\nB[{row['b_chunk_id']}]: {row['b_excerpt']}"
        )
    prompt = (
        "Classify each pair with exactly one label from: "
        "direct_contradiction, partial_overlap, amplification, conditional_applicability, no_conflict.\n"
        "Return strict JSON object mapping PAIR_n -> label only.\n\n"
        + "\n\n".join(lines)
    )
    try:
        resp = client.chat(model=model, message=prompt, max_tokens=350, temperature=0.0)
        raw = (resp.text or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        payload = json.loads(raw[start : end + 1])
        refined: Dict[str, str] = {}
        for k, v in payload.items():
            label = str(v).strip()
            if label in LABELS:
                refined[str(k)] = label
        return refined
    except Exception:
        return {}


def analyze_conflicts(
    client: cohere.Client,
    retrieved: Sequence[Tuple[Chunk, float]],
    max_pairs: int = CONTRADICTION_MAX_PAIRS,
    use_llm_classifier: bool = ENABLE_CONTRADICTION_LLM_CLASSIFIER,
    classifier_model: str = CONTRADICTION_CLASSIFIER_MODEL,
) -> List[Dict[str, Any]]:
    """Classify relationship between retrieved chunk pairs."""
    if len(retrieved) < 2:
        return []

    chunks = [chunk for chunk, _ in retrieved]
    texts = [chunk.text for chunk in chunks]
    vecs = embed_texts(client, texts, input_type="search_document")

    candidate_pairs: List[Tuple[int, int, float, float]] = []
    for i, j in combinations(range(len(chunks)), 2):
        sim = float(vecs[i] @ vecs[j])
        overlap = _token_overlap_ratio(chunks[i].text, chunks[j].text)
        if sim >= CONTRADICTION_MIN_SIMILARITY or overlap >= CONTRADICTION_MIN_TOKEN_OVERLAP:
            candidate_pairs.append((i, j, sim, overlap))

    candidate_pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
    candidate_pairs = candidate_pairs[: max(0, max_pairs)]

    rows: List[Dict[str, Any]] = []
    for i, j, sim, overlap in candidate_pairs:
        a = chunks[i]
        b = chunks[j]
        label = _heuristic_label(a.text, b.text, sim, overlap)
        rows.append(
            {
                "pair_key": f"PAIR_{len(rows) + 1}",
                "label": label,
                "confidence": round((sim + overlap) / 2.0, 4),
                "embedding_similarity": round(sim, 4),
                "token_overlap": round(overlap, 4),
                "a_chunk_id": a.chunk_id,
                "b_chunk_id": b.chunk_id,
                "a_excerpt": _best_sentence(a.text),
                "b_excerpt": _best_sentence(b.text),
            }
        )

    if use_llm_classifier and rows:
        overrides = _llm_refine_labels(client, rows, model=classifier_model)
        for row in rows:
            label = overrides.get(row["pair_key"])
            if label in LABELS:
                row["label"] = label

    return rows


def build_conflict_prompt_section(conflicts: Sequence[Dict[str, Any]]) -> str:
    """
    Build the conflict section to inject into model prompt.
    Always emits a section, even when no conflicts were found.
    """
    lines = ["Conflict detected:"]
    surfaced = [c for c in conflicts if c.get("label") != "no_conflict"]
    if not surfaced:
        lines.append("- none identified in the retrieved excerpts.")
        return "\n".join(lines)

    for row in surfaced[:5]:
        lines.append(
            "- "
            f"{row['label']} between [{row['a_chunk_id']}] and [{row['b_chunk_id']}] "
            f"(confidence={row['confidence']})."
        )
    lines.append("When conflicts appear, explicitly label them in your answer.")
    return "\n".join(lines)
