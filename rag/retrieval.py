"""Minimal hybrid retrieval over embedded policy chunks."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import cohere
import numpy as np

from .app_config import (
    ENABLE_RERANK,
    MAX_CHUNKS_PER_SOURCE,
    RERANK_ALPHA,
    RERANK_MODEL,
    RETRIEVAL_ALPHA,
)
from .embedding_client import embed_texts
from .rag_types import Chunk

LOGGER = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
CLAUSE_SPLIT_RE = re.compile(r"[?;:]|\b(?:and|or|but|while)\b", flags=re.IGNORECASE)
MAX_CLAUSE_COVERAGE = 2
CLAUSE_COVERAGE_MIN_OVERLAP = 0.45

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = NON_ALNUM_RE.sub(" ", lowered)
    return SPACE_RE.sub(" ", cleaned).strip()


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(_normalize_text(text))


def _content_tokens(text: str) -> List[str]:
    return [token for token in _tokenize(text) if len(token) > 2 and token not in STOPWORDS]


def _merge_query_texts(query: str, query_expansions: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [query, *(query_expansions or [])]:
        normalized = value.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(normalized)
    return merged or [query]


def _lexical_overlap_vocab(query_tokens: List[str], chunk_vocab: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    query_vocab = set(query_tokens)
    overlap = sum(1 for token in query_vocab if token in chunk_vocab)
    return overlap / max(1, len(query_vocab))


def _compute_dense_scores(
    client: cohere.Client,
    query_texts: List[str],
    chunk_vecs: np.ndarray,
) -> np.ndarray:
    qvecs = embed_texts(client, query_texts, input_type="search_query")
    dense_matrix = chunk_vecs @ qvecs.T
    if dense_matrix.shape[1] == 1:
        dense_scores = dense_matrix[:, 0]
    else:
        dense_scores = np.max(dense_matrix, axis=1)
    return (dense_scores + 1.0) / 2.0


def _extract_query_clauses(query: str, max_clauses: int = 3) -> List[str]:
    normalized = SPACE_RE.sub(" ", query or "").strip()
    if not normalized:
        return []

    clauses: List[str] = []
    seen = set()
    for value in CLAUSE_SPLIT_RE.split(normalized):
        clause = value.strip(" ,")
        if not clause:
            continue
        tokens = _content_tokens(clause)
        if len(tokens) < 3:
            continue
        key = " ".join(tokens)
        if key in seen:
            continue
        seen.add(key)
        clauses.append(clause)
        if len(clauses) >= max_clauses:
            break
    if len(clauses) <= 1:
        return []
    return clauses


def _clause_coverage_indices(
    query: str,
    ranked_idx: np.ndarray,
    chunk_vocabs: List[Set[str]],
) -> List[int]:
    clauses = _extract_query_clauses(query)
    if not clauses:
        return []

    pool = [int(idx) for idx in ranked_idx[: min(len(ranked_idx), 200)]]
    selected: List[int] = []
    for clause in clauses:
        tokens = _content_tokens(clause)
        if not tokens:
            continue
        best_idx = -1
        best_overlap = 0.0
        for idx in pool:
            overlap = _lexical_overlap_vocab(tokens, chunk_vocabs[idx])
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx >= 0 and best_overlap >= CLAUSE_COVERAGE_MIN_OVERLAP and best_idx not in selected:
            selected.append(best_idx)
        if len(selected) >= MAX_CLAUSE_COVERAGE:
            break
    return selected


def _apply_rerank_scores(
    client: cohere.Client,
    query: str,
    chunks: List[Chunk],
    candidate_idx: List[int],
    combined_scores: np.ndarray,
) -> np.ndarray:
    if (not ENABLE_RERANK) or (not candidate_idx):
        return combined_scores

    rerank_alpha = min(max(RERANK_ALPHA, 0.0), 1.0)
    if rerank_alpha <= 0:
        return combined_scores

    documents = [f"{chunks[idx].title}\n{chunks[idx].text}" for idx in candidate_idx]
    try:
        response = client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
        )
    except Exception as exc:
        LOGGER.warning("Cohere rerank failed; using hybrid scores only: %s", exc)
        return combined_scores

    rerank_scores: Dict[int, float] = {}
    for row in response.results:
        local_idx = int(row.index)
        if local_idx < 0 or local_idx >= len(candidate_idx):
            continue
        rerank_scores[candidate_idx[local_idx]] = float(row.relevance_score)
    if not rerank_scores:
        return combined_scores

    values = list(rerank_scores.values())
    min_score = min(values)
    max_score = max(values)
    denom = (max_score - min_score) + 1e-12

    updated = combined_scores.copy()
    for global_idx, value in rerank_scores.items():
        normalized = (value - min_score) / denom
        blended = ((1.0 - rerank_alpha) * float(updated[global_idx])) + (
            rerank_alpha * float(normalized)
        )
        updated[global_idx] = np.float32(min(max(blended, 0.0), 1.0))
    return updated


def rerank_retrieved_chunks(
    client: cohere.Client,
    query: str,
    rows: List[Tuple[Chunk, float]],
    top_n: int,
    model: str = RERANK_MODEL,
) -> List[Tuple[Chunk, float]]:
    """
    Re-rank an already retrieved candidate list and return top_n rows.

    This is useful for prompt-side condensation: retrieve broadly for recall,
    then compress to a smaller high-precision set before sending to chat.
    """
    target_n = max(1, int(top_n))
    if not rows:
        return []
    if len(rows) <= target_n:
        return rows[:target_n]

    documents = [f"{chunk.title}\n{chunk.text}" for chunk, _score in rows]
    try:
        response = client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=min(target_n, len(documents)),
            return_documents=False,
        )
    except Exception as exc:
        LOGGER.warning("Prompt-side rerank failed; falling back to initial order: %s", exc)
        return rows[:target_n]

    selected: List[Tuple[Chunk, float]] = []
    seen = set()
    for row in response.results:
        local_idx = int(row.index)
        if local_idx < 0 or local_idx >= len(rows) or local_idx in seen:
            continue
        seen.add(local_idx)
        chunk, _orig_score = rows[local_idx]
        selected.append((chunk, float(row.relevance_score)))
        if len(selected) >= target_n:
            return selected

    if len(selected) < target_n:
        for idx, (chunk, score) in enumerate(rows):
            if idx in seen:
                continue
            selected.append((chunk, float(score)))
            if len(selected) >= target_n:
                break
    return selected[:target_n]


def retrieve(
    client: cohere.Client,
    query: str,
    chunks: List[Chunk],
    chunk_vecs: np.ndarray,
    k: int,
    query_expansions: Optional[List[str]] = None,
    chunk_vocabs: Optional[List[Set[str]]] = None,
    chunk_modes: Optional[List[Set[str]]] = None,
    chunk_meta_vocabs: Optional[List[Set[str]]] = None,
) -> List[Tuple[Chunk, float]]:
    """
    Return top-k chunks with minimal hybrid retrieval.

    Inputs `chunk_modes` and `chunk_meta_vocabs` are accepted for backward
    compatibility with existing callsites but are intentionally ignored.
    """
    if not chunks or chunk_vecs.size == 0:
        return []
    target_k = max(1, int(k))

    query_texts = _merge_query_texts(query, query_expansions)
    dense_scores = _compute_dense_scores(client, query_texts, chunk_vecs)

    if chunk_vocabs is None:
        chunk_vocabs = build_chunk_vocabs(chunks)
    if len(chunk_vocabs) != len(chunks):
        raise ValueError("chunk_vocabs length must match chunks length")

    query_tokens = _content_tokens(" ".join(query_texts))
    lexical_scores = np.array(
        [_lexical_overlap_vocab(query_tokens, vocab) for vocab in chunk_vocabs],
        dtype=np.float32,
    )

    alpha = min(max(RETRIEVAL_ALPHA, 0.0), 1.0)
    combined_scores = np.clip((alpha * dense_scores) + ((1.0 - alpha) * lexical_scores), 0.0, 1.0)

    ranked_idx = np.argsort(-combined_scores)
    candidate_pool = min(len(chunks), max(target_k * 4, 40))
    candidate_idx = [int(idx) for idx in ranked_idx[:candidate_pool]]

    combined_scores = _apply_rerank_scores(
        client=client,
        query=query,
        chunks=chunks,
        candidate_idx=candidate_idx,
        combined_scores=combined_scores,
    )
    ranked_idx = np.argsort(-combined_scores)
    clause_idx = _clause_coverage_indices(query=query, ranked_idx=ranked_idx, chunk_vocabs=chunk_vocabs)
    if clause_idx:
        clause_set = set(clause_idx)
        ranked_idx = np.array(
            clause_idx + [int(idx) for idx in ranked_idx if int(idx) not in clause_set],
            dtype=np.int64,
        )

    selected: List[int] = []
    max_per_source = int(MAX_CHUNKS_PER_SOURCE)
    if max_per_source <= 0:
        # Explicitly uncapped mode: keep pure score ordering.
        selected = [int(idx) for idx in ranked_idx[:target_k]]
    else:
        # Lightweight diversity cap by source to prevent full domination by one file.
        per_source_count: Dict[str, int] = {}
        for raw_idx in ranked_idx:
            idx = int(raw_idx)
            source = chunks[idx].source_path
            if per_source_count.get(source, 0) >= max_per_source:
                continue
            selected.append(idx)
            per_source_count[source] = per_source_count.get(source, 0) + 1
            if len(selected) >= target_k:
                break

        # Final fallback if source cap prevented enough results.
        if len(selected) < target_k:
            seen = set(selected)
            for raw_idx in ranked_idx:
                idx = int(raw_idx)
                if idx in seen:
                    continue
                selected.append(idx)
                if len(selected) >= target_k:
                    break

    return [(chunks[idx], float(combined_scores[idx])) for idx in selected[:target_k]]


def build_chunk_vocabs(chunks: List[Chunk]) -> List[Set[str]]:
    """Precompute lexical vocab per chunk and reuse across queries."""
    return [
        set(
            _content_tokens(
                " ".join(
                    [
                        chunk.title,
                        chunk.section_heading or "",
                        " ".join(chunk.heading_path or []),
                        chunk.text,
                    ]
                )
            )
        )
        for chunk in chunks
    ]


def build_chunk_modes(chunks: List[Chunk]) -> List[Set[str]]:
    """Compatibility shim. Minimal retrieval does not use explicit mode routing."""
    return [set() for _ in chunks]


def build_chunk_meta_vocabs(chunks: List[Chunk]) -> List[Set[str]]:
    """Compatibility shim. Minimal retrieval does not use separate metadata vocab."""
    return [set() for _ in chunks]


def build_chunk_features(chunks: List[Chunk]) -> Tuple[List[Set[str]], List[Set[str]], List[Set[str]]]:
    """Return reusable feature tuples expected by existing callsites."""
    return build_chunk_vocabs(chunks), build_chunk_modes(chunks), build_chunk_meta_vocabs(chunks)
