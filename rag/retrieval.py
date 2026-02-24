"""Hybrid retrieval over dense embeddings + Elasticsearch BM25."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cohere
import numpy as np
from elasticsearch import Elasticsearch, helpers

from .app_config import (
    ELASTIC_INDEX_NAME,
    ELASTIC_REQUEST_TIMEOUT,
    ELASTIC_URL,
    ENABLE_MMR_DIVERSITY,
    ENABLE_RERANK,
    MMR_LAMBDA,
    MAX_CHUNKS_PER_SOURCE,
    RERANK_ALPHA,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_ALPHA,
    RETRIEVAL_CANDIDATE_K,
)
from .embedding_client import embed_texts
from .rag_types import Chunk

LOGGER = logging.getLogger(__name__)

_ES_CLIENT: Optional[Elasticsearch] = None
_ES_INDEX_SIGNATURE: Optional[str] = None
INDEX_SCHEMA_VERSION = "metadata_boost_v1"


def _merge_query_texts(query: str, query_expansions: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [query, *(query_expansions or [])]:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(normalized)
    return merged or [query]


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
    return np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def _chunk_signature(chunks: Sequence[Chunk]) -> str:
    lines = [f"schema={INDEX_SCHEMA_VERSION}"]
    lines.extend(chunk.chunk_id for chunk in chunks)
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _get_es_client() -> Elasticsearch:
    global _ES_CLIENT
    if _ES_CLIENT is None:
        _ES_CLIENT = Elasticsearch(ELASTIC_URL, request_timeout=ELASTIC_REQUEST_TIMEOUT)
    return _ES_CLIENT


def _create_index(es: Elasticsearch, index_name: str) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        body={
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "dynamic": "false",
                "properties": {
                    "doc_kind": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "section_id": {"type": "keyword"},
                    "is_exception": {"type": "boolean"},
                    "scope_tags": {"type": "keyword"},
                    "authority_rank": {"type": "integer"},
                    "title": {"type": "text", "analyzer": "english"},
                    "section_heading": {"type": "text", "analyzer": "english"},
                    "heading_path": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                    "full_text": {"type": "text", "analyzer": "english"},
                    "chunk_count": {"type": "integer"},
                    "chunk_signature": {"type": "keyword"},
                },
            },
        },
    )


def _bulk_index_chunks(
    es: Elasticsearch,
    index_name: str,
    chunks: Sequence[Chunk],
    chunk_signature: str,
) -> None:
    actions: List[Dict[str, Any]] = [
        {
            "_index": index_name,
            "_id": "__rag_meta__",
            "_source": {
                "doc_kind": "meta",
                "chunk_signature": chunk_signature,
                "chunk_count": len(chunks),
            },
        }
    ]

    for chunk in chunks:
        actions.append(
            {
                "_index": index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    "doc_kind": "chunk",
                    "chunk_id": chunk.chunk_id,
                    "doc_type": chunk.doc_type or "",
                    "chunk_type": chunk.chunk_type or "",
                    "section_id": chunk.section_id or "",
                    "is_exception": bool(chunk.is_exception),
                    "scope_tags": list(chunk.scope_tags or []),
                    "authority_rank": int(chunk.authority_rank),
                    "title": chunk.title or "",
                    "section_heading": chunk.section_heading or "",
                    "heading_path": " ".join(chunk.heading_path or []),
                    "text": chunk.text or "",
                    "full_text": " ".join(
                        [
                            chunk.doc_type or "",
                            chunk.chunk_type or "",
                            chunk.section_id or "",
                            " ".join(chunk.scope_tags or []),
                            chunk.title or "",
                            chunk.section_heading or "",
                            " ".join(chunk.heading_path or []),
                            chunk.text or "",
                        ]
                    ),
                },
            }
        )
    helpers.bulk(es, actions, refresh="wait_for", request_timeout=120)


def _index_matches_signature(
    es: Elasticsearch,
    index_name: str,
    chunk_signature: str,
    chunk_count: int,
) -> bool:
    if not es.indices.exists(index=index_name):
        return False
    try:
        meta = es.get(index=index_name, id="__rag_meta__", source=True)
        source = meta.get("_source", {})
        sig = str(source.get("chunk_signature", ""))
        count = int(source.get("chunk_count", -1))
        return sig == chunk_signature and count == int(chunk_count)
    except Exception:
        return False


def _ensure_index(es: Elasticsearch, index_name: str, chunks: Sequence[Chunk]) -> None:
    global _ES_INDEX_SIGNATURE
    chunk_signature = _chunk_signature(chunks)

    if _ES_INDEX_SIGNATURE == chunk_signature:
        return

    if _index_matches_signature(es, index_name, chunk_signature, len(chunks)):
        _ES_INDEX_SIGNATURE = chunk_signature
        return

    _create_index(es, index_name)
    _bulk_index_chunks(es, index_name, chunks, chunk_signature)
    _ES_INDEX_SIGNATURE = chunk_signature


def _elastic_bm25_scores(
    es: Elasticsearch,
    index_name: str,
    query_texts: Sequence[str],
    chunk_ids: Sequence[str],
    chunk_id_to_idx: Dict[str, int],
) -> np.ndarray:
    def metadata_should_clauses(query_text: str) -> List[Dict[str, Any]]:
        normalized = str(query_text or "").lower()
        if not normalized:
            return []

        clauses: List[Dict[str, Any]] = []

        def add_term(field: str, value: Any, boost: float) -> None:
            clauses.append({"term": {field: {"value": value, "boost": float(boost)}}})

        def has_any(terms: Sequence[str]) -> bool:
            return any(term in normalized for term in terms)

        if has_any(("table", "limits table", "appendix a")):
            add_term("chunk_type", "table", 1.35)

        if has_any(
            (
                "appendix a",
                "contracting limits",
                "treasury board",
                "non-competitive limit",
                "competitive limit",
                "4.10",
                "procurement-file",
                "procurement file",
            )
        ):
            add_term("doc_type", "tbs_directive", 0.95)

        if has_any(("4.10", "procurement-file", "procurement file", "audit")):
            clauses.append(
                {
                    "match_phrase": {
                        "section_heading": {"query": "4. Requirements", "boost": 0.9}
                    }
                }
            )

        if has_any(("rfsa", "supply arrangement", "request for supply arrangement")):
            add_term("doc_type", "buyers_guide", 0.75)

        if "exception" in normalized:
            add_term("is_exception", True, 0.5)

        return clauses

    bm25_raw = np.zeros(len(chunk_ids), dtype=np.float32)
    size = len(chunk_ids)

    for text in query_texts:
        query = str(text or "").strip()
        if not query:
            continue
        resp = es.search(
            index=index_name,
            size=size,
            track_total_hits=False,
            source=False,
            query={
                "bool": {
                    "filter": [{"term": {"doc_kind": "chunk"}}],
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^2.0",
                                    "section_heading^1.5",
                                    "heading_path^1.2",
                                    "text",
                                    "full_text",
                                ],
                                "type": "best_fields",
                                "operator": "or",
                            }
                        }
                    ],
                    "should": metadata_should_clauses(query),
                    "minimum_should_match": 0,
                }
            },
            request_timeout=max(10, int(ELASTIC_REQUEST_TIMEOUT)),
        )
        for hit in resp.get("hits", {}).get("hits", []):
            chunk_id = str(hit.get("_id", "")).strip()
            if not chunk_id:
                continue
            idx = chunk_id_to_idx.get(chunk_id)
            if idx is None:
                continue
            score = float(hit.get("_score") or 0.0)
            if score > bm25_raw[idx]:
                bm25_raw[idx] = score

    min_score = float(np.min(bm25_raw))
    max_score = float(np.max(bm25_raw))
    if max_score - min_score <= 1e-12:
        return np.zeros_like(bm25_raw, dtype=np.float32)
    return ((bm25_raw - min_score) / (max_score - min_score)).astype(np.float32)


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
    # RERANK_TOP_N == 0 means rerank all considered candidates.
    configured_top_n = int(RERANK_TOP_N)
    if configured_top_n <= 0:
        rerank_top_n = len(documents)
    else:
        rerank_top_n = min(len(documents), configured_top_n)
    try:
        response = client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=rerank_top_n,
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


def _apply_mmr_diversity(
    candidate_idx: List[int],
    combined_scores: np.ndarray,
    chunk_vecs: np.ndarray,
    lambda_mult: float,
) -> List[int]:
    """
    Reorder candidates with MMR to reduce near-duplicate semantic picks.
    """
    if len(candidate_idx) <= 1:
        return candidate_idx

    lam = min(max(float(lambda_mult), 0.0), 1.0)
    if lam >= 1.0:
        return candidate_idx

    candidate_arr = np.asarray(candidate_idx, dtype=np.int32)
    rel = np.asarray([float(combined_scores[idx]) for idx in candidate_idx], dtype=np.float32)

    vecs = np.asarray(chunk_vecs[candidate_arr], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    unit_vecs = vecs / safe_norms
    zero_rows = norms[:, 0] <= 1e-12
    if np.any(zero_rows):
        unit_vecs[zero_rows] = 0.0

    remaining = list(range(len(candidate_idx)))
    selected_local: List[int] = []

    first = int(np.argmax(rel))
    selected_local.append(first)
    remaining.remove(first)

    max_sim = np.zeros(len(candidate_idx), dtype=np.float32)
    while remaining:
        last = selected_local[-1]
        sims = (unit_vecs @ unit_vecs[last]).astype(np.float32)
        max_sim = np.maximum(max_sim, sims)

        best_local = max(
            remaining,
            key=lambda local_idx: float(
                (lam * rel[local_idx]) - ((1.0 - lam) * max_sim[local_idx])
            ),
        )
        selected_local.append(int(best_local))
        remaining.remove(int(best_local))

    return [candidate_idx[local_idx] for local_idx in selected_local]


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
) -> List[Tuple[Chunk, float]]:
    """
    Return top-k chunks with dense + Elasticsearch BM25 hybrid retrieval.
    """
    if not chunks or chunk_vecs.size == 0:
        return []
    target_k = max(1, int(k))

    query_texts = _merge_query_texts(query, query_expansions)
    dense_scores = _compute_dense_scores(client, query_texts, chunk_vecs)

    bm25_ok = False
    bm25_scores = np.zeros(len(chunks), dtype=np.float32)
    try:
        es = _get_es_client()
        _ensure_index(es, ELASTIC_INDEX_NAME, chunks)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
        bm25_scores = _elastic_bm25_scores(
            es=es,
            index_name=ELASTIC_INDEX_NAME,
            query_texts=query_texts,
            chunk_ids=chunk_ids,
            chunk_id_to_idx=chunk_id_to_idx,
        )
        bm25_ok = True
    except Exception as exc:
        LOGGER.warning("Elasticsearch BM25 unavailable; falling back to dense-only ranking: %s", exc)

    if bm25_ok:
        alpha = min(max(RETRIEVAL_ALPHA, 0.0), 1.0)
        combined_scores = np.clip(
            (alpha * dense_scores) + ((1.0 - alpha) * bm25_scores),
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        combined_scores = dense_scores

    ranked_idx = np.argsort(-combined_scores)
    # Keep candidate depth explicit and bounded: consider only the larger of
    # requested top-k and RETRIEVAL_CANDIDATE_K (no hidden 4x expansion).
    candidate_pool = min(len(chunks), max(int(RETRIEVAL_CANDIDATE_K), target_k))
    candidate_idx = [int(idx) for idx in ranked_idx[:candidate_pool]]

    combined_scores = _apply_rerank_scores(
        client=client,
        query=query,
        chunks=chunks,
        candidate_idx=candidate_idx,
        combined_scores=combined_scores,
    )
    ranked_idx = np.argsort(-combined_scores)
    if ENABLE_MMR_DIVERSITY and candidate_pool > 1:
        mmr_candidates = [int(idx) for idx in ranked_idx[:candidate_pool]]
        try:
            diversified = _apply_mmr_diversity(
                candidate_idx=mmr_candidates,
                combined_scores=combined_scores,
                chunk_vecs=chunk_vecs,
                lambda_mult=MMR_LAMBDA,
            )
            tail = [int(idx) for idx in ranked_idx[candidate_pool:]]
            ranked_idx = np.asarray(diversified + tail, dtype=np.int32)
        except Exception as exc:
            LOGGER.warning("MMR diversity pass failed; using score order: %s", exc)

    selected: List[int] = []
    max_per_source = int(MAX_CHUNKS_PER_SOURCE)
    if max_per_source <= 0:
        selected = [int(idx) for idx in ranked_idx[:target_k]]
    else:
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
