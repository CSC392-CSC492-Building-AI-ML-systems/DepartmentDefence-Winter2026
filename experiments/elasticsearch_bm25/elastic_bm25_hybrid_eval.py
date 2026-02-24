"""Evaluate Elasticsearch BM25 as a lexical replacement in dense/BM25 hybrid retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from elasticsearch import Elasticsearch, helpers

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.retrieval_adversarial_runner import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_CASES,
    DEFAULT_CHUNK_CACHE,
    _doc_prefix,
    _load_cases,
    _load_or_chunk_docs,
    _load_or_embed_chunk_vectors,
    _normalize_case,
)
from rag.app_config import RERANK_MODEL  # noqa: E402
from rag.corpus import list_docs  # noqa: E402
from rag.embedding_client import create_client, embed_texts  # noqa: E402
from rag.query_rewrite import generate_query_expansions  # noqa: E402
from rag.rag_types import Chunk  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "elasticsearch_bm25"
    / "runs"
    / "elastic_bm25_hybrid_eval_latest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Elasticsearch BM25 / dense hybrid retrieval experiments."
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        nargs="+",
        default=DEFAULT_CASES,
        help="One or more case files (.json or .jsonl).",
    )
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument(
        "--split",
        default="all",
        help="Optional split filter for JSONL eval-style cases (all/dev/test/train or comma-separated).",
    )
    parser.add_argument(
        "--variants",
        default="bm25,hybrid,dense",
        help="Comma-separated variants from: bm25,hybrid,dense.",
    )
    parser.add_argument(
        "--alphas",
        default="0.0,0.2,0.4,0.6,0.7,0.8,1.0",
        help="Comma-separated alpha values used for hybrid variants.",
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help="Enable LLM query rewrite cache before retrieval.",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Apply Cohere rerank to a candidate pool before final top-k selection.",
    )
    parser.add_argument(
        "--rerank-alphas",
        default="0.0,0.2,0.4,0.6",
        help="Comma-separated rerank blend values (0=no effect, 1=rerank-only in candidate pool).",
    )
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=100,
        help="Candidate depth retrieved before rerank blending (final output still --top-k).",
    )
    parser.add_argument(
        "--rerank-model",
        default=RERANK_MODEL,
        help="Cohere rerank model name.",
    )
    parser.add_argument(
        "--elastic-url",
        default="http://127.0.0.1:9200",
        help="Elasticsearch URL (default local node).",
    )
    parser.add_argument(
        "--index-name",
        default="policy_chunks_bm25_eval",
        help="Elasticsearch index for chunk documents.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Drop and rebuild the Elasticsearch chunk index before evaluation.",
    )
    parser.add_argument(
        "--rewrite-cache-from-run",
        type=Path,
        default=None,
        help=(
            "Optional retrieval-run JSON file with per-case query_expansions to reuse. "
            "If provided and --use-query-rewrite is set, these cached rewrites are used "
            "before calling the rewrite model."
        ),
    )
    return parser.parse_args()


def _select_cases(paths: Iterable[Path], split_arg: str) -> List[Dict[str, Any]]:
    raw_cases = _load_cases(paths)
    cases = [_normalize_case(row) for row in raw_cases]

    split_filter_raw = str(split_arg or "all").strip().lower()
    split_filter = {value.strip() for value in split_filter_raw.split(",") if value.strip()}
    if split_filter and "all" not in split_filter:
        cases = [case for case in cases if case["split"] in split_filter]

    selected = [
        case
        for case in cases
        if case["question"] and (case["expected_chunk_ids"] or case["expected_doc_prefixes"])
    ]
    if not selected:
        raise RuntimeError("No usable cases after filtering.")
    return selected


def _parse_variants(raw: str) -> List[str]:
    allowed = {"bm25", "dense", "hybrid"}
    variants: List[str] = []
    for token in str(raw).split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"Unknown variant '{value}'. Allowed: {sorted(allowed)}")
        if value not in variants:
            variants.append(value)
    if not variants:
        raise ValueError("No variants provided.")
    return variants


def _parse_alphas(raw: str) -> List[float]:
    values: List[float] = []
    for token in str(raw).split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(min(max(float(stripped), 0.0), 1.0))
    if not values:
        raise ValueError("No alpha values provided.")
    return values


def _merge_query_texts(question: str, expansions: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for value in [question, *expansions]:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(normalized)
    return merged or [question]


def _dense_scores(client, chunk_vecs: np.ndarray, query_texts: Sequence[str]) -> np.ndarray:
    qvecs = embed_texts(client, list(query_texts), input_type="search_query")
    dense_matrix = chunk_vecs @ qvecs.T
    if dense_matrix.shape[1] == 1:
        dense_scores = dense_matrix[:, 0]
    else:
        dense_scores = np.max(dense_matrix, axis=1)
    return np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


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
                    "chunk_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "english"},
                    "section_heading": {"type": "text", "analyzer": "english"},
                    "heading_path": {"type": "text", "analyzer": "english"},
                    "text": {"type": "text", "analyzer": "english"},
                    "full_text": {"type": "text", "analyzer": "english"},
                },
            },
        },
    )


def _bulk_index_chunks(es: Elasticsearch, index_name: str, chunks: Sequence[Chunk]) -> None:
    actions = []
    for chunk in chunks:
        actions.append(
            {
                "_index": index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title or "",
                    "section_heading": chunk.section_heading or "",
                    "heading_path": " ".join(chunk.heading_path or []),
                    "text": chunk.text or "",
                    "full_text": " ".join(
                        [
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


def _ensure_index(
    es: Elasticsearch,
    index_name: str,
    chunks: Sequence[Chunk],
    reindex: bool,
) -> None:
    if reindex or (not es.indices.exists(index=index_name)):
        _create_index(es, index_name)
        _bulk_index_chunks(es, index_name, chunks)
        return

    doc_count = es.count(index=index_name).get("count", 0)
    if int(doc_count) != len(chunks):
        _create_index(es, index_name)
        _bulk_index_chunks(es, index_name, chunks)


def _elastic_bm25_scores(
    es: Elasticsearch,
    index_name: str,
    query_texts: Sequence[str],
    chunk_ids: Sequence[str],
    chunk_id_to_idx: Dict[str, int],
) -> np.ndarray:
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
                "multi_match": {
                    "query": query,
                    "fields": ["title^2.0", "section_heading^1.5", "heading_path^1.2", "text", "full_text"],
                    "type": "best_fields",
                    "operator": "or",
                }
            },
            request_timeout=60,
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
    *,
    client: Any,
    query: str,
    chunks: Sequence[Chunk],
    candidate_idx: Sequence[int],
    base_scores: np.ndarray,
    rerank_alpha: float,
    rerank_model: str,
) -> np.ndarray:
    if client is None or (not candidate_idx):
        return base_scores
    alpha = min(max(float(rerank_alpha), 0.0), 1.0)
    if alpha <= 0.0:
        return base_scores

    documents = [f"{chunks[idx].title}\n{chunks[idx].text}" for idx in candidate_idx]
    try:
        response = client.rerank(
            model=rerank_model,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
        )
    except Exception as exc:
        LOGGER.warning("Cohere rerank failed; keeping base scores: %s", exc)
        return base_scores

    rerank_scores: Dict[int, float] = {}
    for row in response.results:
        local_idx = int(row.index)
        if local_idx < 0 or local_idx >= len(candidate_idx):
            continue
        rerank_scores[int(candidate_idx[local_idx])] = float(row.relevance_score)
    if not rerank_scores:
        return base_scores

    values = list(rerank_scores.values())
    min_score = min(values)
    max_score = max(values)
    denom = (max_score - min_score) + 1e-12

    blended = base_scores.copy()
    for global_idx, value in rerank_scores.items():
        normalized = (value - min_score) / denom
        mixed = ((1.0 - alpha) * float(blended[global_idx])) + (alpha * float(normalized))
        blended[global_idx] = np.float32(min(max(mixed, 0.0), 1.0))
    return blended


def _evaluate_variant(
    *,
    name: str,
    variant: str,
    alpha: float | None,
    cases: Sequence[Dict[str, Any]],
    chunks: Sequence[Chunk],
    chunk_ids: Sequence[str],
    chunk_id_to_idx: Dict[str, int],
    chunk_vecs: Optional[np.ndarray],
    client: Any,
    es: Elasticsearch,
    index_name: str,
    top_k: int,
    enable_rerank: bool,
    rerank_alpha: float,
    rerank_candidate_k: int,
    rerank_model: str,
    rewrite_cache: Dict[str, List[str]],
    use_query_rewrite: bool,
    available_prefixes: Set[str],
) -> Dict[str, Any]:
    start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    rewrite_counts: List[int] = []
    target_k = max(1, int(top_k))

    for case in cases:
        question = case["question"]
        expected_ids = list(case["expected_chunk_ids"])
        expected_prefixes = list(case["expected_doc_prefixes"])
        available_expected_prefixes = [
            prefix for prefix in expected_prefixes if prefix in available_prefixes
        ]

        expansions = rewrite_cache.get(question, []) if use_query_rewrite else []
        rewrite_counts.append(len(expansions))
        query_texts = _merge_query_texts(question, expansions)

        dense = None
        bm25 = None
        if variant in {"dense", "hybrid"}:
            if client is None or chunk_vecs is None:
                raise RuntimeError(
                    "Dense/hybrid variant requires Cohere client and chunk embeddings."
                )
            dense = _dense_scores(client=client, chunk_vecs=chunk_vecs, query_texts=query_texts)
        if variant in {"bm25", "hybrid"}:
            bm25 = _elastic_bm25_scores(
                es=es,
                index_name=index_name,
                query_texts=query_texts,
                chunk_ids=chunk_ids,
                chunk_id_to_idx=chunk_id_to_idx,
            )

        if variant == "dense":
            assert dense is not None
            combined = dense
        elif variant == "bm25":
            assert bm25 is not None
            combined = bm25
        elif variant == "hybrid":
            assert dense is not None and bm25 is not None and alpha is not None
            blend = min(max(float(alpha), 0.0), 1.0)
            combined = np.clip((blend * dense) + ((1.0 - blend) * bm25), 0.0, 1.0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        ranked_idx = np.argsort(-combined)
        if enable_rerank:
            if client is None:
                raise RuntimeError("Rerank requested but Cohere client is unavailable.")
            candidate_k = max(target_k, int(rerank_candidate_k))
            candidate_k = min(len(chunk_ids), max(1, candidate_k))
            candidate_idx = [int(idx) for idx in ranked_idx[:candidate_k]]
            combined = _apply_rerank_scores(
                client=client,
                query=question,
                chunks=chunks,
                candidate_idx=candidate_idx,
                base_scores=combined,
                rerank_alpha=rerank_alpha,
                rerank_model=rerank_model,
            )
            ranked_idx = np.argsort(-combined)

        selected_idx = [int(idx) for idx in ranked_idx[:target_k]]
        got_ids = [chunk_ids[idx] for idx in selected_idx]
        got_prefixes = [_doc_prefix(value) for value in got_ids]

        hit_ids = [value for value in expected_ids if value in got_ids]
        hit_prefixes = [value for value in expected_prefixes if value in got_prefixes]
        hit_available_prefixes = [
            value for value in available_expected_prefixes if value in got_prefixes
        ]

        rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "hit_chunk_ids": hit_ids,
                "hit_doc_prefixes": hit_prefixes,
                "chunk_id_recall": (len(hit_ids) / len(expected_ids)) if expected_ids else None,
                "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                if expected_prefixes
                else None,
                "doc_prefix_recall_available": (
                    len(hit_available_prefixes) / len(available_expected_prefixes)
                )
                if available_expected_prefixes
                else None,
                "top3": got_ids[:3],
            }
        )

    elapsed = time.perf_counter() - start
    chunk_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    prefix_available_values = [
        row["doc_prefix_recall_available"]
        for row in rows
        if row["doc_prefix_recall_available"] is not None
    ]
    expected_chunk_total = sum(len(case["expected_chunk_ids"]) for case in cases)
    hit_chunk_total = sum(len(row["hit_chunk_ids"]) for row in rows)
    expected_prefix_total = sum(len(case["expected_doc_prefixes"]) for case in cases)
    hit_prefix_total = sum(len(row["hit_doc_prefixes"]) for row in rows)
    chunk_count_dist = Counter(len(case["expected_chunk_ids"]) for case in cases)
    prefix_count_dist = Counter(len(case["expected_doc_prefixes"]) for case in cases)

    return {
        "name": name,
        "variant": variant,
        "alpha": alpha,
        "enable_rerank": bool(enable_rerank),
        "rerank_alpha": float(rerank_alpha) if enable_rerank else None,
        "summary": {
            "case_count": len(rows),
            "elapsed_seconds": round(elapsed, 2),
            "rerank_enabled": bool(enable_rerank),
            "rerank_alpha": float(rerank_alpha) if enable_rerank else None,
            "rerank_candidate_k": int(rerank_candidate_k) if enable_rerank else None,
            "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values))
            if chunk_values
            else None,
            "chunk_id_recall_micro": (hit_chunk_total / expected_chunk_total)
            if expected_chunk_total
            else None,
            "doc_prefix_recall_mean": (sum(prefix_values) / len(prefix_values))
            if prefix_values
            else None,
            "doc_prefix_recall_micro": (hit_prefix_total / expected_prefix_total)
            if expected_prefix_total
            else None,
            "doc_prefix_recall_available_mean": (
                sum(prefix_available_values) / len(prefix_available_values)
            )
            if prefix_available_values
            else None,
            "zero_chunk_hit_cases": [
                row["id"] for row in rows if row["chunk_id_recall"] == 0.0
            ],
            "zero_prefix_hit_cases": [
                row["id"] for row in rows if row["doc_prefix_recall"] == 0.0
            ],
            "expected_chunk_count_distribution": {
                str(key): int(value) for key, value in sorted(chunk_count_dist.items())
            },
            "expected_doc_prefix_count_distribution": {
                str(key): int(value) for key, value in sorted(prefix_count_dist.items())
            },
            "query_rewrite_avg_expansions_per_question": (
                sum(rewrite_counts) / len(rewrite_counts) if rewrite_counts else 0.0
            ),
            "query_rewrite_questions_with_expansions": sum(1 for count in rewrite_counts if count > 0),
        },
    }


def _load_rewrite_cache_from_run(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases", [])
    if not isinstance(rows, list):
        return {}
    cache: Dict[str, List[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        expansions_raw = row.get("query_expansions", [])
        if not isinstance(expansions_raw, list):
            continue
        expansions = [str(value).strip() for value in expansions_raw if str(value).strip()]
        cache[question] = expansions
    return cache


def run() -> None:
    args = parse_args()
    cases = _select_cases(args.cases_file, args.split)
    variants = _parse_variants(args.variants)
    alphas = _parse_alphas(args.alphas)
    rerank_alphas = _parse_alphas(args.rerank_alphas) if args.enable_rerank else [0.0]

    es = Elasticsearch(args.elastic_url, request_timeout=60)
    info = es.info()

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    if not chunks:
        raise RuntimeError("No chunks available.")

    _ensure_index(es=es, index_name=args.index_name, chunks=chunks, reindex=bool(args.reindex))

    needs_dense = any(variant in {"dense", "hybrid"} for variant in variants)
    needs_client = needs_dense or bool(args.use_query_rewrite) or bool(args.enable_rerank)

    client = create_client() if needs_client else None
    chunk_vecs = (
        _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
        if needs_dense
        else None
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_id_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
    available_prefixes = {_doc_prefix(chunk_id) for chunk_id in chunk_ids}

    rewrite_cache: Dict[str, List[str]] = {}
    if args.use_query_rewrite:
        if args.rewrite_cache_from_run is not None and args.rewrite_cache_from_run.exists():
            rewrite_cache.update(_load_rewrite_cache_from_run(args.rewrite_cache_from_run))
        for case in cases:
            question = case["question"]
            if question in rewrite_cache:
                continue
            if client is None:
                raise RuntimeError("Query rewrite requested but Cohere client is unavailable.")
            rewrite_cache[question] = generate_query_expansions(
                client=client,
                question=question,
                chat_history=[],
            )

    results: List[Dict[str, Any]] = []
    for variant in variants:
        alpha_grid: List[Optional[float]]
        if variant == "hybrid":
            alpha_grid = [float(value) for value in alphas]
        else:
            alpha_grid = [None]

        for alpha in alpha_grid:
            base_name = variant if alpha is None else f"{variant}_alpha_{alpha:.2f}"
            for rerank_alpha in rerank_alphas:
                run_name = base_name
                if args.enable_rerank:
                    run_name = f"{base_name}_rerank_{float(rerank_alpha):.2f}"
                results.append(
                    _evaluate_variant(
                        name=run_name,
                        variant=variant,
                        alpha=alpha,
                        cases=cases,
                        chunks=chunks,
                        chunk_ids=chunk_ids,
                        chunk_id_to_idx=chunk_id_to_idx,
                        chunk_vecs=chunk_vecs,
                        client=client,
                        es=es,
                        index_name=args.index_name,
                        top_k=args.top_k,
                        enable_rerank=bool(args.enable_rerank),
                        rerank_alpha=float(rerank_alpha),
                        rerank_candidate_k=int(args.rerank_candidate_k),
                        rerank_model=str(args.rerank_model),
                        rewrite_cache=rewrite_cache,
                        use_query_rewrite=bool(args.use_query_rewrite),
                        available_prefixes=available_prefixes,
                    )
                )

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "top_k": int(args.top_k),
            "variants": variants,
            "alphas": alphas,
            "enable_rerank": bool(args.enable_rerank),
            "rerank_alphas": rerank_alphas if args.enable_rerank else [],
            "rerank_candidate_k": int(args.rerank_candidate_k),
            "rerank_model": str(args.rerank_model),
            "use_query_rewrite": bool(args.use_query_rewrite),
            "elastic_url": str(args.elastic_url),
            "index_name": str(args.index_name),
            "reindex": bool(args.reindex),
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
            "elastic_version": info.get("version", {}).get("number"),
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _sort_key(row: Dict[str, Any]) -> tuple[float, float]:
        summary = row.get("summary", {})
        return (
            float(summary.get("chunk_id_recall_mean") or -1.0),
            float(summary.get("doc_prefix_recall_mean") or -1.0),
        )

    best = max(results, key=_sort_key)
    best_summary = best["summary"]
    print(
        "Elastic BM25 eval summary: "
        f"best={best['name']} "
        f"chunk={best_summary['chunk_id_recall_mean']} "
        f"prefix={best_summary['doc_prefix_recall_mean']} "
        f"top_k={args.top_k}"
    )
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    run()
