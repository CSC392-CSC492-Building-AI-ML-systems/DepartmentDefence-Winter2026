"""Evaluate third-party BM25 and dense/BM25 hybrid retrieval on the adversarial suite."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

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
from rag.embedding_client import create_client, embed_texts  # noqa: E402
from rag.query_rewrite import generate_query_expansions  # noqa: E402
from rag.rag_types import Chunk  # noqa: E402
from rag.corpus import list_docs  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments" / "bm25_hybrid" / "runs" / "bm25_hybrid_eval_latest.json"
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25/dense/hybrid retrieval experiments.")
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
        "--hybrid-alpha",
        type=float,
        default=0.7,
        help="Dense-vs-BM25 blend for hybrid score. 1.0=dense only, 0.0=BM25 only.",
    )
    parser.add_argument(
        "--variants",
        default="bm25,dense,hybrid",
        help="Comma-separated variants from: bm25,dense,hybrid.",
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help="Enable LLM query rewriting before retrieval.",
    )
    parser.add_argument(
        "--bm25-drop-stopwords",
        action="store_true",
        help="Drop a light stopword list from BM25 tokens.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = NON_ALNUM_RE.sub(" ", lowered)
    return SPACE_RE.sub(" ", cleaned).strip()


def _tokenize(text: str, *, drop_stopwords: bool) -> List[str]:
    tokens = TOKEN_RE.findall(_normalize_text(text))
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


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


def _dense_scores(
    client,
    chunk_vecs: np.ndarray,
    query_texts: Sequence[str],
) -> np.ndarray:
    qvecs = embed_texts(client, list(query_texts), input_type="search_query")
    dense_matrix = chunk_vecs @ qvecs.T
    if dense_matrix.shape[1] == 1:
        dense = dense_matrix[:, 0]
    else:
        dense = np.max(dense_matrix, axis=1)
    return np.clip((dense + 1.0) / 2.0, 0.0, 1.0)


def _bm25_scores(
    bm25: BM25Okapi,
    query_texts: Sequence[str],
    *,
    drop_stopwords: bool,
) -> np.ndarray:
    all_scores: List[np.ndarray] = []
    for text in query_texts:
        tokens = _tokenize(text, drop_stopwords=drop_stopwords)
        if not tokens:
            continue
        all_scores.append(np.asarray(bm25.get_scores(tokens), dtype=np.float32))
    if not all_scores:
        return np.zeros(len(bm25.doc_len), dtype=np.float32)
    raw = all_scores[0] if len(all_scores) == 1 else np.max(np.stack(all_scores, axis=1), axis=1)
    min_score = float(np.min(raw))
    max_score = float(np.max(raw))
    if max_score - min_score <= 1e-12:
        return np.zeros_like(raw, dtype=np.float32)
    return ((raw - min_score) / (max_score - min_score)).astype(np.float32)


def _top_k_indices(scores: np.ndarray, k: int) -> List[int]:
    target_k = max(1, int(k))
    ranked_idx = np.argsort(-scores)
    return [int(idx) for idx in ranked_idx[:target_k]]


def _evaluate_variant(
    *,
    variant: str,
    cases: Sequence[Dict[str, Any]],
    chunks: Sequence[Chunk],
    chunk_vecs: np.ndarray,
    bm25: BM25Okapi,
    client,
    top_k: int,
    hybrid_alpha: float,
    use_query_rewrite: bool,
    rewrite_cache: Dict[str, List[str]],
    bm25_drop_stopwords: bool,
    available_prefixes: Set[str],
) -> Dict[str, Any]:
    start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    rewrite_expansion_counts: List[int] = []

    for case in cases:
        question = case["question"]
        expected_ids = list(case["expected_chunk_ids"])
        expected_prefixes = list(case["expected_doc_prefixes"])
        available_expected_prefixes = [
            prefix for prefix in expected_prefixes if prefix in available_prefixes
        ]

        expansions = rewrite_cache.get(question, []) if use_query_rewrite else []
        rewrite_expansion_counts.append(len(expansions))
        query_texts = _merge_query_texts(question, expansions)

        if variant == "bm25":
            scores = _bm25_scores(
                bm25=bm25,
                query_texts=query_texts,
                drop_stopwords=bm25_drop_stopwords,
            )
        elif variant == "dense":
            scores = _dense_scores(client=client, chunk_vecs=chunk_vecs, query_texts=query_texts)
        elif variant == "hybrid":
            dense = _dense_scores(client=client, chunk_vecs=chunk_vecs, query_texts=query_texts)
            bm25_norm = _bm25_scores(
                bm25=bm25,
                query_texts=query_texts,
                drop_stopwords=bm25_drop_stopwords,
            )
            alpha = min(max(float(hybrid_alpha), 0.0), 1.0)
            scores = np.clip((alpha * dense) + ((1.0 - alpha) * bm25_norm), 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        selected_idx = _top_k_indices(scores=scores, k=top_k)
        got_ids = [chunks[idx].chunk_id for idx in selected_idx]
        got_prefixes = [_doc_prefix(value) for value in got_ids]

        hit_ids = [chunk_id for chunk_id in expected_ids if chunk_id in got_ids]
        hit_prefixes = [prefix for prefix in expected_prefixes if prefix in got_prefixes]
        hit_available_prefixes = [
            prefix for prefix in available_expected_prefixes if prefix in got_prefixes
        ]

        rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "expected_chunk_ids": expected_ids,
                "expected_doc_prefixes": expected_prefixes,
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
                "query_expansions": expansions,
                "top3": got_ids[:3],
            }
        )

    elapsed_seconds = time.perf_counter() - start
    chunk_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    prefix_available_values = [
        row["doc_prefix_recall_available"]
        for row in rows
        if row["doc_prefix_recall_available"] is not None
    ]
    expected_chunk_total = sum(len(row["expected_chunk_ids"]) for row in rows)
    hit_chunk_total = sum(len(row["hit_chunk_ids"]) for row in rows)
    expected_prefix_total = sum(len(row["expected_doc_prefixes"]) for row in rows)
    hit_prefix_total = sum(len(row["hit_doc_prefixes"]) for row in rows)
    chunk_count_dist = Counter(len(row["expected_chunk_ids"]) for row in rows)
    prefix_count_dist = Counter(len(row["expected_doc_prefixes"]) for row in rows)

    return {
        "variant": variant,
        "summary": {
            "case_count": len(rows),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values))
            if chunk_values
            else None,
            "doc_prefix_recall_mean": (sum(prefix_values) / len(prefix_values))
            if prefix_values
            else None,
            "doc_prefix_recall_available_mean": (
                sum(prefix_available_values) / len(prefix_available_values)
            )
            if prefix_available_values
            else None,
            "chunk_id_recall_micro": (hit_chunk_total / expected_chunk_total)
            if expected_chunk_total
            else None,
            "doc_prefix_recall_micro": (hit_prefix_total / expected_prefix_total)
            if expected_prefix_total
            else None,
            "zero_chunk_hit_cases": [
                row["id"] for row in rows if row["expected_chunk_ids"] and (not row["hit_chunk_ids"])
            ],
            "zero_prefix_hit_cases": [row["id"] for row in rows if not row["hit_doc_prefixes"]],
            "expected_chunk_count_distribution": {
                str(key): int(value) for key, value in sorted(chunk_count_dist.items())
            },
            "expected_doc_prefix_count_distribution": {
                str(key): int(value) for key, value in sorted(prefix_count_dist.items())
            },
            "query_rewrite_avg_expansions_per_question": (
                sum(rewrite_expansion_counts) / len(rewrite_expansion_counts)
                if rewrite_expansion_counts
                else 0.0
            ),
            "query_rewrite_questions_with_expansions": sum(
                1 for count in rewrite_expansion_counts if count > 0
            ),
        },
        "cases": rows,
    }


def run() -> None:
    args = parse_args()
    cases = _select_cases(args.cases_file, args.split)
    variants = _parse_variants(args.variants)

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    if not chunks:
        raise RuntimeError("No chunks available.")

    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    bm25_docs = [
        _tokenize(
            " ".join(
                [
                    chunk.title or "",
                    chunk.section_heading or "",
                    " ".join(chunk.heading_path or []),
                    chunk.text or "",
                ]
            ),
            drop_stopwords=bool(args.bm25_drop_stopwords),
        )
        for chunk in chunks
    ]
    bm25 = BM25Okapi(bm25_docs)
    available_prefixes = {_doc_prefix(chunk.chunk_id) for chunk in chunks if chunk.chunk_id}
    rewrite_cache: Dict[str, List[str]] = {}
    if args.use_query_rewrite:
        for case in cases:
            question = case["question"]
            if question in rewrite_cache:
                continue
            rewrite_cache[question] = generate_query_expansions(
                client=client,
                question=question,
                chat_history=[],
            )

    results: List[Dict[str, Any]] = []
    for variant in variants:
        result = _evaluate_variant(
            variant=variant,
            cases=cases,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
            bm25=bm25,
            client=client,
            top_k=args.top_k,
            hybrid_alpha=args.hybrid_alpha,
            use_query_rewrite=bool(args.use_query_rewrite),
            rewrite_cache=rewrite_cache,
            bm25_drop_stopwords=bool(args.bm25_drop_stopwords),
            available_prefixes=available_prefixes,
        )
        results.append(result)

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "top_k": int(args.top_k),
            "hybrid_alpha": float(args.hybrid_alpha),
            "variants": variants,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "bm25_drop_stopwords": bool(args.bm25_drop_stopwords),
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote BM25/hybrid report: {args.output}")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['variant']}: "
            f"chunk={summary['chunk_id_recall_mean']} "
            f"prefix={summary['doc_prefix_recall_mean']} "
            f"seconds={summary['elapsed_seconds']}"
        )


if __name__ == "__main__":
    run()
