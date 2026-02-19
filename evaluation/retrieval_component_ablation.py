"""Ablation runner for minimal retrieval components."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.query_rewrite import generate_query_expansions
import rag.retrieval as retrieval

from evaluation.retrieval_adversarial_runner import (
    DEFAULT_CACHE,
    DEFAULT_CHUNK_CACHE,
    _doc_prefix,
    _load_cases,
    _load_or_chunk_docs,
    _load_or_embed_chunk_vectors,
    _normalize_case,
)

DEFAULT_CASES = [
    REPO_ROOT / "evaluation" / "cases" / "adversarial_retrieval_cases.json",
    REPO_ROOT / "evaluation" / "cases" / "adversarial_chunk_stress_cases.json",
    REPO_ROOT / "evaluation" / "cases" / "adversarial_retrieval_robustness.json",
    REPO_ROOT / "evaluation" / "cases" / "eval_cases.jsonl",
    REPO_ROOT / "evaluation" / "cases" / "eval_cases_reference.jsonl",
]
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "retrieval_component_ablation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run component ablations for retrieval.")
    parser.add_argument(
        "--cases-file",
        type=Path,
        nargs="+",
        default=DEFAULT_CASES,
        help="One or more case files (.json or .jsonl).",
    )
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split",
        default="all",
        help="Optional split filter: all/dev/test/train or comma-separated.",
    )
    parser.add_argument(
        "--with-rerank-variant",
        action="store_true",
        help="Also run a rerank-enabled variant (extra API cost).",
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


def _summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_recall_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_recall_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    prefix_available_values = [
        row["doc_prefix_recall_available"]
        for row in rows
        if row["doc_prefix_recall_available"] is not None
    ]
    return {
        "case_count": len(rows),
        "chunk_id_recall_mean": (sum(chunk_recall_values) / len(chunk_recall_values))
        if chunk_recall_values
        else None,
        "doc_prefix_recall_mean": (sum(prefix_recall_values) / len(prefix_recall_values))
        if prefix_recall_values
        else None,
        "doc_prefix_recall_available_mean": (
            sum(prefix_available_values) / len(prefix_available_values)
        )
        if prefix_available_values
        else None,
        "zero_prefix_hit_cases": [row["id"] for row in rows if not row["hit_doc_prefixes"]],
        "fully_unsupported_cases": [
            row["id"] for row in rows if not row["available_expected_doc_prefixes"]
        ],
    }


def _evaluate_variant(
    *,
    name: str,
    client,
    cases: List[Dict[str, Any]],
    chunks,
    chunk_vecs,
    available_prefixes: Set[str],
    top_k: int,
    use_query_rewrite: bool,
    query_expansion_cache: Dict[str, List[str]],
    retrieval_alpha: Optional[float] = None,
    clear_stopwords: bool = False,
    merge_query_only: bool = False,
    enable_rerank: bool = False,
) -> Dict[str, Any]:
    # Snapshot mutable module state.
    original_alpha = retrieval.RETRIEVAL_ALPHA
    original_enable_rerank = retrieval.ENABLE_RERANK
    original_stopwords = copy.copy(retrieval.STOPWORDS)
    original_merge = retrieval._merge_query_texts

    try:
        if retrieval_alpha is not None:
            retrieval.RETRIEVAL_ALPHA = float(retrieval_alpha)
        retrieval.ENABLE_RERANK = bool(enable_rerank)

        if clear_stopwords:
            retrieval.STOPWORDS = set()
        else:
            retrieval.STOPWORDS = copy.copy(original_stopwords)

        if merge_query_only:
            retrieval._merge_query_texts = lambda query, query_expansions: [query]  # type: ignore[assignment]
        else:
            retrieval._merge_query_texts = original_merge

        chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)

        t0 = time.perf_counter()
        rows: List[Dict[str, Any]] = []
        for case in cases:
            question = case["question"]
            expected_ids = list(case["expected_chunk_ids"])
            expected_prefixes = list(case["expected_doc_prefixes"])
            available_expected_prefixes = [
                prefix for prefix in expected_prefixes if prefix in available_prefixes
            ]

            query_expansions = query_expansion_cache.get(question, []) if use_query_rewrite else []

            retrieved = retrieval.retrieve(
                client=client,
                query=question,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                k=top_k,
                query_expansions=query_expansions,
                chunk_vocabs=chunk_vocabs,
                chunk_modes=chunk_modes,
                chunk_meta_vocabs=chunk_meta_vocabs,
            )

            got_ids = [chunk.chunk_id for chunk, _score in retrieved]
            got_prefixes = [_doc_prefix(chunk_id) for chunk_id in got_ids]
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
                    "available_expected_doc_prefixes": available_expected_prefixes,
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

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        summary = _summarize_results(rows)
        summary["elapsed_ms"] = round(elapsed_ms, 2)
        return {
            "name": name,
            "settings": {
                "use_query_rewrite": use_query_rewrite,
                "retrieval_alpha": retrieval.RETRIEVAL_ALPHA,
                "clear_stopwords": clear_stopwords,
                "merge_query_only": merge_query_only,
                "enable_rerank": enable_rerank,
            },
            "summary": summary,
            "cases": rows,
        }
    finally:
        retrieval.RETRIEVAL_ALPHA = original_alpha
        retrieval.ENABLE_RERANK = original_enable_rerank
        retrieval.STOPWORDS = original_stopwords
        retrieval._merge_query_texts = original_merge


def _build_query_rewrite_cache(client, cases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    cache: Dict[str, List[str]] = {}
    seen = set()
    for case in cases:
        question = case["question"]
        if question in seen:
            continue
        seen.add(question)
        cache[question] = generate_query_expansions(client=client, question=question, chat_history=[])
    return cache


def run() -> None:
    args = parse_args()
    cases = _select_cases(args.cases_file, args.split)
    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    available_prefixes = {_doc_prefix(chunk.chunk_id) for chunk in chunks if chunk.chunk_id}

    rewrite_cache = _build_query_rewrite_cache(client, cases)
    rewrite_lengths = [len(value) for value in rewrite_cache.values()]

    variants = [
        {
            "name": "default_norewrite",
            "use_query_rewrite": False,
        },
        {
            "name": "no_stopwords_norewrite",
            "use_query_rewrite": False,
            "clear_stopwords": True,
        },
        {
            "name": "dense_only_norewrite",
            "use_query_rewrite": False,
            "retrieval_alpha": 1.0,
        },
        {
            "name": "lexical_only_norewrite",
            "use_query_rewrite": False,
            "retrieval_alpha": 0.0,
        },
        {
            "name": "default_rewrite",
            "use_query_rewrite": True,
        },
        {
            "name": "rewrite_no_merge",
            "use_query_rewrite": True,
            "merge_query_only": True,
        },
    ]
    if args.with_rerank_variant:
        variants.append(
            {
                "name": "default_norewrite_rerank",
                "use_query_rewrite": False,
                "enable_rerank": True,
            }
        )

    results: List[Dict[str, Any]] = []
    for variant in variants:
        results.append(
            _evaluate_variant(
                name=variant["name"],
                client=client,
                cases=cases,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                available_prefixes=available_prefixes,
                top_k=args.top_k,
                query_expansion_cache=rewrite_cache,
                use_query_rewrite=bool(variant.get("use_query_rewrite", False)),
                retrieval_alpha=variant.get("retrieval_alpha"),
                clear_stopwords=bool(variant.get("clear_stopwords", False)),
                merge_query_only=bool(variant.get("merge_query_only", False)),
                enable_rerank=bool(variant.get("enable_rerank", False)),
            )
        )

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "top_k": args.top_k,
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
            "variant_count": len(results),
        },
        "query_rewrite_diagnostics": {
            "question_count": len(rewrite_cache),
            "avg_expansions_per_question": (
                sum(rewrite_lengths) / len(rewrite_lengths) if rewrite_lengths else 0.0
            ),
            "questions_with_expansions": sum(1 for count in rewrite_lengths if count > 0),
            "sample": [
                {
                    "question": question,
                    "expansions": rewrite_cache[question],
                }
                for question in list(rewrite_cache.keys())[:8]
            ],
        },
        "variants": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote ablation report: {args.output}")
    for row in results:
        summary = row["summary"]
        print(
            f"{row['name']}: "
            f"prefix_recall_available_mean={summary['doc_prefix_recall_available_mean']} "
            f"prefix_recall_mean={summary['doc_prefix_recall_mean']} "
            f"chunk_recall_mean={summary['chunk_id_recall_mean']}"
        )


if __name__ == "__main__":
    run()
