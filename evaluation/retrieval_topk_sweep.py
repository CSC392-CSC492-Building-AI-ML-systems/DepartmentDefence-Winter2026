"""Sweep top_k and report retrieval metric sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.retrieval_adversarial_runner import (
    DEFAULT_CACHE,
    DEFAULT_CASES,
    DEFAULT_CHUNK_CACHE,
    _doc_prefix,
    _load_cases,
    _load_or_chunk_docs,
    _load_or_embed_chunk_vectors,
    _normalize_case,
)
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.query_rewrite import generate_query_expansions
import rag.retrieval as retrieval

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "retrieval_topk_sweep.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep top_k over retrieval evaluation cases.")
    parser.add_argument(
        "--cases-file",
        type=Path,
        nargs="+",
        default=DEFAULT_CASES,
        help="One or more case files (.json or .jsonl).",
    )
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument(
        "--k-values",
        default="4,8,12,16,24,32",
        help="Comma-separated top_k values.",
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help="Enable query rewrite while sweeping.",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Allow rerank during sweep (off by default for clean top_k isolation).",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Optional split filter: all/dev/test/train or comma-separated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _parse_k_values(raw: str) -> List[int]:
    values: List[int] = []
    for token in str(raw).split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(max(1, int(stripped)))
    if not values:
        raise ValueError("No k-values provided.")
    return values


def _select_cases(paths: List[Path], split_arg: str) -> List[dict]:
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


def _build_query_rewrite_cache(client, cases: List[dict]) -> dict:
    cache = {}
    for case in cases:
        question = case["question"]
        if question in cache:
            continue
        cache[question] = generate_query_expansions(client=client, question=question, chat_history=[])
    return cache


def run() -> None:
    args = parse_args()
    k_values = _parse_k_values(args.k_values)
    cases = _select_cases(args.cases_file, args.split)

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)
    query_rewrite_cache = (
        _build_query_rewrite_cache(client, cases) if args.use_query_rewrite else {}
    )
    available_prefixes = {_doc_prefix(chunk.chunk_id) for chunk in chunks if chunk.chunk_id}
    original_enable_rerank = retrieval.ENABLE_RERANK
    retrieval.ENABLE_RERANK = bool(args.enable_rerank)

    rows = []
    try:
        for k in k_values:
            case_rows = []
            for case in cases:
                question = case["question"]
                expected_ids = list(case["expected_chunk_ids"])
                expected_prefixes = list(case["expected_doc_prefixes"])
                available_expected_prefixes = [
                    prefix for prefix in expected_prefixes if prefix in available_prefixes
                ]
                query_expansions = query_rewrite_cache.get(question, [])
                retrieved = retrieval.retrieve(
                    client=client,
                    query=question,
                    chunks=chunks,
                    chunk_vecs=chunk_vecs,
                    k=k,
                    query_expansions=query_expansions,
                    chunk_vocabs=chunk_vocabs,
                    chunk_modes=chunk_modes,
                    chunk_meta_vocabs=chunk_meta_vocabs,
                )
                got_ids = [chunk.chunk_id for chunk, _ in retrieved]
                got_prefixes = [_doc_prefix(value) for value in got_ids]
                hit_ids = [value for value in expected_ids if value in got_ids]
                hit_prefixes = [value for value in expected_prefixes if value in got_prefixes]
                hit_available_prefixes = [
                    value for value in available_expected_prefixes if value in got_prefixes
                ]
                case_rows.append(
                    {
                        "chunk_id_recall": (len(hit_ids) / len(expected_ids)) if expected_ids else None,
                        "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                        if expected_prefixes
                        else None,
                        "doc_prefix_recall_available": (
                            len(hit_available_prefixes) / len(available_expected_prefixes)
                        )
                        if available_expected_prefixes
                        else None,
                    }
                )

            chunk_values = [row["chunk_id_recall"] for row in case_rows if row["chunk_id_recall"] is not None]
            prefix_values = [
                row["doc_prefix_recall_available"]
                for row in case_rows
                if row["doc_prefix_recall_available"] is not None
            ]
            rows.append(
                {
                    "top_k": k,
                    "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values))
                    if chunk_values
                    else None,
                    "doc_prefix_recall_available_mean": (sum(prefix_values) / len(prefix_values))
                    if prefix_values
                    else None,
                }
            )
    finally:
        retrieval.ENABLE_RERANK = original_enable_rerank

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "k_values": k_values,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "enable_rerank": bool(args.enable_rerank),
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
            "effective_retrieval_alpha": float(retrieval.RETRIEVAL_ALPHA),
            "effective_max_chunks_per_source": int(retrieval.MAX_CHUNKS_PER_SOURCE),
            "effective_enable_rerank": bool(retrieval.ENABLE_RERANK),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    best_prefix_then_chunk = max(
        rows,
        key=lambda row: (
            float(row["doc_prefix_recall_available_mean"] or -1.0),
            float(row["chunk_id_recall_mean"] or -1.0),
        ),
    )
    best_chunk_then_prefix = max(
        rows,
        key=lambda row: (
            float(row["chunk_id_recall_mean"] or -1.0),
            float(row["doc_prefix_recall_available_mean"] or -1.0),
        ),
    )
    payload["best_top_k_by_prefix_then_chunk"] = int(best_prefix_then_chunk["top_k"])
    payload["best_top_k_by_chunk_then_prefix"] = int(best_chunk_then_prefix["top_k"])
    print(
        "Top-k sweep summary: "
        f"cases={len(cases)} "
        f"best_k_prefix_first={best_prefix_then_chunk['top_k']} "
        f"prefix={best_prefix_then_chunk['doc_prefix_recall_available_mean']} "
        f"chunk={best_prefix_then_chunk['chunk_id_recall_mean']} "
        f"| best_k_chunk_first={best_chunk_then_prefix['top_k']} "
        f"chunk={best_chunk_then_prefix['chunk_id_recall_mean']} "
        f"prefix={best_chunk_then_prefix['doc_prefix_recall_available_mean']}"
    )
    print(f"Wrote sweep: {args.output}")


if __name__ == "__main__":
    run()
