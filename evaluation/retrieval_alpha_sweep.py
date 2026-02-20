"""Run a clean sweep over RETRIEVAL_ALPHA for the current retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

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

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "retrieval_alpha_sweep.json"
DEFAULT_ALPHAS = [round(index / 10.0, 2) for index in range(0, 11)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep RETRIEVAL_ALPHA against eval suite.")
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
    parser.add_argument(
        "--alphas",
        default=",".join(str(value) for value in DEFAULT_ALPHAS),
        help="Comma-separated alpha values in [0.0,1.0].",
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help="Include query rewrites while sweeping alpha.",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Allow Cohere rerank during alpha sweep (off by default for clean alpha isolation).",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Optional split filter: all/dev/test/train or comma-separated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def _select_cases(paths: List[Path], split_arg: str) -> List[Dict[str, Any]]:
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


def _build_query_rewrite_cache(client, cases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    cache: Dict[str, List[str]] = {}
    for case in cases:
        question = case["question"]
        if question in cache:
            continue
        cache[question] = generate_query_expansions(
            client=client,
            question=question,
            chat_history=[],
        )
    return cache


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_recall_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_recall_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    return {
        "case_count": len(rows),
        "chunk_id_recall_mean": (sum(chunk_recall_values) / len(chunk_recall_values))
        if chunk_recall_values
        else None,
        "doc_prefix_recall_mean": (sum(prefix_recall_values) / len(prefix_recall_values))
        if prefix_recall_values
        else None,
        "zero_prefix_hit_cases": [row["id"] for row in rows if not row["hit_doc_prefixes"]],
    }


def _evaluate_alpha(
    *,
    client,
    alpha: float,
    cases: List[Dict[str, Any]],
    chunks,
    chunk_vecs,
    top_k: int,
    query_expansion_cache: Dict[str, List[str]],
    use_query_rewrite: bool,
    chunk_vocabs,
    chunk_modes,
    chunk_meta_vocabs,
    available_prefixes: Set[str],
) -> Dict[str, Any]:
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
        got_ids = [chunk.chunk_id for chunk, _ in retrieved]
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
                "chunk_id_recall": (len(hit_ids) / len(expected_ids)) if expected_ids else None,
                "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                if expected_prefixes
                else None,
                "doc_prefix_recall_available": (
                    len(hit_available_prefixes) / len(available_expected_prefixes)
                )
                if available_expected_prefixes
                else None,
                "hit_doc_prefixes": hit_prefixes,
            }
        )

    summary = _summarize(rows)
    available_values = [
        row["doc_prefix_recall_available"]
        for row in rows
        if row["doc_prefix_recall_available"] is not None
    ]
    summary["doc_prefix_recall_available_mean"] = (
        sum(available_values) / len(available_values) if available_values else None
    )
    return {"alpha": alpha, "summary": summary}


def run() -> None:
    args = parse_args()
    alphas = _parse_alphas(args.alphas)
    cases = _select_cases(args.cases_file, args.split)

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    if not chunks:
        raise RuntimeError("No chunks available.")
    available_prefixes = {_doc_prefix(chunk.chunk_id) for chunk in chunks if chunk.chunk_id}

    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)
    query_expansion_cache = (
        _build_query_rewrite_cache(client, cases) if args.use_query_rewrite else {}
    )

    original_alpha = retrieval.RETRIEVAL_ALPHA
    original_enable_rerank = retrieval.ENABLE_RERANK
    try:
        retrieval.ENABLE_RERANK = bool(args.enable_rerank)
        rows: List[Dict[str, Any]] = []
        for alpha in alphas:
            retrieval.RETRIEVAL_ALPHA = float(alpha)
            rows.append(
                _evaluate_alpha(
                    client=client,
                    alpha=float(alpha),
                    cases=cases,
                    chunks=chunks,
                    chunk_vecs=chunk_vecs,
                    top_k=args.top_k,
                    query_expansion_cache=query_expansion_cache,
                    use_query_rewrite=bool(args.use_query_rewrite),
                    chunk_vocabs=chunk_vocabs,
                    chunk_modes=chunk_modes,
                    chunk_meta_vocabs=chunk_meta_vocabs,
                    available_prefixes=available_prefixes,
                )
            )
    finally:
        retrieval.RETRIEVAL_ALPHA = original_alpha
        retrieval.ENABLE_RERANK = original_enable_rerank

    def _sort_key(row: Dict[str, Any]) -> tuple[float, float]:
        summary = row.get("summary", {})
        prefix = float(summary.get("doc_prefix_recall_available_mean") or -1.0)
        chunk = float(summary.get("chunk_id_recall_mean") or -1.0)
        return (prefix, chunk)

    best_prefix_then_chunk = max(rows, key=_sort_key)
    best_chunk_then_prefix = max(
        rows,
        key=lambda row: (
            float(row.get("summary", {}).get("chunk_id_recall_mean") or -1.0),
            float(row.get("summary", {}).get("doc_prefix_recall_available_mean") or -1.0),
        ),
    )
    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "top_k": args.top_k,
            "alphas": alphas,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "enable_rerank": bool(args.enable_rerank),
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
        },
        # Backward-compatible key: prefix-first objective.
        "best_alpha": best_prefix_then_chunk["alpha"],
        "best_alpha_by_prefix_then_chunk": best_prefix_then_chunk["alpha"],
        "best_alpha_by_chunk_then_prefix": best_chunk_then_prefix["alpha"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Alpha sweep summary: "
        f"cases={len(cases)} "
        f"best_alpha_prefix_first={best_prefix_then_chunk['alpha']} "
        f"prefix={best_prefix_then_chunk['summary']['doc_prefix_recall_available_mean']} "
        f"chunk={best_prefix_then_chunk['summary']['chunk_id_recall_mean']} "
        f"| best_alpha_chunk_first={best_chunk_then_prefix['alpha']} "
        f"chunk={best_chunk_then_prefix['summary']['chunk_id_recall_mean']} "
        f"prefix={best_chunk_then_prefix['summary']['doc_prefix_recall_available_mean']}"
    )
    print(f"Wrote sweep: {args.output}")


if __name__ == "__main__":
    run()
