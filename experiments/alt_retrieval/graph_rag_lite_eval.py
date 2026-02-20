"""Evaluate a lightweight GraphRAG-style retrieval expansion in isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

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
from rag.corpus import list_docs  # noqa: E402
from rag.embedding_client import create_client  # noqa: E402
from rag.query_rewrite import generate_query_expansions  # noqa: E402
from rag.rag_types import Chunk  # noqa: E402
import rag.retrieval as retrieval  # noqa: E402

RUNS_DIR = REPO_ROOT / "experiments" / "alt_retrieval" / "runs"
DEFAULT_OUTPUT = RUNS_DIR / "graph_rag_lite_eval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG-lite retrieval expansion.")
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
    parser.add_argument("--pool-k", type=int, default=48)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--graph-inject-k", type=int, default=4)
    parser.add_argument("--use-query-rewrite", action="store_true")
    parser.add_argument("--split", default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _select_cases(paths: List[Path], split_arg: str) -> List[Dict[str, Any]]:
    raw_cases = _load_cases(paths)
    cases = [_normalize_case(row) for row in raw_cases]
    split_filter = {value.strip().lower() for value in str(split_arg).split(",") if value.strip()}
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


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    return {
        "case_count": len(rows),
        "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values)) if chunk_values else None,
        "doc_prefix_recall_mean": (sum(prefix_values) / len(prefix_values)) if prefix_values else None,
    }


def _evaluate_rows(
    *,
    cases: List[Dict[str, Any]],
    chunks: List[Chunk],
    baseline_retrieved_by_case: Dict[str, List[Tuple[Chunk, float]]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        expected_ids = list(case["expected_chunk_ids"])
        expected_prefixes = list(case["expected_doc_prefixes"])
        retrieved = baseline_retrieved_by_case[case["id"]]
        got_ids = [chunk.chunk_id for chunk, _ in retrieved[:top_k]]
        got_prefixes = [_doc_prefix(value) for value in got_ids]
        hit_ids = [value for value in expected_ids if value in got_ids]
        hit_prefixes = [value for value in expected_prefixes if value in got_prefixes]
        rows.append(
            {
                "id": case["id"],
                "chunk_id_recall": (len(hit_ids) / len(expected_ids)) if expected_ids else None,
                "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                if expected_prefixes
                else None,
            }
        )
    return rows, _summary(rows)


def _graph_expand_results(
    pool: List[Tuple[Chunk, float]],
    top_k: int,
    seed_k: int,
    graph_inject_k: int,
) -> List[Tuple[Chunk, float]]:
    if not pool:
        return []

    seed = pool[: max(1, seed_k)]
    related_doc_ids: Set[str] = set()
    for chunk, _ in seed:
        if chunk.parent_doc_id:
            related_doc_ids.add(chunk.parent_doc_id)
        related_doc_ids.update(value for value in chunk.child_doc_ids if value)
        related_doc_ids.update(value for value in chunk.lineage_doc_ids if value and value != chunk.doc_id)

    injected: List[Tuple[Chunk, float]] = []
    seed_ids = {chunk.chunk_id for chunk, _ in seed}
    for chunk, score in pool:
        if chunk.chunk_id in seed_ids:
            continue
        if chunk.doc_id and chunk.doc_id in related_doc_ids:
            injected.append((chunk, score))
        if len(injected) >= max(0, graph_inject_k):
            break

    final: List[Tuple[Chunk, float]] = []
    seen = set()
    for chunk, score in (seed + injected + pool):
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        final.append((chunk, score))
        if len(final) >= max(1, top_k):
            break
    return final


def run() -> None:
    args = parse_args()
    cases = _select_cases(args.cases_file, args.split)
    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)
    original_enable_rerank = retrieval.ENABLE_RERANK
    retrieval.ENABLE_RERANK = False

    try:
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

        baseline_by_case: Dict[str, List[Tuple[Chunk, float]]] = {}
        graph_by_case: Dict[str, List[Tuple[Chunk, float]]] = {}
        for case in cases:
            question = case["question"]
            query_expansions = rewrite_cache.get(question, [])
            pool = retrieval.retrieve(
                client=client,
                query=question,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                k=max(args.pool_k, args.top_k),
                query_expansions=query_expansions,
                chunk_vocabs=chunk_vocabs,
                chunk_modes=chunk_modes,
                chunk_meta_vocabs=chunk_meta_vocabs,
            )
            baseline_by_case[case["id"]] = pool[: args.top_k]
            graph_by_case[case["id"]] = _graph_expand_results(
                pool=pool,
                top_k=args.top_k,
                seed_k=args.seed_k,
                graph_inject_k=args.graph_inject_k,
            )

        baseline_rows, baseline_summary = _evaluate_rows(
            cases=cases,
            chunks=chunks,
            baseline_retrieved_by_case=baseline_by_case,
            top_k=args.top_k,
        )
        graph_rows, graph_summary = _evaluate_rows(
            cases=cases,
            chunks=chunks,
            baseline_retrieved_by_case=graph_by_case,
            top_k=args.top_k,
        )
    finally:
        retrieval.ENABLE_RERANK = original_enable_rerank

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "top_k": args.top_k,
            "pool_k": args.pool_k,
            "seed_k": args.seed_k,
            "graph_inject_k": args.graph_inject_k,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "effective_retrieval_alpha": float(retrieval.RETRIEVAL_ALPHA),
            "rerank_forced_off": True,
        },
        "baseline": {
            "summary": baseline_summary,
            "cases": baseline_rows,
        },
        "graph_rag_lite": {
            "summary": graph_summary,
            "cases": graph_rows,
        },
        "delta": {
            "doc_prefix_recall_mean": (
                (graph_summary["doc_prefix_recall_mean"] or 0.0)
                - (baseline_summary["doc_prefix_recall_mean"] or 0.0)
            ),
            "chunk_id_recall_mean": (
                (graph_summary["chunk_id_recall_mean"] or 0.0)
                - (baseline_summary["chunk_id_recall_mean"] or 0.0)
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        "GraphRAG-lite summary: "
        f"baseline_prefix={baseline_summary['doc_prefix_recall_mean']} "
        f"graph_prefix={graph_summary['doc_prefix_recall_mean']} "
        f"delta_prefix={payload['delta']['doc_prefix_recall_mean']} "
        f"baseline_chunk={baseline_summary['chunk_id_recall_mean']} "
        f"graph_chunk={graph_summary['chunk_id_recall_mean']} "
        f"delta_chunk={payload['delta']['chunk_id_recall_mean']}"
    )
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    run()
