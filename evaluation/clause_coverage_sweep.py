"""Sweep clause-coverage parameters in retrieval and report metric sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
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
import rag.retrieval as retrieval  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "clause_coverage_sweep.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep clause-coverage parameters.")
    parser.add_argument(
        "--cases-file",
        type=Path,
        nargs="+",
        default=DEFAULT_CASES,
        help="One or more case files (.json or .jsonl).",
    )
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--split", default="all")
    parser.add_argument(
        "--coverage-values",
        default="1,2,3",
        help="Comma-separated MAX_CLAUSE_COVERAGE values.",
    )
    parser.add_argument(
        "--overlap-values",
        default="0.35,0.45,0.55",
        help="Comma-separated CLAUSE_COVERAGE_MIN_OVERLAP values.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [max(0, int(token.strip())) for token in str(raw).split(",") if token.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [min(max(float(token.strip()), 0.0), 1.0) for token in str(raw).split(",") if token.strip()]


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


def run() -> None:
    args = parse_args()
    coverage_values = _parse_ints(args.coverage_values)
    overlap_values = _parse_floats(args.overlap_values)
    cases = _select_cases(args.cases_file, args.split)

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)

    original_enable_rerank = retrieval.ENABLE_RERANK
    original_coverage = retrieval.MAX_CLAUSE_COVERAGE
    original_overlap = retrieval.CLAUSE_COVERAGE_MIN_OVERLAP
    retrieval.ENABLE_RERANK = False

    rows = []
    try:
        for coverage in coverage_values:
            for overlap in overlap_values:
                retrieval.MAX_CLAUSE_COVERAGE = int(coverage)
                retrieval.CLAUSE_COVERAGE_MIN_OVERLAP = float(overlap)

                case_rows = []
                for case in cases:
                    expected_ids = list(case["expected_chunk_ids"])
                    expected_prefixes = list(case["expected_doc_prefixes"])
                    retrieved = retrieval.retrieve(
                        client=client,
                        query=case["question"],
                        chunks=chunks,
                        chunk_vecs=chunk_vecs,
                        k=args.top_k,
                        query_expansions=[],
                        chunk_vocabs=chunk_vocabs,
                        chunk_modes=chunk_modes,
                        chunk_meta_vocabs=chunk_meta_vocabs,
                    )
                    got_ids = [chunk.chunk_id for chunk, _ in retrieved]
                    got_prefixes = [_doc_prefix(value) for value in got_ids]
                    hit_ids = [value for value in expected_ids if value in got_ids]
                    hit_prefixes = [value for value in expected_prefixes if value in got_prefixes]
                    case_rows.append(
                        {
                            "chunk_id_recall": (len(hit_ids) / len(expected_ids))
                            if expected_ids
                            else None,
                            "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                            if expected_prefixes
                            else None,
                        }
                    )

                chunk_values = [
                    value["chunk_id_recall"] for value in case_rows if value["chunk_id_recall"] is not None
                ]
                prefix_values = [
                    value["doc_prefix_recall"] for value in case_rows if value["doc_prefix_recall"] is not None
                ]
                rows.append(
                    {
                        "max_clause_coverage": int(coverage),
                        "min_overlap": float(overlap),
                        "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values))
                        if chunk_values
                        else None,
                        "doc_prefix_recall_mean": (sum(prefix_values) / len(prefix_values))
                        if prefix_values
                        else None,
                    }
                )
    finally:
        retrieval.ENABLE_RERANK = original_enable_rerank
        retrieval.MAX_CLAUSE_COVERAGE = original_coverage
        retrieval.CLAUSE_COVERAGE_MIN_OVERLAP = original_overlap

    best = max(
        rows,
        key=lambda row: (
            float(row["doc_prefix_recall_mean"] or -1.0),
            float(row["chunk_id_recall_mean"] or -1.0),
        ),
    )
    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "top_k": args.top_k,
            "coverage_values": coverage_values,
            "overlap_values": overlap_values,
            "effective_retrieval_alpha": float(retrieval.RETRIEVAL_ALPHA),
            "rerank_forced_off": True,
        },
        "best": best,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        "Clause sweep summary: "
        f"best_coverage={best['max_clause_coverage']} "
        f"best_overlap={best['min_overlap']} "
        f"best_prefix={best['doc_prefix_recall_mean']} "
        f"best_chunk={best['chunk_id_recall_mean']}"
    )
    print(f"Wrote sweep: {args.output}")


if __name__ == "__main__":
    run()
