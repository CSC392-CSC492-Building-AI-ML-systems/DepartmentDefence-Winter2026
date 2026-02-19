"""Evaluate contextualized chunk embeddings in isolation from production retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

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
from rag.app_config import EMBED_BATCH, EMBED_MODEL  # noqa: E402
from rag.corpus import list_docs  # noqa: E402
from rag.embedding_client import create_client, embed_texts  # noqa: E402
from rag.query_rewrite import generate_query_expansions  # noqa: E402
import rag.retrieval as retrieval  # noqa: E402

LAB_DIR = REPO_ROOT / "experiments" / "alt_retrieval"
RUNS_DIR = LAB_DIR / "runs"
CACHE_DIR = LAB_DIR / "cache"
DEFAULT_OUTPUT = RUNS_DIR / "contextual_retrieval_eval.json"
DEFAULT_CONTEXT_CACHE = CACHE_DIR / "contextual_embeddings.npz"
CONTEXT_VERSION = "v1_metadata_prefix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate contextual embedding retrieval variant.")
    parser.add_argument(
        "--cases-file",
        type=Path,
        nargs="+",
        default=DEFAULT_CASES,
        help="One or more case files (.json or .jsonl).",
    )
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--context-cache-file", type=Path, default=DEFAULT_CONTEXT_CACHE)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--use-query-rewrite", action="store_true")
    parser.add_argument("--split", default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _chunk_signature(chunk_ids: List[str]) -> str:
    blob = "\n".join(chunk_ids).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


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


def _contextualized_text(chunk) -> str:
    lines = [
        f"DocType: {chunk.doc_type or 'unknown'}",
        f"AuthorityRank: {chunk.authority_rank}",
        f"ChunkType: {chunk.chunk_type}",
    ]
    if chunk.section_heading:
        lines.append(f"SectionHeading: {chunk.section_heading}")
    if chunk.heading_path:
        lines.append(f"HeadingPath: {' > '.join(chunk.heading_path)}")
    if chunk.scope_tags:
        lines.append(f"ScopeTags: {', '.join(chunk.scope_tags)}")
    lines.append(f"IsException: {bool(chunk.is_exception)}")
    lines.append("")
    lines.append(chunk.text)
    return "\n".join(lines)


def _load_or_embed_contextual_vectors(client, chunks, cache_file: Path) -> np.ndarray:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    sig = _chunk_signature(chunk_ids)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        try:
            payload = np.load(cache_file, allow_pickle=True)
            if (
                str(payload["chunk_signature"]) == sig
                and str(payload["embed_model"]) == EMBED_MODEL
                and str(payload["context_version"]) == CONTEXT_VERSION
                and payload["vectors"].shape[0] == len(chunks)
            ):
                return payload["vectors"].astype(np.float32)
        except Exception:
            pass

    texts = [_contextualized_text(chunk) for chunk in chunks]
    rows: List[np.ndarray] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        rows.append(embed_texts(client, batch, input_type="search_document"))
    if rows:
        vecs = np.vstack(rows).astype(np.float32)
    else:
        vecs = np.empty((0, 0), dtype=np.float32)

    np.savez_compressed(
        cache_file,
        vectors=vecs.astype(np.float32),
        chunk_signature=np.array(sig),
        embed_model=np.array(EMBED_MODEL),
        context_version=np.array(CONTEXT_VERSION),
    )
    return vecs


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    return {
        "case_count": len(rows),
        "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values)) if chunk_values else None,
        "doc_prefix_recall_mean": (sum(prefix_values) / len(prefix_values)) if prefix_values else None,
    }


def _evaluate(
    *,
    cases: List[Dict[str, Any]],
    client,
    chunks,
    chunk_vecs,
    top_k: int,
    query_rewrite_cache: Dict[str, List[str]],
    chunk_vocabs,
    chunk_modes,
    chunk_meta_vocabs,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        question = case["question"]
        expected_ids = list(case["expected_chunk_ids"])
        expected_prefixes = list(case["expected_doc_prefixes"])
        retrieved = retrieval.retrieve(
            client=client,
            query=question,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
            k=top_k,
            query_expansions=query_rewrite_cache.get(question, []),
            chunk_vocabs=chunk_vocabs,
            chunk_modes=chunk_modes,
            chunk_meta_vocabs=chunk_meta_vocabs,
        )
        got_ids = [chunk.chunk_id for chunk, _ in retrieved]
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


def run() -> None:
    args = parse_args()
    cases = _select_cases(args.cases_file, args.split)
    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    client = create_client()
    baseline_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    contextual_vecs = _load_or_embed_contextual_vectors(client, chunks, args.context_cache_file)
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
                    client=client, question=question, chat_history=[]
                )

        baseline_rows, baseline_summary = _evaluate(
            cases=cases,
            client=client,
            chunks=chunks,
            chunk_vecs=baseline_vecs,
            top_k=args.top_k,
            query_rewrite_cache=rewrite_cache,
            chunk_vocabs=chunk_vocabs,
            chunk_modes=chunk_modes,
            chunk_meta_vocabs=chunk_meta_vocabs,
        )
        contextual_rows, contextual_summary = _evaluate(
            cases=cases,
            client=client,
            chunks=chunks,
            chunk_vecs=contextual_vecs,
            top_k=args.top_k,
            query_rewrite_cache=rewrite_cache,
            chunk_vocabs=chunk_vocabs,
            chunk_modes=chunk_modes,
            chunk_meta_vocabs=chunk_meta_vocabs,
        )
    finally:
        retrieval.ENABLE_RERANK = original_enable_rerank

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "top_k": args.top_k,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "context_version": CONTEXT_VERSION,
            "effective_retrieval_alpha": float(retrieval.RETRIEVAL_ALPHA),
            "rerank_forced_off": True,
            "cache_file": str(args.cache_file),
            "context_cache_file": str(args.context_cache_file),
        },
        "baseline": {
            "summary": baseline_summary,
            "cases": baseline_rows,
        },
        "contextual": {
            "summary": contextual_summary,
            "cases": contextual_rows,
        },
        "delta": {
            "doc_prefix_recall_mean": (
                (contextual_summary["doc_prefix_recall_mean"] or 0.0)
                - (baseline_summary["doc_prefix_recall_mean"] or 0.0)
            ),
            "chunk_id_recall_mean": (
                (contextual_summary["chunk_id_recall_mean"] or 0.0)
                - (baseline_summary["chunk_id_recall_mean"] or 0.0)
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        "Contextual retrieval summary: "
        f"baseline_prefix={baseline_summary['doc_prefix_recall_mean']} "
        f"contextual_prefix={contextual_summary['doc_prefix_recall_mean']} "
        f"delta_prefix={payload['delta']['doc_prefix_recall_mean']} "
        f"baseline_chunk={baseline_summary['chunk_id_recall_mean']} "
        f"contextual_chunk={contextual_summary['chunk_id_recall_mean']} "
        f"delta_chunk={payload['delta']['chunk_id_recall_mean']}"
    )
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    run()
