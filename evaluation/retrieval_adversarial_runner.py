"""Run adversarial retrieval checks against the current RAG retriever."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = [
    REPO_ROOT / "evaluation" / "cases" / "adversarial_retrieval_cases.json",
    REPO_ROOT / "evaluation" / "cases" / "adversarial_chunk_stress_cases.json",
    REPO_ROOT / "evaluation" / "cases" / "adversarial_retrieval_robustness.json",
    REPO_ROOT / "evaluation" / "cases" / "eval_cases.jsonl",
    REPO_ROOT / "evaluation" / "cases" / "eval_cases_reference.jsonl",
]
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "runs" / "adversarial_retrieval_latest.json"
DEFAULT_CACHE = REPO_ROOT / "evaluation" / "runs" / "adversarial_embeddings.npz"
DEFAULT_CHUNK_CACHE = REPO_ROOT / "evaluation" / "runs" / "adversarial_chunks.json.gz"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.app_config import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    EMBED_MODEL,
    EVAL_TOP_K,
    CHAT_MAX_INPUT_TOKENS,
    CHAT_PREAMBLE,
    CHAT_RESERVED_TOKENS,
    MAX_DOC_TOKENS,
    MAX_PACKED_DOCS,
    QUERY_REWRITE_MODEL,
    RAW_DIR,
    config_diagnostics,
    legacy_retrieval_env_overrides,
)
from rag.corpus import list_docs, load_manifest_index
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
from rag.rag_types import Chunk
import rag.retrieval as retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial retrieval test cases.")
    parser.add_argument(
        "--cases-file",
        type=Path,
        nargs="+",
        default=DEFAULT_CASES,
        help="One or more case files (.json array or .jsonl with one JSON object per line).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-cache-file", type=Path, default=DEFAULT_CHUNK_CACHE)
    parser.add_argument("--top-k", type=int, default=max(24, EVAL_TOP_K))
    parser.add_argument(
        "--retrieval-alpha",
        type=float,
        default=None,
        help="Optional override for RETRIEVAL_ALPHA (0.0 to 1.0) for this run only.",
    )
    parser.add_argument(
        "--use-query-rewrite",
        action="store_true",
        help="Enable LLM query rewriting before retrieval.",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Allow Cohere rerank inside retrieval.",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="Optional split filter for JSONL eval-style cases (all/dev/test/train or comma-separated).",
    )
    parser.add_argument(
        "--measure-packed-recall",
        action="store_true",
        help="Also compute recall against chunks that are actually packed into chat context.",
    )
    parser.add_argument(
        "--pack-max-input-tokens",
        type=int,
        default=CHAT_MAX_INPUT_TOKENS,
        help="Token budget used for packed-recall simulation.",
    )
    parser.add_argument(
        "--pack-reserved-tokens",
        type=int,
        default=CHAT_RESERVED_TOKENS,
        help="Reserved non-document tokens used for packed-recall simulation.",
    )
    parser.add_argument(
        "--pack-max-doc-tokens",
        type=int,
        default=MAX_DOC_TOKENS,
        help="Per-document token cap used for packed-recall simulation.",
    )
    parser.add_argument(
        "--pack-max-docs",
        type=int,
        default=MAX_PACKED_DOCS,
        help="Maximum packed documents used for packed-recall simulation.",
    )
    parser.add_argument(
        "--pack-rerank-top-n",
        type=int,
        default=0,
        help=(
            "If > 0, rerank retrieved chunks to this size before packing "
            "(mirrors app retrieve-wide then rerank-to-app-top-k behavior)."
        ),
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_cases(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            loaded = _load_jsonl(path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                loaded = [row for row in payload if isinstance(row, dict)]
            elif isinstance(payload, dict):
                loaded = [payload]
            else:
                loaded = []

        for row in loaded:
            enriched = dict(row)
            enriched["_source_file"] = path.as_posix()
            rows.append(enriched)
    return rows


def _chunk_signature(chunk_ids: List[str]) -> str:
    blob = "\n".join(chunk_ids).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _file_stat_token(path: Path) -> str:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
        return f"{resolved.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"
    except FileNotFoundError:
        return f"{resolved.as_posix()}|missing"


def _dataset_signature(docs: List[Path]) -> str:
    tokens: List[str] = [
        f"chunk_chars={CHUNK_CHARS}",
        f"chunk_overlap={CHUNK_OVERLAP}",
    ]

    manifest_path = RAW_DIR / "manifest.json"
    if manifest_path.exists():
        tokens.append(_file_stat_token(manifest_path))

    for doc_path in sorted(docs):
        tokens.append(_file_stat_token(doc_path))

    manifest_index = load_manifest_index(RAW_DIR)
    metadata_paths = set()
    for doc_path in docs:
        key = doc_path.resolve().as_posix().lower()
        entry = manifest_index.get(key) or {}
        metadata_path = str(entry.get("_metadata_path_resolved", "")).strip()
        if metadata_path:
            metadata_paths.add(Path(metadata_path))
    for metadata_path in sorted(metadata_paths):
        tokens.append(_file_stat_token(metadata_path))

    payload = "\n".join(tokens).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _deserialize_chunks(rows: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            chunks.append(Chunk(**row))
        except TypeError:
            continue
    return chunks


def _load_or_chunk_docs(docs: List[Path], chunk_cache_file: Path) -> List[Chunk]:
    expected_signature = _dataset_signature(docs)
    chunk_cache_file.parent.mkdir(parents=True, exist_ok=True)

    if chunk_cache_file.exists():
        try:
            with gzip.open(chunk_cache_file, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
            if (
                isinstance(payload, dict)
                and payload.get("dataset_signature") == expected_signature
                and isinstance(payload.get("chunks"), list)
            ):
                cached_chunks = _deserialize_chunks(payload["chunks"])
                if cached_chunks:
                    return cached_chunks
        except Exception:
            pass

    chunks = load_chunks_from_docs(docs)
    payload = {
        "dataset_signature": expected_signature,
        "chunk_count": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    with gzip.open(chunk_cache_file, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True)
    return chunks


def _load_or_embed_chunk_vectors(
    client,
    chunks,
    cache_file: Path,
) -> np.ndarray:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    expected_sig = _chunk_signature(chunk_ids)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        try:
            cached = np.load(cache_file, allow_pickle=True)
            sig = str(cached["chunk_signature"])
            model = str(cached["embed_model"])
            vecs = cached["vectors"]
            if sig == expected_sig and model == EMBED_MODEL and vecs.shape[0] == len(chunks):
                return vecs.astype(np.float32)
        except Exception:
            pass

    vecs = embed_chunks(client, chunks)
    np.savez_compressed(
        cache_file,
        vectors=vecs.astype(np.float32),
        chunk_signature=np.array(expected_sig),
        embed_model=np.array(EMBED_MODEL),
    )
    return vecs


def _doc_prefix(chunk_id: str) -> str:
    if "__" not in chunk_id:
        return chunk_id
    return chunk_id.rsplit("__", 1)[0]


def _normalize_case(row: Dict[str, Any]) -> Dict[str, Any]:
    expected_ids = [
        str(cid).strip()
        for cid in row.get("expected_chunk_ids", row.get("gold_relevant_chunk_ids", []))
        if str(cid).strip()
    ]

    expected_prefixes: List[str] = []
    for key in ("expected_doc_prefixes", "gold_relevant_doc_prefixes"):
        raw_values = row.get(key, [])
        if not isinstance(raw_values, list):
            continue
        expected_prefixes.extend(str(value).strip() for value in raw_values if str(value).strip())

    expected_prefixes.extend(_doc_prefix(chunk_id) for chunk_id in expected_ids)
    expected_prefixes = sorted(set(expected_prefixes))

    return {
        "id": str(row.get("id", "")).strip() or "case_unnamed",
        "question": str(row.get("question", "")).strip(),
        "split": str(row.get("split", "unspecified")).strip().lower(),
        "expected_chunk_ids": expected_ids,
        "expected_doc_prefixes": expected_prefixes,
        "source_file": str(row.get("_source_file", "")),
        "raw": row,
    }


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


def run() -> None:
    args = parse_args()
    raw_cases = _load_cases(args.cases_file)
    normalized_cases = [_normalize_case(row) for row in raw_cases]

    split_filter_raw = str(args.split or "all").strip().lower()
    split_filter = {item.strip() for item in split_filter_raw.split(",") if item.strip()}
    if split_filter and "all" not in split_filter:
        normalized_cases = [
            case for case in normalized_cases if case["split"] in split_filter
        ]

    cases = [
        case
        for case in normalized_cases
        if case["question"] and (case["expected_chunk_ids"] or case["expected_doc_prefixes"])
    ]
    if not cases:
        raise RuntimeError("No usable cases after filtering.")

    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    if not chunks:
        raise RuntimeError("No chunks available.")

    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)
    query_rewrite_cache = (
        _build_query_rewrite_cache(client, cases) if args.use_query_rewrite else {}
    )
    rewrite_lengths = [len(value) for value in query_rewrite_cache.values()]
    original_enable_rerank = retrieval.ENABLE_RERANK
    original_alpha = retrieval.RETRIEVAL_ALPHA
    retrieval.ENABLE_RERANK = bool(args.enable_rerank)
    if args.retrieval_alpha is not None:
        retrieval.RETRIEVAL_ALPHA = min(max(float(args.retrieval_alpha), 0.0), 1.0)
    effective_retrieval_alpha = float(retrieval.RETRIEVAL_ALPHA)
    effective_enable_rerank = bool(retrieval.ENABLE_RERANK)
    effective_max_chunks_per_source = int(retrieval.MAX_CHUNKS_PER_SOURCE)
    available_prefixes = {
        _doc_prefix(chunk.chunk_id) for chunk in chunks if chunk.chunk_id
    }

    results: List[Dict[str, Any]] = []
    try:
        for case in cases:
            question = case["question"]
            expected_ids = list(case["expected_chunk_ids"])
            expected_prefixes = list(case["expected_doc_prefixes"])
            available_expected_prefixes = [
                prefix for prefix in expected_prefixes if prefix in available_prefixes
            ]
            missing_expected_prefixes = [
                prefix for prefix in expected_prefixes if prefix not in available_prefixes
            ]

            query_expansions = query_rewrite_cache.get(question, []) if args.use_query_rewrite else []

            retrieved = retrieval.retrieve(
                client=client,
                query=question,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                k=args.top_k,
                query_expansions=query_expansions,
                chunk_vocabs=chunk_vocabs,
                chunk_modes=chunk_modes,
                chunk_meta_vocabs=chunk_meta_vocabs,
            )

            got_ids = [chunk.chunk_id for chunk, _score in retrieved]
            got_prefixes = [_doc_prefix(cid) for cid in got_ids]

            hit_ids = [cid for cid in expected_ids if cid in got_ids]
            hit_prefixes = [prefix for prefix in expected_prefixes if prefix in got_prefixes]
            hit_available_prefixes = [
                prefix for prefix in available_expected_prefixes if prefix in got_prefixes
            ]

            packed_docs = []
            packed_hit_ids: List[str] = []
            packed_hit_prefixes: List[str] = []
            packed_hit_available_prefixes: List[str] = []
            packed_chunk_ids: List[str] = []
            packed_doc_prefixes: List[str] = []
            packed_stats = None
            if args.measure_packed_recall:
                selected_for_packing = retrieved
                if args.pack_rerank_top_n > 0 and len(retrieved) > args.pack_rerank_top_n:
                    selected_for_packing = retrieval.rerank_retrieved_chunks(
                        client=client,
                        query=question,
                        rows=retrieved,
                        top_n=args.pack_rerank_top_n,
                    )
                packed_docs, packed_stats = pack_retrieved_documents(
                    client=client,
                    question=question,
                    retrieved=selected_for_packing,
                    preamble=CHAT_PREAMBLE,
                    chat_history=[],
                    max_input_tokens=args.pack_max_input_tokens,
                    reserved_tokens=args.pack_reserved_tokens,
                    max_doc_tokens=args.pack_max_doc_tokens,
                    max_packed_docs=args.pack_max_docs,
                )
                packed_chunk_ids = [
                    str(doc.get("chunk_id", "")).strip()
                    for doc in packed_docs
                    if str(doc.get("chunk_id", "")).strip()
                ]
                packed_doc_prefixes = [_doc_prefix(cid) for cid in packed_chunk_ids]
                packed_hit_ids = [cid for cid in expected_ids if cid in packed_chunk_ids]
                packed_hit_prefixes = [
                    prefix for prefix in expected_prefixes if prefix in packed_doc_prefixes
                ]
                packed_hit_available_prefixes = [
                    prefix
                    for prefix in available_expected_prefixes
                    if prefix in packed_doc_prefixes
                ]

            results.append(
                {
                    "id": case["id"],
                    "split": case["split"],
                    "source_file": case["source_file"],
                    "question": question,
                    "expected_chunk_ids": expected_ids,
                    "expected_doc_prefixes": expected_prefixes,
                    "available_expected_doc_prefixes": available_expected_prefixes,
                    "missing_expected_doc_prefixes": missing_expected_prefixes,
                    "query_expansions": query_expansions,
                    "top_k": args.top_k,
                    "retrieved": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "title": chunk.title,
                            "score": float(score),
                            "doc_type": chunk.doc_type,
                            "chunk_type": chunk.chunk_type,
                            "is_exception": bool(chunk.is_exception),
                        }
                        for chunk, score in retrieved
                    ],
                    "hit_chunk_ids": hit_ids,
                    "hit_doc_prefixes": hit_prefixes,
                    "hit_available_doc_prefixes": hit_available_prefixes,
                    "chunk_id_recall": (len(hit_ids) / len(expected_ids)) if expected_ids else None,
                    "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                    if expected_prefixes
                    else None,
                    "doc_prefix_recall_available": (
                        len(hit_available_prefixes) / len(available_expected_prefixes)
                    )
                    if available_expected_prefixes
                    else None,
                    "packed_chunk_ids": packed_chunk_ids,
                    "packed_doc_prefixes": packed_doc_prefixes,
                    "packed_hit_chunk_ids": packed_hit_ids,
                    "packed_hit_doc_prefixes": packed_hit_prefixes,
                    "packed_hit_available_doc_prefixes": packed_hit_available_prefixes,
                    "packed_chunk_id_recall": (
                        (len(packed_hit_ids) / len(expected_ids)) if expected_ids else None
                    ),
                    "packed_doc_prefix_recall": (
                        (len(packed_hit_prefixes) / len(expected_prefixes))
                        if expected_prefixes
                        else None
                    ),
                    "packed_doc_prefix_recall_available": (
                        (
                            len(packed_hit_available_prefixes)
                            / len(available_expected_prefixes)
                        )
                        if available_expected_prefixes
                        else None
                    ),
                    "packing_stats": packed_stats,
                }
            )
    finally:
        retrieval.ENABLE_RERANK = original_enable_rerank
        retrieval.RETRIEVAL_ALPHA = original_alpha

    chunk_recall_values = [row["chunk_id_recall"] for row in results if row["chunk_id_recall"] is not None]
    prefix_recall_values = [row["doc_prefix_recall"] for row in results if row["doc_prefix_recall"] is not None]
    prefix_recall_available_values = [
        row["doc_prefix_recall_available"]
        for row in results
        if row["doc_prefix_recall_available"] is not None
    ]
    packed_chunk_recall_values = [
        row["packed_chunk_id_recall"] for row in results if row["packed_chunk_id_recall"] is not None
    ]
    packed_prefix_recall_values = [
        row["packed_doc_prefix_recall"]
        for row in results
        if row["packed_doc_prefix_recall"] is not None
    ]
    packed_prefix_recall_available_values = [
        row["packed_doc_prefix_recall_available"]
        for row in results
        if row["packed_doc_prefix_recall_available"] is not None
    ]
    expected_chunk_total = sum(len(row["expected_chunk_ids"]) for row in results)
    hit_chunk_total = sum(len(row["hit_chunk_ids"]) for row in results)
    expected_prefix_total = sum(len(row["expected_doc_prefixes"]) for row in results)
    hit_prefix_total = sum(len(row["hit_doc_prefixes"]) for row in results)
    packed_hit_chunk_total = sum(len(row["packed_hit_chunk_ids"]) for row in results)
    packed_hit_prefix_total = sum(len(row["packed_hit_doc_prefixes"]) for row in results)
    chunk_count_dist = Counter(len(row["expected_chunk_ids"]) for row in results)
    prefix_count_dist = Counter(len(row["expected_doc_prefixes"]) for row in results)
    packed_doc_counts = [
        int((row.get("packing_stats") or {}).get("packed_docs", 0))
        for row in results
        if row.get("packing_stats") is not None
    ]

    summary = {
        "case_count": len(results),
        "split_filter": sorted(split_filter) if split_filter else ["all"],
        "chunk_scored_case_count": len(chunk_recall_values),
        "doc_prefix_scored_case_count": len(prefix_recall_values),
        "expected_chunk_total": expected_chunk_total,
        "hit_chunk_total": hit_chunk_total,
        "chunk_id_recall_micro": (hit_chunk_total / expected_chunk_total)
        if expected_chunk_total
        else None,
        "chunk_id_recall_mean": (sum(chunk_recall_values) / len(chunk_recall_values))
        if chunk_recall_values
        else None,
        "expected_doc_prefix_total": expected_prefix_total,
        "hit_doc_prefix_total": hit_prefix_total,
        "doc_prefix_recall_micro": (hit_prefix_total / expected_prefix_total)
        if expected_prefix_total
        else None,
        "doc_prefix_recall_mean": (sum(prefix_recall_values) / len(prefix_recall_values))
        if prefix_recall_values
        else None,
        "doc_prefix_recall_available_mean": (
            sum(prefix_recall_available_values) / len(prefix_recall_available_values)
        )
        if prefix_recall_available_values
        else None,
        "available_prefix_count": len(available_prefixes),
        "zero_chunk_hit_cases": [
            row["id"]
            for row in results
            if row["expected_chunk_ids"] and (not row["hit_chunk_ids"])
        ],
        "zero_prefix_hit_cases": [row["id"] for row in results if not row["hit_doc_prefixes"]],
        "unsupported_prefix_cases": [
            row["id"] for row in results if row["missing_expected_doc_prefixes"]
        ],
        "fully_unsupported_cases": [
            row["id"] for row in results if not row["available_expected_doc_prefixes"]
        ],
        "expected_chunk_count_distribution": {
            str(key): int(value) for key, value in sorted(chunk_count_dist.items())
        },
        "expected_doc_prefix_count_distribution": {
            str(key): int(value) for key, value in sorted(prefix_count_dist.items())
        },
        "packed_chunk_id_recall_micro": (packed_hit_chunk_total / expected_chunk_total)
        if expected_chunk_total
        else None,
        "packed_chunk_id_recall_mean": (
            sum(packed_chunk_recall_values) / len(packed_chunk_recall_values)
        )
        if packed_chunk_recall_values
        else None,
        "packed_doc_prefix_recall_micro": (packed_hit_prefix_total / expected_prefix_total)
        if expected_prefix_total
        else None,
        "packed_doc_prefix_recall_mean": (
            sum(packed_prefix_recall_values) / len(packed_prefix_recall_values)
        )
        if packed_prefix_recall_values
        else None,
        "packed_doc_prefix_recall_available_mean": (
            sum(packed_prefix_recall_available_values)
            / len(packed_prefix_recall_available_values)
        )
        if packed_prefix_recall_available_values
        else None,
        "packed_zero_chunk_hit_cases": [
            row["id"]
            for row in results
            if row["expected_chunk_ids"] and (not row["packed_hit_chunk_ids"])
        ],
        "packed_zero_prefix_hit_cases": [
            row["id"]
            for row in results
            if row["expected_doc_prefixes"] and (not row["packed_hit_doc_prefixes"])
        ],
        "packed_docs_avg": (sum(packed_doc_counts) / len(packed_doc_counts))
        if packed_doc_counts
        else None,
        "packed_docs_min": min(packed_doc_counts) if packed_doc_counts else None,
        "packed_docs_max": max(packed_doc_counts) if packed_doc_counts else None,
    }

    payload = {
        "config": {
            "cases_file": [str(path) for path in args.cases_file],
            "top_k": args.top_k,
            "retrieval_alpha_override": args.retrieval_alpha,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "enable_rerank": bool(args.enable_rerank),
            "embed_model": EMBED_MODEL,
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
            "effective_retrieval_alpha": effective_retrieval_alpha,
            "effective_max_chunks_per_source": effective_max_chunks_per_source,
            "effective_enable_rerank": effective_enable_rerank,
            "env_snapshot": {
                "TOP_K": os.getenv("TOP_K"),
                "RETRIEVAL_ALPHA": os.getenv("RETRIEVAL_ALPHA"),
                "MAX_CHUNKS_PER_SOURCE": os.getenv("MAX_CHUNKS_PER_SOURCE"),
                "ENABLE_RERANK": os.getenv("ENABLE_RERANK"),
                "ENABLE_LLM_QUERY_REWRITE": os.getenv("ENABLE_LLM_QUERY_REWRITE"),
                "QUERY_REWRITE_MODEL": os.getenv("QUERY_REWRITE_MODEL"),
            },
            "effective_query_rewrite_model": QUERY_REWRITE_MODEL,
            "measure_packed_recall": bool(args.measure_packed_recall),
            "pack_max_input_tokens": int(args.pack_max_input_tokens),
            "pack_reserved_tokens": int(args.pack_reserved_tokens),
            "pack_max_doc_tokens": int(args.pack_max_doc_tokens),
            "pack_max_docs": int(args.pack_max_docs),
            "pack_rerank_top_n": int(args.pack_rerank_top_n),
            "config_diagnostics": config_diagnostics(),
            "legacy_retrieval_env_overrides": legacy_retrieval_env_overrides(),
        },
        "query_rewrite_diagnostics": {
            "question_count": len(query_rewrite_cache),
            "avg_expansions_per_question": (
                sum(rewrite_lengths) / len(rewrite_lengths) if rewrite_lengths else 0.0
            ),
            "questions_with_expansions": sum(1 for count in rewrite_lengths if count > 0),
            "sample": [
                {"question": question, "expansions": query_rewrite_cache[question]}
                for question in list(query_rewrite_cache.keys())[:8]
            ],
        },
        "summary": summary,
        "cases": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Adversarial retrieval summary: "
        f"cases={summary['case_count']} "
        f"chunk_id_recall_mean={summary['chunk_id_recall_mean']} "
        f"doc_prefix_recall_mean={summary['doc_prefix_recall_mean']}"
    )
    if args.measure_packed_recall:
        print(
            "Packed-context summary: "
            f"packed_chunk_id_recall_mean={summary['packed_chunk_id_recall_mean']} "
            f"packed_doc_prefix_recall_mean={summary['packed_doc_prefix_recall_mean']} "
            f"packed_docs_avg={summary['packed_docs_avg']}"
        )
    if summary["zero_chunk_hit_cases"]:
        print("Zero chunk-hit cases:", ", ".join(summary["zero_chunk_hit_cases"]))
    if summary["zero_prefix_hit_cases"]:
        print("Zero doc-prefix-hit cases:", ", ".join(summary["zero_prefix_hit_cases"]))
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    run()
