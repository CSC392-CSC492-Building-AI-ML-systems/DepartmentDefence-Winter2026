"""Evaluate expensive retrieval strategies one-at-a-time against the same baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
from rag.app_config import CHAT_MODEL, MAX_CHUNKS_PER_SOURCE, RERANK_MODEL  # noqa: E402
from rag.corpus import list_docs  # noqa: E402
from rag.embedding_client import create_client  # noqa: E402
from rag.query_rewrite import generate_query_expansions  # noqa: E402
from rag.rag_types import Chunk  # noqa: E402
import rag.retrieval as retrieval  # noqa: E402

RUNS_DIR = REPO_ROOT / "experiments" / "alt_retrieval" / "runs"
DEFAULT_OUTPUT = RUNS_DIR / "expensive_methods_eval.json"
METHOD_CHOICES = (
    "large_pool_rerank",
    "decomposition_fusion",
    "two_pass_coverage",
    "llm_reselection",
)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an expensive retrieval method vs baseline.")
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
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
    parser.add_argument("--split", default="all")
    parser.add_argument("--pool-k", type=int, default=180)
    parser.add_argument("--subquery-k", type=int, default=48)
    parser.add_argument("--max-subqueries", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rescue-k", type=int, default=16)
    parser.add_argument("--rescue-weight", type=float, default=0.85)
    parser.add_argument("--clause-overlap-threshold", type=float, default=0.45)
    parser.add_argument("--llm-pool-k", type=int, default=64)
    parser.add_argument("--rerank-model", default=RERANK_MODEL)
    parser.add_argument("--selector-model", default=CHAT_MODEL)
    parser.add_argument("--use-query-rewrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (defaults under experiments/alt_retrieval/runs/).",
    )
    return parser.parse_args()


def _select_cases(paths: Iterable[Path], split_arg: str) -> List[Dict[str, Any]]:
    raw_cases = _load_cases(paths)
    cases = [_normalize_case(row) for row in raw_cases]
    split_filter = {token.strip().lower() for token in str(split_arg).split(",") if token.strip()}
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


def _apply_source_cap(
    ranked: Sequence[Tuple[Chunk, float]],
    *,
    top_k: int,
    max_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> List[Tuple[Chunk, float]]:
    selected: List[Tuple[Chunk, float]] = []
    selected_ids: Set[str] = set()
    per_source_count: Dict[str, int] = {}
    cap = max(1, int(max_per_source))
    target_k = max(1, int(top_k))

    for chunk, score in ranked:
        if chunk.chunk_id in selected_ids:
            continue
        if per_source_count.get(chunk.source_path, 0) >= cap:
            continue
        selected.append((chunk, float(score)))
        selected_ids.add(chunk.chunk_id)
        per_source_count[chunk.source_path] = per_source_count.get(chunk.source_path, 0) + 1
        if len(selected) >= target_k:
            return selected

    # Fallback fill when source cap blocks too many items.
    for chunk, score in ranked:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append((chunk, float(score)))
        selected_ids.add(chunk.chunk_id)
        if len(selected) >= target_k:
            break
    return selected


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(raw)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _summarize_case_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_values = [row["chunk_id_recall"] for row in rows if row["chunk_id_recall"] is not None]
    prefix_values = [row["doc_prefix_recall"] for row in rows if row["doc_prefix_recall"] is not None]
    expected_chunk_total = sum(len(row["expected_chunk_ids"]) for row in rows)
    hit_chunk_total = sum(len(row["hit_chunk_ids"]) for row in rows)
    expected_prefix_total = sum(len(row["expected_doc_prefixes"]) for row in rows)
    hit_prefix_total = sum(len(row["hit_doc_prefixes"]) for row in rows)
    return {
        "case_count": len(rows),
        "chunk_scored_case_count": len(chunk_values),
        "doc_prefix_scored_case_count": len(prefix_values),
        "expected_chunk_total": expected_chunk_total,
        "hit_chunk_total": hit_chunk_total,
        "chunk_id_recall_mean": (sum(chunk_values) / len(chunk_values)) if chunk_values else None,
        "chunk_id_recall_micro": (hit_chunk_total / expected_chunk_total)
        if expected_chunk_total
        else None,
        "expected_doc_prefix_total": expected_prefix_total,
        "hit_doc_prefix_total": hit_prefix_total,
        "doc_prefix_recall_mean": (sum(prefix_values) / len(prefix_values)) if prefix_values else None,
        "doc_prefix_recall_micro": (hit_prefix_total / expected_prefix_total)
        if expected_prefix_total
        else None,
        "zero_chunk_hit_cases": [
            row["id"] for row in rows if row["expected_chunk_ids"] and (not row["hit_chunk_ids"])
        ],
        "zero_prefix_hit_cases": [row["id"] for row in rows if not row["hit_doc_prefixes"]],
    }


def _evaluate_cases(
    *,
    cases: Sequence[Dict[str, Any]],
    retrieve_fn: Callable[[Dict[str, Any]], List[Tuple[Chunk, float]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        expected_ids = list(case["expected_chunk_ids"])
        expected_prefixes = list(case["expected_doc_prefixes"])
        retrieved = retrieve_fn(case)
        got_ids = [chunk.chunk_id for chunk, _ in retrieved]
        got_prefixes = [_doc_prefix(chunk_id) for chunk_id in got_ids]
        hit_ids = [chunk_id for chunk_id in expected_ids if chunk_id in got_ids]
        hit_prefixes = [prefix for prefix in expected_prefixes if prefix in got_prefixes]
        rows.append(
            {
                "id": case["id"],
                "split": case["split"],
                "question": case["question"],
                "expected_chunk_ids": expected_ids,
                "expected_doc_prefixes": expected_prefixes,
                "hit_chunk_ids": hit_ids,
                "hit_doc_prefixes": hit_prefixes,
                "chunk_id_recall": (len(hit_ids) / len(expected_ids)) if expected_ids else None,
                "doc_prefix_recall": (len(hit_prefixes) / len(expected_prefixes))
                if expected_prefixes
                else None,
                "retrieved_top": got_ids[: min(12, len(got_ids))],
            }
        )
    return rows, _summarize_case_rows(rows)


def _build_case_delta(
    baseline_rows: Sequence[Dict[str, Any]],
    method_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    out: List[Dict[str, Any]] = []
    for row in method_rows:
        ref = baseline_by_id.get(row["id"])
        if not ref:
            continue
        out.append(
            {
                "id": row["id"],
                "baseline_chunk_id_recall": ref["chunk_id_recall"],
                "method_chunk_id_recall": row["chunk_id_recall"],
                "delta_chunk_id_recall": (
                    None
                    if (ref["chunk_id_recall"] is None or row["chunk_id_recall"] is None)
                    else (row["chunk_id_recall"] - ref["chunk_id_recall"])
                ),
                "baseline_doc_prefix_recall": ref["doc_prefix_recall"],
                "method_doc_prefix_recall": row["doc_prefix_recall"],
                "delta_doc_prefix_recall": (
                    None
                    if (ref["doc_prefix_recall"] is None or row["doc_prefix_recall"] is None)
                    else (row["doc_prefix_recall"] - ref["doc_prefix_recall"])
                ),
            }
        )
    return out


def _summary_delta(baseline: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Optional[float]]:
    keys = (
        "chunk_id_recall_mean",
        "chunk_id_recall_micro",
        "doc_prefix_recall_mean",
        "doc_prefix_recall_micro",
    )
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        lhs = baseline.get(key)
        rhs = method.get(key)
        delta_key = f"delta_{key}"
        out[delta_key] = None if (lhs is None or rhs is None) else float(rhs) - float(lhs)
    return out


def run() -> None:
    args = parse_args()
    cases = _select_cases(args.cases_file, args.split)
    docs = list_docs()
    chunks = _load_or_chunk_docs(docs, args.chunk_cache_file)
    client = create_client()
    chunk_vecs = _load_or_embed_chunk_vectors(client, chunks, args.cache_file)
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = retrieval.build_chunk_features(chunks)
    chunk_by_id: Dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_vocab_by_id: Dict[str, Set[str]] = {
        chunk.chunk_id: chunk_vocabs[index] for index, chunk in enumerate(chunks)
    }

    original_enable_rerank = retrieval.ENABLE_RERANK
    retrieval.ENABLE_RERANK = False
    original_alpha = retrieval.RETRIEVAL_ALPHA

    rewrite_cache: Dict[str, List[str]] = {}
    if args.use_query_rewrite:
        for case in cases:
            question = case["question"]
            if question in rewrite_cache:
                continue
            rewrite_cache[question] = generate_query_expansions(
                client=client, question=question, chat_history=[]
            )

    def baseline_retrieve(case: Dict[str, Any]) -> List[Tuple[Chunk, float]]:
        question = case["question"]
        return retrieval.retrieve(
            client=client,
            query=question,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
            k=args.top_k,
            query_expansions=[],
            chunk_vocabs=chunk_vocabs,
            chunk_modes=chunk_modes,
            chunk_meta_vocabs=chunk_meta_vocabs,
        )

    def method_large_pool_rerank(case: Dict[str, Any]) -> List[Tuple[Chunk, float]]:
        question = case["question"]
        pool = retrieval.retrieve(
            client=client,
            query=question,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
            k=max(args.pool_k, args.top_k),
            query_expansions=[],
            chunk_vocabs=chunk_vocabs,
            chunk_modes=chunk_modes,
            chunk_meta_vocabs=chunk_meta_vocabs,
        )
        if not pool:
            return []

        try:
            response = client.rerank(
                model=args.rerank_model,
                query=question,
                documents=[f"{chunk.title}\n{chunk.text}" for chunk, _ in pool],
                top_n=len(pool),
                return_documents=False,
            )
            rerank_rows = sorted(
                (
                    (int(row.index), float(row.relevance_score))
                    for row in response.results
                    if 0 <= int(row.index) < len(pool)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            ranked = [(pool[index][0], score) for index, score in rerank_rows]
        except Exception:
            ranked = list(pool)
        return _apply_source_cap(ranked, top_k=args.top_k)

    def _decomposition_queries(question: str) -> List[str]:
        out: List[str] = [question]
        seen = {question.strip().lower()}
        if args.use_query_rewrite:
            for value in rewrite_cache.get(question, []):
                key = value.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(value.strip())
                if len(out) >= args.max_subqueries:
                    return out
        clauses = retrieval._extract_query_clauses(question, max_clauses=args.max_subqueries)  # noqa: SLF001
        for value in clauses:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(value.strip())
            if len(out) >= args.max_subqueries:
                break
        return out

    def method_decomposition_fusion(case: Dict[str, Any]) -> List[Tuple[Chunk, float]]:
        question = case["question"]
        subqueries = _decomposition_queries(question)
        rrf_scores: Dict[str, float] = defaultdict(float)
        for subquery in subqueries:
            rows = retrieval.retrieve(
                client=client,
                query=subquery,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                k=max(args.subquery_k, args.top_k),
                query_expansions=[],
                chunk_vocabs=chunk_vocabs,
                chunk_modes=chunk_modes,
                chunk_meta_vocabs=chunk_meta_vocabs,
            )
            for rank, (chunk, _score) in enumerate(rows, start=1):
                rrf_scores[chunk.chunk_id] += 1.0 / (args.rrf_k + rank)

        ranked = sorted(
            (
                (chunk_by_id[chunk_id], score)
                for chunk_id, score in rrf_scores.items()
                if chunk_id in chunk_by_id
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked:
            return baseline_retrieve(case)
        return _apply_source_cap(ranked, top_k=args.top_k)

    def method_two_pass_coverage(case: Dict[str, Any]) -> List[Tuple[Chunk, float]]:
        question = case["question"]
        primary = baseline_retrieve(case)
        clauses = retrieval._extract_query_clauses(question, max_clauses=4)  # noqa: SLF001
        if not clauses:
            return primary

        missing_clauses: List[str] = []
        for clause in clauses:
            tokens = retrieval._content_tokens(clause)  # noqa: SLF001
            if not tokens:
                continue
            best_overlap = 0.0
            for chunk, _ in primary:
                vocab = chunk_vocab_by_id.get(chunk.chunk_id, set())
                overlap = retrieval._lexical_overlap_vocab(tokens, vocab)  # noqa: SLF001
                if overlap > best_overlap:
                    best_overlap = overlap
            if best_overlap < args.clause_overlap_threshold:
                missing_clauses.append(clause)

        if not missing_clauses:
            return primary

        fusion: Dict[str, float] = defaultdict(float)
        for rank, (chunk, _score) in enumerate(primary, start=1):
            fusion[chunk.chunk_id] += 1.0 / (args.rrf_k + rank)

        for clause in missing_clauses:
            rescue_rows = retrieval.retrieve(
                client=client,
                query=clause,
                chunks=chunks,
                chunk_vecs=chunk_vecs,
                k=max(args.rescue_k, args.top_k // 2),
                query_expansions=[],
                chunk_vocabs=chunk_vocabs,
                chunk_modes=chunk_modes,
                chunk_meta_vocabs=chunk_meta_vocabs,
            )
            for rank, (chunk, _score) in enumerate(rescue_rows, start=1):
                fusion[chunk.chunk_id] += args.rescue_weight / (args.rrf_k + rank)

        ranked = sorted(
            (
                (chunk_by_id[chunk_id], score)
                for chunk_id, score in fusion.items()
                if chunk_id in chunk_by_id
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return _apply_source_cap(ranked, top_k=args.top_k)

    def method_llm_reselection(case: Dict[str, Any]) -> List[Tuple[Chunk, float]]:
        question = case["question"]
        pool = retrieval.retrieve(
            client=client,
            query=question,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
            k=max(args.llm_pool_k, args.top_k),
            query_expansions=[],
            chunk_vocabs=chunk_vocabs,
            chunk_modes=chunk_modes,
            chunk_meta_vocabs=chunk_meta_vocabs,
        )
        if not pool:
            return []

        candidate_lines: List[str] = []
        pool_map = {chunk.chunk_id: (chunk, float(score)) for chunk, score in pool}
        for index, (chunk, _score) in enumerate(pool, start=1):
            snippet = " ".join(chunk.text.split())[:180]
            candidate_lines.append(
                f"{index}. CHUNK_ID={chunk.chunk_id} | TITLE={chunk.title} | "
                f"SECTION={chunk.section_heading or 'n/a'} | SNIPPET={snippet}"
            )

        prompt = (
            "Select evidence chunks for policy retrieval.\n"
            f"Return ONLY JSON object {{\"chunk_ids\": [..]}} with at most {args.top_k} chunk IDs.\n"
            "Goal: maximize coverage across all parts of the question, avoid near-duplicates.\n\n"
            f"Question:\n{question}\n\n"
            "Candidates:\n"
            + "\n".join(candidate_lines)
        )

        chosen_ids: List[str] = []
        try:
            response = client.chat(
                model=args.selector_model,
                message=prompt,
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            payload = _parse_json_object(response.text or "")
            raw_ids = payload.get("chunk_ids", [])
            if isinstance(raw_ids, list):
                for value in raw_ids:
                    chunk_id = str(value).strip()
                    if not chunk_id or chunk_id in chosen_ids or chunk_id not in pool_map:
                        continue
                    chosen_ids.append(chunk_id)
                    if len(chosen_ids) >= args.top_k:
                        break
        except Exception:
            chosen_ids = []

        ranked: List[Tuple[Chunk, float]] = []
        seen: Set[str] = set()
        for chunk_id in chosen_ids:
            chunk, score = pool_map[chunk_id]
            ranked.append((chunk, score))
            seen.add(chunk_id)
        for chunk, score in pool:
            if chunk.chunk_id in seen:
                continue
            ranked.append((chunk, float(score)))
            if len(ranked) >= max(args.top_k * 2, args.llm_pool_k):
                break
        return _apply_source_cap(ranked, top_k=args.top_k)

    method_fn_map: Dict[str, Callable[[Dict[str, Any]], List[Tuple[Chunk, float]]]] = {
        "large_pool_rerank": method_large_pool_rerank,
        "decomposition_fusion": method_decomposition_fusion,
        "two_pass_coverage": method_two_pass_coverage,
        "llm_reselection": method_llm_reselection,
    }
    method_fn = method_fn_map[args.method]

    try:
        baseline_rows, baseline_summary = _evaluate_cases(cases=cases, retrieve_fn=baseline_retrieve)
        method_rows, method_summary = _evaluate_cases(cases=cases, retrieve_fn=method_fn)
    finally:
        retrieval.ENABLE_RERANK = original_enable_rerank
        retrieval.RETRIEVAL_ALPHA = original_alpha

    summary_delta = _summary_delta(baseline_summary, method_summary)
    case_delta = _build_case_delta(baseline_rows, method_rows)

    payload = {
        "config": {
            "method": args.method,
            "cases_file": [str(path) for path in args.cases_file],
            "split": args.split,
            "top_k": args.top_k,
            "pool_k": args.pool_k,
            "subquery_k": args.subquery_k,
            "max_subqueries": args.max_subqueries,
            "rrf_k": args.rrf_k,
            "rescue_k": args.rescue_k,
            "rescue_weight": args.rescue_weight,
            "clause_overlap_threshold": args.clause_overlap_threshold,
            "llm_pool_k": args.llm_pool_k,
            "rerank_model": args.rerank_model,
            "selector_model": args.selector_model,
            "use_query_rewrite": bool(args.use_query_rewrite),
            "cache_file": str(args.cache_file),
            "chunk_cache_file": str(args.chunk_cache_file),
            "effective_retrieval_alpha": float(retrieval.RETRIEVAL_ALPHA),
            "effective_max_chunks_per_source": int(retrieval.MAX_CHUNKS_PER_SOURCE),
        },
        "baseline": {
            "summary": baseline_summary,
            "cases": baseline_rows,
        },
        "method": {
            "name": args.method,
            "summary": method_summary,
            "cases": method_rows,
        },
        "delta": {
            **summary_delta,
            "case_level": case_delta,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Expensive-method summary: "
        f"method={args.method} "
        f"baseline_chunk={baseline_summary['chunk_id_recall_mean']} "
        f"method_chunk={method_summary['chunk_id_recall_mean']} "
        f"baseline_prefix={baseline_summary['doc_prefix_recall_mean']} "
        f"method_prefix={method_summary['doc_prefix_recall_mean']}"
    )
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    run()
