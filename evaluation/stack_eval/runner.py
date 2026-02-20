"""Main orchestration for structured PolicyRAG stack evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_metrics import (
    abstention_accuracy,
    citation_support_metrics,
    forbidden_claim_metrics,
    reference_answer_similarity,
    required_claim_metrics,
)
from .cases import case_forbidden_claims, case_list, case_required_claims, load_cases
from .common import round_float, split_sentences
from .judge import judge_answer
from .paths import CASES_DIR, REPO_ROOT, RUNS_DIR
from .retrieval_metrics import doc_prefix_from_chunk_id, retrieval_evidence_coverage
from .summary import summarize_case_rows, summarize_ci95, timing_summary

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.app_config import (  # noqa: E402
    CHAT_MAX_INPUT_TOKENS,
    CHAT_MAX_OUTPUT_TOKENS,
    CHAT_MODEL,
    CHAT_PREAMBLE,
    EVAL_TOP_K,
)
from rag.corpus import list_docs  # noqa: E402
from rag.embedding_client import create_client  # noqa: E402
from rag.pipeline import embed_chunks, load_chunks_from_docs  # noqa: E402
from rag.prompting import pack_retrieved_documents  # noqa: E402
from rag.query_rewrite import generate_query_expansions  # noqa: E402
from rag.rag_types import Chunk  # noqa: E402
from rag.retrieval import build_chunk_features, retrieve  # noqa: E402


def serialize_retrieved(retrieved: List[Tuple[Chunk, float]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source_path": chunk.source_path,
            "score": round_float(score),
            "snippet": chunk.text[:320].replace("\n", " "),
        }
        for chunk, score in retrieved
    ]


def case_metrics(
    client,
    case: Dict[str, Any],
    retrieved: List[Tuple[Chunk, float]],
    answer: Optional[str],
    required_claim_threshold: float,
    forbidden_claim_threshold: float,
    citation_support_threshold: float,
) -> Dict[str, Any]:
    retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]
    retrieved_prefix_list = [doc_prefix_from_chunk_id(cid) for cid in retrieved_ids]
    retrieved_prefixes = set(retrieved_prefix_list)

    gold_doc_prefixes: List[str] = case_list(case, "gold_relevant_doc_prefixes")
    if not gold_doc_prefixes:
        gold_doc_prefixes = case_list(case, "expected_doc_prefixes")

    contradiction_prefixes: List[str] = case_list(case, "contradiction_doc_prefixes")
    noise_prefixes: List[str] = case_list(case, "noise_doc_prefixes")
    k = len(retrieved_ids)

    gold_doc_hits = 0
    gold_doc_recall_at_k = None
    if gold_doc_prefixes:
        gold_doc_hits = sum(1 for prefix in gold_doc_prefixes if prefix in retrieved_prefixes)
        gold_doc_recall_at_k = gold_doc_hits / len(gold_doc_prefixes)

    contradiction_rate = None
    if contradiction_prefixes and k > 0:
        contradiction_set = set(contradiction_prefixes)
        contradiction_rate = sum(1 for p in retrieved_prefix_list if p in contradiction_set) / k

    noise_rate = None
    if noise_prefixes and k > 0:
        noise_set = set(noise_prefixes)
        noise_rate = sum(1 for p in retrieved_prefix_list if p in noise_set) / k

    evidence_coverage = retrieval_evidence_coverage(
        case=case,
        retrieved_ids=retrieved_ids,
        retrieved_prefixes=retrieved_prefix_list,
    )

    answer_block: Dict[str, Any] = {
        "required_claim_recall": None,
        "forbidden_claim_violation_rate": None,
        "citation_support_rate": None,
        "reference_answer_similarity": None,
        "abstention_correct": None,
    }

    if answer is not None:
        answer_sentences = split_sentences(answer)
        required = case_required_claims(case)
        forbidden = case_forbidden_claims(case)

        required_block = required_claim_metrics(
            client=client,
            answer_sentences=answer_sentences,
            required_claims=required,
            threshold=required_claim_threshold,
        )
        forbidden_block = forbidden_claim_metrics(
            client=client,
            answer_sentences=answer_sentences,
            forbidden_claims=forbidden,
            threshold=forbidden_claim_threshold,
        )
        citation_block = citation_support_metrics(
            client=client,
            answer_sentences=answer_sentences,
            retrieved=retrieved,
            threshold=citation_support_threshold,
        )

        ref_answer = str(case.get("reference_answer", "")).strip()
        ref_similarity = None
        if ref_answer:
            ref_similarity = reference_answer_similarity(client, answer=answer, reference_answer=ref_answer)

        answer_block = {
            "required_claim_recall": required_block["required_claim_recall"],
            "forbidden_claim_violation_rate": forbidden_block["forbidden_claim_violation_rate"],
            "citation_support_rate": citation_block["citation_support_rate"],
            "reference_answer_similarity": round_float(ref_similarity),
            "abstention_correct": abstention_accuracy(answer=answer, expect_abstain=case.get("expect_abstain")),
        }

    return {
        "retrieval": {
            "k": k,
            "gold_doc_prefix_hits": gold_doc_hits,
            "gold_doc_prefix_total": len(gold_doc_prefixes),
            "gold_doc_recall_at_k": round_float(gold_doc_recall_at_k),
            "contradiction_rate": round_float(contradiction_rate),
            "noise_rate": round_float(noise_rate),
            **evidence_coverage,
        },
        "answer": answer_block,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured RAG eval stack.")
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=CASES_DIR / "eval_cases.jsonl",
        help="JSONL file containing structured test cases.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=EVAL_TOP_K,
        help="Top-k retrieval size for this run.",
    )
    parser.add_argument(
        "--with-chat",
        action="store_true",
        help="Generate answers with chat model (required for answer/judge scoring).",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Run optional LLM-as-judge scoring (requires --with-chat).",
    )
    parser.add_argument(
        "--judge-model",
        default=CHAT_MODEL,
        help="Judge model name for optional LLM scoring.",
    )
    parser.add_argument(
        "--required-claim-threshold",
        type=float,
        default=0.7,
        help="Semantic similarity threshold for counting required claim coverage.",
    )
    parser.add_argument(
        "--forbidden-claim-threshold",
        type=float,
        default=0.8,
        help="Semantic similarity threshold for forbidden claim violation detection.",
    )
    parser.add_argument(
        "--citation-support-threshold",
        type=float,
        default=0.65,
        help="Semantic threshold for support between cited sentence and cited chunk windows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of cases to run.",
    )
    parser.add_argument(
        "--split",
        default="all",
        help=(
            "Optional case split filter: one of all/dev/test/train or comma-separated values. "
            "Cases use the `split` field in eval_cases.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "stack_eval_latest.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.with_judge and not args.with_chat:
        raise ValueError("--with-judge requires --with-chat")

    cases = load_cases(args.cases_file)
    split_filter_raw = str(args.split or "all").strip().lower()
    split_filter = {item.strip() for item in split_filter_raw.split(",") if item.strip()}
    if split_filter and "all" not in split_filter:
        cases = [
            case
            for case in cases
            if str(case.get("split", "unspecified")).strip().lower() in split_filter
        ]
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("No eval cases loaded.")

    docs = list_docs()
    if not docs:
        raise RuntimeError("No docs found under RAW_DIR (default: data/).")
    chunks = load_chunks_from_docs(docs)
    if not chunks:
        raise RuntimeError("No chunks produced from source docs.")

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    client = create_client()
    chunk_vecs = embed_chunks(client, chunks)
    if chunk_vecs.size == 0:
        raise RuntimeError("Embedding matrix is empty.")
    chunk_vocabs, chunk_modes, chunk_meta_vocabs = build_chunk_features(chunks)

    case_rows: List[Dict[str, Any]] = []
    retrieval_timings: List[float] = []
    chat_timings: List[float] = []
    judge_timings: List[float] = []
    total_timings: List[float] = []

    for idx, case in enumerate(cases, start=1):
        question = str(case["question"]).strip()
        print(f"[{idx}/{len(cases)}] {case['id']}: {question}")
        t_case_start = time.perf_counter()

        t_retrieve_start = time.perf_counter()
        query_expansions = generate_query_expansions(
            client=client,
            question=question,
            chat_history=[],
        )
        retrieved = retrieve(
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
        packed_docs, packing_stats = pack_retrieved_documents(
            client=client,
            question=question,
            retrieved=retrieved,
            preamble=CHAT_PREAMBLE,
            chat_history=[],
            max_input_tokens=CHAT_MAX_INPUT_TOKENS,
        )
        retrieval_ms = (time.perf_counter() - t_retrieve_start) * 1000.0

        answer: Optional[str] = None
        chat_error: Optional[str] = None
        chat_ms = 0.0
        if args.with_chat:
            t_chat_start = time.perf_counter()
            chat_message = (
                f"{question}\n\n"
                "Instructions: Use only the provided documents. "
                "Cite CHUNK_ID values in square brackets for every policy claim. "
                "If evidence is missing, explicitly say so."
            )
            try:
                resp = client.chat(
                    model=CHAT_MODEL,
                    preamble=CHAT_PREAMBLE,
                    message=chat_message,
                    documents=packed_docs,
                    temperature=0.2,
                    max_tokens=CHAT_MAX_OUTPUT_TOKENS,
                    max_input_tokens=CHAT_MAX_INPUT_TOKENS,
                    citation_quality="off",
                    prompt_truncation="AUTO_PRESERVE_ORDER",
                )
                answer = (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                chat_error = str(exc)
                answer = None
            chat_ms = (time.perf_counter() - t_chat_start) * 1000.0

        judge_scores = None
        judge_error = None
        judge_ms = 0.0
        if args.with_judge and answer:
            judge_scores, judge_error, judge_ms = judge_answer(
                client=client,
                case=case,
                answer=answer,
                retrieved=retrieved,
                judge_model=args.judge_model,
                reference_answer=str(case.get("reference_answer", "")).strip() or None,
            )

        total_ms = (time.perf_counter() - t_case_start) * 1000.0

        retrieval_timings.append(retrieval_ms)
        total_timings.append(total_ms)
        if args.with_chat:
            chat_timings.append(chat_ms)
        if args.with_judge:
            judge_timings.append(judge_ms)

        metrics = case_metrics(
            client=client,
            case=case,
            retrieved=retrieved,
            answer=answer,
            required_claim_threshold=args.required_claim_threshold,
            forbidden_claim_threshold=args.forbidden_claim_threshold,
            citation_support_threshold=args.citation_support_threshold,
        )

        case_rows.append(
            {
                "case": case,
                "query_expansions": query_expansions,
                "timing_ms": {
                    "retrieval": round_float(retrieval_ms, 3),
                    "chat": round_float(chat_ms, 3) if args.with_chat else None,
                    "judge": round_float(judge_ms, 3) if args.with_judge else None,
                    "total": round_float(total_ms, 3),
                },
                "packing_stats": packing_stats,
                "chat_error": chat_error,
                "answer": answer,
                "retrieved": serialize_retrieved(retrieved),
                "metrics": metrics,
                "judge": {"scores": judge_scores, "error": judge_error} if args.with_judge else None,
            }
        )

    overall_metrics = summarize_case_rows(case_rows)

    subgroup_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key_name in ("split", "mode", "question_type"):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in case_rows:
            grouped[str(row["case"].get(key_name, "unknown"))].append(row)
        subgroup_metrics[f"by_{key_name}"] = {
            key: {"count": len(rows), "metrics": summarize_case_rows(rows)}
            for key, rows in sorted(grouped.items())
        }

    intersection_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        mode = str(row["case"].get("mode", "unknown"))
        qtype = str(row["case"].get("question_type", "unknown"))
        intersection_groups[f"{mode}__{qtype}"].append(row)
    subgroup_metrics["by_mode_x_question_type"] = {
        key: {"count": len(rows), "metrics": summarize_case_rows(rows)}
        for key, rows in sorted(intersection_groups.items())
    }

    payload = {
        "config": {
            "cases_file": str(args.cases_file),
            "split_filter": sorted(split_filter) if split_filter else ["all"],
            "top_k": args.top_k,
            "with_chat": bool(args.with_chat),
            "with_judge": bool(args.with_judge),
            "judge_model": args.judge_model if args.with_judge else None,
            "required_claim_threshold": args.required_claim_threshold,
            "forbidden_claim_threshold": args.forbidden_claim_threshold,
            "citation_support_threshold": args.citation_support_threshold,
            "case_count": len(case_rows),
        },
        "timing_summary_ms": {
            "retrieval": timing_summary(retrieval_timings),
            "chat": timing_summary(chat_timings) if args.with_chat else None,
            "judge": timing_summary(judge_timings) if args.with_judge else None,
            "total": timing_summary(total_timings),
        },
        "overall_metrics": overall_metrics,
        "overall_metric_ci95": summarize_ci95(case_rows),
        "subgroup_metrics": subgroup_metrics,
        "cases": case_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Overall metrics: "
        f"doc_recall@k={overall_metrics['retrieval_gold_doc_recall_at_k_mean']}, "
        f"claim_evidence_cov={overall_metrics['retrieval_claim_evidence_coverage_mean']}, "
        f"noise_rate={overall_metrics['retrieval_noise_rate_mean']}, "
        f"required_claim_recall={overall_metrics['answer_required_claim_recall_mean']}, "
        f"citation_support={overall_metrics['answer_citation_support_rate_mean']}, "
        f"forbidden_rate={overall_metrics['answer_forbidden_violation_rate']}, "
        f"abstention_acc={overall_metrics['answer_abstention_accuracy']}"
    )
    print(
        "Timing p50(ms): "
        f"retrieval={payload['timing_summary_ms']['retrieval']['p50']}, "
        f"chat={(payload['timing_summary_ms']['chat']['p50'] if payload['timing_summary_ms']['chat'] else None)}, "
        f"judge={(payload['timing_summary_ms']['judge']['p50'] if payload['timing_summary_ms']['judge'] else None)}, "
        f"total={payload['timing_summary_ms']['total']['p50']}"
    )
    print(f"Wrote report: {args.output}")
