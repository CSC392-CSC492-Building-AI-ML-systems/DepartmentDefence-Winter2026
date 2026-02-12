"""Batch evaluation runner for retrieval/answer checks without manual CLI loops."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

from rag.app_config import (
    CHAT_MAX_INPUT_TOKENS,
    CHAT_MAX_OUTPUT_TOKENS,
    CHAT_MODEL,
    CHAT_PREAMBLE,
    TOP_K,
)
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
from rag.retrieval import retrieve
from rag.rag_types import Chunk


DEFAULT_QUESTIONS = [
    (
        "We are issuing a non-defence RFP under an existing supply arrangement renewal, "
        "and a supplier from a non-trade-partner country submits an offer after closing "
        "with evidence of system delay; under reciprocal procurement and late-offer rules, "
        "is the supplier eligible, can the offer be considered, and what exact file "
        "documentation and approvals are mandatory before award?"
    )
]


def _load_questions_from_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        questions.append(stripped)
    return questions


def _serialize_retrieved(retrieved: List[tuple[Chunk, float]]) -> List[dict]:
    rows = []
    for chunk, score in retrieved:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "source_path": chunk.source_path,
                "score": round(float(score), 6),
                "snippet": chunk.text[:320].replace("\n", " "),
            }
        )
    return rows


def _percentile(values: List[float], p: float) -> float:
    """Compute percentile with linear interpolation on a sorted copy."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    position = (len(sorted_vals) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_vals[lower])
    weight = position - lower
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)


def _round_ms(value: float) -> float:
    return round(value, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch RAG evaluations.")
    parser.add_argument(
        "--questions-file",
        type=Path,
        help=(
            "Text file containing one question per line "
            "(blank lines and # comments ignored), "
            "e.g. eval_questions.txt or eval_questions_hard.txt."
        ),
    )
    parser.add_argument(
        "--question",
        action="append",
        help="Inline question (repeatable). If omitted, default benchmark questions are used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Override retrieval top-k for this run.",
    )
    parser.add_argument(
        "--with-chat",
        action="store_true",
        help="Call chat model and include final answer in report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_runs/latest_eval.json"),
        help="Path to write JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    questions: List[str] = []
    if args.questions_file:
        if not args.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {args.questions_file}")
        questions.extend(_load_questions_from_file(args.questions_file))
    if args.question:
        questions.extend(q.strip() for q in args.question if q and q.strip())
    if not questions:
        questions = DEFAULT_QUESTIONS

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

    results = []
    retrieval_times_ms: List[float] = []
    chat_times_ms: List[float] = []
    total_times_ms: List[float] = []
    for idx, question in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] {question}")
        t_question_start = time.perf_counter()
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

        answer = None
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
                answer = f"CHAT_ERROR: {exc}"
            chat_ms = (time.perf_counter() - t_chat_start) * 1000.0

        total_ms = (time.perf_counter() - t_question_start) * 1000.0

        retrieval_times_ms.append(retrieval_ms)
        total_times_ms.append(total_ms)
        if args.with_chat:
            chat_times_ms.append(chat_ms)

        results.append(
            {
                "question": question,
                "answer": answer,
                "timing_ms": {
                    "retrieval": _round_ms(retrieval_ms),
                    "chat": _round_ms(chat_ms) if args.with_chat else None,
                    "total": _round_ms(total_ms),
                },
                "query_expansions": query_expansions,
                "packing_stats": packing_stats,
                "retrieved": _serialize_retrieved(retrieved),
            }
        )

    timing_summary = {
        "retrieval": {
            "p50": _round_ms(_percentile(retrieval_times_ms, 0.50)),
            "p95": _round_ms(_percentile(retrieval_times_ms, 0.95)),
            "mean": _round_ms(sum(retrieval_times_ms) / max(1, len(retrieval_times_ms))),
        },
        "total": {
            "p50": _round_ms(_percentile(total_times_ms, 0.50)),
            "p95": _round_ms(_percentile(total_times_ms, 0.95)),
            "mean": _round_ms(sum(total_times_ms) / max(1, len(total_times_ms))),
        },
        "chat": None,
    }
    if args.with_chat:
        timing_summary["chat"] = {
            "p50": _round_ms(_percentile(chat_times_ms, 0.50)),
            "p95": _round_ms(_percentile(chat_times_ms, 0.95)),
            "mean": _round_ms(sum(chat_times_ms) / max(1, len(chat_times_ms))),
        }

    payload = {
        "config": {
            "top_k": args.top_k,
            "with_chat": bool(args.with_chat),
            "question_count": len(questions),
        },
        "timing_summary_ms": timing_summary,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        "Timing summary (ms): "
        f"retrieval p50={timing_summary['retrieval']['p50']}, "
        f"p95={timing_summary['retrieval']['p95']}; "
        f"total p50={timing_summary['total']['p50']}, "
        f"p95={timing_summary['total']['p95']}"
    )
    if timing_summary["chat"] is not None:
        print(
            "Timing summary (ms): "
            f"chat p50={timing_summary['chat']['p50']}, "
            f"p95={timing_summary['chat']['p95']}"
        )
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
