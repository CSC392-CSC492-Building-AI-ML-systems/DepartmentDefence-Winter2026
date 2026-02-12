"""Batch evaluation runner for retrieval/answer checks without manual CLI loops."""

import argparse
import json
from pathlib import Path
from typing import List

from rag.app_config import CHAT_MODEL, TOP_K
from rag.corpus import list_docs
from rag.embedding_client import create_client
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import build_prompt
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
    for idx, question in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] {question}")
        retrieved = retrieve(client, question, chunks, chunk_vecs, args.top_k)

        answer = None
        if args.with_chat:
            prompt = build_prompt(question, retrieved)
            try:
                resp = client.chat(model=CHAT_MODEL, message=prompt, temperature=0.2)
                answer = (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                answer = f"CHAT_ERROR: {exc}"

        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieved": _serialize_retrieved(retrieved),
            }
        )

    payload = {
        "config": {
            "top_k": args.top_k,
            "with_chat": bool(args.with_chat),
            "question_count": len(questions),
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
