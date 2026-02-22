"""Prompt and context-packing helpers for grounded policy answers."""

import hashlib
from typing import Dict, List, Sequence, Tuple

import cohere

from .app_config import (
    CHAT_MAX_INPUT_TOKENS,
    CHAT_RESERVED_TOKENS,
    MAX_DOC_TOKENS,
    MAX_PACKED_DOCS,
    TOKENIZER_MODEL,
)

from .rag_types import Chunk


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Build the final model prompt from a question and retrieved chunks."""
    ctx_blocks = []
    for ch, _score in retrieved:
        # Include identifiers so the model can cite specific chunk IDs.
        ctx_blocks.append(
            f"CHUNK_ID: {ch.chunk_id}\n"
            f"TITLE: {ch.title}\n"
            f"SOURCE: {ch.source_path}\n"
            f"TEXT:\n{ch.text}\n"
        )
    context = "\n---\n".join(ctx_blocks)

    return f"""
You are a policy assistant. Answer the user's question using ONLY the provided excerpts.

Rules:
- If the excerpts do not contain enough information, say: "I don't have enough information in the provided policies to answer that."
- Do NOT use outside knowledge.
- Every factual/policy statement MUST cite at least one chunk_id in square brackets, like [chunk_id].
- For multi-part questions, answer each part separately and say when a specific part lacks evidence in the excerpts.
- Keep the answer concise.

User question:
{question}

Policy excerpts:
{context}
""".strip()


def _count_tokens(client: cohere.Client, text: str, model: str = TOKENIZER_MODEL) -> int:
    """Count tokens with Cohere offline tokenizer; fall back to char heuristic."""
    if not text:
        return 0
    try:
        response = client.tokenize(text=text, model=model, offline=True)
        return len(response.tokens)
    except Exception:
        # Conservative fallback for environments where offline tokenization is unavailable.
        return max(1, len(text) // 4)


def _truncate_to_tokens(
    client: cohere.Client,
    text: str,
    max_tokens: int,
    model: str = TOKENIZER_MODEL,
) -> str:
    """Truncate text to at most max_tokens using Cohere tokenizer/detokenizer."""
    if max_tokens <= 0 or not text:
        return ""
    try:
        tokenized = client.tokenize(text=text, model=model, offline=True)
        if len(tokenized.tokens) <= max_tokens:
            return text
        detokenized = client.detokenize(
            tokens=tokenized.tokens[:max_tokens],
            model=model,
            offline=True,
        )
        return (detokenized.text or "").strip()
    except Exception:
        # Fallback: proportional char trim.
        approx_chars = max(1, max_tokens * 4)
        return text[:approx_chars].strip()


def _history_to_text(chat_history: Sequence[dict]) -> str:
    lines: List[str] = []
    for item in chat_history:
        role = str(item.get("role", "")).upper()
        message = str(item.get("message", "")).strip()
        if not message:
            continue
        lines.append(f"{role}: {message}")
    return "\n".join(lines)


def pack_retrieved_documents(
    client: cohere.Client,
    question: str,
    retrieved: List[Tuple[Chunk, float]],
    preamble: str,
    chat_history: Sequence[dict] | None = None,
    max_input_tokens: int = CHAT_MAX_INPUT_TOKENS,
    reserved_tokens: int = CHAT_RESERVED_TOKENS,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_packed_docs: int = MAX_PACKED_DOCS,
    tokenizer_model: str = TOKENIZER_MODEL,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Convert retrieved chunks into Cohere chat documents within a token budget.

    Returns:
      - list of packed document dicts (id/title/text/source_path)
      - packing stats for observability
    """
    history = list(chat_history or [])
    fixed_text = (
        f"PREAMBLE:\n{preamble}\n\n"
        f"HISTORY:\n{_history_to_text(history)}\n\n"
        f"QUESTION:\n{question}\n"
    )
    fixed_tokens = _count_tokens(client, fixed_text, model=tokenizer_model)
    budget_for_docs = max(256, max_input_tokens - reserved_tokens - fixed_tokens)

    packed_docs: List[Dict[str, str]] = []
    used_doc_tokens = 0
    truncated_docs = 0

    for chunk, _score in retrieved:
        if max_packed_docs > 0 and len(packed_docs) >= max_packed_docs:
            break
        # Cohere document IDs must be short; keep full chunk_id in metadata and text header.
        short_doc_id = f"doc_{hashlib.sha1(chunk.chunk_id.encode('utf-8')).hexdigest()[:16]}"
        chunk_header = (
            f"CHUNK_ID: {chunk.chunk_id}\n"
            f"TITLE: {chunk.title}\n"
            f"SOURCE_PATH: {chunk.source_path}\n"
        )
        doc_text = f"{chunk_header}\n{chunk.text}"
        if max_doc_tokens > 0:
            trimmed = _truncate_to_tokens(
                client,
                doc_text,
                max_doc_tokens,
                model=tokenizer_model,
            )
            if trimmed != doc_text:
                truncated_docs += 1
            doc_text = trimmed

        doc_tokens = _count_tokens(client, doc_text, model=tokenizer_model)
        remaining = budget_for_docs - used_doc_tokens
        if remaining <= 0:
            break
        if doc_tokens > remaining:
            # Fit a final partial document if there is enough room to remain useful.
            if remaining < 80:
                break
            partial = _truncate_to_tokens(
                client,
                doc_text,
                remaining,
                model=tokenizer_model,
            )
            partial_tokens = _count_tokens(client, partial, model=tokenizer_model)
            if partial_tokens <= 0:
                break
            packed_docs.append(
                {
                    "id": short_doc_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "source_url": chunk.source_url,
                    "source_title": chunk.source_title,
                    "text": partial,
                }
            )
            used_doc_tokens += partial_tokens
            truncated_docs += 1
            break

        packed_docs.append(
            {
                "id": short_doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "source_path": chunk.source_path,
                "source_url": chunk.source_url,
                "source_title": chunk.source_title,
                "text": doc_text,
            }
        )
        used_doc_tokens += doc_tokens

    stats = {
        "budget_for_docs_tokens": budget_for_docs,
        "used_doc_tokens": used_doc_tokens,
        "packed_docs": len(packed_docs),
        "retrieved_docs": len(retrieved),
        "truncated_docs": truncated_docs,
    }
    return packed_docs, stats
