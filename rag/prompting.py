"""Prompt-construction helpers for grounded policy answers with citations."""

from typing import List, Tuple

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
