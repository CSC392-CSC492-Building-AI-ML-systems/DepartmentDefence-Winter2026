"""Self-RAG critique and revision loop helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence, Tuple

import cohere

from .app_config import (
    ENABLE_SELF_RAG_CRITIQUE_LOOP,
    SELF_RAG_REVISER_MODEL,
    SELF_RAG_VERIFIER_MAX_TOKENS,
    SELF_RAG_VERIFIER_MODEL,
)


def _safe_json_from_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def _as_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _chat_answer(
    client: cohere.Client,
    model: str,
    preamble: str,
    message: str,
    documents: list[dict],
    chat_history: Sequence[dict],
    max_tokens: int,
    max_input_tokens: int,
    temperature: float = 0.2,
) -> str:
    resp = client.chat(
        model=model,
        preamble=preamble,
        message=message,
        documents=documents,
        chat_history=list(chat_history),
        temperature=temperature,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        citation_quality="off",
        prompt_truncation="AUTO_PRESERVE_ORDER",
    )
    return (resp.text or "").strip()


def generate_answer_with_critique_loop(
    client: cohere.Client,
    chat_model: str,
    preamble: str,
    question: str,
    chat_message: str,
    documents: list[dict],
    chat_history: Sequence[dict],
    max_tokens: int,
    max_input_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a first draft, verify groundedness/citations, and revise if needed.
    """
    draft = _chat_answer(
        client=client,
        model=chat_model,
        preamble=preamble,
        message=chat_message,
        documents=documents,
        chat_history=chat_history,
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        temperature=0.2,
    )

    meta: Dict[str, Any] = {
        "self_rag_enabled": bool(ENABLE_SELF_RAG_CRITIQUE_LOOP),
        "revision_applied": False,
        "unsupported_claim_count": 0,
        "missing_citation_count": 0,
    }
    if not ENABLE_SELF_RAG_CRITIQUE_LOOP:
        return draft, meta

    verify_prompt = (
        "You are a strict policy answer verifier.\n"
        "Given QUESTION and DRAFT_ANSWER, detect unsupported claims and citation gaps.\n"
        "Return strict JSON only with keys:\n"
        "{"
        "\"revision_needed\": boolean, "
        "\"unsupported_claims\": [string], "
        "\"missing_citations\": [string], "
        "\"notes\": [string]"
        "}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"DRAFT_ANSWER:\n{draft}\n"
    )
    verify_text = _chat_answer(
        client=client,
        model=SELF_RAG_VERIFIER_MODEL,
        preamble="You audit grounded policy answers.",
        message=verify_prompt,
        documents=documents,
        chat_history=[],
        max_tokens=SELF_RAG_VERIFIER_MAX_TOKENS,
        max_input_tokens=max_input_tokens,
        temperature=0.0,
    )
    verdict = _safe_json_from_text(verify_text)
    unsupported = _as_list_of_str(verdict.get("unsupported_claims"))
    missing_citations = _as_list_of_str(verdict.get("missing_citations"))
    notes = _as_list_of_str(verdict.get("notes"))
    revision_needed = bool(verdict.get("revision_needed")) or bool(unsupported) or bool(missing_citations)

    meta["unsupported_claim_count"] = len(unsupported)
    meta["missing_citation_count"] = len(missing_citations)
    meta["verifier_notes"] = notes

    if not revision_needed:
        return draft, meta

    revision_prompt = (
        "Revise the DRAFT_ANSWER to fix verifier issues using ONLY provided documents.\n"
        "Rules:\n"
        "- Remove unsupported claims.\n"
        "- Add CHUNK_ID citations for all policy claims.\n"
        "- Preserve required structure, including a short 'Conflict detected' section when present.\n"
        "- If evidence is insufficient, explicitly say so.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"DRAFT_ANSWER:\n{draft}\n\n"
        "VERIFIER_FINDINGS:\n"
        f"- unsupported_claims: {unsupported}\n"
        f"- missing_citations: {missing_citations}\n"
        f"- notes: {notes}\n"
    )
    revised = _chat_answer(
        client=client,
        model=SELF_RAG_REVISER_MODEL,
        preamble=preamble,
        message=revision_prompt,
        documents=documents,
        chat_history=list(chat_history),
        max_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
        temperature=0.1,
    )
    if revised:
        meta["revision_applied"] = True
        return revised, meta
    return draft, meta
