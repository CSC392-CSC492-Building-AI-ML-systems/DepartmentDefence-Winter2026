"""LLM-assisted retrieval query rewrite/expansion helpers."""

import json
import re
from typing import List, Sequence

import cohere

from .app_config import (
    CHAT_MODEL,
    ENABLE_LLM_QUERY_REWRITE,
    QUERY_REWRITE_MAX_QUERIES,
    QUERY_REWRITE_MAX_TOKENS,
    QUERY_REWRITE_MODEL,
)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _parse_queries_from_json(text: str) -> List[str]:
    candidate = text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(candidate)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(payload, list):
        raw_queries = payload
    elif isinstance(payload, dict):
        raw_queries = payload.get("queries", [])
    else:
        raw_queries = []

    if not isinstance(raw_queries, list):
        return []
    return _dedupe_keep_order([str(query) for query in raw_queries])


def generate_query_expansions(
    client: cohere.Client,
    question: str,
    chat_history: Sequence[dict] | None = None,
    max_queries: int = QUERY_REWRITE_MAX_QUERIES,
) -> List[str]:
    """Generate a few focused retrieval queries from the user question/history."""
    if (not ENABLE_LLM_QUERY_REWRITE) or max_queries <= 0:
        return []

    # Include only the most recent turns to keep rewrite prompt compact.
    recent_history = list(chat_history or [])[-6:]
    history_lines: List[str] = []
    for item in recent_history:
        role = str(item.get("role", "")).upper()
        message = str(item.get("message", "")).strip()
        if not message:
            continue
        history_lines.append(f"{role}: {message}")

    rewrite_prompt = (
        "You generate retrieval search queries for policy RAG.\n"
        f"Return ONLY JSON: {{\"queries\": [\"...\", \"...\"]}} with up to {max_queries} short queries.\n"
        "Rules:\n"
        "- Keep each query factual and specific.\n"
        "- Avoid duplicate wording.\n"
        "- Prefer terms likely to appear in policy text.\n\n"
        "Recent conversation context (may be empty):\n"
        f"{chr(10).join(history_lines) if history_lines else '(none)'}\n\n"
        f"Current user question:\n{question}\n"
    )

    try:
        response = client.chat(
            model=QUERY_REWRITE_MODEL or CHAT_MODEL,
            message=rewrite_prompt,
            temperature=0.0,
            max_tokens=QUERY_REWRITE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception:
        return []

    text = (response.text or "").strip()
    queries = _parse_queries_from_json(text)

    # Filter away exact repeats of the original question.
    filtered = [query for query in queries if query.lower() != question.lower()]
    return _dedupe_keep_order(filtered)[:max_queries]
