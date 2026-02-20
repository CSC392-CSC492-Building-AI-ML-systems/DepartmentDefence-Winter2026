"""LLM-assisted retrieval query rewrite/expansion helpers."""

from collections import Counter
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
from .rag_types import Chunk


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")
GENERIC_PREFIX_RE = re.compile(
    r"^(are there|is there|what is the authority|who has the authority|what are the exceptions)\b",
    flags=re.IGNORECASE,
)
LIGHT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def build_rewrite_corpus_context(
    chunks: Sequence[Chunk],
    *,
    max_terms: int = 40,
    max_doc_types: int = 8,
    max_chars: int = 900,
) -> str:
    """Build compact corpus hints to ground rewrite vocabulary."""
    if not chunks:
        return ""

    doc_types: List[str] = []
    seen_doc_types = set()
    term_counts: Counter[str] = Counter()

    for chunk in chunks:
        doc_type = (chunk.doc_type or "").strip()
        if doc_type and doc_type.lower() not in seen_doc_types:
            seen_doc_types.add(doc_type.lower())
            doc_types.append(doc_type)

        text_parts = [
            chunk.title or "",
            chunk.section_heading or "",
            " ".join(chunk.heading_path or []),
            " ".join(chunk.scope_tags or []),
        ]
        for token in _content_tokens(" ".join(text_parts)):
            if token in {"policy", "procurement", "guide", "directive", "contract", "contracts"}:
                continue
            term_counts[token] += 1

    top_terms = [value for value, _count in term_counts.most_common(max(0, int(max_terms)))]

    lines: List[str] = []
    if doc_types:
        lines.append("Doc types: " + " | ".join(doc_types[: max(1, int(max_doc_types))]))
    if top_terms:
        lines.append("Vocabulary hints: " + " | ".join(top_terms))

    context = "\n".join(lines).strip()
    if len(context) <= max_chars:
        return context

    truncated = context[: max_chars]
    cutoff = truncated.rfind(" ")
    if cutoff > 0:
        truncated = truncated[:cutoff]
    return truncated.strip()


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


def _content_tokens(text: str) -> List[str]:
    return [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if len(token) > 2 and token not in LIGHT_STOPWORDS
    ]


def _is_overly_generic(query: str) -> bool:
    normalized = (query or "").strip()
    if not normalized:
        return True
    if GENERIC_PREFIX_RE.search(normalized):
        return True
    # Guard against template-y question-style rewrites that do not add retrieval value.
    if normalized.endswith("?") and len(_content_tokens(normalized)) <= 6:
        return True
    return False


def _query_relevance_overlap(query: str, question_vocab: set[str]) -> float:
    if not question_vocab:
        return 1.0
    query_vocab = set(_content_tokens(query))
    if not query_vocab:
        return 0.0
    overlap = len(query_vocab & question_vocab)
    return overlap / max(1, len(question_vocab))


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
    corpus_context: str | None = None,
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

    context_block = ""
    if (corpus_context or "").strip():
        context_block = (
            "Corpus hints (optional):\n"
            f"{(corpus_context or '').strip()}\n\n"
            "Use corpus hints only to align wording with likely policy vocabulary; do not invent facts.\n\n"
        )

    rewrite_prompt = (
        "You generate high-precision retrieval queries for policy/legal documentation.\n"
        f"Return ONLY JSON object: {{\"queries\": [\"...\", \"...\"]}} with up to {max_queries} queries.\n"
        "Professional rules:\n"
        "- Preserve exact policy terms, acronyms, section labels, dollar thresholds, and dates when present.\n"
        "- Keep queries factual and neutral; do not add assumptions or invented facts.\n"
        "- Cover distinct facets in multi-part questions (for example: authority, exception, threshold, documentation requirement).\n"
        "- Prefer language likely to appear verbatim in policy text.\n"
        "- Avoid near-duplicates and boilerplate templates.\n"
        "- Use concise keyword-rich queries, not full answers.\n\n"
        "Target query mix (adapt to question):\n"
        "1) one near-literal query from the user wording,\n"
        "2) one query focused on approvals/authority/obligations,\n"
        "3) one query focused on exceptions/thresholds/documentation if relevant.\n\n"
        f"{context_block}"
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
    question_vocab = set(_content_tokens(question))
    selected: List[str] = []
    for query in filtered:
        if _is_overly_generic(query):
            continue
        overlap = _query_relevance_overlap(query, question_vocab)
        # Keep rewrites anchored to the user question while allowing partial reframes.
        if overlap < 0.20:
            continue
        selected.append(query)
    return _dedupe_keep_order(selected)[:max_queries]
