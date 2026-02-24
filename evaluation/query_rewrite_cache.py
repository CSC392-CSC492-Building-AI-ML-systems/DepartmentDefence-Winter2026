"""Persistent query-rewrite cache for evaluation scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY_REWRITE_CACHE = REPO_ROOT / "evaluation" / "runs" / "query_rewrite_cache.json"
_CACHE_VERSION = 1


def _cache_key(question: str, model_name: str, max_queries: int) -> str:
    payload = f"{model_name}\n{max_queries}\n{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_expansions(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        lowered = text.lower()
        if not text or lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
    return out


def _load_entries(cache_file: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_file.exists():
        return {}
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if int(payload.get("version", 0) or 0) != _CACHE_VERSION:
        return {}
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    valid: Dict[str, Dict[str, Any]] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        valid[key] = value
    return valid


def _save_entries(cache_file: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _CACHE_VERSION,
        "entries": entries,
    }
    cache_file.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_query_rewrite_cache(
    *,
    questions: Iterable[str],
    cache_file: Path,
    model_name: str,
    max_queries: int,
    generator: Callable[[str], List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    Resolve per-question rewrites with persistent cache reuse.

    Cache key is based on question + effective model + max_queries.
    """
    entries = _load_entries(cache_file)
    rewrites_by_question: Dict[str, List[str]] = {}

    seen_questions = set()
    ordered_questions: List[str] = []
    for raw_question in questions:
        question = str(raw_question).strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        ordered_questions.append(question)

    cache_hits = 0
    cache_misses = 0
    generated = 0
    entry_updates = 0

    for question in ordered_questions:
        key = _cache_key(question=question, model_name=model_name, max_queries=max_queries)
        entry = entries.get(key, {})
        expansions = entry.get("expansions")
        if isinstance(expansions, list):
            rewrites_by_question[question] = _normalize_expansions(expansions)
            cache_hits += 1
            continue

        cache_misses += 1
        generated += 1
        fresh = _normalize_expansions(generator(question))
        rewrites_by_question[question] = fresh
        entries[key] = {
            "question": question,
            "model_name": model_name,
            "max_queries": int(max_queries),
            "expansions": fresh,
        }
        entry_updates += 1

    if entry_updates > 0:
        _save_entries(cache_file, entries)

    stats = {
        "cache_file": str(cache_file),
        "effective_model": model_name,
        "max_queries": int(max_queries),
        "question_count": len(ordered_questions),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "generated_count": int(generated),
        "written_entries": int(entry_updates),
    }
    return rewrites_by_question, stats
