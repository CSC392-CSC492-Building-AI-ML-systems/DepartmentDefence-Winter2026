"""Similarity retrieval over embedded policy chunks."""

import re
from typing import Dict, List, Set, Tuple

import cohere
import numpy as np

from .app_config import (
    ACAN_MISMATCH_PENALTY,
    CLAUSE_MIN_OVERLAP,
    ENABLE_CLAUSE_COVERAGE,
    ENABLE_MODE_COVERAGE,
    ENABLE_MODE_ROUTING,
    ENABLE_QUERY_EXPANSION,
    EXPANSION_MIN_OVERLAP,
    MAX_CHUNKS_PER_SOURCE,
    MAX_EXPANSION_QUERIES,
    MAX_SUBQUERIES,
    MIN_CHUNKS_PER_EXPANSION,
    MIN_CHUNKS_PER_CLAUSE,
    MIN_CHUNKS_PER_MODE,
    MODE_MATCH_BONUS,
    MODE_MISMATCH_PENALTY,
    RETRIEVAL_CANDIDATE_POOL,
    RETRIEVAL_ALPHA,
)
from .embedding_client import embed_texts
from .rag_types import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
QUESTION_SPLIT_RE = re.compile(
    r"(?:,\s*(?=(?:what|when|where|which|who|why|how|can|must|should|is|are|do|does)\b)"
    r"|\band\s+(?=(?:what|when|where|which|who|why|how|can|must|should|is|are|do|does)\b))",
    flags=re.IGNORECASE,
)

# Lightweight routing labels for procurement instruments/workflows.
MODE_PATTERNS = {
    "acan": [
        "acan",
        "advance contract award notice",
        "advanced contract award notice",
        "single known business",
    ],
    "rfp": [
        "rfp",
        "request for proposal",
        "solicitation of offers",
    ],
    "rfsa": [
        "rfsa",
        "request for supply arrangement",
        "supply arrangement",
    ],
    "rfso": [
        "rfso",
        "request for standing offer",
        "standing offer",
    ],
    "late_offer": [
        "late offer",
        "delayed offer",
        "handle late or delayed offers",
        "offer closing",
        "after closing",
        "system delay",
    ],
}

STOPWORDS = {
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
    "its",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

COMPLIANCE_TERMS = {
    "approval",
    "approvals",
    "authorize",
    "authorized",
    "documentation",
    "document",
    "documents",
    "evidence",
    "file",
    "mandatory",
    "must",
    "procurement file",
    "record",
    "records",
    "required",
}

PRE_AWARD_TERMS = {
    "before award",
    "before contract award",
    "contract award",
    "pre-award",
    "prior to award",
}

MODE_EXPANSION_QUERIES = {
    "late_offer": "late or delayed offers acceptable evidence timely submission after closing",
    "rfp": "request for proposals solicitation clauses supplier eligibility and evaluation",
    "rfsa": "request for supply arrangement renewal clauses and eligibility limits",
    "rfso": "request for standing offers right of first refusal and call-up eligibility",
    "acan": "advance contract award notice required justifications and procurement file",
}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> List[str]:
    tokens = _tokenize(text)
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    # Normalize slug-like text (e.g., "late-or-delayed-offers") into token-separated words.
    cleaned = NON_ALNUM_RE.sub(" ", lowered)
    return SPACE_RE.sub(" ", cleaned).strip()


def _infer_modes(text: str) -> Set[str]:
    # Return all modes hinted by a query/title/path string.
    normalized = _normalize_text(text)
    modes = set()
    for mode, patterns in MODE_PATTERNS.items():
        if any(_normalize_text(pattern) in normalized for pattern in patterns):
            modes.add(mode)
    return modes


def _extract_subqueries(query: str, max_subqueries: int) -> List[str]:
    # Split multi-part questions into clause-like segments for coverage.
    normalized = _normalize_text(query)
    if not normalized:
        return []

    clauses: List[str] = []
    for part in re.split(r"[?;]+", normalized):
        part = part.strip(" ,")
        if not part:
            continue
        for segment in QUESTION_SPLIT_RE.split(part):
            segment = segment.strip(" ,")
            if len(segment) < 20:
                continue
            clauses.append(segment)

    # Only treat as multi-clause coverage when there is real segmentation.
    deduped: List[str] = []
    seen = set()
    for clause in clauses:
        if clause in seen:
            continue
        seen.add(clause)
        deduped.append(clause)
    if len(deduped) <= 1:
        return []
    return deduped[: max(1, max_subqueries)]


def _contains_any_phrase(text: str, phrases: Set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _dedupe_keep_order(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _build_expansion_queries(
    query: str,
    clauses: List[str],
    query_modes: Set[str],
    max_expansions: int,
) -> List[str]:
    """Synthesize a few focused subqueries for coverage-oriented retrieval."""
    expansions: List[str] = []
    normalized = _normalize_text(query)

    has_compliance = _contains_any_phrase(normalized, COMPLIANCE_TERMS)
    has_pre_award = _contains_any_phrase(normalized, PRE_AWARD_TERMS)
    mentions_reciprocal = "reciprocal procurement" in normalized
    mentions_non_trade_partner = (
        ("non trading partner" in normalized)
        or ("non trade partner" in normalized)
        or ("applicable trading partner" in normalized)
    )

    if has_compliance or has_pre_award:
        expansions.append(
            "mandatory required documentation approvals before contract award procurement file"
        )
    if mentions_reciprocal or mentions_non_trade_partner:
        expansions.append(
            "reciprocal procurement declarations supplier eligibility and contract award requirements"
        )
    if mentions_reciprocal and has_compliance:
        expansions.append(
            "reciprocal procurement exceptions approvals in writing and procurement file requirements"
        )
    if "exception" in normalized and has_compliance:
        expansions.append(
            "exceptions approvals in writing assistant deputy minister and procurement file"
        )

    # Add one compact clause-derived subquery to preserve query-specific language.
    for clause in clauses:
        clause_tokens = _content_tokens(clause)
        if len(clause_tokens) < 4:
            continue
        distilled = " ".join(clause_tokens[:12])
        expansions.append(distilled)
        break

    for mode in sorted(query_modes):
        mode_query = MODE_EXPANSION_QUERIES.get(mode)
        if mode_query:
            expansions.append(mode_query)

    deduped = _dedupe_keep_order(expansions)
    return deduped[: max(0, max_expansions)]


def _lexical_overlap_vocab(query_tokens: List[str], chunk_vocab: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    query_vocab = set(query_tokens)
    overlap = sum(1 for token in query_vocab if token in chunk_vocab)
    return overlap / max(1, len(query_vocab))


def _mode_adjustment(query_modes: Set[str], chunk_modes: Set[str]) -> float:
    # No detected mode in query means "no routing opinion".
    if not query_modes:
        return 0.0
    if query_modes & chunk_modes:
        return MODE_MATCH_BONUS
    # Strongly penalize ACAN content unless ACAN is explicitly requested.
    if "acan" in chunk_modes and "acan" not in query_modes:
        return -ACAN_MISMATCH_PENALTY
    # Mild penalty for other mismatched procurement modes.
    if chunk_modes:
        # Multi-mode questions are intentionally broad, so use a softer mismatch penalty.
        broad_query_scale = 0.4 if len(query_modes) >= 2 else 1.0
        return -(MODE_MISMATCH_PENALTY * broad_query_scale)
    return 0.0


def _has_acan_mode(modes: Set[str]) -> bool:
    return "acan" in modes


def retrieve(
    client: cohere.Client,
    query: str,
    chunks: List[Chunk],
    chunk_vecs: np.ndarray,
    k: int,
) -> List[Tuple[Chunk, float]]:
    """Return top-k chunks using hybrid semantic+lexical scoring with mode-aware routing."""
    # Use the query embedding with search_query input type.
    qvec = embed_texts(client, [query], input_type="search_query")[0]
    # Vectors are normalized, so dot product equals cosine similarity.
    dense_scores = chunk_vecs @ qvec
    dense_scores = (dense_scores + 1.0) / 2.0

    query_tokens = _content_tokens(query)
    chunk_vocabs: List[Set[str]] = [
        set(_content_tokens(f"{chunk.title} {chunk.source_path} {chunk.text}"))
        for chunk in chunks
    ]
    lexical_scores = np.array(
        [_lexical_overlap_vocab(query_tokens, vocab) for vocab in chunk_vocabs],
        dtype=np.float32,
    )
    max_lex = float(np.max(lexical_scores)) if lexical_scores.size else 0.0
    if max_lex > 0:
        lexical_scores = lexical_scores / max_lex

    query_modes = _infer_modes(query)
    chunk_modes: List[Set[str]] = [
        # Include a prefix of chunk text to catch labels not visible in filename/path.
        _infer_modes(f"{chunk.title} {chunk.source_path} {chunk.text[:240]}")
        for chunk in chunks
    ]

    alpha = min(max(RETRIEVAL_ALPHA, 0.0), 1.0)
    # Base hybrid score before optional routing adjustments.
    combined_scores = (alpha * dense_scores) + ((1.0 - alpha) * lexical_scores)

    if ENABLE_MODE_ROUTING:
        # Use title+path signals to avoid heavy per-chunk text parsing overhead.
        mode_adjustments = np.array(
            [
                _mode_adjustment(query_modes, chunk_modes[idx])
                for idx in range(len(chunks))
            ],
            dtype=np.float32,
        )
        combined_scores = np.clip(combined_scores + mode_adjustments, 0.0, 1.0)

    ranked_idx = np.argsort(-combined_scores)
    candidate_pool = max(k, RETRIEVAL_CANDIDATE_POOL)
    candidate_idx = [int(idx) for idx in ranked_idx[:candidate_pool]]

    # Diversify results so one long source file doesn't dominate all retrieved chunks.
    selected: List[int] = []
    selected_set = set()
    per_source_count: Dict[str, int] = {}
    max_per_source = max(1, MAX_CHUNKS_PER_SOURCE)
    min_per_mode = max(1, MIN_CHUNKS_PER_MODE)
    min_per_clause = max(1, MIN_CHUNKS_PER_CLAUSE)
    min_per_expansion = max(1, MIN_CHUNKS_PER_EXPANSION)
    clauses: List[str] = []

    def try_select(idx: int, ignore_source_cap: bool = False) -> bool:
        if idx in selected_set:
            return False
        source_path = chunks[idx].source_path
        count = per_source_count.get(source_path, 0)
        if (not ignore_source_cap) and count >= max_per_source:
            return False
        selected.append(idx)
        selected_set.add(idx)
        per_source_count[source_path] = count + 1
        return True

    # 1) Mode coverage pass: ensure at least one chunk for each detected mode if available.
    mode_counts = {mode: 0 for mode in query_modes}
    if ENABLE_MODE_ROUTING and ENABLE_MODE_COVERAGE and query_modes:
        for mode in sorted(query_modes):
            needed = min_per_mode
            for idx in candidate_idx:
                if mode not in chunk_modes[idx]:
                    continue
                if try_select(idx):
                    mode_counts[mode] += 1
                    needed -= 1
                if needed == 0 or len(selected) == k:
                    break
            if len(selected) == k:
                break

    # 2) Targeted supplementation pass across full ranking if a mode is still uncovered.
    if ENABLE_MODE_ROUTING and ENABLE_MODE_COVERAGE and len(selected) < k:
        for mode in sorted(query_modes):
            if mode_counts.get(mode, 0) >= min_per_mode:
                continue
            for raw_idx in ranked_idx:
                idx = int(raw_idx)
                if mode not in chunk_modes[idx]:
                    continue
                # Allow bypassing source cap to preserve coverage of query subtopics.
                if try_select(idx, ignore_source_cap=True):
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
                    break
            if len(selected) == k:
                break

    # 3) Clause coverage pass for multi-part prompts.
    if ENABLE_CLAUSE_COVERAGE and len(selected) < k:
        clauses = _extract_subqueries(query, MAX_SUBQUERIES)
        clause_vectors = (
            embed_texts(client, clauses, input_type="search_query") if clauses else np.empty((0, 0))
        )

        for clause_idx, clause in enumerate(clauses):
            clause_tokens = _content_tokens(clause)
            if len(clause_tokens) < 3:
                continue
            context_tokens = [token for token in query_tokens if token not in set(clause_tokens)]
            context_tokens = context_tokens[:8]
            clause_dense = ((chunk_vecs @ clause_vectors[clause_idx]) + 1.0) / 2.0
            needed = min_per_clause
            prefer_mode_match = bool(query_modes)
            while needed > 0 and len(selected) < k:
                best_idx = None
                best_score = 0.0
                for raw_idx in ranked_idx:
                    idx = int(raw_idx)
                    if idx in selected_set:
                        continue
                    if prefer_mode_match and not (query_modes & chunk_modes[idx]):
                        continue
                    overlap = _lexical_overlap_vocab(clause_tokens, chunk_vocabs[idx])
                    if overlap < CLAUSE_MIN_OVERLAP:
                        continue
                    # Blend clause lexical relevance and clause semantic relevance.
                    score = (0.75 * overlap) + (0.25 * float(clause_dense[idx]))
                    # Keep clause context tied to the full query (e.g., reciprocal/late-offer).
                    if context_tokens:
                        score += 0.10 * _lexical_overlap_vocab(context_tokens, chunk_vocabs[idx])
                    # Keep slight preference for globally strong candidates.
                    score += 0.05 * float(combined_scores[idx])
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    # Fallback to any mode on second attempt if strict mode match found none.
                    if prefer_mode_match:
                        prefer_mode_match = False
                        continue
                    break
                # Prioritize sub-question coverage over source diversity.
                try_select(best_idx, ignore_source_cap=True)
                needed -= 1

    # 4) Query-expansion coverage pass (generic compliance/mode expansion queries).
    if ENABLE_QUERY_EXPANSION and len(selected) < k:
        expansions = _build_expansion_queries(
            query=query,
            clauses=clauses,
            query_modes=query_modes,
            max_expansions=MAX_EXPANSION_QUERIES,
        )
        expansion_vectors = (
            embed_texts(client, expansions, input_type="search_query")
            if expansions
            else np.empty((0, 0))
        )

        for expansion_idx, expansion in enumerate(expansions):
            expansion_tokens = _content_tokens(expansion)
            if len(expansion_tokens) < 3:
                continue
            expansion_dense = ((chunk_vecs @ expansion_vectors[expansion_idx]) + 1.0) / 2.0
            needed = min_per_expansion
            while needed > 0 and len(selected) < k:
                best_idx = None
                best_score = 0.0
                for raw_idx in ranked_idx:
                    idx = int(raw_idx)
                    if idx in selected_set:
                        continue
                    overlap = _lexical_overlap_vocab(expansion_tokens, chunk_vocabs[idx])
                    if overlap < EXPANSION_MIN_OVERLAP:
                        continue
                    score = (
                        (0.55 * overlap)
                        + (0.35 * float(expansion_dense[idx]))
                        + (0.10 * float(combined_scores[idx]))
                    )
                    if query_modes and (query_modes & chunk_modes[idx]):
                        score += 0.08
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is None:
                    break
                # Expansion coverage is meant to supplement gaps, so bypass source cap.
                try_select(best_idx, ignore_source_cap=True)
                needed -= 1

    # 5) Fill remaining slots from globally best chunks, deferring ACAN when not requested.
    acan_deferred: List[int] = []
    for raw_idx in ranked_idx:
        idx = int(raw_idx)
        if idx in selected_set:
            continue
        if ENABLE_MODE_ROUTING and (not _has_acan_mode(query_modes)) and _has_acan_mode(chunk_modes[idx]):
            acan_deferred.append(idx)
            continue
        if try_select(idx):
            if len(selected) == k:
                break

    # 6) Fallback: if we still need slots, allow deferred ACAN chunks.
    if len(selected) < k:
        for idx in acan_deferred:
            if try_select(idx):
                if len(selected) == k:
                    break

    # 7) Final fallback: ignore source cap to guarantee exactly k when possible.
    if len(selected) < k:
        final_deferred_acan: List[int] = []
        for raw_idx in ranked_idx:
            idx = int(raw_idx)
            if idx in selected_set:
                continue
            if ENABLE_MODE_ROUTING and (not _has_acan_mode(query_modes)) and _has_acan_mode(chunk_modes[idx]):
                final_deferred_acan.append(idx)
                continue
            if try_select(idx, ignore_source_cap=True):
                if len(selected) == k:
                    break
        # Last-resort fill with ACAN chunks only if still short.
        if len(selected) < k:
            for idx in final_deferred_acan:
                if try_select(idx, ignore_source_cap=True):
                    if len(selected) == k:
                        break

    return [(chunks[i], float(combined_scores[i])) for i in selected]
