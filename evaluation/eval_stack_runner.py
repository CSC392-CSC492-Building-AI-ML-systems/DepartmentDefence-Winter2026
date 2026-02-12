"""Structured evaluation stack runner for PolicyRAG.

This runner supports professional-style evaluation layers:
- Retrieval metrics (evidence/source/mode coverage)
- Deterministic answer checks (citation validity, abstention, forbidden claims)
- Claim-level semantic scoring (required/forbidden claims via embeddings)
- Optional LLM-as-judge scoring
- Subgroup and intersection summaries
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repo-root imports work when running:
#   python evaluation/eval_stack_runner.py
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
CASES_DIR = EVAL_DIR / "cases"
RUNS_DIR = EVAL_DIR / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.app_config import (
    CHAT_MAX_INPUT_TOKENS,
    CHAT_MAX_OUTPUT_TOKENS,
    CHAT_MODEL,
    CHAT_PREAMBLE,
    TOP_K,
)
from rag.corpus import list_docs
from rag.embedding_client import create_client, embed_texts
from rag.pipeline import embed_chunks, load_chunks_from_docs
from rag.prompting import pack_retrieved_documents
from rag.query_rewrite import generate_query_expansions
from rag.rag_types import Chunk
from rag.retrieval import retrieve

MODE_PATTERNS: Dict[str, List[str]] = {
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
        "offer validity period",
        "handle late or delayed offers",
    ],
    "award": [
        "before award",
        "contract award",
        "elements to consider before award",
    ],
}

ABSTAIN_PATTERNS = [
    "i don't have enough information",
    "i do not have enough information",
    "insufficient evidence",
    "insufficient information",
    "cannot determine",
    "not enough information",
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal stopword list for lexical overlap checks.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "under",
    "was",
    "were",
    "with",
}


def _round(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def _try_parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    candidates = [raw]

    # Common markdown-wrapped JSON.
    if raw.startswith("```"):
        stripped = raw.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    # Extract the outermost JSON object if extra text exists.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Unable to parse judge JSON payload")


def _percentile(values: Sequence[float], p: float) -> float:
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
    w = position - lower
    return float(sorted_vals[lower] * (1 - w) + sorted_vals[upper] * w)


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _bootstrap_ci95(values: Sequence[float], samples: int = 1000) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 1:
        return {"low": arr[0], "high": arr[0], "n": 1}
    # Deterministic pseudo-randomness for reproducible reports.
    seed = 1337
    boot_means: List[float] = []
    for _ in range(samples):
        # xorshift32-style simple deterministic RNG.
        sample = []
        for _ in range(n):
            seed ^= (seed << 13) & 0xFFFFFFFF
            seed ^= (seed >> 17) & 0xFFFFFFFF
            seed ^= (seed << 5) & 0xFFFFFFFF
            idx = seed % n
            sample.append(arr[idx])
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    low = _percentile(boot_means, 0.025)
    high = _percentile(boot_means, 0.975)
    return {"low": _round(low), "high": _round(high), "n": n}


def _binary_ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = list(relevance[:k])
    if not rel:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(rel):
        if r <= 0:
            continue
        dcg += 1.0 / math.log2(i + 2.0)
    ideal_rel = sorted(rel, reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal_rel):
        if r <= 0:
            continue
        idcg += 1.0 / math.log2(i + 2.0)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        payload = json.loads(raw)
        if "id" not in payload or "question" not in payload:
            raise ValueError(f"Case must include 'id' and 'question': {raw}")
        out.append(payload)
    return out


def _infer_modes(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    modes: List[str] = []
    for mode, phrases in MODE_PATTERNS.items():
        if any(phrase in normalized for phrase in phrases):
            modes.append(mode)
    return modes


def _source_family_for_chunk(chunk: Chunk) -> str:
    stem = Path(chunk.source_path).stem.lower()
    if stem.startswith("buyers_guide__"):
        return "buyers_guide"
    if stem.startswith("buy_canadian_policy__"):
        return "buy_canadian_policy"
    if stem.startswith("tbs_directive__"):
        return "tbs_directive"
    return "other"


def _doc_prefix_from_chunk_id(chunk_id: str) -> str:
    if "__" not in chunk_id:
        return chunk_id
    return chunk_id.rsplit("__", 1)[0]


def _split_sentences(text: str) -> List[str]:
    parts = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    return [part for part in parts if len(part) >= 10]


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _content_tokens(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS and len(tok) > 2]


def _max_lexical_overlap(claim: str, answer_sentences: Sequence[str]) -> float:
    claim_tokens = set(_content_tokens(claim))
    if not claim_tokens:
        return 0.0
    best = 0.0
    for sentence in answer_sentences:
        sentence_tokens = set(_content_tokens(sentence))
        if not sentence_tokens:
            continue
        overlap = len(claim_tokens & sentence_tokens) / len(claim_tokens)
        if overlap > best:
            best = overlap
    return float(best)


def _parse_citations(text: str) -> List[str]:
    citations: List[str] = []
    for content in re.findall(r"\[([^\]]+)\]", text):
        for part in content.split(","):
            token = part.strip()
            token = re.sub(r"^chunk_id:\s*", "", token, flags=re.IGNORECASE)
            if "__" in token:
                citations.append(token)
    deduped: List[str] = []
    seen = set()
    for item in citations:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _citation_matches_retrieved(citation: str, retrieved_ids: Sequence[str]) -> bool:
    if citation in retrieved_ids:
        return True
    return any(rid.endswith(citation) for rid in retrieved_ids)


def _abstained(answer_lower: str) -> bool:
    return any(pattern in answer_lower for pattern in ABSTAIN_PATTERNS)


def _case_required_claims(case: Dict[str, Any]) -> List[str]:
    claims = [str(item).strip() for item in case.get("required_claims", []) if str(item).strip()]
    if claims:
        return claims
    groups = case.get("required_phrase_groups", [])
    fallback: List[str] = []
    for group in groups:
        if not group:
            continue
        fallback.append(str(group[0]).strip())
    return [c for c in fallback if c]


def _case_forbidden_claims(case: Dict[str, Any]) -> List[str]:
    claims = [str(item).strip() for item in case.get("forbidden_claims", []) if str(item).strip()]
    if claims:
        return claims
    return [str(item).strip() for item in case.get("forbidden_phrases", []) if str(item).strip()]


def _semantic_matrix(client, texts: Sequence[str], claims: Sequence[str]):
    if not texts or not claims:
        return None
    text_vecs = embed_texts(client, list(texts), input_type="search_document")
    claim_vecs = embed_texts(client, list(claims), input_type="search_query")
    return text_vecs @ claim_vecs.T


def _required_claim_metrics(
    client,
    answer_sentences: Sequence[str],
    required_claims: Sequence[str],
    threshold: float,
    lexical_threshold: float,
) -> Dict[str, Any]:
    if not required_claims:
        return {
            "required_claim_hits": None,
            "required_claim_total": None,
            "required_claim_recall": None,
            "required_claim_similarity_mean": None,
            "required_claim_lexical_overlap_mean": None,
            "required_claim_max_similarity": [],
            "required_claim_max_lexical_overlap": [],
            "required_claim_hit_methods": [],
        }

    if not answer_sentences:
        return {
            "required_claim_hits": 0,
            "required_claim_total": len(required_claims),
            "required_claim_recall": 0.0,
            "required_claim_similarity_mean": 0.0,
            "required_claim_lexical_overlap_mean": 0.0,
            "required_claim_max_similarity": [0.0 for _ in required_claims],
            "required_claim_max_lexical_overlap": [0.0 for _ in required_claims],
            "required_claim_hit_methods": ["none" for _ in required_claims],
        }

    sim = _semantic_matrix(client, answer_sentences, required_claims)
    max_sim_list = [0.0 for _ in required_claims]
    if sim is not None:
        max_per_claim = sim.max(axis=0)
        max_sim_list = [float(x) for x in max_per_claim]

    max_lex_list = [_max_lexical_overlap(claim, answer_sentences) for claim in required_claims]

    answer_joined = _normalize(" ".join(answer_sentences))
    hit_methods: List[str] = []
    hits = 0
    for claim, max_sim, max_lex in zip(required_claims, max_sim_list, max_lex_list):
        claim_norm = _normalize(claim)
        exact = bool(claim_norm) and claim_norm in answer_joined
        lexical = max_lex >= lexical_threshold
        semantic = max_sim >= threshold
        if exact:
            hit_methods.append("exact")
            hits += 1
        elif lexical and semantic:
            hit_methods.append("lexical+semantic")
            hits += 1
        elif lexical:
            hit_methods.append("lexical")
            hits += 1
        elif semantic:
            hit_methods.append("semantic")
            hits += 1
        else:
            hit_methods.append("none")

    total = len(required_claims)
    recall = hits / total if total else None
    sim_mean = float(sum(max_sim_list) / total) if total else None
    lex_mean = float(sum(max_lex_list) / total) if total else None
    return {
        "required_claim_hits": hits,
        "required_claim_total": total,
        "required_claim_recall": _round(recall),
        "required_claim_similarity_mean": _round(sim_mean),
        "required_claim_lexical_overlap_mean": _round(lex_mean),
        "required_claim_max_similarity": [_round(x) for x in max_sim_list],
        "required_claim_max_lexical_overlap": [_round(x) for x in max_lex_list],
        "required_claim_hit_methods": hit_methods,
    }


def _forbidden_claim_metrics(
    client,
    answer_sentences: Sequence[str],
    forbidden_claims: Sequence[str],
    threshold: float,
    lexical_threshold: float,
) -> Dict[str, Any]:
    if not forbidden_claims:
        return {
            "forbidden_claim_total": 0,
            "forbidden_claim_violations": 0,
            "forbidden_claim_violation_rate": None,
            "forbidden_claim_max_similarity": [],
            "forbidden_claim_max_lexical_overlap": [],
            "forbidden_claim_violation_methods": [],
        }

    if not answer_sentences:
        return {
            "forbidden_claim_total": len(forbidden_claims),
            "forbidden_claim_violations": 0,
            "forbidden_claim_violation_rate": 0.0,
            "forbidden_claim_max_similarity": [0.0 for _ in forbidden_claims],
            "forbidden_claim_max_lexical_overlap": [0.0 for _ in forbidden_claims],
            "forbidden_claim_violation_methods": ["none" for _ in forbidden_claims],
        }

    sim = _semantic_matrix(client, answer_sentences, forbidden_claims)
    max_sim_list = [0.0 for _ in forbidden_claims]
    if sim is not None:
        max_per_claim = sim.max(axis=0)
        max_sim_list = [float(x) for x in max_per_claim]
    max_lex_list = [_max_lexical_overlap(claim, answer_sentences) for claim in forbidden_claims]

    answer_joined = _normalize(" ".join(answer_sentences))
    violation_methods: List[str] = []
    violations = 0
    for claim, max_sim, max_lex in zip(forbidden_claims, max_sim_list, max_lex_list):
        claim_norm = _normalize(claim)
        exact = bool(claim_norm) and claim_norm in answer_joined
        lexical = max_lex >= lexical_threshold
        semantic = max_sim >= threshold
        if exact:
            violation_methods.append("exact")
            violations += 1
        elif lexical and semantic:
            violation_methods.append("lexical+semantic")
            violations += 1
        elif lexical:
            violation_methods.append("lexical")
            violations += 1
        elif semantic:
            violation_methods.append("semantic")
            violations += 1
        else:
            violation_methods.append("none")

    total = len(forbidden_claims)
    rate = violations / total if total else None
    return {
        "forbidden_claim_total": total,
        "forbidden_claim_violations": violations,
        "forbidden_claim_violation_rate": _round(rate),
        "forbidden_claim_max_similarity": [_round(x) for x in max_sim_list],
        "forbidden_claim_max_lexical_overlap": [_round(x) for x in max_lex_list],
        "forbidden_claim_violation_methods": violation_methods,
    }


def _citation_support_metrics(
    client,
    answer_sentences: Sequence[str],
    retrieved: List[Tuple[Chunk, float]],
    threshold: float,
) -> Dict[str, Any]:
    retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]
    by_id: Dict[str, Chunk] = {chunk.chunk_id: chunk for chunk, _ in retrieved}

    sentence_citation_pairs: List[Tuple[str, str]] = []
    for sentence in answer_sentences:
        citation_ids = _parse_citations(sentence)
        for cid in citation_ids:
            matched = None
            if cid in by_id:
                matched = cid
            else:
                for rid in retrieved_ids:
                    if rid.endswith(cid):
                        matched = rid
                        break
            if matched is not None:
                sentence_citation_pairs.append((sentence, matched))

    if not sentence_citation_pairs:
        return {
            "citation_supported_count": None,
            "citation_scored_count": None,
            "citation_support_rate": None,
            "citation_sentence_similarity_mean": None,
        }

    sentences = [pair[0] for pair in sentence_citation_pairs]
    chunk_texts = [by_id[pair[1]].text[:1200] for pair in sentence_citation_pairs]
    sentence_vecs = embed_texts(client, sentences, input_type="search_document")
    chunk_vecs = embed_texts(client, chunk_texts, input_type="search_document")
    sims = (sentence_vecs * chunk_vecs).sum(axis=1)

    supported = sum(1 for sim in sims if float(sim) >= threshold)
    total = len(sims)
    rate = supported / total if total else None
    sim_mean = float(sims.mean()) if total else None
    return {
        "citation_supported_count": supported,
        "citation_scored_count": total,
        "citation_support_rate": _round(rate),
        "citation_sentence_similarity_mean": _round(sim_mean),
    }


def _case_list(case: Dict[str, Any], key: str) -> List[str]:
    return [str(item).strip() for item in case.get(key, []) if str(item).strip()]


def _retrieval_evidence_coverage(
    case: Dict[str, Any],
    retrieved_ids: Sequence[str],
    retrieved_prefixes: Sequence[str],
) -> Dict[str, Any]:
    claim_evidence = case.get("claim_evidence", [])
    if not isinstance(claim_evidence, list) or not claim_evidence:
        return {
            "claim_evidence_total": None,
            "claim_evidence_covered": None,
            "claim_evidence_coverage": None,
        }

    retrieved_id_set = set(retrieved_ids)
    retrieved_prefix_set = set(retrieved_prefixes)
    covered = 0
    for item in claim_evidence:
        if not isinstance(item, dict):
            continue
        ev_ids = [str(v).strip() for v in item.get("evidence_chunk_ids", []) if str(v).strip()]
        ev_prefixes = [
            str(v).strip() for v in item.get("evidence_doc_prefixes", []) if str(v).strip()
        ]
        hit = any(ev_id in retrieved_id_set for ev_id in ev_ids) or any(
            ev_prefix in retrieved_prefix_set for ev_prefix in ev_prefixes
        )
        if hit:
            covered += 1

    total = len(claim_evidence)
    return {
        "claim_evidence_total": total,
        "claim_evidence_covered": covered,
        "claim_evidence_coverage": _round(covered / total if total else None),
    }


def _judge_answer(
    client,
    case: Dict[str, Any],
    answer: str,
    retrieved: List[Tuple[Chunk, float]],
    judge_model: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], float]:
    required_claims = _case_required_claims(case)
    forbidden_claims = _case_forbidden_claims(case)

    snippets = []
    for chunk, score in retrieved[:6]:
        text = chunk.text.replace("\n", " ").strip()
        snippets.append(f"- {chunk.chunk_id} (score={score:.3f}): {text[:600]}")
    evidence_text = "\n".join(snippets) if snippets else "(none)"

    prompt = (
        "You are evaluating a policy RAG answer.\n"
        "Use ONLY the question, answer, and evidence snippets below.\n"
        "Return ONLY JSON with this schema:\n"
        "{"
        "\"decision_correctness\": 0|1|2, "
        "\"groundedness\": 0|1|2, "
        "\"completeness\": 0|1|2, "
        "\"mandatory_optional_precision\": 0|1|2, "
        "\"uncertainty_handling\": 0|1|2, "
        "\"required_claim_recall\": number, "
        "\"forbidden_claim_violation\": 0|1, "
        "\"required_claim_checks\": [{\"claim\":\"...\",\"covered\":0|1,\"reason\":\"...\"}], "
        "\"forbidden_claim_checks\": [{\"claim\":\"...\",\"violated\":0|1,\"reason\":\"...\"}], "
        "\"notes\": \"short rationale\""
        "}\n\n"
        "Scoring guide:\n"
        "- 2 = strong/correct\n"
        "- 1 = partial/mixed\n"
        "- 0 = wrong/unsupported\n"
        "- required_claim_recall in [0,1]\n"
        "- forbidden_claim_violation: 1 if answer asserts forbidden claim meaningfully\n\n"
        "Coverage rules:\n"
        "- Mark a required claim covered only if the answer states it with policy-faithful meaning.\n"
        "- Prefer claims that are explicitly supported by the provided evidence snippets.\n"
        "- Mark forbidden claim violated when the answer clearly asserts that meaning.\n\n"
        f"Question:\n{case['question']}\n\n"
        f"Required claims:\n{json.dumps(required_claims)}\n\n"
        f"Forbidden claims:\n{json.dumps(forbidden_claims)}\n\n"
        f"Expect abstain:\n{json.dumps(case.get('expect_abstain'))}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Evidence snippets:\n{evidence_text}\n"
    )

    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=judge_model,
            message=prompt,
            temperature=0.0,
            max_tokens=420,
            response_format={"type": "json_object"},
        )
        raw = (resp.text or "").strip()
        payload = _try_parse_json_object(raw)
        for key in (
            "decision_correctness",
            "groundedness",
            "completeness",
            "mandatory_optional_precision",
            "uncertainty_handling",
        ):
            value = int(payload.get(key, 0))
            payload[key] = max(0, min(2, value))
        required_checks_raw = payload.get("required_claim_checks", [])
        required_checks: List[Dict[str, Any]] = []
        if isinstance(required_checks_raw, list):
            for item in required_checks_raw[: len(required_claims)]:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                covered = 1 if int(item.get("covered", 0)) else 0
                reason = str(item.get("reason", "")).strip()
                required_checks.append({"claim": claim, "covered": covered, "reason": reason})
        payload["required_claim_checks"] = required_checks

        forbidden_checks_raw = payload.get("forbidden_claim_checks", [])
        forbidden_checks: List[Dict[str, Any]] = []
        if isinstance(forbidden_checks_raw, list):
            for item in forbidden_checks_raw[: len(forbidden_claims)]:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                violated = 1 if int(item.get("violated", 0)) else 0
                reason = str(item.get("reason", "")).strip()
                forbidden_checks.append({"claim": claim, "violated": violated, "reason": reason})
        payload["forbidden_claim_checks"] = forbidden_checks

        recall = float(payload.get("required_claim_recall", 0.0))
        if required_checks:
            recall = sum(int(item["covered"]) for item in required_checks) / len(required_checks)
        payload["required_claim_recall"] = min(max(recall, 0.0), 1.0)
        forbidden = int(payload.get("forbidden_claim_violation", 0))
        if forbidden_checks:
            forbidden = 1 if any(int(item["violated"]) for item in forbidden_checks) else 0
        payload["forbidden_claim_violation"] = 1 if forbidden else 0
        payload["notes"] = str(payload.get("notes", "")).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return payload, None, elapsed_ms
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return None, str(exc), elapsed_ms


def _serialize_retrieved(retrieved: List[Tuple[Chunk, float]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source_path": chunk.source_path,
            "score": _round(score),
            "snippet": chunk.text[:320].replace("\n", " "),
        }
        for chunk, score in retrieved
    ]


def _case_metrics(
    client,
    case: Dict[str, Any],
    retrieved: List[Tuple[Chunk, float]],
    answer: Optional[str],
    required_claim_threshold: float,
    required_lexical_threshold: float,
    forbidden_claim_threshold: float,
    forbidden_lexical_threshold: float,
    citation_support_threshold: float,
) -> Dict[str, Any]:
    retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]
    retrieved_prefix_list = [_doc_prefix_from_chunk_id(cid) for cid in retrieved_ids]
    retrieved_prefixes = set(retrieved_prefix_list)
    retrieved_families = {_source_family_for_chunk(chunk) for chunk, _ in retrieved}

    expected_prefixes: List[str] = _case_list(case, "expected_doc_prefixes")
    gold_doc_prefixes: List[str] = _case_list(case, "gold_relevant_doc_prefixes")
    if not gold_doc_prefixes:
        # Backward compatibility: treat expected prefixes as gold document labels.
        gold_doc_prefixes = expected_prefixes
    gold_chunk_ids: List[str] = _case_list(case, "gold_relevant_chunk_ids")
    contradiction_prefixes: List[str] = _case_list(case, "contradiction_doc_prefixes")
    noise_prefixes: List[str] = _case_list(case, "noise_doc_prefixes")
    needed_families: List[str] = _case_list(case, "source_family_needed")
    case_mode = str(case.get("mode", "")).strip().lower()
    k = len(retrieved_ids)

    prefix_hits = 0
    prefix_recall = None
    if expected_prefixes:
        prefix_hits = sum(1 for prefix in expected_prefixes if prefix in retrieved_prefixes)
        prefix_recall = prefix_hits / len(expected_prefixes)

    gold_doc_hits = 0
    gold_doc_recall_at_k = None
    if gold_doc_prefixes:
        gold_doc_hits = sum(1 for prefix in gold_doc_prefixes if prefix in retrieved_prefixes)
        gold_doc_recall_at_k = gold_doc_hits / len(gold_doc_prefixes)

    gold_chunk_hit_positions: List[int] = []
    if gold_chunk_ids:
        gold_chunk_set = set(gold_chunk_ids)
        for idx, rid in enumerate(retrieved_ids):
            if rid in gold_chunk_set:
                gold_chunk_hit_positions.append(idx)

    chunk_recall_at_k = None
    chunk_precision_at_k = None
    chunk_mrr_at_k = None
    chunk_ndcg_at_k = None
    if gold_chunk_ids and k > 0:
        hit_count = len(set(gold_chunk_ids) & set(retrieved_ids))
        chunk_recall_at_k = hit_count / len(gold_chunk_ids)
        chunk_precision_at_k = hit_count / k
        if gold_chunk_hit_positions:
            chunk_mrr_at_k = 1.0 / (min(gold_chunk_hit_positions) + 1.0)
        rel = [1 if rid in set(gold_chunk_ids) else 0 for rid in retrieved_ids]
        chunk_ndcg_at_k = _binary_ndcg_at_k(rel, k)

    # Proxy IR metrics when chunk-level labels are unavailable.
    doc_proxy_precision_at_k = None
    doc_proxy_mrr_at_k = None
    doc_proxy_ndcg_at_k = None
    if gold_doc_prefixes and k > 0:
        rel = [1 if prefix in set(gold_doc_prefixes) else 0 for prefix in retrieved_prefix_list]
        doc_proxy_precision_at_k = sum(rel) / k
        first_rel = next((idx for idx, flag in enumerate(rel) if flag > 0), None)
        if first_rel is not None:
            doc_proxy_mrr_at_k = 1.0 / (first_rel + 1.0)
        doc_proxy_ndcg_at_k = _binary_ndcg_at_k(rel, k)

    family_hits = 0
    family_coverage = None
    if needed_families:
        family_hits = sum(1 for fam in needed_families if fam in retrieved_families)
        family_coverage = family_hits / len(needed_families)

    mode_match_rate = None
    mode_diversity = None
    if retrieved:
        retrieved_modes: List[List[str]] = [
            _infer_modes(f"{chunk.title} {chunk.source_path} {chunk.text[:220]}")
            for chunk, _ in retrieved
        ]
        all_modes = sorted({m for modes in retrieved_modes for m in modes})
        mode_diversity = len(all_modes)
        if case_mode and case_mode != "cross_mode":
            matches = sum(1 for modes in retrieved_modes if case_mode in modes)
            mode_match_rate = matches / len(retrieved_modes)
        elif case_mode == "cross_mode":
            mode_match_rate = min(mode_diversity / 2.0, 1.0)

    contradiction_rate = None
    if contradiction_prefixes and k > 0:
        contradiction_rate = sum(1 for prefix in retrieved_prefix_list if prefix in set(contradiction_prefixes)) / k

    noise_rate = None
    if noise_prefixes and k > 0:
        noise_rate = sum(1 for prefix in retrieved_prefix_list if prefix in set(noise_prefixes)) / k

    evidence_coverage = _retrieval_evidence_coverage(
        case=case,
        retrieved_ids=retrieved_ids,
        retrieved_prefixes=retrieved_prefix_list,
    )

    answer_metrics: Dict[str, Any] = {
        "required_claim_hits": None,
        "required_claim_total": None,
        "required_claim_recall": None,
        "required_claim_similarity_mean": None,
        "required_claim_lexical_overlap_mean": None,
        "required_claim_max_similarity": [],
        "required_claim_max_lexical_overlap": [],
        "required_claim_hit_methods": [],
        "forbidden_claim_total": None,
        "forbidden_claim_violations": None,
        "forbidden_claim_violation_rate": None,
        "forbidden_claim_max_similarity": [],
        "forbidden_claim_max_lexical_overlap": [],
        "forbidden_claim_violation_methods": [],
        "has_citation": None,
        "citation_count": None,
        "citation_validity": None,
        "citation_support_rate": None,
        "citation_sentence_similarity_mean": None,
        "abstained": None,
        "abstention_correct": None,
    }

    if answer is not None:
        answer_lower = answer.lower()
        answer_sentences = _split_sentences(answer)
        required_claims = _case_required_claims(case)
        forbidden_claims = _case_forbidden_claims(case)

        required_metrics = _required_claim_metrics(
            client=client,
            answer_sentences=answer_sentences,
            required_claims=required_claims,
            threshold=required_claim_threshold,
            lexical_threshold=required_lexical_threshold,
        )
        forbidden_metrics = _forbidden_claim_metrics(
            client=client,
            answer_sentences=answer_sentences,
            forbidden_claims=forbidden_claims,
            threshold=forbidden_claim_threshold,
            lexical_threshold=forbidden_lexical_threshold,
        )

        citations = _parse_citations(answer)
        has_citation = len(citations) > 0
        citation_validity = None
        if citations:
            valid = sum(1 for cid in citations if _citation_matches_retrieved(cid, retrieved_ids))
            citation_validity = valid / len(citations)

        citation_support = _citation_support_metrics(
            client=client,
            answer_sentences=answer_sentences,
            retrieved=retrieved,
            threshold=citation_support_threshold,
        )

        abstained = _abstained(answer_lower)
        abstention_correct = None
        if case.get("expect_abstain") is not None:
            abstention_correct = abstained == bool(case.get("expect_abstain"))

        answer_metrics = {
            **required_metrics,
            **forbidden_metrics,
            "has_citation": has_citation,
            "citation_count": len(citations),
            "citation_validity": _round(citation_validity),
            "citation_support_rate": citation_support["citation_support_rate"],
            "citation_sentence_similarity_mean": citation_support["citation_sentence_similarity_mean"],
            "abstained": abstained,
            "abstention_correct": abstention_correct,
        }

    return {
        "retrieval": {
            "k": k,
            "expected_doc_prefix_hits": prefix_hits,
            "expected_doc_prefix_total": len(expected_prefixes),
            "expected_doc_prefix_recall": _round(prefix_recall),
            "gold_doc_prefix_hits": gold_doc_hits,
            "gold_doc_prefix_total": len(gold_doc_prefixes),
            "gold_doc_recall_at_k": _round(gold_doc_recall_at_k),
            "gold_chunk_total": len(gold_chunk_ids),
            "gold_chunk_recall_at_k": _round(chunk_recall_at_k),
            "gold_chunk_precision_at_k": _round(chunk_precision_at_k),
            "gold_chunk_mrr_at_k": _round(chunk_mrr_at_k),
            "gold_chunk_ndcg_at_k": _round(chunk_ndcg_at_k),
            "doc_proxy_precision_at_k": _round(doc_proxy_precision_at_k),
            "doc_proxy_mrr_at_k": _round(doc_proxy_mrr_at_k),
            "doc_proxy_ndcg_at_k": _round(doc_proxy_ndcg_at_k),
            "source_family_hits": family_hits,
            "source_family_total": len(needed_families),
            "source_family_coverage": _round(family_coverage),
            "mode_match_rate": _round(mode_match_rate),
            "mode_diversity": mode_diversity,
            "contradiction_rate": _round(contradiction_rate),
            "noise_rate": _round(noise_rate),
            **evidence_coverage,
        },
        "answer": answer_metrics,
    }


def _collect_numeric(rows: Iterable[Dict[str, Any]], path: Sequence[str]) -> List[float]:
    out: List[float] = []
    for row in rows:
        value: Any = row
        missing = False
        for key in path:
            if not isinstance(value, dict) or key not in value:
                missing = True
                break
            value = value[key]
        if missing or value is None:
            continue
        if isinstance(value, bool):
            out.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _summarize_case_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "retrieval_gold_doc_recall_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "gold_doc_recall_at_k")))
        ),
        "retrieval_gold_chunk_recall_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "gold_chunk_recall_at_k")))
        ),
        "retrieval_gold_chunk_precision_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "gold_chunk_precision_at_k")))
        ),
        "retrieval_gold_chunk_mrr_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "gold_chunk_mrr_at_k")))
        ),
        "retrieval_gold_chunk_ndcg_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "gold_chunk_ndcg_at_k")))
        ),
        "retrieval_doc_proxy_precision_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "doc_proxy_precision_at_k")))
        ),
        "retrieval_doc_proxy_mrr_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "doc_proxy_mrr_at_k")))
        ),
        "retrieval_doc_proxy_ndcg_at_k_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "doc_proxy_ndcg_at_k")))
        ),
        "retrieval_claim_evidence_coverage_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "claim_evidence_coverage")))
        ),
        "retrieval_contradiction_rate_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "contradiction_rate")))
        ),
        "retrieval_noise_rate_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "noise_rate")))
        ),
        "retrieval_expected_doc_prefix_recall_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "expected_doc_prefix_recall")))
        ),
        "retrieval_source_family_coverage_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "source_family_coverage")))
        ),
        "retrieval_mode_match_rate_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "retrieval", "mode_match_rate")))
        ),
        "answer_required_claim_recall_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "required_claim_recall")))
        ),
        "answer_required_claim_similarity_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "required_claim_similarity_mean")))
        ),
        "answer_required_claim_lexical_overlap_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "required_claim_lexical_overlap_mean")))
        ),
        "answer_citation_presence_rate": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "has_citation")))
        ),
        "answer_citation_validity_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "citation_validity")))
        ),
        "answer_citation_support_rate_mean": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "citation_support_rate")))
        ),
        "answer_abstention_accuracy": _round(
            _mean(_collect_numeric(rows, ("metrics", "answer", "abstention_correct")))
        ),
    }

    forbidden_rates = _collect_numeric(
        rows, ("metrics", "answer", "forbidden_claim_violation_rate")
    )
    metrics["answer_forbidden_violation_rate"] = _round(_mean(forbidden_rates))

    # Backward-compatible alias for earlier reports.
    metrics["answer_required_group_recall_mean"] = metrics["answer_required_claim_recall_mean"]

    for judge_key in (
        "decision_correctness",
        "groundedness",
        "completeness",
        "mandatory_optional_precision",
        "uncertainty_handling",
        "required_claim_recall",
        "forbidden_claim_violation",
    ):
        metrics[f"judge_{judge_key}_mean"] = _round(
            _mean(_collect_numeric(rows, ("judge", "scores", judge_key)))
        )
    return metrics


def _summarize_ci95(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    ci_metrics: Dict[str, Sequence[str]] = {
        "retrieval_gold_doc_recall_at_k_mean": ("metrics", "retrieval", "gold_doc_recall_at_k"),
        "retrieval_doc_proxy_precision_at_k_mean": (
            "metrics",
            "retrieval",
            "doc_proxy_precision_at_k",
        ),
        "retrieval_doc_proxy_mrr_at_k_mean": ("metrics", "retrieval", "doc_proxy_mrr_at_k"),
        "retrieval_doc_proxy_ndcg_at_k_mean": ("metrics", "retrieval", "doc_proxy_ndcg_at_k"),
        "retrieval_claim_evidence_coverage_mean": (
            "metrics",
            "retrieval",
            "claim_evidence_coverage",
        ),
        "retrieval_contradiction_rate_mean": ("metrics", "retrieval", "contradiction_rate"),
        "retrieval_noise_rate_mean": ("metrics", "retrieval", "noise_rate"),
        "answer_required_claim_recall_mean": ("metrics", "answer", "required_claim_recall"),
        "answer_citation_support_rate_mean": ("metrics", "answer", "citation_support_rate"),
        "judge_decision_correctness_mean": ("judge", "scores", "decision_correctness"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for metric_name, path in ci_metrics.items():
        values = _collect_numeric(rows, path)
        ci = _bootstrap_ci95(values)
        if ci is not None:
            out[metric_name] = ci
    return out


def _timing_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": _round(_percentile(values, 0.50), 3) or 0.0,
        "p95": _round(_percentile(values, 0.95), 3) or 0.0,
        "mean": _round(_mean(values), 3) or 0.0,
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
        default=TOP_K,
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
        default=0.66,
        help="Semantic similarity threshold for counting a required claim as covered.",
    )
    parser.add_argument(
        "--required-lexical-threshold",
        type=float,
        default=0.58,
        help="Lexical overlap threshold for counting a required claim as covered.",
    )
    parser.add_argument(
        "--forbidden-claim-threshold",
        type=float,
        default=0.78,
        help="Semantic similarity threshold for forbidden claim violation detection.",
    )
    parser.add_argument(
        "--forbidden-lexical-threshold",
        type=float,
        default=0.62,
        help="Lexical overlap threshold for forbidden claim violation detection.",
    )
    parser.add_argument(
        "--citation-support-threshold",
        type=float,
        default=0.60,
        help="Semantic similarity threshold between citation sentence and cited chunk text.",
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

    cases = _load_cases(args.cases_file)
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
            judge_scores, judge_error, judge_ms = _judge_answer(
                client=client,
                case=case,
                answer=answer,
                retrieved=retrieved,
                judge_model=args.judge_model,
            )

        total_ms = (time.perf_counter() - t_case_start) * 1000.0

        retrieval_timings.append(retrieval_ms)
        total_timings.append(total_ms)
        if args.with_chat:
            chat_timings.append(chat_ms)
        if args.with_judge:
            judge_timings.append(judge_ms)

        metrics = _case_metrics(
            client=client,
            case=case,
            retrieved=retrieved,
            answer=answer,
            required_claim_threshold=args.required_claim_threshold,
            required_lexical_threshold=args.required_lexical_threshold,
            forbidden_claim_threshold=args.forbidden_claim_threshold,
            forbidden_lexical_threshold=args.forbidden_lexical_threshold,
            citation_support_threshold=args.citation_support_threshold,
        )
        row = {
            "case": case,
            "query_expansions": query_expansions,
            "timing_ms": {
                "retrieval": _round(retrieval_ms, 3),
                "chat": _round(chat_ms, 3) if args.with_chat else None,
                "judge": _round(judge_ms, 3) if args.with_judge else None,
                "total": _round(total_ms, 3),
            },
            "packing_stats": packing_stats,
            "chat_error": chat_error,
            "answer": answer,
            "retrieved": _serialize_retrieved(retrieved),
            "metrics": metrics,
            "judge": {
                "scores": judge_scores,
                "error": judge_error,
            }
            if args.with_judge
            else None,
        }
        case_rows.append(row)

    overall_metrics = _summarize_case_rows(case_rows)

    by_mode: Dict[str, Dict[str, Any]] = {}
    mode_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        mode_groups[str(row["case"].get("mode", "unknown"))].append(row)
    for mode, rows in sorted(mode_groups.items()):
        by_mode[mode] = {"count": len(rows), "metrics": _summarize_case_rows(rows)}

    by_split: Dict[str, Dict[str, Any]] = {}
    split_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        split_groups[str(row["case"].get("split", "unspecified"))].append(row)
    for split_name, rows in sorted(split_groups.items()):
        by_split[split_name] = {"count": len(rows), "metrics": _summarize_case_rows(rows)}

    by_question_type: Dict[str, Dict[str, Any]] = {}
    qtype_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        qtype_groups[str(row["case"].get("question_type", "unknown"))].append(row)
    for qtype, rows in sorted(qtype_groups.items()):
        by_question_type[qtype] = {"count": len(rows), "metrics": _summarize_case_rows(rows)}

    by_mode_x_qtype: Dict[str, Dict[str, Any]] = {}
    intersection_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        mode = str(row["case"].get("mode", "unknown"))
        qtype = str(row["case"].get("question_type", "unknown"))
        intersection_groups[f"{mode}__{qtype}"].append(row)
    for key, rows in sorted(intersection_groups.items()):
        by_mode_x_qtype[key] = {"count": len(rows), "metrics": _summarize_case_rows(rows)}

    payload = {
        "config": {
            "cases_file": str(args.cases_file),
            "split_filter": sorted(split_filter) if split_filter else ["all"],
            "top_k": args.top_k,
            "with_chat": bool(args.with_chat),
            "with_judge": bool(args.with_judge),
            "judge_model": args.judge_model if args.with_judge else None,
            "required_claim_threshold": args.required_claim_threshold,
            "required_lexical_threshold": args.required_lexical_threshold,
            "forbidden_claim_threshold": args.forbidden_claim_threshold,
            "forbidden_lexical_threshold": args.forbidden_lexical_threshold,
            "citation_support_threshold": args.citation_support_threshold,
            "case_count": len(case_rows),
        },
        "timing_summary_ms": {
            "retrieval": _timing_summary(retrieval_timings),
            "chat": _timing_summary(chat_timings) if args.with_chat else None,
            "judge": _timing_summary(judge_timings) if args.with_judge else None,
            "total": _timing_summary(total_timings),
        },
        "overall_metrics": overall_metrics,
        "overall_metric_ci95": _summarize_ci95(case_rows),
        "subgroup_metrics": {
            "by_split": by_split,
            "by_mode": by_mode,
            "by_question_type": by_question_type,
            "by_mode_x_question_type": by_mode_x_qtype,
        },
        "cases": case_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(
        "Overall metrics: "
        f"doc_recall@k={overall_metrics['retrieval_gold_doc_recall_at_k_mean']}, "
        f"doc_proxy_precision@k={overall_metrics['retrieval_doc_proxy_precision_at_k_mean']}, "
        f"doc_proxy_mrr@k={overall_metrics['retrieval_doc_proxy_mrr_at_k_mean']}, "
        f"doc_proxy_ndcg@k={overall_metrics['retrieval_doc_proxy_ndcg_at_k_mean']}, "
        f"claim_evidence_cov={overall_metrics['retrieval_claim_evidence_coverage_mean']}, "
        f"contradiction_rate={overall_metrics['retrieval_contradiction_rate_mean']}, "
        f"noise_rate={overall_metrics['retrieval_noise_rate_mean']}, "
        f"source_cov={overall_metrics['retrieval_source_family_coverage_mean']}, "
        f"required_claim_recall={overall_metrics['answer_required_claim_recall_mean']}, "
        f"citation_presence={overall_metrics['answer_citation_presence_rate']}, "
        f"citation_support={overall_metrics['answer_citation_support_rate_mean']}, "
        f"forbidden_rate={overall_metrics['answer_forbidden_violation_rate']}"
    )
    print(
        "Timing p50(ms): "
        f"retrieval={payload['timing_summary_ms']['retrieval']['p50']}, "
        f"chat={(payload['timing_summary_ms']['chat']['p50'] if payload['timing_summary_ms']['chat'] else None)}, "
        f"judge={(payload['timing_summary_ms']['judge']['p50'] if payload['timing_summary_ms']['judge'] else None)}, "
        f"total={payload['timing_summary_ms']['total']['p50']}"
    )
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
