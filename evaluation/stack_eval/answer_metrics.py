"""Answer-side metric helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from rag.rag_types import Chunk

from .common import abstained, parse_citations, round_float, split_sentences

MAX_CITATION_WINDOWS_PER_CHUNK = 24


def _embed_texts(client, texts: Sequence[str], input_type: str):
    # Imported lazily so script import order can set repo root path first.
    from rag.embedding_client import embed_texts

    return embed_texts(client, list(texts), input_type=input_type)


def semantic_matrix(client, texts: Sequence[str], claims: Sequence[str]):
    if not texts or not claims:
        return None
    text_vecs = _embed_texts(client, texts, input_type="search_document")
    claim_vecs = _embed_texts(client, claims, input_type="search_query")
    return text_vecs @ claim_vecs.T


def required_claim_metrics(
    client,
    answer_sentences: Sequence[str],
    required_claims: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    if not required_claims:
        return {
            "required_claim_hits": None,
            "required_claim_total": None,
            "required_claim_recall": None,
            "required_claim_similarity_mean": None,
            "required_claim_max_similarity": [],
            "required_claim_hit_methods": [],
        }

    if not answer_sentences:
        return {
            "required_claim_hits": 0,
            "required_claim_total": len(required_claims),
            "required_claim_recall": 0.0,
            "required_claim_similarity_mean": 0.0,
            "required_claim_max_similarity": [0.0 for _ in required_claims],
            "required_claim_hit_methods": ["none" for _ in required_claims],
        }

    sim = semantic_matrix(client, answer_sentences, required_claims)
    max_sim_list = [0.0 for _ in required_claims]
    if sim is not None:
        max_sim_list = [float(x) for x in sim.max(axis=0)]

    hit_methods: List[str] = []
    hits = 0
    for max_sim in max_sim_list:
        if max_sim >= threshold:
            hit_methods.append("semantic")
            hits += 1
        else:
            hit_methods.append("none")

    total = len(required_claims)
    recall = hits / total if total else None
    sim_mean = float(sum(max_sim_list) / total) if total else None
    return {
        "required_claim_hits": hits,
        "required_claim_total": total,
        "required_claim_recall": round_float(recall),
        "required_claim_similarity_mean": round_float(sim_mean),
        "required_claim_max_similarity": [round_float(x) for x in max_sim_list],
        "required_claim_hit_methods": hit_methods,
    }


def forbidden_claim_metrics(
    client,
    answer_sentences: Sequence[str],
    forbidden_claims: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    if not forbidden_claims:
        return {
            "forbidden_claim_total": 0,
            "forbidden_claim_violations": 0,
            "forbidden_claim_violation_rate": None,
            "forbidden_claim_max_similarity": [],
            "forbidden_claim_violation_methods": [],
        }

    if not answer_sentences:
        return {
            "forbidden_claim_total": len(forbidden_claims),
            "forbidden_claim_violations": 0,
            "forbidden_claim_violation_rate": 0.0,
            "forbidden_claim_max_similarity": [0.0 for _ in forbidden_claims],
            "forbidden_claim_violation_methods": ["none" for _ in forbidden_claims],
        }

    sim = semantic_matrix(client, answer_sentences, forbidden_claims)
    max_sim_list = [0.0 for _ in forbidden_claims]
    if sim is not None:
        max_sim_list = [float(x) for x in sim.max(axis=0)]

    violation_methods: List[str] = []
    violations = 0
    for max_sim in max_sim_list:
        if max_sim >= threshold:
            violation_methods.append("semantic")
            violations += 1
        else:
            violation_methods.append("none")

    total = len(forbidden_claims)
    rate = violations / total if total else None
    return {
        "forbidden_claim_total": total,
        "forbidden_claim_violations": violations,
        "forbidden_claim_violation_rate": round_float(rate),
        "forbidden_claim_max_similarity": [round_float(x) for x in max_sim_list],
        "forbidden_claim_violation_methods": violation_methods,
    }


def _chunk_sentence_windows(text: str) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        compact = " ".join(text.split())
        return [compact[:1200]] if compact else []

    windows: List[str] = []
    for idx, sentence in enumerate(sentences):
        windows.append(sentence)
        if idx + 1 < len(sentences):
            windows.append(f"{sentence} {sentences[idx + 1]}")

    deduped: List[str] = []
    seen = set()
    for window in windows:
        key = " ".join(window.lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
        if len(deduped) >= MAX_CITATION_WINDOWS_PER_CHUNK:
            break
    return deduped


def citation_support_metrics(
    client,
    answer_sentences: Sequence[str],
    retrieved: List[Tuple["Chunk", float]],
    threshold: float,
) -> Dict[str, Any]:
    by_id: Dict[str, Chunk] = {chunk.chunk_id: chunk for chunk, _ in retrieved}
    retrieved_ids = list(by_id.keys())

    sentence_citation_pairs: List[Tuple[str, str]] = []
    for sentence in answer_sentences:
        citation_ids = parse_citations(sentence)
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

    unique_sentences = sorted({sentence for sentence, _ in sentence_citation_pairs})
    sentence_vecs = _embed_texts(client, unique_sentences, input_type="search_document")
    sentence_vec_by_text = {
        sentence: sentence_vecs[idx] for idx, sentence in enumerate(unique_sentences)
    }

    cited_chunk_ids = sorted({chunk_id for _, chunk_id in sentence_citation_pairs})
    chunk_window_vecs: Dict[str, Any] = {}
    for chunk_id in cited_chunk_ids:
        windows = _chunk_sentence_windows(by_id[chunk_id].text)
        if not windows:
            continue
        chunk_window_vecs[chunk_id] = _embed_texts(client, windows, input_type="search_document")

    sims: List[float] = []
    for sentence, chunk_id in sentence_citation_pairs:
        sent_vec = sentence_vec_by_text[sentence]
        window_vecs = chunk_window_vecs.get(chunk_id)
        if window_vecs is None:
            continue
        pair_sims = window_vecs @ sent_vec
        sims.append(float(pair_sims.max()))

    if not sims:
        return {
            "citation_supported_count": 0,
            "citation_scored_count": 0,
            "citation_support_rate": 0.0,
            "citation_sentence_similarity_mean": 0.0,
        }

    supported = sum(1 for sim in sims if sim >= threshold)
    total = len(sims)
    rate = supported / total
    sim_mean = sum(sims) / total
    return {
        "citation_supported_count": supported,
        "citation_scored_count": total,
        "citation_support_rate": round_float(rate),
        "citation_sentence_similarity_mean": round_float(sim_mean),
    }


def reference_answer_similarity(client, answer: str, reference_answer: str) -> float:
    ref_vec = _embed_texts(client, [reference_answer], input_type="search_document")[0]
    ans_vec = _embed_texts(client, [answer], input_type="search_document")[0]
    return float((ref_vec * ans_vec).sum())


def abstention_accuracy(answer: str, expect_abstain: Any) -> Any:
    if expect_abstain is None:
        return None
    return abstained(answer.lower()) == bool(expect_abstain)
