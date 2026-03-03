"""Run-level aggregation and summary helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from .common import bootstrap_ci95, mean, percentile, round_float


def collect_numeric(rows: Iterable[Dict[str, Any]], path: Sequence[str]) -> List[float]:
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


def summarize_case_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "retrieval_gold_doc_recall_at_k_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "gold_doc_recall_at_k")))
        ),
        "retrieval_gold_doc_precision_at_k_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "gold_doc_precision_at_k")))
        ),
        "retrieval_gold_doc_top1_hit_rate": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "gold_doc_top1_hit")))
        ),
        "retrieval_gold_doc_mrr_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "gold_doc_mrr")))
        ),
        "retrieval_gold_doc_ndcg_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "gold_doc_ndcg")))
        ),
        "retrieval_unique_prefix_fraction_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "unique_prefix_fraction")))
        ),
        "retrieval_claim_evidence_coverage_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "claim_evidence_coverage")))
        ),
        "retrieval_contradiction_rate_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "contradiction_rate")))
        ),
        "retrieval_noise_rate_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "retrieval", "noise_rate")))
        ),
        "answer_required_claim_recall_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "required_claim_recall")))
        ),
        "answer_citation_support_rate_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "citation_support_rate")))
        ),
        "answer_reference_similarity_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "reference_answer_similarity")))
        ),
        "answer_abstention_accuracy": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "abstention_correct")))
        ),
        "answer_forbidden_violation_rate": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "forbidden_claim_violation_rate")))
        ),
        "answer_citation_sentence_rate_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "citation_sentence_rate")))
        ),
        "answer_citation_count_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "citation_count")))
        ),
        "answer_sentence_count_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "answer_sentence_count")))
        ),
        "answer_word_count_mean": round_float(
            mean(collect_numeric(rows, ("metrics", "answer", "answer_word_count")))
        ),
    }

    for judge_key in (
        "decision_correctness",
        "reference_alignment",
        "required_claim_recall",
        "forbidden_claim_violation",
    ):
        metrics[f"judge_{judge_key}_mean"] = round_float(
            mean(collect_numeric(rows, ("judge", "scores", judge_key)))
        )
    return metrics


def summarize_ci95(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    ci_metrics: Dict[str, Sequence[str]] = {
        "retrieval_gold_doc_recall_at_k_mean": ("metrics", "retrieval", "gold_doc_recall_at_k"),
        "retrieval_gold_doc_precision_at_k_mean": (
            "metrics",
            "retrieval",
            "gold_doc_precision_at_k",
        ),
        "retrieval_gold_doc_mrr_mean": ("metrics", "retrieval", "gold_doc_mrr"),
        "retrieval_gold_doc_ndcg_mean": ("metrics", "retrieval", "gold_doc_ndcg"),
        "retrieval_claim_evidence_coverage_mean": (
            "metrics",
            "retrieval",
            "claim_evidence_coverage",
        ),
        "retrieval_contradiction_rate_mean": ("metrics", "retrieval", "contradiction_rate"),
        "retrieval_noise_rate_mean": ("metrics", "retrieval", "noise_rate"),
        "answer_required_claim_recall_mean": ("metrics", "answer", "required_claim_recall"),
        "answer_citation_support_rate_mean": ("metrics", "answer", "citation_support_rate"),
        "answer_reference_similarity_mean": ("metrics", "answer", "reference_answer_similarity"),
        "answer_citation_sentence_rate_mean": ("metrics", "answer", "citation_sentence_rate"),
        "judge_decision_correctness_mean": ("judge", "scores", "decision_correctness"),
        "judge_reference_alignment_mean": ("judge", "scores", "reference_alignment"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for metric_name, path in ci_metrics.items():
        values = collect_numeric(rows, path)
        ci = bootstrap_ci95(values)
        if ci is not None:
            out[metric_name] = ci
    return out


def timing_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": round_float(percentile(values, 0.50), 3) or 0.0,
        "p95": round_float(percentile(values, 0.95), 3) or 0.0,
        "mean": round_float(mean(values), 3) or 0.0,
    }
