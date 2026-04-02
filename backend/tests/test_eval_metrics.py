import unittest

from evaluation.stack_eval import retrieval_metrics, summary


class EvalMetricHelperTests(unittest.TestCase):
    def test_precision_top1_and_mrr_follow_existing_rank_logic(self):
        gold_prefixes = ["doc-a", "doc-b"]
        retrieved_prefixes = ["doc-z", "doc-b", "doc-a"]

        self.assertAlmostEqual(
            retrieval_metrics.precision_at_k(gold_prefixes, retrieved_prefixes),
            2 / 3,
        )
        self.assertEqual(retrieval_metrics.top1_hit(gold_prefixes, retrieved_prefixes), 0.0)
        self.assertEqual(retrieval_metrics.mrr_at_k(gold_prefixes, retrieved_prefixes), 0.5)

    def test_retrieval_evidence_coverage_accepts_doc_prefix_and_chunk_hits(self):
        case = {
            "claim_evidence": [
                {"evidence_doc_prefixes": ["doc-a"]},
                {"evidence_chunk_ids": ["doc-b__chunk_2"]},
                {"evidence_doc_prefixes": ["doc-missing"]},
            ]
        }

        coverage = retrieval_metrics.retrieval_evidence_coverage(
            case=case,
            retrieved_ids=["doc-b__chunk_2"],
            retrieved_prefixes=["doc-a", "doc-b"],
        )

        self.assertEqual(coverage["claim_evidence_total"], 3)
        self.assertEqual(coverage["claim_evidence_covered"], 2)
        self.assertAlmostEqual(coverage["claim_evidence_coverage"], 2 / 3, places=6)

    def test_summarize_case_rows_aggregates_existing_dashboard_metrics(self):
        rows = [
            {
                "metrics": {
                    "retrieval": {
                        "gold_doc_recall_at_k": 1.0,
                        "gold_doc_precision_at_k": 0.5,
                        "gold_doc_top1_hit": 1.0,
                        "gold_doc_mrr": 1.0,
                        "gold_doc_ndcg": 0.9,
                        "unique_prefix_fraction": 0.75,
                        "claim_evidence_coverage": 1.0,
                        "contradiction_rate": 0.0,
                        "noise_rate": 0.1,
                    },
                    "answer": {
                        "required_claim_recall": 0.8,
                        "citation_support_rate": 0.9,
                        "reference_answer_similarity": 0.85,
                        "abstention_correct": True,
                        "forbidden_claim_violation_rate": 0.0,
                        "citation_sentence_rate": 0.7,
                        "citation_count": 3,
                        "answer_sentence_count": 5,
                        "answer_word_count": 80,
                    },
                },
                "judge": {
                    "scores": {
                        "decision_correctness": 1.0,
                        "reference_alignment": 0.9,
                        "required_claim_recall": 0.8,
                        "forbidden_claim_violation": 0.0,
                    }
                },
            },
            {
                "metrics": {
                    "retrieval": {
                        "gold_doc_recall_at_k": 0.5,
                        "gold_doc_precision_at_k": 0.25,
                        "gold_doc_top1_hit": 0.0,
                        "gold_doc_mrr": 0.5,
                        "gold_doc_ndcg": 0.6,
                        "unique_prefix_fraction": 1.0,
                        "claim_evidence_coverage": 0.5,
                        "contradiction_rate": 0.1,
                        "noise_rate": 0.2,
                    },
                    "answer": {
                        "required_claim_recall": 0.6,
                        "citation_support_rate": 0.7,
                        "reference_answer_similarity": 0.75,
                        "abstention_correct": False,
                        "forbidden_claim_violation_rate": 0.2,
                        "citation_sentence_rate": 0.5,
                        "citation_count": 2,
                        "answer_sentence_count": 4,
                        "answer_word_count": 60,
                    },
                },
                "judge": {
                    "scores": {
                        "decision_correctness": 0.0,
                        "reference_alignment": 0.7,
                        "required_claim_recall": 0.6,
                        "forbidden_claim_violation": 0.2,
                    }
                },
            },
        ]

        metrics = summary.summarize_case_rows(rows)

        self.assertEqual(metrics["retrieval_gold_doc_recall_at_k_mean"], 0.75)
        self.assertEqual(metrics["retrieval_claim_evidence_coverage_mean"], 0.75)
        self.assertEqual(metrics["answer_required_claim_recall_mean"], 0.7)
        self.assertEqual(metrics["answer_abstention_accuracy"], 0.5)
        self.assertEqual(metrics["judge_decision_correctness_mean"], 0.5)

    def test_timing_summary_uses_existing_percentile_shape(self):
        timing = summary.timing_summary([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(timing, {"p50": 25.0, "p95": 38.5, "mean": 25.0})

    # We intentionally avoid asserting more controversial metric behavior here,
    # such as the current nDCG output bounds, because backend alignment work is
    # still in progress and these tests are meant to protect stable behavior.


if __name__ == "__main__":
    unittest.main()
