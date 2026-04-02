import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from dashboard import eval_api


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RUN_FIXTURES_DIR = FIXTURES_DIR / "eval_runs"
CASE_FIXTURES_DIR = FIXTURES_DIR / "eval_cases"
FEEDBACK_FIXTURE_PATH = FIXTURES_DIR / "feedback" / "feedback.jsonl"


class FixedDateTime:
    @classmethod
    def now(cls, tz=None):
        dt = datetime.fromtimestamp(2_000_000_000, tz=timezone.utc)
        if tz is None:
            return dt.replace(tzinfo=None)
        return dt.astimezone(tz)


class DashboardEvalApiTests(unittest.TestCase):
    def setUp(self):
        eval_api._chunk_feedback_lookup.cache_clear()

    def _make_client(self):
        app = Flask(__name__)
        app.register_blueprint(eval_api.dashboard_bp)
        return app.test_client()

    def test_load_case_modes_reads_existing_fixture_files(self):
        with patch.object(eval_api, "EVAL_CASES_DIR", CASE_FIXTURES_DIR):
            self.assertEqual(
                eval_api._load_case_modes(),
                ["cross_mode", "late_offer", "reference_mode"],
            )

    def test_normalize_feedback_record_stabilizes_shape(self):
        normalized = eval_api._normalize_feedback_record(
            {
                "timestamp": "1999999500",
                "feedback_type": "invalid",
                "thumb": "DOWN",
                "comment": " Needs work ",
                "conversation_id": " conv-1 ",
                "turn_id": " turn-9 ",
                "question": " What happened? ",
                "answer": " Something ",
                "cited_chunk_ids": ["chunk-1", " ", "chunk-2"],
            }
        )

        self.assertEqual(normalized["timestamp"], 1_999_999_500)
        self.assertEqual(normalized["feedback_type"], "citation")
        self.assertEqual(normalized["thumb"], "down")
        self.assertEqual(normalized["comment"], "Needs work")
        self.assertEqual(normalized["conversation_id"], "conv-1")
        self.assertEqual(normalized["turn_id"], "turn-9")
        self.assertEqual(normalized["question"], "What happened?")
        self.assertEqual(normalized["answer"], "Something")
        self.assertEqual(normalized["cited_chunk_ids"], ["chunk-1", "chunk-2"])
        self.assertEqual(normalized["target_chunk_id"], "chunk-1")

    def test_load_feedback_summary_keeps_feedback_streams_split_and_latest_only(self):
        chunk_lookup = {
            "chunk-1": {
                "source_title": "Buyer Guide A",
                "source_url": "https://example.test/a",
                "source_path": "data/source_a.txt",
                "section_title": "Section A",
                "doc_type": "guide",
                "authority_rank": 2,
                "chunk_preview": "Preview A",
            },
            "chunk-2": {
                "source_title": "Buyer Guide B",
                "source_url": "https://example.test/b",
                "source_path": "data/source_b.txt",
                "section_title": "Section B",
                "doc_type": "guide",
                "authority_rank": 3,
                "chunk_preview": "Preview B",
            },
        }

        with patch.object(eval_api, "FEEDBACK_JSONL_PATH", FEEDBACK_FIXTURE_PATH), patch.object(
            eval_api, "datetime", FixedDateTime
        ), patch.object(eval_api, "_chunk_feedback_lookup", return_value=chunk_lookup):
            summary = eval_api._load_feedback_summary()

        citation_summary = summary["citation"]
        answer_summary = summary["answer"]

        self.assertEqual(citation_summary["total_count"], 2)
        self.assertEqual(citation_summary["thumb_counts"], {"up": 0, "side": 1, "down": 1})
        self.assertEqual(citation_summary["count_last_24h"], 2)
        self.assertEqual(citation_summary["count_last_7d"], 2)
        self.assertEqual(citation_summary["attention_rate"], 1.0)
        self.assertEqual(len(citation_summary["recent_negative_feedback"]), 2)
        self.assertEqual(citation_summary["recent_negative_feedback"][0]["source_title"], "Buyer Guide B")

        self.assertEqual(answer_summary["total_count"], 2)
        self.assertEqual(answer_summary["thumb_counts"], {"up": 1, "side": 1, "down": 0})
        self.assertEqual(answer_summary["count_last_24h"], 2)
        self.assertEqual(answer_summary["count_last_7d"], 2)
        self.assertEqual(answer_summary["attention_rate"], 0.5)
        self.assertEqual(len(answer_summary["recent_negative_feedback"]), 1)
        self.assertEqual(answer_summary["recent_negative_feedback"][0]["comment"], "Needs more detail")

    def test_build_error_breakdown_counts_chat_judge_and_empty_answers(self):
        payload = {
            "cases": [
                {"chat_error": "boom", "judge": {"error": "judge failed"}, "answer": "Filled"},
                {"chat_error": None, "judge": {"error": None}, "answer": ""},
                {"chat_error": None, "judge": None, "answer": "  "},
            ]
        }

        self.assertEqual(
            eval_api._build_error_breakdown(payload),
            {
                "total_cases": 3,
                "chat_error_count": 1,
                "judge_error_count": 1,
                "empty_answer_count": 2,
            },
        )

    def test_build_key_metrics_groups_existing_overall_metrics(self):
        payload = {
            "overall_metrics": {
                "retrieval_gold_doc_recall_at_k_mean": 0.8,
                "retrieval_gold_doc_mrr_mean": 0.7,
                "retrieval_noise_rate_mean": 0.1,
                "retrieval_claim_evidence_coverage_mean": 0.95,
                "answer_citation_support_rate_mean": 0.9,
                "answer_forbidden_violation_rate": 0.05,
                "answer_abstention_accuracy": 1.0,
            }
        }

        self.assertEqual(
            eval_api._build_key_metrics(payload),
            {
                "retrieval": {"recall_at_k": 0.8, "mrr": 0.7, "noise_rate": 0.1},
                "grounding_citation": {
                    "claim_evidence_coverage": 0.95,
                    "citation_support_rate": 0.9,
                },
                "safety": {
                    "forbidden_violation_rate": 0.05,
                    "abstention_accuracy": 1.0,
                },
            },
        )

    def test_eval_health_returns_404_without_dashboard_secret(self):
        client = self._make_client()
        with patch.dict(os.environ, {"DASHBOARD_ACCESS_KEY": ""}, clear=False):
            response = client.get("/api/eval/health")

        self.assertEqual(response.status_code, 404)

    def test_list_runs_sorts_newest_first_and_keeps_headline_metrics(self):
        client = self._make_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            (runs_dir / "retrieval_only.json").write_text(
                (RUN_FIXTURES_DIR / "retrieval_only.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (runs_dir / "chat_judge.json").write_text(
                (RUN_FIXTURES_DIR / "chat_judge.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (runs_dir / "broken.json").write_text("{not-valid", encoding="utf-8")

            older = 1_900_000_000
            newer = 1_900_000_100
            broken = 1_900_000_050
            os.utime(runs_dir / "retrieval_only.json", (older, older))
            os.utime(runs_dir / "chat_judge.json", (newer, newer))
            os.utime(runs_dir / "broken.json", (broken, broken))

            with patch.dict(os.environ, {"DASHBOARD_ACCESS_KEY": "enabled"}, clear=False), patch.object(
                eval_api, "EVAL_RUNS_DIR", runs_dir
            ):
                response = client.get("/api/eval/runs")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual([run["run_id"] for run in payload["runs"]], ["chat_judge", "broken", "retrieval_only"])
        self.assertEqual(payload["runs"][0]["headline"]["retrieval_gold_doc_recall_at_k_mean"], 0.9)
        self.assertEqual(payload["runs"][1]["case_count"], 0)
        self.assertFalse(payload["runs"][1]["with_chat"])

    def test_get_run_summary_returns_existing_dashboard_shape(self):
        client = self._make_client()
        with patch.dict(os.environ, {"DASHBOARD_ACCESS_KEY": "enabled"}, clear=False), patch.object(
            eval_api, "EVAL_RUNS_DIR", RUN_FIXTURES_DIR
        ):
            response = client.get("/api/eval/runs/chat_judge/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("config", payload)
        self.assertIn("timing_summary_ms", payload)
        self.assertIn("overall_metrics", payload)
        self.assertIn("subgroup_metrics", payload)
        self.assertEqual(
            payload["error_breakdown"],
            {
                "total_cases": 2,
                "chat_error_count": 1,
                "judge_error_count": 1,
                "empty_answer_count": 1,
            },
        )
        self.assertEqual(payload["key_metrics"]["retrieval"]["recall_at_k"], 0.9)

    def test_feedback_summary_route_uses_existing_feedback_shape(self):
        client = self._make_client()
        with patch.dict(os.environ, {"DASHBOARD_ACCESS_KEY": "enabled"}, clear=False), patch.object(
            eval_api, "FEEDBACK_JSONL_PATH", FEEDBACK_FIXTURE_PATH
        ), patch.object(eval_api, "datetime", FixedDateTime), patch.object(
            eval_api,
            "_chunk_feedback_lookup",
            return_value={},
        ):
            response = client.get("/api/eval/feedback/summary")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["citation"]["total_count"], 2)
        self.assertEqual(payload["answer"]["total_count"], 2)

    def test_meta_route_reports_defaults_and_case_modes(self):
        client = self._make_client()
        with patch.dict(os.environ, {"DASHBOARD_ACCESS_KEY": "enabled"}, clear=False), patch.object(
            eval_api, "EVAL_CASES_DIR", CASE_FIXTURES_DIR
        ), patch.object(eval_api, "_read_env_str", side_effect=lambda name, default: default):
            response = client.get("/api/eval/meta")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["models"]["chat_model"], "command-r-plus-08-2024")
        self.assertEqual(payload["execution_modes"], ["retrieval-only", "chat", "chat+judge"])
        self.assertEqual(payload["case_modes"], ["cross_mode", "late_offer", "reference_mode"])


if __name__ == "__main__":
    unittest.main()
