import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_support import import_backend_app_module


class FeedbackRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_db = Path(tempfile.mkdtemp()) / "test_dummy_database.db"
        cls.app_module = import_backend_app_module(cls.temp_db)

    def setUp(self):
        self.feedback_tmpdir = tempfile.TemporaryDirectory()
        self.feedback_dir = Path(self.feedback_tmpdir.name)
        self.app_module.FEEDBACK_DIR = self.feedback_dir
        self.app_module.latest_feedback = {}
        self.app_module.feedback_weights = {}
        self.app_module.backup_weights = {}
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        self.feedback_tmpdir.cleanup()

    def test_citation_feedback_reweights_and_none_restores_baseline(self):
        with patch.object(self.app_module.time, "time", side_effect=[101, 102, 103]):
            up_response = self.client.post(
                "/api/feedback",
                json={
                    "thumb": "up",
                    "conversation_id": "conv-1",
                    "turn_id": "turn-1",
                    "feedback_type": "citation",
                    "cited_chunk_ids": ["chunk-1"],
                },
            )
            down_response = self.client.post(
                "/api/feedback",
                json={
                    "thumb": "down",
                    "conversation_id": "conv-1",
                    "turn_id": "turn-1",
                    "feedback_type": "citation",
                    "cited_chunk_ids": ["chunk-1"],
                },
            )
            none_response = self.client.post(
                "/api/feedback",
                json={
                    "thumb": "none",
                    "conversation_id": "conv-1",
                    "turn_id": "turn-1",
                    "feedback_type": "citation",
                    "cited_chunk_ids": ["chunk-1"],
                },
            )

        self.assertEqual(up_response.status_code, 200)
        self.assertEqual(down_response.status_code, 200)
        self.assertEqual(none_response.status_code, 200)

        weight_map = self.app_module.feedback_weights["conv-1"]
        self.assertEqual(self.app_module.backup_weights["conv-1"]["citation::turn-1::chunk-1"]["chunk-1"], 1.0)
        self.assertEqual(weight_map["chunk-1"], 1.0)

        log_lines = (
            self.feedback_dir / "feedback.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(log_lines), 3)

    def test_answer_feedback_tracks_latest_negative_and_clears_on_positive(self):
        with patch.object(self.app_module.time, "time", side_effect=[201, 202]):
            negative_response = self.client.post(
                "/api/feedback",
                json={
                    "thumb": "down",
                    "conversation_id": "conv-2",
                    "turn_id": "turn-9",
                    "feedback_type": "answer",
                    "comment": "Needs more detail",
                    "cited_chunk_ids": ["chunk-9"],
                },
            )
            clear_response = self.client.post(
                "/api/feedback",
                json={
                    "thumb": "up",
                    "conversation_id": "conv-2",
                    "turn_id": "turn-9",
                    "feedback_type": "answer",
                    "cited_chunk_ids": ["chunk-9"],
                },
            )

        self.assertEqual(negative_response.status_code, 200)
        self.assertEqual(clear_response.status_code, 200)
        self.assertNotIn("conv-2", self.app_module.latest_feedback)

        records = [
            json.loads(line)
            for line in (self.feedback_dir / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["thumb"] for record in records], ["down", "up"])
        self.assertTrue(all(record["feedback_type"] == "answer" for record in records))

    def test_invalid_feedback_payload_is_rejected(self):
        bad_thumb = self.client.post(
            "/api/feedback",
            json={"thumb": "bad-value", "conversation_id": "conv-3", "turn_id": "turn-1"},
        )
        bad_type = self.client.post(
            "/api/feedback",
            json={
                "thumb": "up",
                "conversation_id": "conv-3",
                "turn_id": "turn-1",
                "feedback_type": "unknown",
            },
        )

        self.assertEqual(bad_thumb.status_code, 400)
        self.assertEqual(bad_type.status_code, 400)


if __name__ == "__main__":
    unittest.main()
