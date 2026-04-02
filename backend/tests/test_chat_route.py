import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_support import import_backend_app_module


class ChatRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_db = Path(tempfile.mkdtemp()) / "test_chat_database.db"
        cls.app_module = import_backend_app_module(cls.temp_db)

    def setUp(self):
        self.app_module.chat_history = []
        self.app_module.latest_feedback = {}
        self.app_module.feedback_weights = {}
        self.app_module.backup_weights = {}

        conn = self.app_module.get_db()
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()

        self.client = self.app_module.app.test_client()

    def test_empty_message_returns_400(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "   ", "user_id": 1, "language": "en"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["reply"], "Please enter a question.")

    def test_non_policy_message_uses_shortcut_path_and_persists_messages(self):
        with patch.object(
            self.app_module,
            "classify_message_intent",
            return_value={"route": "greeting"},
        ), patch.object(
            self.app_module,
            "build_intent_reply",
            return_value="Hello. Ask me a procurement policy question.",
        ):
            response = self.client.post(
                "/api/chat",
                json={"message": "hello", "user_id": 1, "language": "en"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["reply"], "Hello. Ask me a procurement policy question.")
        self.assertEqual(payload["stats"]["intent_route"], "greeting")
        self.assertEqual(payload["citations"], [])
        self.assertIsNotNone(payload["conversation_id"])

        conn = self.app_module.get_db()
        rows = conn.execute(
            "SELECT type, text FROM messages WHERE conversation_id = ? ORDER BY id",
            (payload["conversation_id"],),
        ).fetchall()
        conn.close()

        self.assertEqual([(row["type"], row["text"]) for row in rows], [
            ("user", "hello"),
            ("bot", "Hello. Ask me a procurement policy question."),
        ])

    def test_policy_question_returns_expected_live_shape_and_stores_citations(self):
        retrieved = [
            (SimpleNamespace(chunk_id="doc-a__chunk_1"), 0.4),
            (SimpleNamespace(chunk_id="doc-b__chunk_2"), 0.3),
        ]
        packed_docs = [
            {
                "chunk_id": "doc-a__chunk_1",
                "title": "Doc A",
                "source_title": "Source A",
                "source_url": "https://example.test/a",
            },
            {
                "chunk_id": "doc-b__chunk_2",
                "title": "Doc B",
                "source_title": "Source B",
                "source_url": "https://example.test/b",
            },
        ]
        self_rag_meta = {
            "revision_applied": True,
            "unsupported_claim_count": 1,
            "missing_citation_count": 2,
        }

        with patch.object(
            self.app_module,
            "classify_message_intent",
            return_value={"route": "policy_question"},
        ), patch.object(
            self.app_module,
            "generate_query_expansions",
            return_value=["procurement policy"],
        ), patch.object(
            self.app_module,
            "retrieve",
            return_value=retrieved,
        ), patch.object(
            self.app_module,
            "pack_retrieved_documents",
            return_value=(packed_docs, {"packed_docs": 2, "retrieved_docs": 2}),
        ), patch.object(
            self.app_module,
            "generate_answer_with_critique_loop",
            return_value=("Policy answer [doc-a__chunk_1].", self_rag_meta),
        ), patch.object(self.app_module, "ENABLE_CONTRADICTION_ANALYSIS", False):
            response = self.client.post(
                "/api/chat",
                json={"message": "What is the procurement rule?", "user_id": 1, "language": "en"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["reply"], "Policy answer [doc-a__chunk_1].")
        self.assertEqual(payload["stats"]["retrieved"], 2)
        self.assertEqual(payload["stats"]["packed_docs"], 2)
        self.assertEqual(payload["stats"]["intent_route"], "policy_question")
        self.assertTrue(payload["stats"]["self_rag_revision_applied"])
        self.assertEqual(payload["stats"]["self_rag_unsupported_claims"], 1)
        self.assertEqual(payload["stats"]["self_rag_missing_citations"], 2)
        self.assertEqual(
            payload["citations"],
            [
                {"title": "Source A", "link": "https://example.test/a", "chunk_id": "doc-a__chunk_1"},
                {"title": "Source B", "link": "https://example.test/b", "chunk_id": "doc-b__chunk_2"},
            ],
        )

        conn = self.app_module.get_db()
        row = conn.execute(
            "SELECT text, citations FROM messages WHERE conversation_id = ? AND type = 'bot' ORDER BY id DESC LIMIT 1",
            (payload["conversation_id"],),
        ).fetchone()
        conn.close()

        self.assertEqual(row["text"], "Policy answer [doc-a__chunk_1].")
        self.assertEqual(
            json.loads(row["citations"]),
            payload["citations"],
        )

    def test_policy_question_adds_feedback_note_and_french_instruction(self):
        retrieved = [
            (SimpleNamespace(chunk_id="doc-a__chunk_1"), 0.4),
        ]
        packed_docs = [
            {
                "chunk_id": "doc-a__chunk_1",
                "title": "Doc A",
                "source_title": "Source A",
                "source_url": "https://example.test/a",
            },
        ]
        captured = {}

        def fake_generate_answer_with_critique_loop(**kwargs):
            captured["chat_message"] = kwargs["chat_message"]
            return "Revised French answer [doc-a__chunk_1].", {
                "revision_applied": False,
                "unsupported_claim_count": 0,
                "missing_citation_count": 0,
            }

        self.app_module.latest_feedback["conv-55"] = {
            "thumb": "down",
            "comment": "Missing exact clause",
        }

        with patch.object(
            self.app_module,
            "classify_message_intent",
            return_value={"route": "policy_question"},
        ), patch.object(
            self.app_module,
            "generate_query_expansions",
            return_value=["procurement policy"],
        ), patch.object(
            self.app_module,
            "retrieve",
            return_value=retrieved,
        ), patch.object(
            self.app_module,
            "pack_retrieved_documents",
            return_value=(packed_docs, {"packed_docs": 1, "retrieved_docs": 1}),
        ), patch.object(
            self.app_module,
            "generate_answer_with_critique_loop",
            side_effect=fake_generate_answer_with_critique_loop,
        ), patch.object(self.app_module, "ENABLE_CONTRADICTION_ANALYSIS", False):
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Quelle est la politique?",
                    "conversation_id": "conv-55",
                    "language": "fr",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("FEEDBACK:", captured["chat_message"])
        self.assertIn("Missing exact clause", captured["chat_message"])
        self.assertIn("respond entirely in French", captured["chat_message"])

    def test_policy_question_returns_no_context_message_when_packer_returns_empty(self):
        with patch.object(
            self.app_module,
            "classify_message_intent",
            return_value={"route": "policy_question"},
        ), patch.object(
            self.app_module,
            "generate_query_expansions",
            return_value=["procurement policy"],
        ), patch.object(
            self.app_module,
            "retrieve",
            return_value=[(SimpleNamespace(chunk_id="doc-a__chunk_1"), 0.4)],
        ), patch.object(
            self.app_module,
            "pack_retrieved_documents",
            return_value=([], {"packed_docs": 0, "retrieved_docs": 1}),
        ):
            response = self.client.post(
                "/api/chat",
                json={"message": "What is the procurement rule?", "language": "en"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["reply"], "No context docs fit. Try a shorter question.")


if __name__ == "__main__":
    unittest.main()
