import os
import tempfile
import unittest

from app import db
from app import retrieval
from app import runtime_state


class RetrievalAndStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db.DB_PATH = os.path.join(self.tmp.name, "test.db")
        retrieval._FTS_READY = False
        retrieval._FTS_MODE = "uninitialized"
        db.init_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranked_external_retrieval_finds_relevant_older_item(self):
        db.add_external_item(
            "Test",
            "Battery recycling startup expands in Japan",
            "https://example.com/battery-recycling",
            "2026-01-01T00:00:00+00:00",
            "closed-loop lithium battery recycling market",
            {},
        )
        for i in range(20):
            db.add_external_item(
                "Noise",
                f"Unrelated sports item {i}",
                f"https://noise.example/{i}",
                "2026-09-01T00:00:00+00:00",
                "sports result",
                {},
            )
        rows = retrieval.search_external_ranked(["battery recycling"], limit=5)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["url"], "https://example.com/battery-recycling")

    def test_runtime_state_roundtrip(self):
        runtime_state.set_state("test", {"goal": "finish second brain", "n": 2})
        value = runtime_state.get_state("test")
        self.assertEqual(value["goal"], "finish second brain")
        self.assertEqual(value["n"], 2)

    def test_duplicate_syndication_does_not_count_as_independent_sources(self):
        claim_id = db.upsert_claim("test-claim", "A test claim")
        first = db.add_external_item(
            "Publisher A",
            "Same wire story - Publisher A",
            "https://a.example/story",
            summary="same report",
        )
        second = db.add_external_item(
            "Publisher B",
            "Same wire story - Publisher B",
            "https://b.example/story",
            summary="same report",
        )
        db.add_claim_evidence(claim_id, first, "a.example", "news", "supports", 0.8)
        db.add_claim_evidence(claim_id, second, "b.example", "news", "supports", 0.8)
        row = db.search_claims(["test claim"], limit=1)[0]
        self.assertEqual(row["independent_sources"], 1)
        self.assertEqual(row["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
