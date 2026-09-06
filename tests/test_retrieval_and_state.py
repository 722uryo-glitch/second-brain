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


if __name__ == "__main__":
    unittest.main()
