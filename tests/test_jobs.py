import os
import tempfile
import unittest

from app import jobs


class PersistentJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        jobs.DB_PATH = os.path.join(self.tmp.name, "jobs.db")
        jobs.init_jobs()

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_is_persisted_and_reloaded(self):
        created = jobs.create_job("海外市場を調査して比較記事を書いて")
        self.assertEqual(created["status"], "queued")
        loaded = jobs.get_job(created["id"])
        self.assertEqual(loaded["request"], "海外市場を調査して比較記事を書いて")
        self.assertFalse(loaded["cancel_requested"])
        events = jobs.job_events(created["id"])
        self.assertTrue(events)
        self.assertEqual(events[0]["event_type"], "accepted")

    def test_queued_cancel_becomes_final_cancelled_state(self):
        created = jobs.create_job("長い調査")
        self.assertTrue(jobs.request_cancel(created["id"]))
        loaded = jobs.get_job(created["id"])
        self.assertTrue(loaded["cancel_requested"])
        self.assertEqual(loaded["status"], "cancelled")

    def test_running_job_is_requeued_on_process_restart(self):
        created = jobs.create_job("再起動テスト")
        claimed = jobs._claim_job("old-worker")
        self.assertEqual(claimed["id"], created["id"])
        self.assertEqual(jobs.get_job(created["id"])["status"], "running")
        jobs.init_jobs()
        recovered = jobs.get_job(created["id"])
        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["current_step"], "recovered_after_restart")
        self.assertIsNone(recovered["worker_id"])

    def test_research_is_queued_but_small_talk_is_not(self):
        self.assertTrue(jobs.should_enqueue("海外の市場を調べて比較して"))
        self.assertFalse(jobs.should_enqueue("こんにちは"))


if __name__ == "__main__":
    unittest.main()
