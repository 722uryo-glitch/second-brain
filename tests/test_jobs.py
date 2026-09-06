import asyncio
import os
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

from app import jobs, executive, db, web_research
from app.job_context import BudgetExceeded, JobStopped, checkpoint, current, gather_owned, reserve_call


class PersistentJobTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = jobs.DB_PATH, db.DB_PATH
        jobs.DB_PATH = db.DB_PATH = os.path.join(self.tmp.name, 'jobs.db')
        db.init_db()
        jobs.init_jobs()

    def tearDown(self):
        jobs.DB_PATH, db.DB_PATH = self.previous
        self.tmp.cleanup()

    def claim(self):
        created = jobs.create_job('調査して記事を書いて')
        claimed = jobs._claim_job('worker')
        self.assertEqual(created['id'], claimed['id'])
        return claimed

    def update(self, job, **values):
        with closing(jobs._connect()) as conn:
            conn.execute('UPDATE jobs SET '+','.join(k+'=?' for k in values)+' WHERE id=?', (*values.values(), job['id']))
            conn.commit()

    def expire(self, job):
        self.update(job, lease_expires_at='2000-01-01T00:00:00+00:00')

    def test_job_is_persisted_and_reloaded(self):
        job = jobs.create_job('依頼')
        jobs.init_jobs()
        self.assertEqual(jobs.get_job(job['id'])['request'], '依頼')
        self.assertEqual(jobs.job_events(job['id'])[0]['event_type'], 'accepted')

    def test_live_lease_survives_initialization(self):
        job = self.claim()
        jobs.init_jobs()
        self.assertEqual(jobs.get_job(job['id'])['worker_id'], job['worker_id'])
        self.assertIsNone(jobs._claim_job('second-worker'))

    def test_expired_lease_recovery_does_not_reset_budget(self):
        job = self.claim()
        self.update(job, calls_used=11, retries_used=3)
        self.expire(job)
        jobs.init_jobs()
        newer = jobs._claim_job('worker')
        self.assertNotEqual(newer['worker_id'], job['worker_id'])
        self.assertEqual(newer['deadline_at'], job['deadline_at'])
        self.assertEqual(newer['calls_used'], 11)
        self.assertEqual(newer['retries_used'], 3)

    def test_stale_worker_cannot_heartbeat_finish_or_write_artifact(self):
        job = self.claim()
        self.expire(job)
        newer = jobs._claim_job('worker')
        with self.assertRaises(JobStopped):
            jobs._heartbeat(job['id'], job['worker_id'])
        self.assertFalse(jobs._finish(job['id'], job['worker_id'], 'succeeded', 'stale'))
        self.assertEqual(jobs.artifacts(job['id']), [])
        self.assertEqual(jobs.get_job(job['id'])['worker_id'], newer['worker_id'])

    def test_claim_is_exclusive(self):
        jobs.create_job('one')
        with ThreadPoolExecutor(max_workers=4) as pool:
            claims = list(pool.map(jobs._claim_job, ['a','b','c','d']))
        self.assertEqual(sum(x is not None for x in claims), 1)

    def test_cancel_wins_completion_race(self):
        job = self.claim()
        self.assertTrue(jobs.request_cancel(job['id']))
        self.assertFalse(jobs._finish(job['id'], job['worker_id'], 'succeeded', 'late'))
        self.assertEqual(jobs.get_job(job['id'])['status'], 'cancelled')
        self.assertEqual(jobs.artifacts(job['id']), [])

    def test_queued_cancel_is_terminal(self):
        job = jobs.create_job('one')
        jobs.request_cancel(job['id'])
        self.assertIsNone(jobs._claim_job('worker'))
        self.assertEqual(jobs.get_job(job['id'])['status'], 'cancelled')

    async def test_completed_step_reused_after_restart(self):
        job = self.claim()
        first = jobs.Execution(job, job['worker_id'])
        operation = AsyncMock(return_value={'evidence':['source']})
        result = await first.step('research', {'q':'one'}, operation, 'research_pack')
        self.expire(job)
        recovered = jobs._claim_job('new')
        second = jobs.Execution(recovered, recovered['worker_id'])
        self.assertEqual(await second.step('research', {'q':'one'}, operation, 'research_pack'), result)
        self.assertEqual(operation.await_count, 1)
        self.assertEqual(len(jobs.artifacts(job['id'])), 1)
        with self.assertRaises(RuntimeError):
            await second.step('research', {'q':'different'}, operation, 'research_pack')

    async def test_stale_step_output_is_rejected(self):
        job = self.claim()
        context = jobs.Execution(job, job['worker_id'])
        async def operation():
            self.expire(job)
            jobs._claim_job('new')
            return 'stale'
        with self.assertRaises(JobStopped):
            await context.step('draft', {}, operation, 'draft')
        self.assertEqual(jobs.artifacts(job['id']), [])

    async def test_cancel_propagates_and_awaits_children(self):
        job = self.claim()
        ready = asyncio.Event()
        cleaned = []
        async def child(n):
            ready.set()
            try:
                await asyncio.sleep(60)
            finally:
                cleaned.append(n)
        async def run(_):
            await gather_owned(child(1), child(2))
        with patch.object(jobs, 'executive_run', run):
            task = asyncio.create_task(jobs._execute(job, job['worker_id']))
            await ready.wait()
            jobs.request_cancel(job['id'])
            await asyncio.wait_for(task, 2)
        self.assertEqual(sorted(cleaned), [1,2])
        self.assertEqual(jobs.get_job(job['id'])['status'], 'cancelled')

    async def test_budget_exhaustion_is_partial_and_keeps_draft(self):
        job = self.claim()
        self.update(job, max_calls=1)
        async def run(_):
            await checkpoint('draft', {}, lambda:'unfinished draft', 'draft')
            reserve_call()
            reserve_call(retry=True)
        with patch.object(jobs, 'executive_run', run):
            await jobs._execute(job, job['worker_id'])
        loaded = jobs.get_job(job['id'])
        self.assertEqual(loaded['status'], 'partial')
        self.assertEqual(loaded['calls_used'], 1)
        self.assertEqual(jobs.artifacts(job['id'])[0]['content'], 'unfinished draft')

    async def test_expired_deadline_never_starts_model_call(self):
        job = self.claim()
        self.update(job, deadline_at='2000-01-01T00:00:00+00:00')
        called = []
        async def run(_):
            reserve_call()
            called.append(True)
        with patch.object(jobs, 'executive_run', run):
            await jobs._execute(job, job['worker_id'])
        self.assertEqual(called, [])
        self.assertEqual(jobs.get_job(job['id'])['status'], 'partial')

    async def test_retry_budget_shared_across_children(self):
        job = self.claim()
        self.update(job, max_retries=1)
        context = jobs.Execution(job, job['worker_id'])
        token = current.set(context)
        try:
            reserve_call(retry=True)
            with self.assertRaises(BudgetExceeded):
                reserve_call(retry=True)
        finally:
            current.reset(token)
        self.assertEqual(jobs.get_job(job['id'])['retries_used'], 1)

    async def test_executive_resumes_draft_without_repeating_research(self):
        job = self.claim()
        plan = {'mode':'research','needs_memory':False,'needs_research':True,'complexity':'complex'}
        research = AsyncMock(return_value=([],[],True,{'sufficient':True},1))
        draft = AsyncMock(side_effect=[RuntimeError('restart here'), 'finished'])
        with patch.object(executive, '_history', return_value=[]), patch.object(executive, '_make_plan', AsyncMock(return_value=plan)) as planner, patch.object(executive, '_gather_memory', AsyncMock(return_value=[])), patch.object(executive, '_gather_research', research), patch.object(executive, '_draft', draft), patch.object(executive, '_critique', AsyncMock(return_value={'pass':True})), patch.object(executive, '_store_conversation_later', AsyncMock()):
            await jobs._execute(job, job['worker_id'])
            self.assertEqual(jobs.get_job(job['id'])['status'], 'partial')
            self.assertTrue(jobs.resume_job(job['id']))
            again = jobs._claim_job('new')
            await jobs._execute(again, again['worker_id'])
        self.assertEqual(jobs.get_job(job['id'])['status'], 'succeeded')
        self.assertEqual(planner.await_count, 1)
        self.assertEqual(research.await_count, 1)
        self.assertEqual(draft.await_count, 2)
        saved = jobs.steps(job['id'])
        self.assertEqual(next(x for x in saved if x['step_key']=='v1:draft')['attempts'], 2)

    async def test_failed_audit_is_not_success(self):
        job = self.claim()
        with patch.object(jobs, 'executive_run', AsyncMock(return_value=executive.ExecutiveResult('limitations',[],None,'research',{}, {'pass':False}))):
            await jobs._execute(job, job['worker_id'])
        self.assertEqual(jobs.get_job(job['id'])['status'], 'partial')

    async def test_graceful_shutdown_releases_lease(self):
        job = self.claim()
        ready = asyncio.Event()
        closed = asyncio.Event()
        async def run(_):
            ready.set()
            try:
                await asyncio.sleep(60)
            finally:
                closed.set()
        with patch.object(jobs, 'executive_run', run):
            task = asyncio.create_task(jobs._execute(job, job['worker_id']))
            await ready.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(closed.is_set())
        self.assertIsNotNone(jobs._claim_job('new'))

    def test_latest_events_and_cursor(self):
        job = jobs.create_job('one')
        for n in range(120):
            jobs.add_event(job['id'], 'progress', str(n))
        latest = jobs.job_events(job['id'], limit=5, latest=True)
        self.assertEqual(latest[-1]['message'], '119')
        self.assertEqual(jobs.job_events(job['id'], after_id=latest[-1]['id']), [])

    def test_resume_budget_requires_explicit_extension(self):
        job = self.claim()
        self.update(job, deadline_at='2000-01-01T00:00:00+00:00')
        jobs._finish(job['id'], job['worker_id'], 'partial', error='budget')
        self.assertFalse(jobs.resume_job(job['id']))
        self.assertTrue(jobs.resume_job(job['id'], extend_budget=True))
        self.assertEqual(jobs.get_job(job['id'])['max_calls'], 240)

    def test_small_talk_not_queued(self):
        self.assertTrue(jobs.should_enqueue('海外の市場を調べて'))
        self.assertFalse(jobs.should_enqueue('こんにちは'))


if __name__ == '__main__':
    unittest.main()
