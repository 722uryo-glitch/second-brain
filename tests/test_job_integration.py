import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from app import jobs, db, main, ollama_client, web_research
from app.job_context import current, BudgetExceeded


class JobIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = jobs.DB_PATH, db.DB_PATH
        jobs.DB_PATH = db.DB_PATH = os.path.join(self.tmp.name, 'jobs.db')
        db.init_db()
        jobs.init_jobs()
        # No lifespan: integration tests must never start real collectors/models.
        self.client = TestClient(main.app)

    def tearDown(self):
        self.client.close()
        jobs.DB_PATH, db.DB_PATH = self.previous
        self.tmp.cleanup()

    def test_api_accept_detail_cancel_and_resume(self):
        r = self.client.post('/api/chat', json={'message':'海外市場を調査して'})
        self.assertEqual(r.status_code, 200)
        job_id = r.json()['job_id']
        j = self.client.get('/api/jobs/'+job_id).json()
        self.assertEqual(j['job']['status'], 'queued')
        self.assertIn('steps',j)
        self.assertIn('artifacts',j)
        self.assertTrue(self.client.post('/api/jobs/'+job_id+'/cancel').json()['ok'])
        self.assertEqual(self.client.post('/api/jobs/'+job_id+'/resume',json={'extend_budget':True}).status_code,200)
        self.assertEqual(self.client.post('/api/jobs/'+job_id+'/resume',json={}).status_code,409)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code,404)

    def test_process_exit_preserves_checkpoint(self):
        job = jobs.create_job('crash test')
        script = '''
import asyncio, os
from app import jobs
async def run():
    job = jobs._claim_job('crashing-worker')
    ctx = jobs.Execution(job, job['worker_id'])
    await ctx.step('research', {'q':'test'}, lambda:{'sources':['saved']}, 'research_pack')
    os._exit(17)
asyncio.run(run())
'''
        env = dict(os.environ, SECOND_BRAIN_DB=jobs.DB_PATH)
        result = subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,17,result.stderr)
        with closing(jobs._connect()) as conn:
            conn.execute("UPDATE jobs SET lease_expires_at='2000-01-01' WHERE id=?", (job['id'],))
            conn.commit()
        recovered = jobs._claim_job('restart')
        op = AsyncMock(side_effect=AssertionError('must not repeat saved research'))
        value = asyncio.run(jobs.Execution(recovered,recovered['worker_id']).step('research',{'q':'test'},op,'research_pack'))
        self.assertEqual(value,{'sources':['saved']})
        self.assertEqual(op.await_count,0)

    def test_db_effects_are_idempotent_across_checkpoint_crash_window(self):
        jobs.create_job('one')
        job = jobs._claim_job('worker')
        token = current.set(jobs.Execution(job,job['worker_id']))
        try:
            first = db.add_memory('conversation','user','same content')
            second = db.add_memory('conversation','user','same content')
            self.assertEqual(first,second)
            run = db.start_agent_run('one')
            self.assertEqual(run,db.start_agent_run('one'))
            self.assertEqual(db.add_agent_step(run,1,'plan'),db.add_agent_step(run,1,'plan'))
        finally:
            current.reset(token)
        self.assertEqual(len(db.recent_memories()),1)

    def test_nontransient_http_error_is_not_retried(self):
        response = httpx.Response(401,request=httpx.Request('GET','https://example.test'))
        client = AsyncMock()
        client.request.return_value=response
        with self.assertRaises(httpx.HTTPStatusError):
            asyncio.run(web_research._request_with_retry(client,'GET','https://example.test'))
        self.assertEqual(client.request.await_count,1)

    def test_router_budget_stop_never_falls_back(self):
        with patch.object(ollama_client,'UNOROUTER_ENABLED',True), patch.object(ollama_client,'UNOROUTER_API_KEY','test'), patch.object(ollama_client,'_uno_chat',AsyncMock(side_effect=BudgetExceeded('stop'))), patch.object(ollama_client,'_ollama_chat',AsyncMock()) as local:
            with self.assertRaises(BudgetExceeded):
                asyncio.run(ollama_client.chat([],route='reasoning'))
            self.assertEqual(local.await_count,0)

    def test_ui_javascript_parses(self):
        import shutil
        if not shutil.which('node'):
            self.skipTest('Node unavailable')
        html = Path('app/static/index.html').read_text(encoding='utf-8')
        script = html.split('<script>',1)[1].split('</script>',1)[0]
        path = Path(self.tmp.name)/'ui.js'
        path.write_text(script,encoding='utf-8')
        result = subprocess.run(['node','--check',str(path)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()
