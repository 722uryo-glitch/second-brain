import asyncio
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from app import db, jobs, knowledge as k, autonomy, knowledge_vault as vault, v1_storage as storage, global_intelligence, main, ollama_client
from app.job_context import current, checkpoint, YieldForUser


class KnowledgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.previous=(db.DB_PATH,jobs.DB_PATH,storage.DB_PATH,vault.OBSIDIAN_VAULT_PATH)
        db.DB_PATH=jobs.DB_PATH=storage.DB_PATH=os.path.join(self.tmp.name,'test.db')
        vault.OBSIDIAN_VAULT_PATH=os.path.join(self.tmp.name,'vault')
        db.init_db();jobs.init_jobs();storage.init_v1_storage();k.init_knowledge()
        k.configure(enabled=True,topics=['battery'])

    def tearDown(self):
        db.DB_PATH,jobs.DB_PATH,storage.DB_PATH,vault.OBSIDIAN_VAULT_PATH=self.previous
        self.tmp.cleanup()

    def source(self,body=None):
        body=body or ('Battery recycling can recover lithium. This is a source statement. '*5)
        eid=db.add_external_item('test','Battery recycling study','https://example.test/battery',summary=body)
        storage.save_document(eid,'https://example.test/battery',body)
        return eid

    def schedule_read(self,eid):
        job=jobs.create_job('read',metadata={'autonomous':True,'autonomy_kind':'read','external_id':eid,'depth':0})
        with db._connect() as conn:
            conn.execute("INSERT INTO knowledge_sources(external_id,state,next_at,job_id) VALUES(?,'queued',?,?)",(eid,k.now(),job['id']))
        return jobs._claim_job('test')

    async def decoder(self,text):
        return {'summary':'Battery recycling can recover lithium.','topics':['battery'],
                'claims':[{'text':'Battery recycling can recover lithium.','kind':'fact','quote':'Battery recycling can recover lithium.','status':'source_statement'}],
                'questions':[{'query':'battery recycling recovery costs','reason':'Check feasibility','quote':'Battery recycling can recover lithium.'}]}

    async def test_collect_read_connect_followup_end_to_end(self):
        old=k.save_note('idea:old','Old battery idea','Battery recycling can recover lithium. Investigate costs.',kind='idea')
        async def search(query,limit=6):
            self.source()
            return [{'url':'https://example.test/battery','title':'Battery recycling study','body':'Battery recycling can recover lithium. '*6}]
        with patch.object(autonomy,'research_web',search):
            first=autonomy.cycle();self.assertTrue(first['scheduled'])
            job=jobs._claim_job('test');await jobs._execute(job,job['worker_id'])
        self.assertEqual(jobs.get_job(first['job_id'])['status'],'succeeded')
        second=autonomy.cycle();job=jobs._claim_job('test')
        self.assertEqual(job['metadata']['autonomy_kind'],'read')
        links={'links':[{'target_id':old,'relation':'applies_to','reason':'New evidence makes the old cost question relevant.',
          'left_quote':'Battery recycling can recover lithium.','right_quote':'Battery recycling can recover lithium.'}]}
        with patch.object(autonomy,'decode_chunk',self.decoder),patch.object(autonomy,'embed',AsyncMock(return_value=[1.,0.])),patch.object(autonomy,'chat',AsyncMock(return_value=json.dumps(links))):
            await jobs._execute(job,job['worker_id'])
        self.assertEqual(jobs.get_job(second['job_id'])['status'],'succeeded')
        snap=k.snapshot()
        self.assertEqual(len(snap['links']),1)
        self.assertTrue(any(q['query']=='battery recycling recovery costs' for q in snap['questions']))
        vault.export_knowledge()
        self.assertIn('Old battery idea',(vault.root()/'Resurfaced.md').read_text(encoding='utf-8'))
        # Third autonomous action comes from an extracted question, without user instruction.
        third=autonomy.cycle();self.assertTrue(third['scheduled'])
        claimed=jobs._claim_job('test')
        self.assertIn('recovery costs',claimed['request'])

    async def test_decoder_rejects_fabricated_quotes(self):
        bad={'summary':'summary','claims':[{'text':'unsupported','kind':'fact','quote':'invented passage'}]}
        with patch.object(autonomy,'chat',AsyncMock(return_value=json.dumps(bad))):
            with self.assertRaises(ValueError):
                await autonomy.decode_chunk('actual source text')

    async def test_decode_failure_stays_unread_without_title_support(self):
        eid=self.source();job=self.schedule_read(eid)
        with patch.object(autonomy,'decode_chunk',AsyncMock(side_effect=ValueError('bad JSON'))):
            await jobs._execute(job,job['worker_id'])
        self.assertEqual(k.snapshot()['counts']['knowledge_notes'],0)
        await global_intelligence.factcheck_batch()
        with db._connect() as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM claim_evidence').fetchone()[0],0)
        self.assertEqual(jobs.get_job(job['id'])['status'],'partial')

    async def test_reads_tail_and_resumes_saved_chunks(self):
        body='Battery recycling can recover lithium. '+'a'*5000+' THE TAIL MATTERS'
        eid=self.source(body);job=self.schedule_read(eid)
        calls=[]
        fail=True
        async def decode(text):
            nonlocal fail
            calls.append(text)
            if 'THE TAIL' in text and fail:
                fail=False
                raise ValueError('temporary model parse failure')
            return {'summary':text[-40:],'topics':[],'claims':[],'questions':[]}
        with patch.object(autonomy,'decode_chunk',decode),patch.object(autonomy,'embed',AsyncMock(return_value=None)):
            await jobs._execute(job,job['worker_id'])
            jobs.resume_job(job['id'],extend_budget=True)
            again=jobs._claim_job('test');await jobs._execute(again,again['worker_id'])
        self.assertEqual(len(calls),3)
        note=k.snapshot()['notes'][0]
        self.assertIn('THE TAIL MATTERS',note['content'])
        self.assertEqual(note['coverage']['read_chars'],len(body))

    def test_old_notes_retrieved_and_human_edits_preserved(self):
        nid=k.save_note('idea:old','battery idea','battery recycling costs',kind='idea')
        with db._connect() as conn:
            conn.execute("UPDATE knowledge_notes SET created_at='2001-01-01' WHERE id=?",(nid,))
        self.assertEqual(k.related({'id':-1,'title':'battery recycling','content':'','topics':[]})[0]['id'],nid)
        vault.export_knowledge()
        path=vault.root()/f'Knowledge/{nid:08d}.md'
        path.write_text(path.read_text(encoding='utf-8')+'\nHuman correction\n',encoding='utf-8')
        k.save_note('idea:old','battery idea','battery recycling costs changed',kind='idea')
        vault.export_knowledge();vault.import_personal_notes()
        self.assertIn('Human correction',path.read_text(encoding='utf-8'))
        self.assertTrue(list(path.parent.glob('*.pending-*.md')))
        self.assertTrue(any(n['kind']=='correction' for n in k.snapshot()['notes']))

    def test_private_note_cannot_create_outbound_question(self):
        nid=k.save_note('private','Secret','private info')
        self.assertIsNone(k.add_question('search secret info','reason','topic',note=k.get_note(nid)))
        self.assertEqual(k.snapshot()['questions'],[])

    def test_links_require_real_quotes_and_versions(self):
        a=k.get_note(k.save_note('a','a','shared battery hypothesis'))
        b=k.get_note(k.save_note('b','b','shared battery evidence'))
        proposal={'target_id':b['id'],'relation':'related','reason':'same topic','left_quote':'shared battery','right_quote':'fake'}
        self.assertEqual(k.save_links(a,[b],[proposal]),0)
        proposal['right_quote']='shared battery'
        self.assertEqual(k.save_links(a,[b],[proposal]),1)
        k.save_note('b','b','corrected information')
        self.assertEqual(k.save_links(a,[b],[proposal]),0)
        self.assertEqual(k.snapshot()['links'],[])
        self.assertEqual(k.snapshot()['counts']['knowledge_versions'],3)

    def test_daily_limit_and_concurrent_dispatch(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            result=list(pool.map(lambda _:autonomy.cycle(),range(2)))
        self.assertEqual(sum(r['scheduled'] for r in result),1)
        for job in jobs.list_jobs():
            jobs.request_cancel(job['id'])
        with db._connect() as conn:
            for n in range(12):
                conn.execute('INSERT INTO knowledge_dispatch(job_id,day,created_at) VALUES(?,?,?)',(str(n),k.now()[:10],k.now()))
        self.assertEqual(autonomy.cycle()['reason'],'daily_limit')

    def test_foreground_priority(self):
        autonomy.cycle()
        human=jobs.create_job('user request')
        self.assertEqual(jobs._claim_job('worker')['id'],human['id'])

    async def test_autonomous_work_yields_to_user(self):
        autonomy.cycle();background=jobs._claim_job('worker')
        human=jobs.create_job('user request')
        with patch.object(autonomy,'run_job',AsyncMock()):
            await jobs._execute(background,background['worker_id'])
        self.assertEqual(jobs.get_job(background['id'])['status'],'queued')
        self.assertEqual(jobs._claim_job('worker')['id'],human['id'])

    async def test_autonomous_context_forbids_cloud_even_explicit_route(self):
        autonomy.cycle();job=jobs._claim_job('worker')
        token=current.set(jobs.Execution(job,job['worker_id']))
        try:
            with patch.object(ollama_client,'UNOROUTER_ENABLED',True),patch.object(ollama_client,'UNOROUTER_API_KEY','test'),patch.object(ollama_client,'_uno_chat',AsyncMock()) as cloud,patch.object(ollama_client,'_ollama_chat',AsyncMock(return_value='local')):
                self.assertEqual(await ollama_client.chat([],route='reasoning'),'local')
                cloud.assert_not_called()
        finally:
            current.reset(token)

    def test_api_pause_does_not_cancel_human_job(self):
        autonomy.cycle();human=jobs.create_job('human')
        client=TestClient(main.app)
        try:
            r=client.post('/api/knowledge/settings',json={'enabled':False})
            self.assertEqual(r.status_code,200)
            self.assertEqual(jobs.get_job(human['id'])['status'],'queued')
            self.assertEqual(client.post('/api/knowledge/cycle').json()['reason'],'paused')
            self.assertEqual(client.get('/api/knowledge').status_code,200)
        finally:
            client.close()

    async def test_weekly_refresh_uses_new_search_not_completed_checkpoint(self):
        first=autonomy.cycle();job=jobs._claim_job('test')
        qid=job['metadata']['question_id']
        jobs._finish(job['id'],job['worker_id'],'succeeded','done')
        with db._connect() as conn:
            conn.execute("UPDATE knowledge_questions SET state='done',next_at='2000-01-01' WHERE id=?",(qid,))
        second=autonomy.cycle()
        self.assertTrue(second['scheduled'])
        self.assertNotEqual(first['job_id'],second['job_id'])

    def test_global_pause_and_resume_restores_question(self):
        first=autonomy.cycle()
        client=TestClient(main.app)
        try:
            client.post('/api/knowledge/settings',json={'enabled':False})
            client.post('/api/knowledge/settings',json={'enabled':True})
            second=client.post('/api/knowledge/cycle').json()
        finally:
            client.close()
        self.assertTrue(second['scheduled'])
        self.assertEqual(first['job_id'],second['job_id'])

    def test_changed_topics_do_not_run_old_seed(self):
        k.add_question('old topic search','why','old')
        k.configure(topics=['newtopic'])
        action=autonomy.cycle()
        self.assertIn('newtopic',jobs.get_job(action['job_id'])['request'])

    def test_depth_limit_stops_recursive_expansion(self):
        n=k.get_note(k.save_note('source','source','source',privacy='public'))
        self.assertIsNone(k.add_question('another public question','why','topic',note=n,depth=3))


if __name__=='__main__':
    unittest.main()
