"""Versioned knowledge, tentative relationships and a persistent curiosity queue."""
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

from . import db
from .job_context import current


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def init_knowledge():
    with db._connect() as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS knowledge_notes (
          id INTEGER PRIMARY KEY, origin TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
          content TEXT NOT NULL, source_url TEXT NOT NULL DEFAULT '', privacy TEXT NOT NULL DEFAULT 'private',
          kind TEXT NOT NULL, topics_json TEXT NOT NULL DEFAULT '[]', claims_json TEXT NOT NULL DEFAULT '[]',
          questions_json TEXT NOT NULL DEFAULT '[]', coverage_json TEXT NOT NULL DEFAULT '{}',
          version INTEGER NOT NULL DEFAULT 1, content_hash TEXT NOT NULL,
          embedding_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_versions (
          note_id INTEGER NOT NULL, version INTEGER NOT NULL, snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL,
          PRIMARY KEY(note_id,version));
        CREATE TABLE IF NOT EXISTS knowledge_links (
          id INTEGER PRIMARY KEY, from_id INTEGER NOT NULL, to_id INTEGER NOT NULL,
          from_version INTEGER NOT NULL, to_version INTEGER NOT NULL, relation TEXT NOT NULL,
          reason TEXT NOT NULL, left_quote TEXT NOT NULL, right_quote TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'tentative', created_at TEXT NOT NULL,
          UNIQUE(from_id,to_id,from_version,to_version,relation));
        CREATE TABLE IF NOT EXISTS knowledge_questions (
          id INTEGER PRIMARY KEY, query TEXT NOT NULL, query_key TEXT NOT NULL UNIQUE,
          reason TEXT NOT NULL, note_id INTEGER, note_version INTEGER, topic TEXT NOT NULL,
          depth INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0, next_at TEXT NOT NULL, job_id TEXT,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_sources (
          external_id INTEGER PRIMARY KEY, depth INTEGER NOT NULL DEFAULT 0,
          state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          next_at TEXT NOT NULL, job_id TEXT, error TEXT);
        CREATE TABLE IF NOT EXISTS knowledge_dispatch (
          id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, day TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_recipes (
          id INTEGER PRIMARY KEY, topic TEXT NOT NULL UNIQUE, query TEXT NOT NULL,
          steps_json TEXT NOT NULL, successes INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge_recipe_runs (job_id TEXT PRIMARY KEY, recipe_id INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS knowledge_question_queue ON knowledge_questions(state,next_at);
        CREATE INDEX IF NOT EXISTS knowledge_source_queue ON knowledge_sources(state,next_at);
        ''')


def guard(conn):
    conn.execute('BEGIN IMMEDIATE')
    if current.get() is not None:
        current.get().check(conn)


def decode(row):
    if row is None:
        return None
    out = dict(row)
    for key in ('topics','claims','questions','coverage','embedding'):
        raw = out.pop(key+'_json', None)
        if raw is not None:
            out[key] = json.loads(raw)
    return out


def get_note(note_id):
    with db._connect() as conn:
        return decode(conn.execute('SELECT * FROM knowledge_notes WHERE id=?',(note_id,)).fetchone())


def save_note(origin, title, content, *, kind='source', privacy='private', source_url='', topics=None,
              claims=None, questions=None, coverage=None, embedding=None):
    values = dict(title=title,content=content,kind=kind,privacy=privacy,source_url=source_url,
                  topics=topics or [],claims=claims or [],questions=questions or [],coverage=coverage or {})
    fingerprint = digest(encode(values))
    with db._connect() as conn:
        guard(conn)
        old = conn.execute('SELECT * FROM knowledge_notes WHERE origin=?',(origin,)).fetchone()
        if old and old['content_hash'] == fingerprint:
            return int(old['id'])
        version = int(old['version'])+1 if old else 1
        stamp = now()
        conn.execute('''INSERT INTO knowledge_notes(origin,title,content,source_url,privacy,kind,topics_json,
          claims_json,questions_json,coverage_json,version,content_hash,embedding_json,created_at,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(origin) DO UPDATE SET
          title=excluded.title,content=excluded.content,source_url=excluded.source_url,privacy=excluded.privacy,
          kind=excluded.kind,topics_json=excluded.topics_json,claims_json=excluded.claims_json,
          questions_json=excluded.questions_json,coverage_json=excluded.coverage_json,version=excluded.version,
          content_hash=excluded.content_hash,embedding_json=excluded.embedding_json,updated_at=excluded.updated_at''',
          (origin,title,content,source_url,privacy,kind,encode(topics or []),encode(claims or []),
           encode(questions or []),encode(coverage or {}),version,fingerprint,encode(embedding) if embedding else None,stamp,stamp))
        note_id = conn.execute('SELECT id FROM knowledge_notes WHERE origin=?',(origin,)).fetchone()[0]
        conn.execute('INSERT INTO knowledge_versions VALUES(?,?,?,?)',(note_id,version,encode(values),stamp))
        if old:
            conn.execute("UPDATE knowledge_links SET status='stale' WHERE from_id=? OR to_id=?",(note_id,note_id))
            conn.execute("UPDATE knowledge_questions SET state='stale',updated_at=? WHERE note_id=? AND state='pending'",(stamp,note_id))
        return int(note_id)


def terms(text):
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{2,}|[一-龥ぁ-んァ-ヶー]{2,}', text.lower())
    result = set(words)
    for word in words:
        if re.search('[一-龥ぁ-んァ-ヶ]',word):
            result.update(word[i:i+2] for i in range(len(word)-1))
    return result


def related(note, limit=6):
    """Search across old notes; age alone must never exclude a useful idea."""
    from .memory import cosine
    query = terms(note['title']+' '+note['content']+' '+ ' '.join(note.get('topics',[])))
    with db._connect() as conn:
        rows = conn.execute('SELECT * FROM knowledge_notes WHERE id<>?',(note['id'],)).fetchall()
    ranked = []
    for row in rows:
        candidate = decode(row)
        other = terms(candidate['title']+' '+candidate['content']+' '+' '.join(candidate['topics']))
        overlap = len(query & other) / max(1, min(len(query), len(other)))
        semantic = cosine(note.get('embedding'), candidate.get('embedding'))
        if overlap >= .08 or semantic >= .65:
            ranked.append((max(overlap, semantic if semantic >= .65 else 0),candidate))
    return [n for _,n in sorted(ranked,key=lambda x:x[0],reverse=True)[:limit]]


def save_links(note, candidates, proposed):
    targets = {n['id']:n for n in candidates}
    saved = 0
    with db._connect() as conn:
        guard(conn)
        for item in proposed[:6]:
            if not isinstance(item,dict):
                continue
            target = targets.get(item.get('target_id'))
            relation = item.get('relation')
            left,right = item.get('left_quote'),item.get('right_quote')
            reason = str(item.get('reason','')).strip()[:1000]
            if not target or relation not in {'supports','contradicts','extends','applies_to','related'}:
                continue
            if not isinstance(left,str) or not isinstance(right,str) or len(left)<4 or len(right)<4 or not reason:
                continue
            if left not in note['content'] or right not in target['content']:
                continue
            # Concurrent human edits invalidate the candidate snapshot.
            versions = dict(conn.execute('SELECT id,version FROM knowledge_notes WHERE id IN (?,?)',(note['id'],target['id'])).fetchall())
            if versions.get(note['id'])!=note['version'] or versions.get(target['id'])!=target['version']:
                continue
            cur = conn.execute('''INSERT OR IGNORE INTO knowledge_links
              (from_id,to_id,from_version,to_version,relation,reason,left_quote,right_quote,created_at)
              VALUES(?,?,?,?,?,?,?,?,?)''',(note['id'],target['id'],note['version'],target['version'],relation,reason,left,right,now()))
            saved += cur.rowcount
    return saved


def add_question(query, reason, topic, *, note=None, depth=0):
    query = re.sub(r'\s+',' ',str(query)).strip()[:300]
    if len(query)<4 or depth>2:
        return None
    if note is not None and note.get('privacy')!='public':
        return None
    with db._connect() as conn:
        guard(conn)
        existing=conn.execute('SELECT id FROM knowledge_questions WHERE query_key=?',(digest(query.casefold()),)).fetchone()
        if existing:
            return existing[0]
        if conn.execute("SELECT COUNT(*) FROM knowledge_questions WHERE state IN ('pending','queued')").fetchone()[0]>=100:
            return None
        conn.execute('''INSERT OR IGNORE INTO knowledge_questions(query,query_key,reason,note_id,note_version,
          topic,depth,next_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)''',
          (query,digest(query.casefold()),reason[:1000],note['id'] if note else None,note['version'] if note else None,
           topic[:120],depth,now(),now(),now()))
        return conn.execute('SELECT id FROM knowledge_questions WHERE query_key=?',(digest(query.casefold()),)).fetchone()[0]


def settings():
    import os
    with db._connect() as conn:
        saved = {r['key']:json.loads(r['value']) for r in conn.execute('SELECT * FROM knowledge_settings')}
    return {'enabled':saved.get('enabled',os.getenv('AUTONOMY_ENABLED','true').lower()=='true'),
            'topics':saved.get('topics',[s.strip() for s in os.getenv('AUTONOMY_TOPICS','AI agents,knowledge management,Obsidian').split(',') if s.strip()]),
            'daily_jobs':max(1,min(48,int(os.getenv('AUTONOMY_DAILY_JOBS','12')))),
            'interval_seconds':max(60,int(os.getenv('AUTONOMY_INTERVAL_SECONDS','600')))}


def configure(enabled=None, topics=None):
    with db._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if topics is not None:
            conn.execute("UPDATE knowledge_questions SET state='out_of_scope' WHERE state IN ('pending','paused')")
            conn.execute("UPDATE knowledge_sources SET state='out_of_scope' WHERE state IN ('pending','paused')")
            for topic in topics:
                conn.execute("UPDATE knowledge_questions SET state='pending' WHERE state='out_of_scope' AND note_id IS NULL AND topic=?",(topic,))
        if enabled is True:
            conn.execute("UPDATE knowledge_questions SET state='pending' WHERE state='paused'")
            conn.execute("UPDATE knowledge_sources SET state='pending' WHERE state='paused'")
        for key,value in [('enabled',enabled),('topics',topics)]:
            if value is not None:
                conn.execute('INSERT INTO knowledge_settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,encode(value)))
    return settings()


def snapshot(limit=30):
    with db._connect() as conn:
        notes = [decode(r) for r in conn.execute('SELECT * FROM knowledge_notes ORDER BY updated_at DESC LIMIT ?',(limit,))]
        links = [dict(r) for r in conn.execute('''SELECT l.*, a.title AS from_title,b.title AS to_title,
          b.created_at AS old_note_created_at FROM knowledge_links l JOIN knowledge_notes a ON a.id=l.from_id
          JOIN knowledge_notes b ON b.id=l.to_id WHERE l.status='tentative' ORDER BY l.id DESC LIMIT ?''',(limit,))]
        questions = [dict(r) for r in conn.execute('SELECT * FROM knowledge_questions ORDER BY id DESC LIMIT ?',(limit,))]
        recipes = [dict(r) for r in conn.execute('SELECT * FROM knowledge_recipes ORDER BY updated_at DESC LIMIT ?',(limit,))]
        counts = {table:conn.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in
                  ('knowledge_notes','knowledge_links','knowledge_questions','knowledge_versions')}
        sources = {r['state']:r['n'] for r in conn.execute('SELECT state,COUNT(*) n FROM knowledge_sources GROUP BY state')}
        dispatched = conn.execute('SELECT COUNT(*) FROM knowledge_dispatch WHERE day=?',(now()[:10],)).fetchone()[0]
    return dict(settings=settings(),notes=notes,links=links,questions=questions,recipes=recipes,
                counts=counts,sources=sources,today_jobs=dispatched)
