"""Obsidian knowledge workspace. Human Inbox is imported; generated edits are preserved."""
import json
import os
import re
import threading
from pathlib import Path

from . import db, knowledge as k
from .config import OBSIDIAN_VAULT_PATH

_export_lock=threading.RLock()


def root():
    return Path(OBSIDIAN_VAULT_PATH).expanduser().resolve() / 'Second Brain'


def import_personal_notes():
    # Existing explicit user memories become candidates even if they are old.
    with db._connect() as conn:
        memories=conn.execute("SELECT * FROM memories WHERE kind IN ('note','goal','decision','preference','task') AND source IN ('user','manual') ORDER BY id").fetchall()
    for memory in memories:
        k.save_note('memory:'+str(memory['id']),memory['content'][:80],memory['content'],kind=memory['kind'],privacy='private')
    inbox=root()/'Inbox';inbox.mkdir(parents=True,exist_ok=True)
    for path in sorted(inbox.rglob('*.md'))[:200]:
        if path.is_symlink() or not path.resolve().is_relative_to(inbox.resolve()) or path.stat().st_size>256000:
            continue
        content=path.read_text(encoding='utf-8-sig')
        k.save_note('obsidian:'+str(path.relative_to(inbox)),path.stem,content,kind='idea',privacy='private')

    # Human edits in a generated knowledge note are imported as a private correction,
    # rather than silently replacing the original source-backed reading.
    managed=root()/'Knowledge'
    if managed.exists():
        for path in sorted(managed.glob('*.md'))[:500]:
            if path.is_symlink() or '.pending-' in path.name or path.stat().st_size>256000:
                continue
            relative='Knowledge/'+path.name
            with db._connect() as conn:
                old=conn.execute('SELECT value FROM knowledge_settings WHERE key=?',('export_hash:'+relative,)).fetchone()
            content=path.read_text(encoding='utf-8-sig')
            if old and json.loads(old[0])!=k.digest(content):
                k.save_note('obsidian-edit:'+path.stem,'人による訂正・追記 '+path.stem,content,kind='correction',privacy='private')


def write_managed(relative,text):
    """Do not erase edits to generated notes. Offer a separately named new version."""
    target=root()/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.resolve().is_relative_to(root().resolve()) or target.is_symlink():
        raise ValueError('export path outside managed vault')
    key='export_hash:'+relative
    fingerprint=k.digest(text)
    with db._connect() as conn:
        previous=conn.execute('SELECT value FROM knowledge_settings WHERE key=?',(key,)).fetchone()
        known=json.loads(previous[0]) if previous else None
    if target.exists():
        existing=target.read_text(encoding='utf-8')
        if existing==text:
            return
        if known!=k.digest(existing):
            target=target.with_name(target.stem+'.pending-'+fingerprint[:10]+'.md')
            if target.exists():
                return
            target.write_text(text,encoding='utf-8')
            return
    temporary=target.with_suffix('.tmp')
    temporary.write_text(text,encoding='utf-8')
    os.replace(temporary,target)
    with db._connect() as conn:
        conn.execute('INSERT INTO knowledge_settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,k.encode(fingerprint)))


def label(text):
    return re.sub(r'[\[\]|\r\n]',' ',str(text))


def wiki(note_id,title):
    return f'[[Second Brain/Knowledge/{int(note_id):08d}|{label(title)}]]'


def export_knowledge():
    with _export_lock:
        return _export()


def _export():
    with db._connect() as conn:
        notes=[k.decode(r) for r in conn.execute('SELECT * FROM knowledge_notes ORDER BY id')]
        links=[dict(r) for r in conn.execute("SELECT * FROM knowledge_links WHERE status='tentative' ORDER BY id")]
        questions=[dict(r) for r in conn.execute('SELECT * FROM knowledge_questions ORDER BY id')]
        recipes=[dict(r) for r in conn.execute('SELECT * FROM knowledge_recipes ORDER BY id')]
    by_id={n['id']:n for n in notes}
    for note in notes:
        lines=['---','type: second-brain-knowledge',f'id: {note["id"]}',f'version: {note["version"]}',
          'privacy: '+note['privacy'],'tags: '+k.encode(note['topics']),'---','',f'# {note["title"]}','',
          '資料の解釈です。主張の真偽と、AIによる関連付けは未確定です。','',note['content'],'',
          '## 出所','',note['source_url'] or note['origin'],'','## 関連する知識','']
        for link in links:
            if note['id'] not in (link['from_id'],link['to_id']):
                continue
            other=by_id.get(link['to_id'] if link['from_id']==note['id'] else link['from_id'])
            if other:
                lines.append(f'- {wiki(other["id"],other["title"])} — {link["relation"]}（仮説）: {link["reason"]}')
        lines+=['','## 根拠となる原文','']
        for claim in note['claims']:
            lines.extend([f'- {claim["kind"]}: {claim["text"]}',f'  - 原文位置 {claim.get("offset",0)}: '+claim['quote']])
        lines+=['','## 読み込み範囲','',k.encode(note['coverage']),'','## 訂正・追記','',
          '編集内容は上書きしません。このノートへの編集は次の巡回で私的な訂正ノートとして取り込みます。自由なメモは Second Brain/Inbox に保存できます。','']
        write_managed(f'Knowledge/{note["id"]:08d}.md','\n'.join(lines))
    resurfaced=['# 再浮上した知識','','新しい資料をきっかけに見つかった、過去の知識との関係です。すべて仮説です。','']
    for link in reversed(links[-100:]):
        a,b=by_id.get(link['from_id']),by_id.get(link['to_id'])
        if a and b:
            resurfaced.append(f'- {wiki(b["id"],b["title"])} ← {wiki(a["id"],a["title"])}: {link["reason"]}')
    write_managed('Resurfaced.md','\n'.join(resurfaced)+'\n')
    frontier=['# 次に調べる疑問','','公開資料から生まれた疑問だけを、自動検索へ送ります。','']
    for q in questions:
        source=wiki(q['note_id'],by_id[q['note_id']]['title']) if q['note_id'] in by_id else '関心分野'
        frontier.extend([f'## {q["query"]}',f'- 状態: {q["state"]}',f'- 理由: {q["reason"]}',f'- 起点: {source}',''])
    write_managed('Questions.md','\n'.join(frontier))
    workflows=['# 再利用する調査手順','','収集に成功した検索条件と固定手順です。正確性が実証されたワークフローという意味ではありません。','']
    for recipe in recipes:
        workflows.extend([f'## {recipe["topic"]}',f'- 検索条件: {recipe["query"]}',f'- 収集成功回数: {recipe["successes"]}',
          '- 手順: 検索 → 本文取得 → 分割解読 → 関連付け → 次の疑問',''])
    write_managed('Workflows.md','\n'.join(workflows))
    home=['# 自主的に育つSecond Brain','','- [[Second Brain/Resurfaced|再浮上した知識]]','- [[Second Brain/Questions|次に調べる疑問]]',
          '- [[Second Brain/Workflows|再利用する調査手順]]','','## 知識','']
    home += ['- '+wiki(n['id'],n['title']) for n in notes]
    write_managed('HOME.md','\n'.join(home)+'\n')
    inbox=root()/'Inbox';inbox.mkdir(parents=True,exist_ok=True)
    return {'notes':len(notes),'links':len(links),'questions':len(questions),'vault':str(root())}
