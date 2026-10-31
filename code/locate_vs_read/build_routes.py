"""Routing fixes for on-demand materialisation. The read side is unchanged from the
naive lazy arm: only the cached native summaries of the routed states are read (plus the
notes of the first three routed trajectories). Only the route that picks states changes.

  lazy_rewrite : 9B rewrites the question into page/field vocabulary, then structure BM25
  lazy_embed   : Qwen3-Embedding-0.6B over structure chunks (dense retrieval)
  lazy_coarse  : one cheap 9B call per trajectory writes a coarse summary from goal, page
                 titles and actions; BM25 over coarse summaries picks trajectories, structure
                 BM25 orders states inside them

Stages: rewrite | coarse | embed | contexts. Each stage writes its own file and records cost."""
import argparse, collections, json, sys, time, urllib.request
import concurrent.futures as cf
from pathlib import Path
import numpy as np
sys.argv, _argv = ['x'], sys.argv
exec(open(Path(__file__).with_name('build_arms.py')).read().replace("if __name__ == '__main__':\n    main()", ''))
sys.argv = _argv
R = OUT / 'routes'


def post(port, path, body, timeout=1800):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(prompt, max_tokens):
    resp = post(8122, '/v1/chat/completions', {'model': 'Qwen/Qwen3.5-9B', 'temperature': 0, 'max_tokens': max_tokens,
                                                'chat_template_kwargs': {'enable_thinking': False},
                                                'messages': [{'role': 'user', 'content': prompt}]})
    return resp['choices'][0]['message']['content'], resp.get('usage') or {}


def qset():
    split = json.loads((OUT / 'question_split.json').read_text()); _, _, _, rows = select_questions()
    return split['old_dev12'] + split['new_dev24'], rows


def stage_rewrite():
    qids, rows = qset()
    def one(q):
        text, u = chat('An AI agent earlier worked in a ServiceNow web portal. We will search the accessibility trees of the pages it saw '
                       'to answer the question below. Write search keywords: the likely page names, form or list names, tab names, field labels, '
                       'button labels and record values that would appear on the relevant pages. Output one line of space-separated keywords, nothing else.\n\n'
                       f'Question: {clean_question(rows[q]["question"])}', 200)
        return q, text.strip(), u
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, qids))
    out = {q: {'rewrite': t, 'usage': u} for q, t, u in res}
    (R / 'rewrite.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print('rewrite tokens', sum(v['usage'].get('prompt_tokens', 0) for v in out.values()), sum(v['usage'].get('completion_tokens', 0) for v in out.values()))


TITLE = re.compile(r"^\t*(?:\[[^\]]*\]\s*)?RootWebArea '((?:[^'\\]|\\.)*)'", re.M)


def stage_coarse():
    per = collections.defaultdict(list)
    for line in STATES.open():
        s = json.loads(line)
        titles = list(dict.fromkeys(TITLE.findall(s['accessibility_tree'])))[:2]
        per[s['trajectory_id']].append((s['state_index'], ' / '.join(titles), (s.get('action') or '')[:160]))
    goals = {}
    for d in POOL.iterdir():
        goals[d.name] = json.loads((d / 'agentrunbook_r_artifact/procedure_note.json').read_text()).get('goal', '')
    goal_of = {}
    for line in (ROOT / 'trajectories.jsonl').open():
        t = json.loads(line)
        if t['id'] in per:
            goal_of[t['id']] = str(t.get('goal') or t.get('task') or t.get('instruction') or '')[:800]
    def one(tid):
        lines, prev = [], None
        for st, title, act in sorted(per[tid]):
            if title != prev:
                lines.append(f'state {st}: page "{title}"'); prev = title
            if act:
                lines.append(f'  action: {act}')
        body = '\n'.join(lines)[:9000]
        text, u = chat('Below is a compressed log of one AI agent session in a ServiceNow portal: its goal, the page titles it visited and its actions. '
                       'Write a coarse index summary (at most 120 words) listing which forms, lists, records, tabs and fields it worked with and what it did. '
                       f'Use the exact names that appear in the log.\n\nGoal: {goal_of.get(tid) or goals.get(tid, "")}\n\n{body}', 300)
        return tid, text.strip(), u
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, sorted(per)))
    out = {t: {'coarse': x, 'usage': u} for t, x, u in res}
    (R / 'coarse.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print('coarse calls', len(out), 'tokens', sum(v['usage'].get('prompt_tokens', 0) for v in out.values()), sum(v['usage'].get('completion_tokens', 0) for v in out.values()))


INSTR = "Instruct: Given a question about an AI agent's earlier ServiceNow sessions, retrieve the page states that contain the answer\nQuery: "


def embed(texts, batch=32):
    vecs, tokens = [], 0
    for i in range(0, len(texts), batch):
        r = post(8119, '/v1/embeddings', {'model': 'memory-qwen3-embedding-0.6b', 'input': texts[i:i + batch], 'truncate_prompt_tokens': 8000})
        vecs += [d['embedding'] for d in r['data']]; tokens += (r.get('usage') or {}).get('prompt_tokens', 0)
    v = np.asarray(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True), tokens


def stage_embed():
    units, _ = build_units(); S = units['structure']
    t = time.time(); docs, dt = embed([u['text'] for u in S]); ds = time.time() - t
    qids, rows = qset(); qv, qt = embed([INSTR + clean_question(rows[q]['question']) for q in qids])
    np.save(R / 'structure_emb.npy', docs); np.save(R / 'query_emb.npy', qv)
    (R / 'embed_cost.json').write_text(json.dumps({'doc_tokens': dt, 'doc_seconds': ds, 'query_tokens': qt, 'units': len(S), 'model': 'Qwen3-Embedding-0.6B'}, indent=1))
    print('embed', dt, round(ds, 1), qt)


def stage_contexts():
    qids, rows = qset(); units, _ = build_units(); S, M = units['structure'], units['summary']
    bS = BM25([u['text'] for u in S])
    by_state, notes = collections.defaultdict(list), collections.defaultdict(list)
    for i, u in enumerate(M):
        tid, kind, _ = u['key']
        if kind.startswith('event_'):
            a, b = kind.split('_')[1:3]; by_state[(tid, int(a))].append(i); by_state[(tid, int(b))].append(i)
        else:
            notes[tid].append(i)
    rewrite = json.loads((R / 'rewrite.json').read_text()); coarse = json.loads((R / 'coarse.json').read_text())
    ctids = sorted(coarse); bC = BM25([coarse[t]['coarse'] for t in ctids])
    docs, qv = np.load(R / 'structure_emb.npy'), np.load(R / 'query_emb.npy')

    def fill(order, budget):
        picked, used = [], 0
        for i in order:
            n = ntok(M[i]['text'])
            if used + n > budget:
                continue
            picked.append(i); used += n
            if used > budget - 300:
                break
        return picked, used

    def read_states(states, budget):
        order, seen, trajs = [], set(), []
        for tid, st in states:
            if tid not in trajs:
                trajs.append(tid)
            for j in by_state.get((tid, int(st)), []) + (notes[tid] if len(trajs) <= 3 else []):
                if j not in seen:
                    seen.add(j); order.append(j)
        return fill(order, budget)

    out, touched = [], collections.defaultdict(set)
    for n, q in enumerate(qids):
        qt = clean_question(rows[q]['question'])
        routes = {}
        rw = re.sub(r'(?i)\b(servicenow|portal|form names?|list names?|tab names?|field labels?|button labels?|record values?|page names?)\b', ' ', rewrite[q]['rewrite'])
        routes['lazy_rewrite'] = [S[i]['key'][:2] for i in bS.top(qt + ' ' + rw, 400)]
        routes['lazy_embed'] = [S[i]['key'][:2] for i in np.argsort(-(docs @ qv[n]))[:400]]
        top_t = [ctids[i] for i in bC.top(qt, 10)]
        ranked = [S[i]['key'][:2] for i in bS.top(qt, 3000)]
        routes['lazy_coarse'] = [k for t in top_t for k in ranked if k[0] == t]
        for arm, states in routes.items():
            for budget in BUDGETS:
                pl, ul = read_states(states, budget); touched[(arm, budget)] |= set(pl)
                ctx = '\n\n'.join(M[i]['text'] for i in sorted(pl, key=lambda i: tuple(map(str, M[i]['key']))))
                out.append({'id': q, 'arm': arm, 'budget': budget, 'context_tokens': ul, 'context': ctx})
    with (OUT / 'contexts.jsonl').open('a') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    rep = {f'{a}_{b}': {'distinct_summary_units_materialised': len(t), 'of_total_units': len(M)} for (a, b), t in sorted(touched.items())}
    (R / 'route_materialisation_cost.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    R.mkdir(exist_ok=True)
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['rewrite', 'coarse', 'embed', 'contexts'])
    globals()['stage_' + ap.parse_args().stage]()
