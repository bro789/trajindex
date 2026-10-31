"""New direction, step 2: can a much cheaper semantic index keep the routing quality of the
full native summaries? Reading side is fixed to raw AX chunks of the routed states (the best
reader content in the first test); only the index changes.

  key0_raw  : deterministic per-event key, no model call (goal head, action, page titles,
              first new structure lines of the post state)
  key1_raw  : one 9B call per event writes a one-line key from cheap inputs only
              (action, page titles, structure diff); no screenshots, no full AX
  key1c_raw : key1 plus the 100 coarse per-trajectory summaries from the routing round
  key0t_raw : key0 plus the agent's own recorded thoughts of the two states (free, logged at run time)
  thought_raw : the recorded thoughts only

Stages: inputs | keygen | contexts | contexts2 (the two thought arms)."""
import argparse, collections, json, sys, urllib.request
import concurrent.futures as cf
from pathlib import Path
sys.argv, _argv = ['x'], sys.argv
exec(open(Path(__file__).with_name('build_arms.py')).read().replace("if __name__ == '__main__':\n    main()", ''))
sys.argv = _argv
K = OUT / 'cheap_index'
TITLE = re.compile(r"^\t*(?:\[[^\]]*\]\s*)?RootWebArea '((?:[^'\\]|\\.)*)'", re.M)


def stage_inputs():
    views, titles, goals = {}, {}, {}
    for line in STATES.open():
        s = json.loads(line); k = (s['trajectory_id'], int(s['state_index']))
        views[k] = structure_view(s['accessibility_tree']).split('\n')
        titles[k] = ' / '.join(list(dict.fromkeys(TITLE.findall(s['accessibility_tree'])))[:2])
    rows = []
    for d in sorted(POOL.iterdir()):
        for line in (d / 'agentrunbook_r_artifact/event_pool.jsonl').open():
            e = json.loads(line); tid = e['trajectory_id']
            a, b = int(e['pre_state_index']), int(e['post_state_index'])
            before = set(views.get((tid, a), []))
            diff = [x[:120] for x in views.get((tid, b), []) if x not in before][:40]
            goals[tid] = e['goal']
            rows.append({'trajectory_id': tid, 'event_id': e['event_id'], 'pre': a, 'post': b, 'goal_head': e['goal'][:300],
                         'action': e['local_action'][:200], 'title_pre': titles.get((tid, a), ''), 'title_post': titles.get((tid, b), ''), 'diff': diff})
    with (K / 'event_inputs.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('events', len(rows))


def key0(r):
    return (f"goal: {r['goal_head'][:200]}\naction: {r['action']}\npage: {r['title_pre']} -> {r['title_post']}\n" + '\n'.join(r['diff'][:15]))


def prompt1(r):
    return ('One step of an AI agent in a ServiceNow portal is described below. Write ONE line (at most 30 words) that would help someone later '
            'find this step: name the form, list, record, tab or field involved and what the agent did or saw. Use exact names. Output only the line.\n\n'
            f"Task: {r['goal_head']}\nAction: {r['action']}\nPage before: {r['title_pre']}\nPage after: {r['title_post']}\n"
            'New items on the page after the action:\n' + '\n'.join(r['diff']))


def stage_keygen():
    rows = [json.loads(l) for l in (K / 'event_inputs.jsonl').open()]
    def one(r):
        body = {'model': 'Qwen/Qwen3.5-9B', 'temperature': 0, 'max_tokens': 80, 'chat_template_kwargs': {'enable_thinking': False},
                'messages': [{'role': 'user', 'content': prompt1(r)}]}
        req = urllib.request.Request('http://127.0.0.1:8122/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as x:
            resp = json.loads(x.read())
        return {'trajectory_id': r['trajectory_id'], 'event_id': r['event_id'], 'key': resp['choices'][0]['message']['content'].strip(),
                'usage': resp.get('usage') or {}}
    t = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        out = list(ex.map(one, rows))
    with (K / 'key1.jsonl').open('w') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    cost = {'calls': len(out), 'prompt_tokens': sum(o['usage'].get('prompt_tokens', 0) for o in out),
            'completion_tokens': sum(o['usage'].get('completion_tokens', 0) for o in out), 'seconds': time.time() - t, 'concurrency': 16}
    (K / 'key1_cost.json').write_text(json.dumps(cost, indent=1)); print(cost)


def stage_contexts():
    split = json.loads((OUT / 'question_split.json').read_text()); _, _, _, qrows = select_questions()
    qids = split['old_dev12'] + split['new_dev24']
    units, _ = build_units(); RAW = units['raw']
    chunks, states_of = collections.defaultdict(list), collections.defaultdict(set)
    for i, u in enumerate(RAW):
        k = (u['key'][0], int(u['key'][1])); chunks[k].append(i); states_of[k[0]].add(k[1])
    ev = [json.loads(l) for l in (K / 'event_inputs.jsonl').open()]
    if not ARMS2:
        k1 = {(x['trajectory_id'], x['event_id']): x['key'] for x in map(json.loads, (K / 'key1.jsonl').open())}
        coarse = json.loads((OUT / 'routes/coarse.json').read_text())
        indexes = {
            'key0_raw': [((r['trajectory_id'], [r['pre'], r['post']]), key0(r)) for r in ev],
            'key1_raw': [((r['trajectory_id'], [r['pre'], r['post']]), k1[(r['trajectory_id'], r['event_id'])]) for r in ev],
        }
        indexes['key1c_raw'] = indexes['key1_raw'] + [((t, None), v['coarse']) for t, v in coarse.items()]
    else:
        thought = {}
        wanted = {r['trajectory_id'] for r in ev}
        for line in (ROOT / 'trajectories.jsonl').open():
            t = json.loads(line)
            if t['id'] in wanted:
                for st in t['states']:
                    thought[(t['id'], int(st['state_index']))] = st.get('thought') or ''
        th = lambda r: (f"thought before: {thought.get((r['trajectory_id'], r['pre']), '')}\n"
                        f"thought after: {thought.get((r['trajectory_id'], r['post']), '')}")
        indexes = {'key0t_raw': [((r['trajectory_id'], [r['pre'], r['post']]), key0(r) + '\n' + th(r)) for r in ev],
                   'thought_raw': [((r['trajectory_id'], [r['pre'], r['post']]), th(r)) for r in ev]}
    out = []
    for arm, idx in indexes.items():
        bm = BM25([text for _, text in idx])
        for qid in qids:
            order, seen = [], set()
            for i in bm.top(clean_question(qrows[qid]['question']), 400):
                tid, sts = idx[i][0]
                for s in (sts if sts is not None else sorted(states_of[tid])):
                    if (tid, s) not in seen:
                        seen.add((tid, s)); order.append((tid, s))
            for budget in BUDGETS:
                picked, used = [], 0
                for st in order:
                    for i in chunks.get(st, []):
                        n = ntok(RAW[i]['text'])
                        if used + n <= budget:
                            picked.append(i); used += n
                    if used > budget - 300:
                        break
                ctx = '\n\n'.join(RAW[i]['text'] for i in sorted(picked, key=lambda i: tuple(map(str, RAW[i]['key']))))
                out.append({'id': qid, 'arm': arm, 'budget': budget, 'context_tokens': used, 'context': ctx})
    with (OUT / 'contexts.jsonl').open('a') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print('added', len(out))


if __name__ == '__main__':
    K.mkdir(exist_ok=True)
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['inputs', 'keygen', 'contexts', 'contexts2'])
    st = ap.parse_args().stage; ARMS2 = st == 'contexts2'
    globals()['stage_' + st.rstrip('2')]()
