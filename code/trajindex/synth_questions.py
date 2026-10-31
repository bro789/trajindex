"""Synthetic tuning questions from trajectories outside every benchmark haystack.

For each domain, unused trajectories are ordered by sha256(salt|id); the first 100 form
haystack h1 and the next 100 form h2. Seeds are sampled per haystack, 30 per question type:
  static     one page (structure view) -> a fact shown on that page
  dynamic    one action and the page lines it added/removed -> what changed
  procedure  the trajectory's page/action list -> which step or value the agent used
The 9B writes a question and a short answer; a question is kept only if the answer appears
verbatim in the source text it was given and not in the question. Style examples come from the
enterprise development questions only; the 258 test questions are never read."""
import argparse, hashlib, json, random, re, sys, urllib.request
import concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import zc_index as z

SALT = 'synthetic_tuning_haystacks_20261024'
OUT = Path(__file__).parent
EVAL = 'norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true'
SUFFIX = '\n\nMark your final answer (should be one or more short phrases) in \\boxed{}.'
TASK = {
    'static-environment': 'Ask about a specific fact shown on this page (a field value, a label, a tab name, an option, a count of items).',
    'dynamic-environment': 'Ask about what changed on the page after the action (a new value, a status, a message, an item that appeared).',
    'procedure': 'Ask about how the task was carried out (which page, menu, filter, button or value the agent used at some point).',
}


def haystacks(domain):
    hs = json.loads((z.ROOT / 'lme_v2_small.json').read_text()); used = set()
    for v in hs.values():
        used |= set(v)
    ids = []
    for line in (z.ROOT / 'trajectories.jsonl').open():
        t = json.loads(line)
        if t['domain'] == domain and t['id'] not in used and sum(1 for s in t['states']) >= 3:
            ids.append(t['id'])
    ids.sort(key=lambda i: hashlib.sha256(f'{SALT}|{i}'.encode()).hexdigest())
    return {'h1': ids[:100], 'h2': ids[100:200]}


def examples(qtype):
    split = json.loads((z.ROOT / 'lazy_materialization_v1/improve_v1/dev_question_split.json').read_text())
    dev = set(split['old_dev12'] + split['new_dev24'])
    ex = [json.loads(l) for l in (z.ROOT / 'questions.jsonl').open()]
    ex = [q for q in ex if q['id'] in dev and q['question_type'] == qtype and q['eval_function'].startswith('norm_phrase')]
    return [z.clean_question(q['question']) for q in ex[:2]]


def source(mem, traj, qtype, rng):
    tid = traj['id']; sts = sorted(int(s['state_index']) for s in traj['states'])
    goal = (traj.get('goal') or '')[:600]
    if qtype == 'static-environment':
        k = (tid, rng.choice(sts[1:]))
        return f"Task goal: {goal}\nPage: {mem.titles[k]}\nPage content:\n" + '\n'.join(mem.views[k])[:5000]
    if qtype == 'dynamic-environment':
        evs = [e for e in mem.events if e['trajectory_id'] == tid and e['diff']]
        e = rng.choice(evs)
        removed = [x for x in mem.views[(tid, e['pre'])] if x not in set(mem.views[(tid, e['post'])])][:20]
        return (f"Task goal: {goal}\nAction: {e['action']}\nPage before: {e['page']}\nLines added by the action:\n" +
                '\n'.join(e['diff']) + '\nLines removed by the action:\n' + '\n'.join(removed))[:5000]
    steps = [f"step {i + 1}: page \"{e['page']}\", action {e['action']}" for i, e in enumerate(e for e in mem.events if e['trajectory_id'] == tid)]
    return f"Task goal: {goal}\nSteps:\n" + '\n'.join(steps)[:5000]


def prompt(src, qtype, domain):
    where = 'a ServiceNow enterprise portal' if domain == 'enterprise' else 'websites'
    ex = '\n'.join(f'- {x}' for x in examples(qtype))
    return (f"An AI agent earlier worked in {where}. Below is part of what it saw and did. Write ONE question that the agent's user might "
            f"ask much later, as if recalling past work, whose answer is in this material. {TASK[qtype]}\n"
            f"Rules: the answer must be a short phrase (at most 6 words) copied exactly from the material; the question must not contain the "
            f"answer; do not mention steps, states, trajectories or 'the material'; name the page, form or site so the question is specific.\n"
            f"Examples of the question style:\n{ex}\n\nMaterial:\n{src}\n\n"
            'Output JSON only: {"question": "...", "answer": "..."}')


def call(port, text):
    body = {'model': 'Qwen/Qwen3.5-9B', 'temperature': 0.3, 'max_tokens': 300, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': text}]}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


def norm(s):
    return re.sub(r'\s+', ' ', s.lower()).strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--domain', required=True); ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--per-type', type=int, default=30); a = ap.parse_args()
    trajs_all, kept, stats = {}, [], {'seeds': 0, 'parsed': 0, 'kept': 0}
    for h, ids in haystacks(a.domain).items():
        trajs = z.load_trajectories(ids); mem = z.Memory(trajs)
        (OUT / f'haystack_{a.domain}_{h}.json').write_text(json.dumps(ids))
        rng = random.Random(f'{SALT}|{a.domain}|{h}'); jobs = []
        for qtype in TASK:
            for n in range(a.per_type):
                t = trajs[rng.choice(ids)]
                jobs.append((qtype, n, t['id'], source(mem, t, qtype, rng)))
        stats['seeds'] += len(jobs)
        with cf.ThreadPoolExecutor(12) as ex:
            outs = list(ex.map(lambda j: call(a.port, prompt(j[3], j[0], a.domain)), jobs))
        for (qtype, n, tid, src), o in zip(jobs, outs):
            m = re.search(r'\{.*\}', o, re.S)
            try:
                d = json.loads(m.group(0)); q, ans = d['question'].strip(), d['answer'].strip()
            except Exception:
                continue
            stats['parsed'] += 1
            if not ans or len(ans.split()) > 6 or norm(ans) not in norm(src) or norm(ans) in norm(q) or not 6 <= len(q.split()) <= 70:
                continue
            if re.search(r'\b(step|state|trajector|material)\b', q, re.I):
                continue
            kept.append({'id': f'syn-{a.domain}-{h}-{qtype[:3]}-{n}', 'domain': a.domain, 'haystack': h, 'source_trajectory': tid,
                         'question_type': qtype, 'question': q + SUFFIX, 'answer': ans, 'eval_function': EVAL, 'image': None})
    stats['kept'] = len(kept)
    with (OUT / f'synth_questions_{a.domain}.jsonl').open('w') as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + '\n')
    (OUT / f'synth_stats_{a.domain}.json').write_text(json.dumps(stats, indent=1)); print(stats)


if __name__ == '__main__':
    main()
