"""Tune the zero-cost index on synthetic questions, then run the chosen configuration on the
real benchmark questions.

  proxy    CPU only: for every route x read x budget, does the gold answer string appear in the
           context given to the reader? Writes proxy.json.
  contexts writes contexts.jsonl for the configurations named by --configs
  read     reader calls for contexts.jsonl (one port), responses/ ; resumable
Scoring is done by score_synth.py (needs the official harness environment).

--set synthetic : questions from synth_questions_<domain>.jsonl over haystacks h1/h2
--set real      : the benchmark's deterministic text questions of the domain over its haystack"""
import argparse, collections, json, os, re, sys, time, urllib.request
import concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import zc_index as z

HERE = Path(__file__).parent
ROUTES, READS, BUDGETS = ('bm25', 'bm25f', 'phrase'), ('state', 'next', 'delta', 'nextdelta'), (8000, 32000)
SYSTEM = ("You answer questions about an AI agent's earlier work sessions in a ServiceNow portal. "
          "You are given memory excerpts retrieved from those sessions. Use only the excerpts. "
          "If the excerpts are insufficient, give your best answer anyway.")


def question_sets(setname, domain):
    """Yield (haystack trajectory ids, [question rows])."""
    if setname == 'synthetic':
        qs = [json.loads(l) for l in (HERE / f'synth_questions_{domain}.jsonl').open()]
        for h in ('h1', 'h2'):
            yield json.loads((HERE / f'haystack_{domain}_{h}.json').read_text()), [q for q in qs if q['haystack'] == h]
    else:
        hs = json.loads((z.ROOT / 'lme_v2_small.json').read_text())
        qs = [json.loads(l) for l in (z.ROOT / 'questions.jsonl').open()]
        qs = [q for q in qs if q['domain'] == domain and q['image'] in (None, 'None') and 'llm' not in q['eval_function']]
        yield hs[qs[0]['id']], qs


def norm(s):
    return re.sub(r'[\s\-]+', ' ', s.lower())


def contexts(setname, domain, configs):
    for ids, qs in question_sets(setname, domain):
        mem = z.Memory(z.load_trajectories(ids))
        for q in qs:
            text = z.clean_question(q['question'])
            routed = {r: mem.route(text, r) for r in {c[0] for c in configs}}
            for route, read, budget in configs:
                ctx, used = mem.read(routed[route], read, budget)
                yield q, f'{route}+{read}', budget, ctx, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['proxy', 'contexts', 'read']); ap.add_argument('--set', default='synthetic')
    ap.add_argument('--domain', required=True); ap.add_argument('--configs', help='comma list route+read')
    ap.add_argument('--port', type=int, default=8122); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    out = HERE / f'{a.set}_{a.domain}'; out.mkdir(exist_ok=True)
    if a.stage == 'proxy':
        configs = [(r, d, b) for r in ROUTES for d in READS for b in BUDGETS]
        if a.configs:
            configs = [(*c.split('+'), b) for c in a.configs.split(',') for b in BUDGETS]
        hit = collections.Counter(); n = collections.Counter(); t = time.time()
        for q, name, budget, ctx, _ in contexts(a.set, a.domain, configs):
            phrases = [p.strip() for p in re.split(r'[,;]', q['answer']) if p.strip()]
            hit[(name, budget)] += all(norm(p) in norm(ctx) for p in phrases); n[(name, budget)] += 1
        res = {f'{k[0]}@{k[1]}': {'hit': hit[k], 'n': n[k], 'rate': round(hit[k] / n[k], 3)} for k in sorted(n)}
        (out / ('proxy.json' if not a.configs else 'proxy_extra.json')).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); print('seconds', round(time.time() - t))
    elif a.stage == 'contexts':
        configs = [(*c.split('+'), b) for c in a.configs.split(',') for b in BUDGETS]
        with (out / 'contexts.jsonl').open('a') as f:
            for q, name, budget, ctx, used in contexts(a.set, a.domain, configs):
                f.write(json.dumps({'id': q['id'], 'arm': name, 'budget': budget, 'context_tokens': used, 'context': ctx,
                                    'question': q['question']}, ensure_ascii=False) + '\n')
    else:
        system = SYSTEM if a.domain == 'enterprise' else SYSTEM.replace('in a ServiceNow portal', 'browsing websites')
        res = out / 'responses'; res.mkdir(exist_ok=True)
        todo = [c for c in map(json.loads, (out / 'contexts.jsonl').open())
                if not (res / f"{c['id']}_{c['arm']}_{c['budget']}.json").exists()]
        if a.configs:
            todo = [c for c in todo if c['arm'] in a.configs.split(',')]
        print('todo', len(todo), flush=True)

        def one(c):
            body = {'model': 'Qwen/Qwen3.5-9B', 'temperature': 0, 'max_tokens': 2048, 'chat_template_kwargs': {'enable_thinking': False},
                    'messages': [{'role': 'system', 'content': system},
                                 {'role': 'user', 'content': f"Memory excerpts:\n\n{c['context']}\n\n---\nQuestion: {c['question']}"}]}
            req = urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
            t = time.time()
            with urllib.request.urlopen(req, timeout=1800) as r:
                resp = json.loads(r.read())
            (res / f"{c['id']}_{c['arm']}_{c['budget']}.json").write_text(json.dumps(
                {'id': c['id'], 'arm': c['arm'], 'budget': c['budget'], 'context_tokens': c['context_tokens'], 'seconds': time.time() - t,
                 'response': resp}, ensure_ascii=False))
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(one, todo), 1):
                if i % 50 == 0:
                    print(i, flush=True)
        print('done', len(todo), flush=True)


if __name__ == '__main__':
    main()
