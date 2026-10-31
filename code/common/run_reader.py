"""Feasibility comparison, GPU stage: one reader call per (question, arm, budget).
Same reader, same prompt, thinking off, greedy. Resumable: finished cases are skipped."""
import argparse, json, os, time, urllib.request, concurrent.futures as cf
from pathlib import Path

OUT = Path(os.environ.get('FEAS_OUT', 'workspace/lazy_materialization_v1/feasibility'))
SYSTEM = ("You answer questions about an AI agent's earlier work sessions in a ServiceNow portal. "
          "You are given memory excerpts retrieved from those sessions. Use only the excerpts. "
          "If the excerpts are insufficient, give your best answer anyway.")
if os.environ.get('FEAS_DOMAIN') == 'web':
    # Same instruction with the domain wording corrected for web-browsing trajectories.
    SYSTEM = SYSTEM.replace('in a ServiceNow portal', 'browsing websites')


def call(port, ctx, question, max_tokens):
    body = {'model': 'Qwen/Qwen3.5-9B', 'temperature': 0, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': f'Memory excerpts:\n\n{ctx}\n\n---\nQuestion: {question}'}]}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read())
    return resp, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8122); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int); ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--tag', default='run'); ap.add_argument('--arms', help='comma list; default all')
    a = ap.parse_args()
    qs = {}
    for line in Path('workspace/questions.jsonl').read_text().splitlines():
        r = json.loads(line); qs[r['id']] = r['question']
    res_dir = OUT / 'responses'; res_dir.mkdir(exist_ok=True)
    todo = []
    for line in (OUT / 'contexts.jsonl').open():
        c = json.loads(line)
        if a.arms and c['arm'] not in a.arms.split(','):
            continue
        p = res_dir / f"{c['id']}_{c['arm']}_{c['budget']}.json"
        if not p.exists():
            todo.append((c, p))
    todo = todo[:a.limit] if a.limit else todo
    print('todo', len(todo), flush=True)
    t0 = time.time()

    def one(item):
        c, p = item
        resp, sec = call(a.port, c['context'], qs[c['id']], a.max_tokens)
        p.write_text(json.dumps({'id': c['id'], 'arm': c['arm'], 'budget': c['budget'], 'context_tokens': c['context_tokens'],
                                 'seconds': sec, 'response': resp, 'tag': a.tag}, ensure_ascii=False))
        return c['id'], c['arm'], c['budget'], sec, resp.get('usage')

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, r in enumerate(ex.map(one, todo), 1):
            print(n, round(time.time() - t0, 1), *r, flush=True)
    print('done', len(todo), round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
