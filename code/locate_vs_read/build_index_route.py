"""New direction, first test: semantic summaries used only as an index.

  idx_struct : BM25 over the native summary units ranks events/notes; each hit is mapped to
               the states it describes (event pre/post state; a note maps to its whole
               trajectory, states in order); the reader sees only the structure view of those
               states, never the summary text.
  idx_raw    : same route, reader sees the raw AX chunks of those states.

If idx_struct reaches the full-summary arm, the value of the expensive summaries is routing."""
import collections, json, sys
from pathlib import Path
sys.argv = ['x']
exec(open(Path(__file__).with_name('build_arms.py')).read().replace("if __name__ == '__main__':\n    main()", ''))


def main():
    split = json.loads((OUT / 'question_split.json').read_text())
    _, _, _, rows = select_questions()
    qids = split['old_dev12'] + split['new_dev24']
    units, _ = build_units()
    M = units['summary']; bM = BM25([u['text'] for u in M])
    states_of_traj = collections.defaultdict(list)
    for u in units['structure']:
        states_of_traj[u['key'][0]].append(int(u['key'][1]))
    chunks = {arm: collections.defaultdict(list) for arm in ('structure', 'raw')}
    for arm in chunks:
        for i, u in enumerate(units[arm]):
            chunks[arm][(u['key'][0], int(u['key'][1]))].append(i)

    def route(q):
        order, seen = [], set()
        for i in bM.top(q, 400):
            tid, kind, _ = M[i]['key']
            if kind.startswith('event_'):
                a, b = kind.split('_')[1:3]; sts = [int(a), int(b)]
            else:
                sts = sorted(set(states_of_traj[tid]))
            for s in sts:
                if (tid, s) not in seen:
                    seen.add((tid, s)); order.append((tid, s))
        return order

    out = []
    for qid in qids:
        states = route(clean_question(rows[qid]['question']))
        for arm, src in (('idx_struct', 'structure'), ('idx_raw', 'raw')):
            U = units[src]
            for budget in BUDGETS:
                picked, used = [], 0
                for st in states:
                    for i in chunks[src].get(st, []):
                        n = ntok(U[i]['text'])
                        if used + n <= budget:
                            picked.append(i); used += n
                    if used > budget - 300:
                        break
                ctx = '\n\n'.join(U[i]['text'] for i in sorted(picked, key=lambda i: tuple(map(str, U[i]['key']))))
                out.append({'id': qid, 'arm': arm, 'budget': budget, 'context_tokens': used, 'context': ctx})
    with (OUT / 'contexts.jsonl').open('a') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print('added', len(out))


if __name__ == '__main__':
    main()
