"""Second CPU stage: two arms that combine the cheap structure view with semantic summaries.

  hybrid : half budget structure units + half budget summary units (both BM25)
  lazy   : structure BM25 routes to states; only the event summaries touching those states
           (plus notes of the touched trajectories) are used. The already-built native
           summaries act as the cache of what on-demand materialisation would compute, so the
           lazy ingest cost is the count of distinct events/notes ever touched by the query set.
"""
import collections, json, sys
from pathlib import Path
sys.argv = ['x']
exec(open(Path(__file__).with_name('build_arms.py')).read().replace("if __name__ == '__main__':\n    main()", ''))


def fill(order, units, budget):
    picked, used = [], 0
    for i in order:
        n = ntok(units[i]['text'])
        if used + n > budget:
            continue
        picked.append(i); used += n
        if used > budget - 300:
            break
    return picked, used


def main():
    split = json.loads((OUT / 'question_split.json').read_text())
    _, _, _, rows = select_questions()
    qids = split['old_dev12'] + split['new_dev24']
    units, _ = build_units()
    S, M = units['structure'], units['summary']
    bS, bM = BM25([u['text'] for u in S]), BM25([u['text'] for u in M])
    # summary units addressed by the states they describe
    by_state, notes = collections.defaultdict(list), collections.defaultdict(list)
    for i, u in enumerate(M):
        tid, kind, _ = u['key']
        if kind.startswith('event_'):
            a, b = kind.split('_')[1:3]
            by_state[(tid, int(a))].append(i); by_state[(tid, int(b))].append(i)
        else:
            notes[tid].append(i)
    out, touched = [], {b: set() for b in BUDGETS}
    for qid in qids:
        q = clean_question(rows[qid]['question'])
        rs, rm = bS.top(q, 400), bM.top(q, 400)
        for budget in BUDGETS:
            ps, us_ = fill(rs, S, budget // 2); pm, um = fill(rm, M, budget - us_)
            ctx = '\n\n'.join([S[i]['text'] for i in sorted(ps, key=lambda i: tuple(map(str, S[i]['key'])))] +
                              [M[i]['text'] for i in sorted(pm, key=lambda i: tuple(map(str, M[i]['key'])))])
            out.append({'id': qid, 'arm': 'hybrid', 'budget': budget, 'context_tokens': us_ + um, 'context': ctx})
            order, seen, trajs = [], set(), []
            for i in rs:
                tid, st, _ = S[i]['key']
                if tid not in trajs:
                    trajs.append(tid)
                for j in by_state.get((tid, int(st)), []) + (notes[tid] if len(trajs) <= 3 else []):
                    if j not in seen:
                        seen.add(j); order.append(j)
            pl, ul = fill(order, M, budget)
            touched[budget] |= set(pl)
            ctx = '\n\n'.join(M[i]['text'] for i in sorted(pl, key=lambda i: tuple(map(str, M[i]['key']))))
            out.append({'id': qid, 'arm': 'lazy', 'budget': budget, 'context_tokens': ul, 'context': ctx})
    with (OUT / 'contexts.jsonl').open('a') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    rep = {b: {'distinct_summary_units_materialised': len(t),
               'events': sum(M[i]['key'][1].startswith('event_') for i in t),
               'notes': sum(not M[i]['key'][1].startswith('event_') for i in t),
               'of_total_units': len(M)} for b, t in touched.items()}
    (OUT / 'lazy_materialisation_cost.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
