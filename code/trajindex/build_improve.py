"""Contexts for the zero-cost index candidates in IMPROVE_PROTOCOL.md. CPU only: no model call
at ingest or retrieval (the tokenizer is used only to count budget tokens).

Environment (same as the earlier builders): FEAS_OUT, FEAS_POOL, FEAS_STATES, FEAS_DOMAIN,
CONF_SPLIT. Default question set is old_dev12 + new_dev24; --conf uses confirmation_untouched.
--arms restricts which candidates are written."""
import argparse, collections, json, os, sys
from pathlib import Path
sys.argv, _argv = ['x'], sys.argv
exec(open(Path(__file__).with_name('build_arms.py')).read().replace("if __name__ == '__main__':\n    main()", ''))
_cheap = open(Path(__file__).with_name('build_cheap_index.py')).read().split("\nif __name__ == '__main__':")[0]
exec(_cheap.split("sys.argv = _argv\n", 1)[1])  # K, TITLE, key0, stage_inputs
sys.argv = _argv
RRF, ROUTE_STATES = 60, 20
STOP = set('the a an and or of to in on for with at by from is are was were be been it this that these those what which who '
           'how many much did does do you your i my me we our they their there here as into than then when where why not no yes '
           'any all some can could would should will one two three first last same other only also about after before'.split())


def scores(bm, weights):
    s = collections.Counter()
    for w, qw in weights.items():
        for i in bm.post.get(w, []):
            f = bm.tf[i][w]
            s[i] += qw * bm.idf[w] * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * bm.len[i] / bm.avg))
    return s


def qterms(q):
    return {w: 1.0 for w in set(WORD.findall(q.lower()))}


def rm3(bm, q, fb_docs=10, fb_terms=10, weight=0.5):
    base = qterms(q); first = scores(bm, base).most_common(fb_docs)
    if not first:
        return base
    top = max(s for _, s in first); exp = collections.Counter()
    for i, s in first:
        for w, f in bm.tf[i].items():
            if w not in base and w not in STOP and len(w) >= 3:
                exp[w] += (s / top) * f / bm.len[i]
    terms = exp.most_common(fb_terms); z = sum(v for _, v in terms) or 1
    out = dict(base)
    for w, v in terms:
        out[w] = weight * len(base) * v / z
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--conf', action='store_true')
    ap.add_argument('--arms', default='bm25ck_raw,prfck_raw,hierck_raw'); a = ap.parse_args()
    arms = a.arms.split(',')
    split = json.loads(Path(os.environ['CONF_SPLIT']).read_text())
    qids = split['confirmation_untouched'] if a.conf else split['old_dev12'] + split['new_dev24']
    _, _, _, qrows = select_questions()
    K.mkdir(parents=True, exist_ok=True)
    if not (K / 'event_inputs.jsonl').exists():
        stage_inputs()
    ev = [json.loads(l) for l in (K / 'event_inputs.jsonl').open()]
    thought, wanted = {}, {r['trajectory_id'] for r in ev}
    for line in (ROOT / 'trajectories.jsonl').open():
        t = json.loads(line)
        if t['id'] in wanted:
            for st in t['states']:
                thought[(t['id'], int(st['state_index']))] = st.get('thought') or ''
    texts = [key0(r) + '\n' + f"thought before: {thought.get((r['trajectory_id'], r['pre']), '')}\n"
             f"thought after: {thought.get((r['trajectory_id'], r['post']), '')}" for r in ev]
    bmK = BM25(texts)
    units, _ = build_units(); RAW = units['raw']; bmR = BM25([u['text'] for u in RAW])
    chunks = collections.defaultdict(list)
    for i, u in enumerate(RAW):
        chunks[(u['key'][0], int(u['key'][1]))].append(i)

    def to_states(unit_order):
        states, seen = [], set()
        for i in unit_order:
            for s in (ev[i]['pre'], ev[i]['post']):
                k = (ev[i]['trajectory_id'], s)
                if k not in seen:
                    seen.add(k); states.append(k)
        return states

    def chunk_ranked(states, q):
        cand = [c for st in states[:ROUTE_STATES] for c in chunks.get(st, [])]
        srank = {st: r for r, st in enumerate(states)}
        cs = scores(bmR, qterms(q))
        crank = {c: r for r, c in enumerate(sorted(cand, key=lambda c: -cs.get(c, 0.0)))}
        key = lambda c: 1 / (RRF + srank[(RAW[c]['key'][0], int(RAW[c]['key'][1]))] + 1) + 1 / (RRF + crank[c] + 1)
        return sorted(cand, key=key, reverse=True)

    out = []
    for qid in qids:
        q = clean_question(qrows[qid]['question'])
        base = scores(bmK, qterms(q))
        routes = {'bm25ck_raw': [i for i, _ in base.most_common(400)],
                  'prfck_raw': [i for i, _ in scores(bmK, rm3(bmK, q)).most_common(400)]}
        per_traj = collections.defaultdict(list)
        for i, s in base.most_common(3000):
            per_traj[ev[i]['trajectory_id']].append((s, i))
        traj_rank = sorted(per_traj, key=lambda t: -sum(s for s, _ in sorted(per_traj[t], reverse=True)[:3]))[:10]
        routes['hierck_raw'] = [i for t in traj_rank for _, i in sorted(per_traj[t], reverse=True)]
        for arm in arms:
            order = chunk_ranked(to_states(routes[arm]), q)
            for budget in BUDGETS:
                picked, used = [], 0
                for c in order:
                    m = ntok(RAW[c]['text'])
                    if used + m <= budget:
                        picked.append(c); used += m
                    if used > budget - 300:
                        break
                ctx = '\n\n'.join(RAW[c]['text'] for c in sorted(picked, key=lambda c: tuple(map(str, RAW[c]['key']))))
                out.append({'id': qid, 'arm': arm, 'budget': budget, 'context_tokens': used, 'context': ctx})
    with (OUT / 'contexts.jsonl').open('a') as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print('added', len(out))


if __name__ == '__main__':
    main()
