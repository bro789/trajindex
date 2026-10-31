"""Everything around the judge batches for the 128 LLM-judged abstention questions.

  python judge_pipeline.py fetch            # pull blinded items, the sealed mapping and earlier scores from machine 4
  python judge_pipeline.py pilot            # Sonnet vs Opus on the same pilot items: agreement, cost, disagreements
  python judge_pipeline.py analyze --tag T  # unblind judgements_T.jsonl and run the pooled 386-question analysis

Batch submission and collection are in judge_batch.py. Prices are the Batches API rates
(50% of standard): Sonnet 5.5 $1/$5, Opus 5.5 $2/$10 per million input/output tokens."""
import argparse, json, random, subprocess
from pathlib import Path

HERE = Path(__file__).parent; DATA = HERE / 'data'
HOST = 'user@server'; R = 'workspace/lazy_materialization_v1'
FILES = {f'{R}/judged_v1/judge_items.jsonl': 'judge_items.jsonl',
         f'{R}/judged_v1/judge_mapping_SEALED.json': 'judge_mapping_SEALED.json',
         f'{R}/feasibility_web/ent_confirm/scores.json': 'det_summary_enterprise.json',
         f'{R}/feasibility_web/confirm/scores.json': 'det_summary_web.json',
         f'{R}/synth_v1/real_enterprise/scores.json': 'det_zc_enterprise.json',
         f'{R}/synth_v1/real_web/scores.json': 'det_zc_web.json',
         f'{R}/improve_v1/ent_question_split.json': 'ent_confirm_split.json',
         f'{R}/feasibility_web/web_question_split.json': 'web_confirm_split.json'}
PRICE = {'claude-sonnet-5-5': (1.0, 5.0), 'claude-opus-5-5': (2.0, 10.0)}
ARMS = {'summary': 'summary', 'zc': 'phrase+nextdelta'}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding='utf-8')]


def fetch(_):
    DATA.mkdir(exist_ok=True)
    for src, dst in FILES.items():
        subprocess.run(['scp', '-q', f'{HOST}:{src}', str(DATA / dst)], check=True)
    items = read_jsonl(DATA / 'judge_items.jsonl')
    sealed = json.loads((DATA / 'judge_mapping_SEALED.json').read_text(encoding='utf-8'))
    (HERE / 'judge_items.jsonl').write_text((DATA / 'judge_items.jsonl').read_text(encoding='utf-8'), encoding='utf-8')
    print('items', len(items), 'mapped', len(sealed['mapping']), 'auto-zero empty', len(sealed['auto_zero_empty']))


def cost(rows, model):
    i = sum(r.get('usage', {}).get('input', 0) for r in rows); o = sum(r.get('usage', {}).get('output', 0) for r in rows)
    pi, po = PRICE[model]
    return i, o, i / 1e6 * pi + o / 1e6 * po


def pilot(_):
    s = {r['custom_id']: r for r in read_jsonl(HERE / 'judgements_pilot_sonnet.jsonl')}
    o = {r['custom_id']: r for r in read_jsonl(HERE / 'judgements_pilot_opus.jsonl')}
    both = [k for k in s if k in o and s[k].get('label') is not None and o[k].get('label') is not None]
    agree = sum(s[k]['label'] == o[k]['label'] for k in both)
    print(f'pilot items {len(s)}; both labelled {len(both)}; agreement {agree}/{len(both)}')
    print('label=1 rate  sonnet', sum(s[k]['label'] for k in both), ' opus', sum(o[k]['label'] for k in both))
    for name, rows, m in (('sonnet', s.values(), 'claude-sonnet-5-5'), ('opus', o.values(), 'claude-opus-5-5')):
        i, out, usd = cost(list(rows), m); n = len(s)
        print(f'{name}: input {i} output {out} -> ${usd:.3f} for {n}; projected for 512: ${usd / n * 512:.2f}')
    items = {it['custom_id']: it for it in read_jsonl(HERE / 'judge_items.jsonl')} if (HERE / 'judge_items.jsonl').exists() else {}
    for k in [k for k in both if s[k]['label'] != o[k]['label']][:5]:
        it = items.get(k, {})
        print('\n--- disagreement', k, 'sonnet', s[k]['label'], 'opus', o[k]['label'])
        print('Q:', it.get('question', '')[:300]); print('REF:', it.get('reference', '')[:200]); print('FINAL:', it.get('final_answer', '')[:300])
        print('sonnet reason:', s[k].get('reason', '')[:200]); print('opus reason:', o[k].get('reason', '')[:200])


def stratified(strata, a, b, B):
    xs = [[(C[(q, a, B)], C[(q, b, B)]) for q in ids] for C, ids in strata]; flat = [p for s in xs for p in s]; n = len(flat)
    rng = random.Random(0); boots = []
    for _ in range(10000):
        tot = 0
        for s in xs:
            for _ in range(len(s)):
                x, y = s[rng.randrange(len(s))]; tot += x - y
        boots.append(tot / n)
    boots.sort()
    return {'a': a, 'b': b, 'budget': B, 'n': n, 'acc_a': sum(x for x, _ in flat) / n, 'acc_b': sum(y for _, y in flat) / n,
            'diff_pp': 100 * sum(x - y for x, y in flat) / n, 'ci95_pp': [100 * boots[249], 100 * boots[9749]],
            'a_only': sum(x and not y for x, y in flat), 'b_only': sum(y and not x for x, y in flat)}


def analyze(a):
    sealed = json.loads((DATA / 'judge_mapping_SEALED.json').read_text(encoding='utf-8'))
    mapping, judged = sealed['mapping'], {r['custom_id']: r for r in read_jsonl(HERE / f'judgements_{a.tag}.jsonl')}
    refused_q = {mapping[k]['id'] for k, r in judged.items() if r.get('stop_reason') == 'refusal'}
    if a.rejudge:  # items the main judge refused are labelled by the re-judge batch instead
        for k, r in ((r['custom_id'], r) for r in read_jsonl(HERE / f'judgements_{a.rejudge}.jsonl')):
            if judged.get(k, {}).get('stop_reason') == 'refusal':
                judged[k] = {**r, 'rejudged': True}
    missing = [k for k in mapping if k not in judged and k not in sealed['auto_zero_empty']]
    unlabelled = [k for k, r in judged.items() if r.get('label') is None]
    J = {}
    for k, m in mapping.items():
        label = 0 if k in sealed['auto_zero_empty'] else (judged.get(k) or {}).get('label')
        J[(m['id'], m['arm'], m['budget'])] = bool(label)  # unlabelled (refusal/parse failure) counts as wrong
    jdom = {d: sorted({m['id'] for m in mapping.values() if m['domain'] == d}) for d in ('enterprise', 'web')}
    need = [(q, arm, B) for ids in jdom.values() for q in ids for arm in ARMS.values() for B in (32000, 8000)]
    absent = [k for k in need if k not in J]
    if absent or len(jdom['enterprise']) != 56 or len(jdom['web']) != 72:
        raise SystemExit(f"incomplete judged set: enterprise {len(jdom['enterprise'])}/56, web {len(jdom['web'])}/72, "
                         f"missing (question, arm, budget) {len(absent)}, e.g. {absent[:3]}")
    D = {}
    for f in ('det_summary_enterprise.json', 'det_summary_web.json', 'det_zc_enterprise.json', 'det_zc_web.json'):
        for r in json.loads((DATA / f).read_text(encoding='utf-8'))['rows']:
            D[(r['id'], r['arm'], r['budget'])] = bool(r['correct'])
    ent = json.loads((DATA / 'ent_confirm_split.json').read_text(encoding='utf-8'))['confirmation_untouched']
    web = json.loads((DATA / 'web_confirm_split.json').read_text(encoding='utf-8'))['confirmation_untouched']
    C = {**D, **J}
    za, sa = ARMS['zc'], ARMS['summary']
    sets = {'judged128': [(C, jdom['enterprise']), (C, jdom['web'])],
            'deterministic258': [(C, ent), (C, web)],
            'pooled386': [(C, ent), (C, web), (C, jdom['enterprise']), (C, jdom['web'])],
            # sensitivity: drop every judged question for which the main judge refused any of its four items
            'pooled_excl_refused': [(C, ent), (C, web)] + [(C, [q for q in jdom[d] if q not in refused_q]) for d in ('enterprise', 'web')]}
    res = {'judge_tag': a.tag, 'rejudge_tag': a.rejudge, 'missing_judgements': len(missing), 'unlabelled_counted_wrong': len(unlabelled),
           'questions_with_main_judge_refusal': len(refused_q), 'rejudged_items': sum(1 for r in judged.values() if r.get('rejudged'))}
    for name, strata in sets.items():
        pairs = [(za, sa)] + ([(za, 'bm25+state')] if name == 'deterministic258' else [])  # judged set has no bm25+state arm
        res[name] = [stratified(strata, x, y, B) for x, y in pairs for B in (32000, 8000)]
    (HERE / f'analysis_{a.tag}{"+" + a.rejudge if a.rejudge else ""}.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('missing', len(missing), 'unlabelled (counted wrong)', len(unlabelled), 'rejudged', res['rejudged_items'],
          'questions with a main-judge refusal', len(refused_q))
    for name in sets:
        print('==', name)
        for r in res[name]:
            print(r['a'], 'vs', r['b'], r['budget'], 'n', r['n'], round(100 * r['acc_a'], 1), round(100 * r['acc_b'], 1),
                  'diff', round(r['diff_pp'], 1), 'CI', [round(v, 1) for v in r['ci95_pp']], r['a_only'], ':', r['b_only'],
                  'NON-INFERIOR' if r['b'] == sa and r['ci95_pp'][0] > -5 else '')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['fetch', 'pilot', 'analyze']); ap.add_argument('--tag')
    ap.add_argument('--rejudge', help='tag of a batch that re-judged the items the main judge refused')
    a = ap.parse_args(); {'fetch': fetch, 'pilot': pilot, 'analyze': analyze}[a.stage](a)

