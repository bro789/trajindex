"""Analysis frozen in WEB_PROTOCOL_ADDENDUM.md: web alone, and enterprise-confirmation + web
pooled with a bootstrap stratified by domain (10,000 resamples, seed 0, margin -5 pp)."""
import json, random, sys
from pathlib import Path
ENT, WEB = Path(sys.argv[1]), Path(sys.argv[2])  # the two scores.json files


def table(p):
    S = json.loads(p.read_text())
    return {(r['id'], r['arm'], r['budget']): bool(r['correct']) for r in S['rows']}, sorted({r['id'] for r in S['rows']})


def compare(strata, a, b, B):
    xs = [[(C[(q, a, B)], C[(q, b, B)]) for q in ids] for C, ids in strata]
    flat = [p for s in xs for p in s]; n = len(flat)
    diff = sum(x - y for x, y in flat) / n
    rng = random.Random(0); boots = []
    for _ in range(10000):
        tot = 0
        for s in xs:  # resample questions within each domain
            for _ in range(len(s)):
                x, y = s[rng.randrange(len(s))]; tot += x - y
        boots.append(tot / n)
    boots.sort()
    return {'a': a, 'b': b, 'budget': B, 'n': n, 'acc_a': sum(x for x, _ in flat) / n, 'acc_b': sum(y for _, y in flat) / n,
            'diff_pp': 100 * diff, 'ci95_pp': [100 * boots[249], 100 * boots[9749]],
            'a_only': sum(x and not y for x, y in flat), 'b_only': sum(y and not x for x, y in flat)}


ent, web = table(ENT), table(WEB)
pairs = [('key0t_raw', 'summary', 32000), ('key0t_raw', 'summary', 8000), ('key0t_raw', 'idx_raw', 32000),
         ('idx_raw', 'summary', 32000), ('structure', 'summary', 32000), ('raw', 'summary', 32000)]
res = {'web': [compare([web], *p) for p in pairs], 'pooled': [compare([ent, web], *p) for p in pairs]}
for k in ('web', 'pooled'):
    res[k][0]['non_inferior_at_minus5pp'] = res[k][0]['ci95_pp'][0] > -5
out = WEB.parent / 'pooled_analysis.json'; out.write_text(json.dumps(res, indent=2))
for k in ('web', 'pooled'):
    print('==', k)
    for r in res[k]:
        print(r['a'], 'vs', r['b'], r['budget'], 'n', r['n'], round(100 * r['acc_a'], 1), round(100 * r['acc_b'], 1),
              'diff', round(r['diff_pp'], 1), 'CI', [round(v, 1) for v in r['ci95_pp']], r['a_only'], ':', r['b_only'],
              ('NONINF' if r.get('non_inferior_at_minus5pp') else ''))
