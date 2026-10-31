"""Score tune.py responses with the official deterministic metric (boxed extraction unchanged)."""
import collections, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path('workspace'); HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'LongMemEval-V2'))
from evaluation import harness
from evaluation import qa_eval_metrics as metrics

d = HERE / sys.argv[1]  # e.g. synthetic_enterprise
refs = {}
for p in [HERE / f'synth_questions_{n}.jsonl' for n in ('enterprise', 'web')] + [ROOT / 'questions.jsonl']:
    if p.exists():
        for l in p.open():
            r = json.loads(l); refs[r['id']] = r
rows = []
for p in sorted((d / 'responses').glob('*.json')):
    x = json.loads(p.read_text()); g = refs[x['id']]; ch = x['response']['choices'][0]
    text = harness.extract_text_from_response_message(SimpleNamespace(**ch['message']))
    parsed = metrics.extract_boxed_answer(text)
    row = {'eval_name': metrics.eval_name(g['eval_function']), 'eval_function': g['eval_function'], 'response_parsed_boxed': parsed,
           'response_raw': text, 'answer_gold': g['answer'], 'is_unknown': metrics.is_unknown(parsed), 'question_item': g}
    rows.append({'id': x['id'], 'type': g['question_type'], 'arm': x['arm'], 'budget': x['budget'],
                 'correct': bool(harness.score_prediction(row, {})[0]), 'finish': ch['finish_reason']})
summ = collections.defaultdict(dict)
for arm in sorted({r['arm'] for r in rows}):
    for b in sorted({r['budget'] for r in rows}):
        sel = [r for r in rows if r['arm'] == arm and r['budget'] == b]
        summ[arm][b] = {'correct': sum(r['correct'] for r in sel), 'n': len(sel), 'length': sum(r['finish'] == 'length' for r in sel)}
(d / 'scores.json').write_text(json.dumps({'summary': summ, 'rows': rows}, indent=1))
for arm, v in summ.items():
    print(arm, {b: f"{x['correct']}/{x['n']}" for b, x in v.items()}, 'total', sum(x['correct'] for x in v.values()))
