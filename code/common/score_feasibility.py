"""Score feasibility responses with the official deterministic metric (boxed extraction unchanged)."""
import collections, json, os, re, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path('workspace'); OUT = Path(os.environ.get('FEAS_OUT', str(ROOT / 'lazy_materialization_v1/feasibility')))
sys.path.insert(0, str(ROOT / 'LongMemEval-V2'))
from evaluation import harness
from evaluation import qa_eval_metrics as metrics

refs = {json.loads(l)['id']: json.loads(l) for l in (ROOT / 'questions.jsonl').read_text().splitlines()}
split = json.loads((OUT / 'question_split.json').read_text())
group = {q: 'old_dev12' for q in split['old_dev12']} | {q: 'new_dev24' for q in split['new_dev24']}
ctx = {}
for line in (OUT / 'contexts.jsonl').open():
    c = json.loads(line); ctx[(c['id'], c['arm'], c['budget'])] = c['context'].lower()
rows = []
for p in sorted((OUT / 'responses').glob('*.json')):
    d = json.loads(p.read_text()); g = refs[d['id']]; spec = g['eval_function']
    ch = d['response']['choices'][0]; msg = ch['message']
    text = harness.extract_text_from_response_message(SimpleNamespace(**msg))
    parsed = metrics.extract_boxed_answer(text)
    row = {'eval_name': metrics.eval_name(spec), 'eval_function': spec, 'response_parsed_boxed': parsed, 'response_raw': text,
           'answer_gold': g['answer'], 'is_unknown': metrics.is_unknown(parsed), 'question_item': g}
    ok = bool(harness.score_prediction(row, {})[0])
    gold_in_ctx = None
    if metrics.eval_name(spec).startswith('norm_phrase'):
        phrases = [x.strip().lower() for x in re.split(r'[,;]', g['answer']) if x.strip()]
        gold_in_ctx = all(x.replace('-', ' ') in ctx[(d['id'], d['arm'], d['budget'])].replace('-', ' ') for x in phrases)
    u = d['response'].get('usage') or {}
    rows.append({'id': d['id'], 'group': group[d['id']], 'type': g['question_type'], 'arm': d['arm'], 'budget': d['budget'], 'correct': ok,
                 'finish': ch['finish_reason'], 'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens'),
                 'seconds': d['seconds'], 'gold_string_in_context': gold_in_ctx, 'parsed': parsed, 'gold': g['answer']})
summ = collections.defaultdict(dict)
for arm in ['raw', 'structure', 'summary', 'hybrid', 'lazy', 'lazy_rewrite', 'lazy_embed', 'lazy_coarse', 'idx_struct', 'idx_raw', 'key0_raw', 'key1_raw', 'key1c_raw', 'key0t_raw', 'thought_raw']:
    for b in sorted({r['budget'] for r in rows}):
        sel = [r for r in rows if r['arm'] == arm and r['budget'] == b]
        if not sel:
            continue
        gs = [r for r in sel if r['gold_string_in_context'] is not None]
        summ[arm][b] = {'correct': sum(r['correct'] for r in sel), 'n': len(sel),
                        'by_type': {t: f"{sum(r['correct'] for r in sel if r['type'] == t)}/{sum(r['type'] == t for r in sel)}" for t in sorted({r['type'] for r in sel})},
                        'by_group': {g: f"{sum(r['correct'] for r in sel if r['group'] == g)}/{sum(r['group'] == g for r in sel)}" for g in ['old_dev12', 'new_dev24']},
                        'length': sum(r['finish'] == 'length' for r in sel),
                        'gold_string_in_context': f"{sum(r['gold_string_in_context'] for r in gs)}/{len(gs)}",
                        'mean_prompt_tokens': round(sum(r['prompt_tokens'] or 0 for r in sel) / len(sel)),
                        'mean_completion_tokens': round(sum(r['completion_tokens'] or 0 for r in sel) / len(sel))}
(OUT / 'scores.json').write_text(json.dumps({'summary': summ, 'rows': rows}, indent=2, ensure_ascii=False))
print(json.dumps(summ, indent=1, ensure_ascii=False))
