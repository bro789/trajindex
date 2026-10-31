"""Feasibility comparison, CPU stage: build three memory representations over the
same 100-trajectory haystack and retrieve a fixed token budget per question.

  raw        : original AX text of every state, chunked
  structure  : deterministic full pre-parse of every state into a compact typed view
  summary    : AgentRunbook-R native memory already built (events + procedure/hint notes)

Retrieval is the same BM25 over each arm's own units, so the arms differ only in
what was materialised at ingest time. No model call happens here."""
import hashlib, json, math, os, re, sys, time, collections
from pathlib import Path

ROOT = Path('workspace')
OUT = Path(os.environ.get('FEAS_OUT', str(ROOT / 'lazy_materialization_v1/feasibility')))
POOL = Path(os.environ.get('FEAS_POOL', str(ROOT / 'artifacts/official_r_c2022362_e97b0c61_v1')))
STATES = Path(os.environ.get('FEAS_STATES', str(ROOT / 'source_structure_pilot_preparation_v1/payload/main12_source_preannotation/source_states_only.jsonl')))
DOMAIN = os.environ.get('FEAS_DOMAIN', 'enterprise')
SALT = 'lazy_materialization_feasibility_dev_20261021'
CHUNK_CHARS = 6000
BUDGETS = [8000, 32000]

sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(ROOT / 'models/Qwen3.5-9B'))
ntok = lambda s: len(tok(s, add_special_tokens=False)['input_ids'])


def select_questions():
    frozen = json.loads((ROOT / 'official_anchor_development_frozen.json').read_text())
    old = [x['id'] for x in frozen['selected']]
    rows = {}
    for line in (ROOT / 'questions.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['domain'] == DOMAIN and r['image'] in (None, 'None') and 'llm' not in r['eval_function']:
            rows[r['id']] = r
    # new development questions: same frozen rule, 8 per type, never from the old 12
    by_type = collections.defaultdict(list)
    for q, r in rows.items():
        if q not in old:
            by_type[r['question_type']].append(q)
    new = []
    for t in sorted(by_type):
        new += sorted(by_type[t], key=lambda q: hashlib.sha256(f'{SALT}|{q}'.encode()).hexdigest())[:8]
    confirmation = sorted(set(rows) - set(old) - set(new))
    return old, new, confirmation, rows


# ---------- structure arm: deterministic parse of one AX line ----------
LINE = re.compile(r"^(\t*)(?:\[[^\]]*\]\s*)?(\w+)(?:\s+'((?:[^'\\]|\\.)*)')?(.*)$")
GLYPH = re.compile(r'\\uf[0-9a-f]{3}|[-]')
DROP_ROLES = {'generic', 'none', 'Section', 'region', 'group', 'list', 'navigation', 'main', 'search',
              'Iframe', 'paragraph', 'banner', 'contentinfo', 'complementary', 'form', 'article', 'image'}
KEEP_FLAGS = ('required', 'readonly', 'disabled', 'checked', 'selected', 'expanded', 'invalid', 'pressed')


def structure_view(ax):
    out, seen, row = [], set(), None
    for raw in ax.split('\n'):
        m = LINE.match(raw)
        if not m:
            continue
        depth, role, name, rest = len(m.group(1)), m.group(2), GLYPH.sub('', m.group(3) or '').strip(), m.group(4)
        value = re.search(r"value='((?:[^'\\]|\\.)*)'", rest)
        if role == 'navigation':
            continue
        flags = [f for f in KEEP_FLAGS if re.search(rf'\b{f}\b', rest) and not re.search(rf"{f}=(?:False|'false')", rest)]
        if role in ('row',):
            if row:
                out.append('row: ' + ' | '.join(row))
            row = []
            continue
        if role in ('gridcell', 'cell', 'columnheader', 'rowheader') and row is not None:
            row.append(name or (value.group(1) if value else ''))
            continue
        if role in DROP_ROLES and not name:
            continue
        if role == 'StaticText' and (name in seen or not name):
            continue
        if role not in ('StaticText', 'heading', 'tab', 'textbox', 'combobox', 'checkbox', 'radio', 'LabelText',
                        'button', 'link', 'option', 'menuitem', 'RootWebArea', 'listitem', 'note', 'alert', 'dialog',
                        'searchbox', 'spinbutton', 'switch', 'tabpanel', 'treeitem', 'DescriptionList', 'term',
                        'definition', 'status', 'tooltip', 'strong', 'emphasis', 'cell', 'gridcell', 'columnheader'):
            if not name:
                continue
        if role == 'option' and 'selected' not in flags:
            continue
        if not name and not value:
            continue
        key = (role, name, value.group(1) if value else None, tuple(flags))
        if key in seen:
            continue
        seen.add(key)
        seen.add(name)
        if value:
            seen.add(value.group(1))
        s = f'{role}: {name}'
        if value:
            s += f" = {value.group(1)}"
        if flags:
            s += ' [' + ','.join(flags) + ']'
        out.append(s)
    if row:
        out.append('row: ' + ' | '.join(row))
    return '\n'.join(out)


def chunk(header, body):
    parts, cur = [], []
    size = 0
    for line in body.split('\n'):
        if size + len(line) > CHUNK_CHARS and cur:
            parts.append(header + '\n' + '\n'.join(cur)); cur, size = [], 0
        cur.append(line[:CHUNK_CHARS]); size += len(line) + 1
    if cur:
        parts.append(header + '\n' + '\n'.join(cur))
    return parts


def build_units():
    t0 = time.time(); raw_units, struct_units = [], []
    struct_seconds = 0.0
    for line in STATES.open():
        s = json.loads(line)
        header = f"[trajectory {s['trajectory_id']} state {s['state_index']}] url: {s['url']}\naction: {s.get('action')}"
        raw_units += [dict(key=(s['trajectory_id'], s['state_index'], i), text=c) for i, c in enumerate(chunk(header, s['accessibility_tree']))]
        a = time.time(); view = structure_view(s['accessibility_tree']); struct_seconds += time.time() - a
        struct_units += [dict(key=(s['trajectory_id'], s['state_index'], i), text=c) for i, c in enumerate(chunk(header, view))]
    summary_units = []
    for d in sorted(POOL.iterdir()):
        art = d / 'agentrunbook_r_artifact'
        for line in (art / 'event_pool.jsonl').open():
            e = json.loads(line)
            text = (f"[trajectory {e['trajectory_id']} {e['event_id']} states {e['pre_state_index']}->{e['post_state_index']} outcome {e['outcome']}]\n"
                    f"action: {e['local_action']}\noverview: {e['overview']}\ntransition: {e['state_transition']}\n"
                    f"description: {e['description']}\nthought before: {e['pre_thoughts']}\nthought after: {e['post_thoughts']}")
            summary_units.append(dict(key=(e['trajectory_id'], e['event_id'], 0), text=text))
        for n in ('procedure_note.json', 'hint_note.json'):
            x = json.loads((art / n).read_text())
            summary_units.append(dict(key=(x['trajectory_id'], x['note_type'], 0),
                                      text=f"[trajectory {x['trajectory_id']} {x['note_type']} note]\ngoal: {x.get('goal','')}\n{x.get('note_text') or x['content']}"))
    return {'raw': raw_units, 'structure': struct_units, 'summary': summary_units}, {'seconds_total': time.time() - t0, 'structure_parse_seconds': struct_seconds}


WORD = re.compile(r'[a-z0-9]+')


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.tf = [collections.Counter(WORD.findall(d.lower())) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]; self.avg = sum(self.len) / len(self.len)
        df = collections.Counter(w for t in self.tf for w in t)
        n = len(docs); self.idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}
        self.k1, self.b = k1, b
        self.post = collections.defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)

    def top(self, query, k):
        score = collections.Counter()
        for w in set(WORD.findall(query.lower())):
            for i in self.post.get(w, []):
                f = self.tf[i][w]
                score[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in score.most_common(k)]


def clean_question(q):
    return q.split('Mark your final answer')[0].strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old, new, conf, rows = select_questions()
    (OUT / 'question_split.json').write_text(json.dumps({'salt': SALT, 'old_dev12': old, 'new_dev24': new, 'confirmation_untouched': conf,
                                                         'rule': 'enterprise, text-only, deterministic metric; new = 8 per type by ascending sha256(salt|id) excluding old 12'}, indent=2))
    units, cost = build_units()
    stats = {}
    for arm, us in units.items():
        chars = sum(len(u['text']) for u in us)
        stats[arm] = {'units': len(us), 'chars': chars, 'tokens_est': int(chars / 3.2)}
    index_t = {}
    contexts = []
    for arm, us in units.items():
        a = time.time(); bm = BM25([u['text'] for u in us]); index_t[arm] = time.time() - a
        for qid in old + new:
            q = clean_question(rows[qid]['question']); ranked = bm.top(q, 400)
            for budget in BUDGETS:
                picked, used = [], 0
                for i in ranked:
                    n = ntok(us[i]['text'])
                    if used + n > budget:
                        continue
                    picked.append(i); used += n
                    if used > budget - 300:
                        break
                picked.sort(key=lambda i: tuple(map(str, us[i]['key'])))
                contexts.append({'id': qid, 'arm': arm, 'budget': budget, 'context_tokens': used, 'units': [list(map(str, us[i]['key'])) for i in picked],
                                 'context': '\n\n'.join(us[i]['text'] for i in picked)})
    with (OUT / 'contexts.jsonl').open('w') as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    (OUT / 'build_report.json').write_text(json.dumps({'unit_stats': stats, 'cpu_cost': cost, 'bm25_index_seconds': index_t,
                                                      'questions': len(old + new), 'budgets': BUDGETS, 'model_calls': 0}, indent=2))
    print(json.dumps({'unit_stats': stats, 'cpu_cost': cost, 'bm25_index_seconds': index_t}, indent=1))


if __name__ == '__main__':
    main()
