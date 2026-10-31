"""Zero-cost index for agent memory, built from recorded trajectories only (CPU, no model call).

An event is a pair of consecutive states (i, i+1). Its index unit (key0t) is the deterministic
key of build_cheap_index.key0 plus the agent's logged thoughts of both states. Variants change
how units are scored (route) and how the routed states are read into the budget (read).

Route options
  bm25        BM25 over the concatenated unit text (the key0t_raw baseline)
  bm25f       BM25F: fields scored separately and summed with fixed field weights
  phrase      bm25 plus a bigram BM25 over the same text (exact multi-word labels)
Read options
  state       raw AX chunks of routed states in route order (baseline)
  next        like state, but each routed event contributes its post state first
  nextdelta   the state order of next, rendered as in delta
  delta       per trajectory, the first chosen state in full and each later chosen state as
              the lines added relative to the previous chosen state of that trajectory
"""
import collections, functools, json, math, re, sys
from pathlib import Path

ROOT = Path('workspace')
# Reuse the exact structure view, chunking and token counter of the earlier arms.
_ns = {'__name__': 'build_arms'}
exec(open(ROOT / 'lazy_materialization_v1/feasibility_web/build_arms.py').read(), _ns)
structure_view, chunk = _ns['structure_view'], _ns['chunk']
ntok = functools.lru_cache(maxsize=None)(_ns['ntok'])  # chunks are counted many times across variants
TITLE = re.compile(r"^\t*(?:\[[^\]]*\]\s*)?RootWebArea '((?:[^'\\]|\\.)*)'", re.M)
WORD = re.compile(r'[a-z0-9]+')
FIELDS = ('goal', 'action', 'page', 'diff', 'thought')
FIELD_W = {'goal': 0.5, 'action': 1.0, 'page': 1.5, 'diff': 1.0, 'thought': 1.0}


def annotate(action, ax):
    """Append the AX line of the acted-on element, as the native event records do."""
    m = re.match(r"\s*\w+\('([^']+)'", action)
    if m:
        for line in ax.split('\n'):
            if f'[{m.group(1)}]' in line:
                return f"{action}  # {line.strip()}"
    return action


def load_trajectories(ids):
    ids, out = set(ids), {}
    for line in (ROOT / 'trajectories.jsonl').open():
        t = json.loads(line)
        if t['id'] in ids:
            out[t['id']] = t
    return out


class Memory:
    """All CPU artefacts for one haystack: states, raw chunks, events and their fields."""

    def __init__(self, trajectories):
        self.views, self.titles, self.raw, self.events = {}, {}, [], []
        self.chunks = collections.defaultdict(list)
        for tid in sorted(trajectories):
            t = trajectories[tid]
            sts = sorted(t['states'], key=lambda s: int(s['state_index']))
            ax_of = {int(s['state_index']): s.get('accessibility_tree') or '' for s in sts}
            for s in sts:
                k = (tid, int(s['state_index'])); ax = ax_of[k[1]]
                self.views[k] = structure_view(ax).split('\n')
                self.titles[k] = ' / '.join(list(dict.fromkeys(TITLE.findall(ax)))[:2])
                header = f"[trajectory {tid} state {k[1]}] url: {s.get('url')}\naction: {s.get('action')}"
                for c in chunk(header, ax):
                    self.chunks[k].append(len(self.raw)); self.raw.append({'key': k, 'text': c, 'lines': c.split('\n')})
            for a, b in zip(sts, sts[1:]):
                ka, kb = (tid, int(a['state_index'])), (tid, int(b['state_index']))
                before = set(self.views[ka])
                diff = [x[:120] for x in self.views[kb] if x not in before][:40]
                self.events.append({'trajectory_id': tid, 'pre': ka[1], 'post': kb[1],
                                    'goal': (t.get('goal') or '')[:300], 'action': annotate(b.get('action') or '', ax_of[ka[1]])[:200],
                                    'page': f"{self.titles[ka]} -> {self.titles[kb]}", 'diff': diff,
                                    'thought': f"thought before: {a.get('thought') or ''}\nthought after: {b.get('thought') or ''}"})
        self.unit_text = [self.key0t(e) for e in self.events]
        self.bm = BM25([self.unit_text[i] for i in range(len(self.events))])
        self.bmF = {f: BM25([self.field(e, f) for e in self.events]) for f in FIELDS}
        self.bm2 = BM25([self.unit_text[i] for i in range(len(self.events))], bigram=True)

    @staticmethod
    def field(e, f):
        return '\n'.join(e['diff'][:15]) if f == 'diff' else e[f]

    @staticmethod
    def key0t(e):
        return (f"goal: {e['goal'][:200]}\naction: {e['action']}\npage: {e['page']}\n" + '\n'.join(e['diff'][:15]) + '\n' + e['thought'])

    def route(self, q, how):
        if how == 'bm25':
            s = self.bm.scores(q)
        elif how == 'bm25f':
            s = collections.Counter()
            for f, w in FIELD_W.items():
                for i, v in self.bmF[f].scores(q).items():
                    s[i] += w * v
        elif how == 'phrase':
            s = self.bm.scores(q)
            for i, v in self.bm2.scores(q).items():
                s[i] += v
        else:
            raise ValueError(how)
        return [i for i, _ in s.most_common(400)]

    def read(self, units, how, budget):
        states, seen = [], set()
        for i in units:
            e = self.events[i]
            pair = [(e['trajectory_id'], e['post']), (e['trajectory_id'], e['pre'])] if how in ('next', 'nextdelta') else \
                   [(e['trajectory_id'], e['pre']), (e['trajectory_id'], e['post'])]
            for k in pair:
                if k not in seen:
                    seen.add(k); states.append(k)
        if how in ('state', 'next'):
            picked, used = [], 0
            for st in states:
                for c in self.chunks.get(st, []):
                    m = ntok(self.raw[c]['text'])
                    if used + m <= budget:
                        picked.append(c); used += m
                if used > budget - 300:
                    break
            return '\n\n'.join(self.raw[c]['text'] for c in sorted(picked, key=lambda c: (self.raw[c]['key'], c))), used
        if how in ('delta', 'nextdelta'):
            chosen, used, last, prev_idx = [], 0, {}, {}
            for st in states:
                full = [l for c in self.chunks.get(st, []) for l in self.raw[c]['lines']]
                prev = last.get(st[0])
                if prev is None:
                    text = '\n'.join(full)
                else:
                    old = set(prev)
                    text = f"[trajectory {st[0]} state {st[1]}] (lines added since state {prev_idx[st[0]]})\n" + \
                           '\n'.join(l for l in full if l not in old)
                m = ntok(text)
                if used + m <= budget:
                    chosen.append((st, text)); used += m
                    last[st[0]] = full
                    prev_idx[st[0]] = st[1]
                if used > budget - 300:
                    break
            return '\n\n'.join(t for _, t in sorted(chosen)), used
        raise ValueError(how)


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75, bigram=False):
        toks = [self.tok(d, bigram) for d in docs]
        self.tf = [collections.Counter(t) for t in toks]
        self.len = [max(1, sum(t.values())) for t in self.tf]; self.avg = sum(self.len) / len(self.len)
        df = collections.Counter(w for t in self.tf for w in t); n = len(docs)
        self.idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}
        self.k1, self.b, self.bigram = k1, b, bigram
        self.post = collections.defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)

    @staticmethod
    def tok(text, bigram):
        w = WORD.findall(text.lower())
        return [a + '_' + b for a, b in zip(w, w[1:])] if bigram else w

    def scores(self, query):
        s = collections.Counter()
        for w in set(self.tok(query, self.bigram)):
            for i in self.post.get(w, []):
                f = self.tf[i][w]
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def clean_question(q):
    return q.split('Mark your final answer')[0].strip()
