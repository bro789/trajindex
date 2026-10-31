"""Real-question test of the synthetic-tuned zero-cost index (see SELECTION.md).
Summary-arm results come from the earlier frozen confirmation runs; the 258 confirmation questions
are enterprise confirmation 105 + web 153. Bootstrap stratified by domain, 10,000 resamples, seed 0."""
import json, random
from pathlib import Path
H = Path(__file__).parent; W = H.parent / "feasibility_web"
def load(p):
    return {(r["id"], r["arm"], r["budget"]): bool(r["correct"]) for r in json.loads(Path(p).read_text())["rows"]}
C = {}
for p in (W / "ent_confirm/scores.json", W / "confirm/scores.json", H / "real_enterprise/scores.json", H / "real_web/scores.json"):
    C.update(load(p))
ent = json.loads((H.parent / "improve_v1/ent_question_split.json").read_text())["confirmation_untouched"]
web = json.loads((W / "web_question_split.json").read_text())["confirmation_untouched"]
def compare(a, b, B, strata):
    xs = [[(C[(q, a, B)], C[(q, b, B)]) for q in s] for s in strata]; flat = [p for s in xs for p in s]; n = len(flat)
    rng = random.Random(0); boots = []
    for _ in range(10000):
        tot = 0
        for s in xs:
            for _ in range(len(s)):
                x, y = s[rng.randrange(len(s))]; tot += x - y
        boots.append(tot / n)
    boots.sort()
    return {"a": a, "b": b, "budget": B, "n": n, "acc_a": sum(x for x, _ in flat) / n, "acc_b": sum(y for _, y in flat) / n,
            "diff_pp": 100 * sum(x - y for x, y in flat) / n, "ci95_pp": [100 * boots[249], 100 * boots[9749]],
            "a_only": sum(x and not y for x, y in flat), "b_only": sum(y and not x for x, y in flat)}
res = {}
for name, strata in (("pooled258", [ent, web]), ("enterprise105", [ent]), ("web153", [web])):
    res[name] = [compare(a, b, B, strata) for a, b in (("phrase+nextdelta", "summary"), ("bm25+state", "summary"),
                 ("phrase+nextdelta", "bm25+state"), ("phrase+nextdelta", "idx_raw")) for B in (32000, 8000)]
(H / "final_analysis.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print("==", k)
    for r in v:
        print(r["a"], "vs", r["b"], r["budget"], round(100 * r["acc_a"], 1), round(100 * r["acc_b"], 1), "diff", round(r["diff_pp"], 1),
              "CI", [round(x, 1) for x in r["ci95_pp"]], r["a_only"], ":", r["b_only"], "NONINF" if r["ci95_pp"][0] > -5 else "")
