"""Primary and secondary analysis exactly as in confirm/CONFIRMATION_PROTOCOL.md."""
import json, random
from pathlib import Path
D = Path(__file__).parent / "confirm"
S = json.loads((D / "scores.json").read_text())
C = {(r["id"], r["arm"], r["budget"]): bool(r["correct"]) for r in S["rows"]}
ids = sorted({r["id"] for r in S["rows"]})
def compare(a, b, B):
    x = [C[(q, a, B)] for q in ids]; y = [C[(q, b, B)] for q in ids]; n = len(ids)
    diff = (sum(x) - sum(y)) / n; rng = random.Random(0); boots = []
    for _ in range(10000):
        s = [rng.randrange(n) for _ in range(n)]
        boots.append((sum(x[i] for i in s) - sum(y[i] for i in s)) / n)
    boots.sort()
    return {"a": a, "b": b, "budget": B, "n": n, "acc_a": sum(x) / n, "acc_b": sum(y) / n, "diff_pp": 100 * diff,
            "ci95_pp": [100 * boots[249], 100 * boots[9749]],
            "a_only": sum(i and not j for i, j in zip(x, y)), "b_only": sum(j and not i for i, j in zip(x, y))}
res = {"primary": compare("key0t_raw", "summary", 32000)}
res["primary"]["non_inferior_at_minus5pp"] = res["primary"]["ci95_pp"][0] > -5
res["secondary"] = [compare("key0t_raw", "summary", 8000), compare("key0t_raw", "idx_raw", 32000), compare("key0t_raw", "idx_raw", 8000),
                    compare("idx_raw", "summary", 32000), compare("structure", "summary", 32000), compare("raw", "summary", 32000)]
res["arm_summary"] = S["summary"]
(D / "confirm_analysis.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res["primary"], indent=1))
for r in res["secondary"]:
    print(r["a"], "vs", r["b"], r["budget"], round(100 * r["acc_a"], 1), round(100 * r["acc_b"], 1), "diff", round(r["diff_pp"], 1), "CI", [round(v, 1) for v in r["ci95_pp"]], r["a_only"], ":", r["b_only"])
for a, v in S["summary"].items():
    for b, x in v.items():
        print(a, b, x["correct"], "/", x["n"], x["by_type"], "len", x["length"], "ptok", x["mean_prompt_tokens"])
