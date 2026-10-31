"""Build confirmation contexts for the frozen arms (see confirm/CONFIRMATION_PROTOCOL.md).
Run with FEAS_OUT=<feasibility>/confirm. raw/structure/summary are built here with the same
code path as build_arms.main; idx_raw and key0t_raw come from build_index_route.py and
build_cheap_index.py contexts2, which read confirm/question_split.json."""
import json, sys
from pathlib import Path
sys.argv = ["x"]
exec(open(Path(__file__).with_name("build_arms.py")).read().replace("if __name__ == \x27__main__\x27:\n    main()", ""))
DEV = Path(__file__).parent
import os
split = json.loads(Path(os.environ.get("CONF_SPLIT", str(DEV / "question_split.json"))).read_text())
conf = split["confirmation_untouched"]
OUT.mkdir(exist_ok=True)
(OUT / "question_split.json").write_text(json.dumps({"old_dev12": [], "new_dev24": conf, "confirmation_untouched": [],
    "note": "confirmation set: all ids listed under new_dev24 so the frozen builders pick them up"}, indent=2))
_, _, _, rows = select_questions()
units, _ = build_units()
out = []
for arm in ("raw", "structure", "summary"):
    us = units[arm]; bm = BM25([u["text"] for u in us])
    for qid in conf:
        ranked = bm.top(clean_question(rows[qid]["question"]), 400)
        for budget in BUDGETS:
            picked, used = [], 0
            for i in ranked:
                n = ntok(us[i]["text"])
                if used + n > budget:
                    continue
                picked.append(i); used += n
                if used > budget - 300:
                    break
            picked.sort(key=lambda i: tuple(map(str, us[i]["key"])))
            out.append({"id": qid, "arm": arm, "budget": budget, "context_tokens": used,
                        "context": "\n\n".join(us[i]["text"] for i in picked)})
with (OUT / "contexts.jsonl").open("w") as f:
    for c in out:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print("confirmation questions", len(conf), "contexts", len(out))
