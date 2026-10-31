"""Export blinded judge items for the LLM-judged abstention questions.

Each response is reduced exactly as the official harness does before calling its judge:
full text via extract_text_from_response_message, final answer via extract_boxed_answer
(falling back to the full text). Empty predictions are scored 0 without a judge call, as
require_non_empty does. Items carry a random id and no arm/budget; the mapping is separate."""
import hashlib, json, random, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path("workspace"); H = Path(__file__).parent
sys.path.insert(0, str(ROOT / "LongMemEval-V2"))
from evaluation import harness
from evaluation import qa_eval_metrics as metrics
qs = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "questions.jsonl").open()}
items, mapping, auto0 = [], {}, []
for d in ("judged_enterprise", "judged_web"):
    for p in sorted((H / d / "responses").glob("*.json")):
        x = json.loads(p.read_text()); q = qs[x["id"]]; ch = x["response"]["choices"][0]
        text = harness.extract_text_from_response_message(SimpleNamespace(**ch["message"])) or ""
        final = metrics.extract_boxed_answer(text) or text
        key = "judge-blind-20261025|%s|%s|%s" % (x["id"], x["arm"], x["budget"])
        cid = hashlib.sha256(key.encode()).hexdigest()[:24]
        mapping[cid] = {"id": x["id"], "arm": x["arm"], "budget": x["budget"], "domain": q["domain"], "type": q["question_type"], "finish": ch["finish_reason"]}
        if not text.strip() or not str(final).strip():
            auto0.append(cid); continue
        items.append({"custom_id": cid, "eval": q["eval_function"].split("|")[0], "question": q["question"].strip(),
                      "reference": q["answer"], "full_response": text, "final_answer": str(final)})
random.Random(20261025).shuffle(items)
with (H / "judge_items.jsonl").open("w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
(H / "judge_mapping_SEALED.json").write_text(json.dumps({"mapping": mapping, "auto_zero_empty": auto0}, indent=1))
print("items", len(items), "auto-zero empty", len(auto0), "mapped", len(mapping))
