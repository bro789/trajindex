"""Judge the blinded abstention items with Claude Sonnet 5.5 through the Message Batches API.

The judge messages and the label parser are the official LongMemEval-V2 ones
(qa_eval_metrics._build_abstention_judge_messages / _parse_llm_binary_judgement, copied
unchanged next to this script); only the judge model differs from the official gpt-5.2.
The API key is read from the environment by the SDK; it is never written anywhere.

  python judge_batch.py submit --items judge_items.jsonl [--limit 24] --tag pilot
  python judge_batch.py collect --tag pilot
"""
import argparse, json, os, sys, time
from pathlib import Path
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import qa_eval_metrics as official  # unchanged copy of LongMemEval-V2/evaluation/qa_eval_metrics.py

MODEL = os.environ.get('JUDGE_MODEL', 'claude-sonnet-5-5')
BUILDERS = {'llm_abstention_checker': official._build_abstention_judge_messages,
            'llm_gotchas_checker': official._build_gotchas_judge_messages}


def params(item):
    system, user = BUILDERS[item['eval']](question_text=item['question'], reference_answer=item['reference'],
                                          model_full_response=item['full_response'], model_final_answer=item['final_answer'])
    return MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=8000, output_config={'effort': 'medium'},
        system=system['content'], messages=[{'role': 'user', 'content': user['content']}])


def submit(a, client):
    items = [json.loads(l) for l in Path(a.items).open(encoding='utf-8')]
    if a.refused_in:  # re-judge only the items another batch's judge declined
        refused = {r['custom_id'] for r in map(json.loads, (HERE / f'judgements_{a.refused_in}.jsonl').open(encoding='utf-8'))
                   if r.get('stop_reason') == 'refusal'}
        items = [i for i in items if i['custom_id'] in refused]
    if a.limit:
        items = items[:a.limit]
    if (HERE / f'batch_{a.tag}.json').exists():
        raise SystemExit(f'batch_{a.tag}.json already exists; pick a new --tag instead of resubmitting')
    batch = client.messages.batches.create(requests=[Request(custom_id=i['custom_id'], params=params(i)) for i in items])
    (HERE / f'batch_{a.tag}.json').write_text(json.dumps({'batch_id': batch.id, 'n': len(items), 'model': MODEL, 'created': time.time()}))
    print('submitted', batch.id, len(items))


def collect(a, client):
    meta = json.loads((HERE / f'batch_{a.tag}.json').read_text(encoding='utf-8'))
    while True:
        b = client.messages.batches.retrieve(meta['batch_id'])
        if b.processing_status == 'ended':
            break
        print('status', b.processing_status, 'processing', b.request_counts.processing, flush=True)
        time.sleep(30)
    rows, usage = [], {'input': 0, 'output': 0}
    for r in client.messages.batches.results(meta['batch_id']):
        row = {'custom_id': r.custom_id, 'result': r.result.type}
        if r.result.type == 'succeeded':
            msg = r.result.message
            usage['input'] += msg.usage.input_tokens; usage['output'] += msg.usage.output_tokens
            row['stop_reason'] = msg.stop_reason
            row['usage'] = {'input': msg.usage.input_tokens, 'output': msg.usage.output_tokens}
            text = next((c.text for c in msg.content if c.type == 'text'), '')
            row['raw'] = text
            if msg.stop_reason == 'refusal':
                row['label'] = None
            else:
                try:
                    row['label'], row['reason'] = official._parse_llm_binary_judgement(text)
                except ValueError as e:
                    row['label'], row['parse_error'] = None, str(e)
        rows.append(row)
    with (HERE / f'judgements_{a.tag}.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = {k: sum(1 for r in rows if r.get('label') == k) for k in (0, 1, None)}
    print('collected', len(rows), 'labels', counts, 'usage', usage, 'batch counts', b.request_counts)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['submit', 'collect'])
    ap.add_argument('--items', default=str(HERE / 'judge_items.jsonl')); ap.add_argument('--limit', type=int); ap.add_argument('--tag', required=True)
    ap.add_argument('--refused-in', help='tag of a collected batch; submit only the items its judge refused')
    a = ap.parse_args()
    # The key is not scoped to a workspace, so every request names the workspace explicitly.
    client = anthropic.Anthropic(default_headers={'anthropic-workspace-id': os.environ['ANTHROPIC_WORKSPACE_ID']})
    submit(a, client) if a.stage == 'submit' else collect(a, client)
