"""Build the native AgentRunbook-R memory pool for the web 100-trajectory haystack with the
same official builder, prompts and configuration as the enterprise pool
(official_r_c2022362_e97b0c61_v1: Qwen3.5-9B controller, Qwen3-Embedding-0.6B).

Only scheduling differs: the haystack is split into two shards, each built against its own
identical reader (8122 on GPU0, 8123 on GPU1). Every HTTP request/response is logged with
official_anchor's hooks so build cost can be counted the same way."""
import argparse, asyncio, hashlib, json, os, sys, time, uuid
from pathlib import Path
ROOT = Path('workspace')
sys.path.insert(0, str(ROOT))
import official_anchor as oa
from official_anchor import ar, httpx, AutoTokenizer, OpenAI, AsyncOpenAI

NAMESPACE = 'official_r_web_full100_e97b0c61_v1'


class LoggedOpenAI(OpenAI):
    def __init__(self, *a, **kw):
        kw['http_client'] = httpx.Client(trust_env=False, event_hooks={'request': [oa.request_hook], 'response': [oa.response_hook]})
        super().__init__(*a, **kw)


class LoggedAsyncOpenAI(AsyncOpenAI):
    def __init__(self, *a, **kw):
        kw['http_client'] = httpx.AsyncClient(trust_env=False, event_hooks={'request': [oa.async_request_hook], 'response': [oa.async_response_hook]})
        super().__init__(*a, **kw)


def make_builder(workspace, port):
    cfg = json.loads((ROOT / 'LongMemEval-V2/evaluation/memory_configs/agentrunbook_r.json').read_text())['memory_params']
    cfg.update({'workspace_dir': str(workspace), 'trajectories_root_dir': str(ROOT / 'data'), 'trajectory_pool_root': None})
    cfg['controller_params'].update({'model': oa.MODEL, 'base_url': f'http://127.0.0.1:{port}/v1', 'api_key_env': 'MEMORY_LOCAL_API_KEY'})
    cfg['embedding_params'].update({'model': oa.EMBEDDER, 'base_url': 'http://127.0.0.1:8119/v1', 'api_key_env': 'MEMORY_LOCAL_API_KEY'})
    memory = ar.AgentRunbookR(cfg)
    memory._embedding_tokenizer = AutoTokenizer.from_pretrained(oa.EMBED_PATH)
    return memory


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shard', type=int, required=True); ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--port', type=int, required=True); ap.add_argument('--event-concurrency', type=int, default=8)
    a = ap.parse_args()
    out = ROOT / 'runs' / NAMESPACE; out.mkdir(parents=True, exist_ok=True)
    oa.AUDIT = out / 'api_calls'; oa.AUDIT.mkdir(exist_ok=True)
    ar.OpenAI, ar.AsyncOpenAI, ar.EVENT_ASYNC_MAX_CONCURRENCY = LoggedOpenAI, LoggedAsyncOpenAI, a.event_concurrency
    os.environ['MEMORY_LOCAL_API_KEY'] = 'EMPTY'
    qs = [json.loads(l) for l in (ROOT / 'questions.jsonl').open()]
    web = [q for q in qs if q['domain'] == 'web' and q['image'] in (None, 'None') and 'llm' not in q['eval_function']]
    hay = json.loads((ROOT / 'lme_v2_small.json').read_text()); poolids = hay[web[0]['id']]
    assert all(hay[q['id']] == poolids for q in web), 'web questions must share one haystack'
    mine = [t for i, t in enumerate(poolids) if i % a.shards == a.shard]
    trajectories = {}
    for line in (ROOT / 'trajectories.jsonl').open():
        t = json.loads(line)
        if t['id'] in set(mine):
            trajectories[t['id']] = t
    pool = ROOT / 'artifacts' / NAMESPACE; pool.mkdir(parents=True, exist_ok=True)
    (out / f'manifest_shard{a.shard}.json').write_text(json.dumps({'namespace': NAMESPACE, 'pool_ids': poolids, 'shard_ids': mine, 'port': a.port,
        'event_concurrency': a.event_concurrency, 'embedder': oa.EMBEDDER, 'model': oa.MODEL, 'same_config_as': 'official_r_c2022362_e97b0c61_v1',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    builder = make_builder(out / f'builder_workspace_{a.shard}', a.port)
    loop = asyncio.new_event_loop(); builder._run_async = loop.run_until_complete
    for n, tid in enumerate(mine):
        target = pool / tid
        if (target / ar.TRAJECTORY_ARTIFACT_DIRNAME / ar.TRAJECTORY_ARTIFACT_INDEX_FILENAME).exists():
            print(json.dumps({'cached': tid}), flush=True); continue
        attempt = pool / (tid + '.attempt.' + uuid.uuid4().hex); started = time.time()
        try:
            info = builder.build_and_write_trajectory_pool_artifact(trajectory=trajectories[tid], trajectories_root_dir=ROOT / 'data', output_dir=attempt)
            attempt.rename(target)
            oa.save(out / f'build_{tid}.json', {'exit_status': 0, 'seconds': time.time() - started, **info})
            print(json.dumps({'built': tid, 'n': n + 1, 'of': len(mine), 'seconds': round(time.time() - started, 1)}), flush=True)
        except Exception as e:
            oa.save(out / f'build_{tid}.error.json', {'exit_status': 1, 'seconds': time.time() - started, 'error': repr(e)})
            print(json.dumps({'error': tid, 'msg': repr(e)[:300]}), flush=True)
    if builder._async_controller_client is not None:
        loop.run_until_complete(builder._async_controller_client.close())
    loop.close()
    print('shard done', flush=True)


if __name__ == '__main__':
    main()
