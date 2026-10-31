"""Run unmodified official AgentRunbook-R with pinned local model endpoints.

Only transport logging, local tokenizer paths, persistent per-trajectory artifacts,
and worker scheduling are adapted. No benchmark reference answer is loaded here.
"""
from pathlib import Path
import argparse,asyncio,base64,concurrent.futures,copy,gzip,hashlib,json,os,shutil,sys,threading,time,uuid,traceback
from types import SimpleNamespace
os.environ['OPENAI_AGENTS_DISABLE_TRACING']='1'
ROOT=Path('workspace')
sys.path.insert(0,str(ROOT/'LongMemEval-V2'))
import numpy as np
from openai import OpenAI,AsyncOpenAI
import openai._base_client as sdk_transport
httpx=getattr(sdk_transport,'httpx',None) or sdk_transport.httpx2
from transformers import AutoProcessor,AutoTokenizer
from evaluation import harness
from memory_modules import agentrunbook_r as ar

NAMESPACE='official_r_c2022362_e97b0c61_v1'
MODEL='Qwen/Qwen3.5-9B';EMBEDDER='memory-qwen3-embedding-0.6b'
MODEL_PATH=ROOT/'models/Qwen3.5-9B';EMBED_PATH=ROOT/'models/Qwen3-Embedding-0.6B'
LOCAL=threading.local();AUDIT=None

def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def externalize_images(value):
    if isinstance(value,str) and value.startswith('data:image/'):
        mime,encoded=value.split(',',1);raw=base64.b64decode(encoded)
        return {'externalized_source_image':True,'mime':mime,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if isinstance(value,list):return [externalize_images(v) for v in value]
    if isinstance(value,dict):return {k:externalize_images(v) for k,v in value.items()}
    return value

def gzsave(path,obj):
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False)

def request_hook(request):
    ident=uuid.uuid4().hex;request.extensions['memory_audit_id']=ident;request.extensions['memory_start']=time.time()
    if hasattr(LOCAL,'request_ids'):LOCAL.request_ids.append(ident)
    try:body=json.loads(request.content)
    except Exception:body={'unparsed_body_sha256':hashlib.sha256(request.content).hexdigest()}
    gzsave(AUDIT/(ident+'.request.json.gz'),{'time':time.time(),'url':str(request.url),'phase':getattr(LOCAL,'phase','build'),'query_invocation_id':getattr(LOCAL,'invocation',None),'payload':externalize_images(body)})

def response_hook(response):
    response.read();ident=response.request.extensions.get('memory_audit_id',uuid.uuid4().hex)
    try:body=response.json()
    except Exception:body={'body':response.text}
    gzsave(AUDIT/(ident+'.response.json.gz'),{'status':response.status_code,'seconds':time.time()-response.request.extensions.get('memory_start',time.time()),'payload':body})

async def async_request_hook(request):request_hook(request)
async def async_response_hook(response):
    await response.aread();response_hook(response)

class NativeEventRoutingTransport(httpx.AsyncHTTPTransport):
    """Schedule existing async event calls on an identical second local reader.

    The original builder and request payload are unchanged. Only one event may
    use the second reader at a time, leaving its other model slot to SQL. The
    response is consumed inside the lease so failures always release the slot.
    """
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.secondary_slot=asyncio.Semaphore(1)

    async def handle_async_request(self,request):
        secondary=False;original_url=str(request.url)
        route_file=ROOT/'native_second_reader_ready.json'
        if request.url.port==8122 and request.url.path.endswith('/chat/completions') and route_file.exists():
            route=json.loads(route_file.read_text())
            if route.get('ready') and not self.secondary_slot.locked():
                await self.secondary_slot.acquire();secondary=True
                request.url=request.url.copy_with(port=8123)
        try:
            ident=request.extensions.get('memory_audit_id')
            if ident:
                path=AUDIT/(ident+'.request.json.gz')
                with gzip.open(path,'rt',encoding='utf-8') as f:record=json.load(f)
                record.update({'requested_url':original_url,'url':str(request.url),'transport_route':'identical_reader_secondary_one_slot' if secondary else 'primary','routing_payload_unchanged':True})
                gzsave(path,record)
            response=await super().handle_async_request(request)
            await response.aread()
            return response
        finally:
            if secondary:self.secondary_slot.release()

class LoggedOpenAI(OpenAI):
    def __init__(self,*a,**kw):
        if not str(kw.get('base_url','')).startswith('http://127.0.0.1:'):raise RuntimeError('Only approved local endpoints are allowed')
        kw['http_client']=httpx.Client(trust_env=False,event_hooks={'request':[request_hook],'response':[response_hook]})
        super().__init__(*a,**kw)

class LoggedAsyncOpenAI(AsyncOpenAI):
    def __init__(self,*a,**kw):
        if not str(kw.get('base_url','')).startswith('http://127.0.0.1:'):raise RuntimeError('Only approved local endpoints are allowed')
        kw['http_client']=httpx.AsyncClient(trust_env=False,transport=NativeEventRoutingTransport(),event_hooks={'request':[async_request_hook],'response':[async_response_hook]})
        super().__init__(*a,**kw)

def make_memory(workspace,pool,rerank=False):
    cfg=json.loads((ROOT/'LongMemEval-V2/evaluation/memory_configs/agentrunbook_r.json').read_text())['memory_params']
    cfg.update({'workspace_dir':str(workspace),'trajectories_root_dir':str(ROOT/'data'),'trajectory_pool_root':str(pool) if pool else None})
    cfg['controller_params'].update({'model':MODEL,'base_url':'http://127.0.0.1:8122/v1','api_key_env':'MEMORY_LOCAL_API_KEY'})
    cfg['embedding_params'].update({'model':EMBEDDER,'base_url':'http://127.0.0.1:8119/v1','api_key_env':'MEMORY_LOCAL_API_KEY'})
    cfg['retrieval_params']['enable_rerank']=rerank
    memory=ar.AgentRunbookR(cfg)
    # The served alias is not a Hub repo. This injects the exact pinned local tokenizer;
    # original truncation, pooling calls, normalization and ranking remain unchanged.
    memory._embedding_tokenizer=AutoTokenizer.from_pretrained(EMBED_PATH)
    return memory

def main():
    global AUDIT
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['build','query','all'],default='all');ap.add_argument('--event-concurrency',type=int,default=4);ap.add_argument('--build-limit',type=int);ap.add_argument('--build-ids',help='Optional existing haystack trajectory IDs for staged native construction only; queries still require the entire pool');ap.add_argument('--arms',default='original,rerank');a=ap.parse_args()
    out=ROOT/'runs'/NAMESPACE;out.mkdir(exist_ok=True,parents=True);AUDIT=out/'api_calls';AUDIT.mkdir(exist_ok=True)
    ar.OpenAI=LoggedOpenAI;ar.AsyncOpenAI=LoggedAsyncOpenAI;ar.EVENT_ASYNC_MAX_CONCURRENCY=a.event_concurrency
    os.environ['MEMORY_LOCAL_API_KEY']='EMPTY'
    identity=json.loads((ROOT/'official_source_identity.json').read_text())
    if identity['git_commit']!='2cc8c540bdb87fe6761629b585e727e1c4704520' or identity['git_diff']:raise RuntimeError('Official source identity is not the pinned clean checkout')
    for row in identity['tracked_files']:
        path=ROOT/'LongMemEval-V2'/row['name']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Official source changed: '+row['name'])
    server=json.loads((ROOT/'reader9b_launch.json').read_text())
    smoke=json.loads((ROOT/'reader9b_api_smoke.json').read_text())
    if smoke.get('exit_status')!=0:raise RuntimeError('Official GPU vision endpoint has not passed actual-image smoke')
    frozen=json.loads((ROOT/'official_anchor_development_frozen.json').read_text());questions=frozen['selected']
    hay=json.loads((ROOT/'lme_v2_small.json').read_text());poolids=hay[questions[0]['id']]
    assert all(hay[q['id']]==poolids for q in questions),'Shared-pool assumption must be checked'
    build_poolids=poolids
    if a.build_ids:
        subset=set(a.build_ids.split(','))
        assert subset<=set(poolids),'Construction subset must stay inside the frozen complete haystack'
        build_poolids=[tid for tid in poolids if tid in subset]
    wanted=set(poolids);trajectories={}
    with (ROOT/'trajectories.jsonl').open() as f:
        for line in f:
            t=json.loads(line)
            if t['id'] in wanted:trajectories[t['id']]=t
    assert set(trajectories)==wanted
    pool=ROOT/'artifacts'/NAMESPACE;pool.mkdir(exist_ok=True,parents=True)
    manifest={'official_source_commit':'2cc8c540bdb87fe6761629b585e727e1c4704520','model_repo':MODEL,'model_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','embedding_repo':'Qwen/Qwen3-Embedding-0.6B','embedding_revision':'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3','embedding_deviation':'Official paper uses Qwen3-Embedding-8B; this 0.6B encoder is a controlled substitution','scope':'enterprise small-tier; 12 frozen development questions; no independent confirmation','question_ids':[q['id'] for q in questions],'pool_ids':poolids,'memory_budget':200000,'reader_output_budget':20000,'event_concurrency':a.event_concurrency,'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_answers_loaded':False,'image_logging':'inline image bytes externalized by SHA256; source screenshots and artifact copies retained','new_algorithm':None}
    manifest.update({'source_verification':{'git_commit':identity['git_commit'],'tracked_files_verified':len(identity['tracked_files']),'archive_sha256':identity['archive_sha256']},'reader_server_launch':server,'reader_server_launch_sha256':hashlib.sha256((ROOT/'reader9b_launch.json').read_bytes()).hexdigest(),'reader_api_smoke':smoke,'expected_cases_per_arm':len(questions),'missing_or_error_cases':'retained in the frozen full denominator; never dropped'})
    manifest['staged_construction']={'requested_build_ids':a.build_ids,'build_limit':a.build_limit,'eligible_build_ids':build_poolids,'query_haystack_still_requires_all':len(poolids)}
    route_file=ROOT/'native_second_reader_ready.json'
    manifest['event_transport_scheduling']={'primary':'http://127.0.0.1:8122/v1','secondary':json.loads(route_file.read_text()) if route_file.exists() else None,'secondary_concurrency_limit':1,'request_payload_unchanged':True,'shared_secondary_load':'SQL capacity checks may occupy the second model slot; do not interpret mixed-load wallclock as isolated speedup'}
    manifest['async_transport_lifecycle_repair']={'observed_problem':'Official builder caches one AsyncOpenAI but calls asyncio.run for each trajectory; a two-loop HTTP GET smoke reproduced Event loop is closed on the second call','adaptation':'Keep one persistent asyncio event loop for the build queue (Python 3.10-compatible); original async event coroutine, prompts, payload, retry and parsing remain unchanged','evidence':'dual_reader_transport_get_smoke.json; persistent_loop_transport_smoke.json'}
    processor=AutoProcessor.from_pretrained(MODEL_PATH)
    image_processor=processor.image_processor.to_dict()
    save(out/'actual_image_processor.json',image_processor)
    manifest['image_processor']=image_processor
    manifest['image_processor_sha256']=hashlib.sha256((out/'actual_image_processor.json').read_bytes()).hexdigest()
    save(out/'manifest.json',manifest)
    if a.stage in ['build','all']:
        LOCAL.phase='official_native_artifact_build';builder=make_memory(out/'builder_workspace',None)
        build_async_loop=asyncio.new_event_loop()
        builder._run_async=build_async_loop.run_until_complete
        for n,tid in enumerate(build_poolids):
            if a.build_limit and n>=a.build_limit:break
            target=pool/tid
            if (target/ar.TRAJECTORY_ARTIFACT_DIRNAME/ar.TRAJECTORY_ARTIFACT_INDEX_FILENAME).exists():
                # Original loader checks fingerprints, prompts/config signature and vector rows.
                from memory_modules.trajectory_store import prepare_trajectory_insert
                prepared=prepare_trajectory_insert(trajectories[tid],trajectories_root_dir=ROOT/'data')
                builder._load_trajectory_artifact_bundle_from_pool(prepared=prepared,pooled_trajectory_dir=target)
                print(json.dumps({'cached_verified':tid}),flush=True);continue
            attempt=pool/(tid+'.attempt.'+uuid.uuid4().hex)
            started=time.time()
            try:
                info=builder.build_and_write_trajectory_pool_artifact(trajectory=trajectories[tid],trajectories_root_dir=ROOT/'data',output_dir=attempt)
                # Atomic publish only after all native outputs were written successfully.
                attempt.rename(target);save(out/('build_'+tid+'.json'),{'exit_status':0,'seconds':time.time()-started,**info})
                print(json.dumps({'built':tid,'n':n+1,'total':len(poolids),'seconds':time.time()-started}),flush=True)
            except Exception as e:
                save(out/('build_'+tid+'.error.json'),{'exit_status':1,'seconds':time.time()-started,'error':repr(e),'incomplete_artifact':str(attempt)})
                raise
        if builder._async_controller_client is not None:
            build_async_loop.run_until_complete(builder._async_controller_client.close())
        build_async_loop.close()
    if a.stage in ['query','all']:
        assert all((pool/tid/ar.TRAJECTORY_ARTIFACT_DIRNAME/ar.TRAJECTORY_ARTIFACT_INDEX_FILENAME).exists() for tid in poolids),'Full public haystack must be built before querying'
        # Inject the pinned local processor into the original context-token accounting helper.
        harness.MEMORY_CONTEXT_PROCESSOR_LOCAL.processor=processor
        reader=LoggedOpenAI(base_url='http://127.0.0.1:8122/v1',api_key='EMPTY',timeout=1200,max_retries=3)
        reader_args=SimpleNamespace(model=MODEL,base_url='http://127.0.0.1:8122/v1',timeout_seconds=1200,max_completion_tokens=20000,reasoning_effort=None,temperature=.6,top_p=.95,presence_penalty=None,top_k=20,repetition_penalty=None,reader_enable_thinking=True)
        for arm in a.arms.split(','):
            workspace=out/('workspace_'+arm)
            # Query workspaces are reconstructed from the verified persistent native pool.
            # Preserve an old workspace on restart; never overwrite native inputs.
            if workspace.exists():workspace=out/('workspace_'+arm+'_'+uuid.uuid4().hex)
            memory=make_memory(workspace,pool,rerank=arm=='rerank')
            for tid in poolids:memory.insert(trajectories[tid])
            save(out/('config_'+arm+'.json'),memory.memory_config)
            for q in questions:
                path=out/(q['id']+'_'+arm+'.json')
                if path.exists():continue
                LOCAL.phase='query_'+arm;invocation=uuid.uuid4().hex;LOCAL.invocation=invocation;LOCAL.request_ids=[];memory.set_query_context(query_invocation_id=invocation);started=time.time();stage='retrieval'
                record={'question_id':q['id'],'arm':arm,'question':q['question'],'query_invocation_id':invocation}
                try:
                    context=memory.query(q['question']);record['memory_context']=context;stage='context_token_accounting'
                    limited,original_tokens,limited_tokens=harness.truncate_memory_context(context,max_tokens=200000,question_id=invocation)
                    record.update({'retained_memory_context':limited,'original_tokens':original_tokens,'retained_tokens':limited_tokens});stage='reader'
                    messages,log_messages=harness.build_messages(system_prompt=harness.get_system_prompt('enterprise'),question_text=q['question'],image_path=None,memory_context=limited)
                    record['messages_for_log']=log_messages;request=harness.build_reader_request(reader_args,messages)
                    response=reader.chat.completions.create(**request)
                    record.update({'response':response.model_dump(mode='json'),'exit_status':0})
                except Exception as e:
                    record.update({'exit_status':1,'error':repr(e),'traceback':traceback.format_exc(),'failed_stage':stage,'denominator_policy':'Counts as unsuccessful/inconclusive infrastructure case in the complete frozen denominator; not excluded'})
                record.update({'seconds':time.time()-started,'api_request_ids':list(LOCAL.request_ids)});save(path,record)
                print(json.dumps({'queried':q['id'],'arm':arm,'exit_status':record['exit_status'],'stage':stage,'request_ids':len(LOCAL.request_ids)}),flush=True)
        expected=[out/(q['id']+'_'+arm+'.json') for arm in a.arms.split(',') for q in questions]
        errors=sum(json.loads(path.read_text()).get('exit_status')!=0 for path in expected if path.exists())
        save(out/'exit.json',{'exit_status':0 if not errors else 1,'arms':a.arms.split(','),'expected_cases':len(expected),'recorded_cases':sum(path.exists() for path in expected),'error_cases':errors,'full_denominator_retained':True})
if __name__=='__main__':main()
