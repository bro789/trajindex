# Web confirmation addendum (frozen before any web answer is read)

Frozen: 2026-10-24, before the web native pool finished building and before any web
reader call. No web question answer has been opened by this line; web questions were never
used for development.

## Questions
All web questions that are text-only with a deterministic metric (153), sharing one
100-trajectory haystack (`lme_v2_small.json`). Listed in `web_question_split.json`.

## Memory pool
`artifacts/official_r_web_full100_e97b0c61_v1`, built by `build_web_pool.py` with the same
official AgentRunbook-R builder, prompts and configuration as the enterprise pool
(Qwen3.5-9B controller, Qwen3-Embedding-0.6B). Only scheduling differs (two identical
readers, event concurrency 8). Build cost is counted from its request logs.

## Arms, reader, scoring
Identical to `confirm/CONFIRMATION_PROTOCOL.md` (raw, structure, summary, idx_raw, key0t_raw;
8k and 32k). One domain change applied to every arm: the reader system prompt says
"browsing websites" instead of "in a ServiceNow portal". The structure view, chunking,
BM25 and key0 definition are unchanged.

## Analysis
1. Web alone: the same primary comparison (key0t_raw vs summary at 32k, paired bootstrap
   95% CI, 10,000 resamples, seed 0, margin -5 pp).
2. Pooled enterprise-confirmation 105 + web 153 = 258 questions: same comparison with the
   bootstrap resampling questions within each domain (stratified). This pooled result is the
   headline claim; the per-domain results are reported alongside it.
The margin, arms and budgets may not change after this file.
