# Confirmation protocol (frozen before any confirmation answer is read)

Frozen: 2026-10-22, by the memory-03 line. Confirmation answers have not been opened; only
the dev36 answers were used while designing the arms.

## Questions
All 105 ids in `question_split.json` -> `confirmation_untouched` (enterprise, text-only,
deterministic metric, same 100-trajectory haystack). No further selection, no removal.

## Arms (code frozen as of this file; budgets 8000 and 32000 context tokens)
- `raw`        : raw AX chunks, BM25 over raw (build_arms.py)
- `structure`  : deterministic structure view, BM25 over structure (build_arms.py)
- `summary`    : native AgentRunbook-R summaries, BM25 over summaries, read summaries (build_arms.py)
- `idx_raw`    : route with native summaries, read raw AX (build_index_route.py)
- `key0t_raw`  : route with deterministic key + recorded agent thoughts, read raw AX; zero model calls at ingest (build_cheap_index.py contexts2)

Reader: Qwen3.5-9B, thinking off, temperature 0, max_tokens 2048, prompt in run_reader.py.
Scoring: official deterministic metric via score_feasibility.py (score_wrap.py on machine 6).

## Primary comparison
`key0t_raw` vs `summary` at 32k. Report accuracy of both, the paired discordant counts,
and a paired bootstrap 95% CI of the difference (10,000 resamples over questions, seed 0).
Pre-set non-inferiority margin: -5 percentage points. key0t_raw is called non-inferior only
if the lower CI bound is above -5 pp.

## Secondary (reported, not used for the claim)
Same comparison at 8k; `key0t_raw` vs `idx_raw`; `raw` and `structure` as references;
by question type; length truncations; mean prompt tokens.

## Not allowed after freezing
Changing arms, prompts, budgets, BM25 parameters, chunking, or the margin; dropping
questions; rerunning with different sampling. Engineering failures (HTTP errors) may be
retried once with identical inputs and are reported.

