# Zero-cost index improvement protocol (frozen before any new reader call)

Frozen: 2026-10-24. "Zero cost" means no model call of any kind at ingest (no generative call,
no embedding): the index is built from the recorded trajectory only, on CPU. Index units are
the same `key0t` units as `key0t_raw` (deterministic key + the agent's logged thoughts of the
two states of each event). All candidates below run on CPU only, at ingest and at retrieval.

## Candidates (fixed now; no others will be tried)
All three use the same read side, within-state chunk ranking: candidate chunks are the raw AX
chunks of the first 20 routed states; rank them by reciprocal-rank fusion (k = 60) of their
state's route rank and their own BM25 rank against the question; fill the budget in that
order; present the chosen chunks in trajectory/state order. They differ only in the route.

- A `bm25ck_raw`: route by BM25 over key0t units (as `key0t_raw`).
- B `prfck_raw`: route by BM25 with RM3 pseudo-relevance feedback over key0t units: top 10
  units, 10 expansion terms (English stopwords and tokens shorter than 3 characters excluded),
  expansion weight 0.5 relative to original query terms.
- C `hierck_raw`: two-level route: score each trajectory by the sum of its top-3 key0t unit
  BM25 scores; keep the top 10 trajectories in that order; inside each, order states by their
  units' BM25 scores.
Budgets 8k and 32k; reader, prompt (per domain) and scoring unchanged.

## Selection (enterprise development 36 only)
Pick the candidate with the highest total correct over both budgets on dev36; ties go to the
simpler candidate in the order A, C, B. If no candidate beats `key0t_raw` on dev36
(30/72: 13 at 8k, 17 at 32k), stop: no held-out run.

## Held-out evaluation (single run)
Run only the selected candidate on enterprise confirmation 105 + web 153 (258). These 258
questions were already used once to evaluate `key0t_raw`; this is their second use and is
reported as such. Analysis: same paired stratified bootstrap vs `summary` (10,000 resamples,
seed 0, margin -5 pp) at 32k as primary, 8k secondary, plus vs `key0t_raw`.
