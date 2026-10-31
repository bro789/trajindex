# Selection on synthetic tuning questions (frozen before any real-question reader call)

Synthetic set: 299 questions (enterprise 140, web 159) over four haystacks built from trajectories outside every
benchmark haystack (synth_questions.py). CPU proxy screened 12 route x read combinations (proxy.json, proxy_extra.json);
the top three plus the baseline were read by Qwen3.5-9B (thinking off) at 8k and 32k and scored with the official metric.

Correct over both budgets (enterprise + web, 598 answers each):
  bm25+state (baseline, = key0t_raw rebuilt from trajectories)  183
  bm25+next                                                     194
  phrase+next                                                   196
  phrase+nextdelta                                              200   <- selected (highest total)

Real-question test: run phrase+nextdelta and bm25+state once on all 294 deterministic text questions
(enterprise 141 incl. dev36, web 153). Primary comparison vs the native summary arm at 32k on the 258
confirmation questions, paired stratified bootstrap (10,000 resamples, seed 0, margin -5 pp), as before.
The 258 questions are used for the third time; no further tuning will follow this run.
