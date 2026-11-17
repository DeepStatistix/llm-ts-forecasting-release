# Results index (which script produces which result)

All results are in `results/`; every script is run as `python -m scripts.<name>` from the repo root with
`E:\envs\tsllm` (Moirai 2.0 runs in `E:\envs\tsllm_moirai` via `scripts/run_moirai.py`).

## Protocol (fixed before looking at 2026 results)
- Evaluation origins start 2026-01 (post-cutoff). Baselines are tuned **only** on 2025 data (`scripts/tune_baselines.py`,
  `results/tuned/`). Truth = final value; FRED/CDC use real-time vintages. FRED 2026-08 vintage dropped (corrupted).
- Metrics: MASE, WQL over tau in {0.1, 0.5, 0.9}, 80% coverage. Significance: origin-level circular block bootstrap
  (B = 10000), Holm correction. Foundation models are deterministic at inference.
- Prompted LLMs (Qwen3-8B, Mistral-7B-Instruct-v0.3, 4-bit nf4, 8 samples, T = 0.8) run on fixed seeded subsets
  (flu 396, FRED 300, Wikipedia 120, EIA 16). Gemma 4 E4B ran only partially (flu, 220 rows) and is reported separately.
- One fixed prompt (not tuned); sensitivity is measured instead (`scripts/agent_sensitivity.py`).

## Main result files
| What | File |
|---|---|
| All models on the LLM-subset rows | `results/summary/llm_compare_<task>.csv` |
| Full-task significance (best FM vs best non-FM) | `results/summary/significance_<task>.csv` (`scripts/significance.py`) |
| LLM-subset significance (categories, trivial baselines, agent) | `results/summary/significance_llm_<task>.csv` |
| Cost: params, est. GFLOPs, tokens, measured sec, VRAM | `results/summary/cost_table.csv` (`scripts/cost_table.py`) |

## Agent (light tool-augmented selector; `src/models/agent.py`)
- One constrained-decoding call per item picks a candidate (naive, seasonal naive, classical, Chronos-Bolt-small) and
  an interval width. Final protocol = `agentc*` files. First free-text runs (`agent*`) had format failures; kept for reference only.
- Same-row comparison incl. controls (best single candidate, best fixed (candidate, width), ensemble, random, oracle):
  `scripts/compare_agent.py` -> `results/summary/agent_compare_<task>_<llm>.csv`.
- Text-context test (series description + holidays; origin-time info only): `scripts/compare_ctx.py` ->
  `results/summary/ctx_compare.csv`. Prompted LLM: no significant effect on flu/FRED/Wikipedia.
- Prompt sensitivity (constrained decoding): `results/summary/agent_sensitivity_constrained_<llm>.csv`.
  Earlier free-text version (`agent_sensitivity_<llm>.csv`) is affected by parse failures: do not cite.

## Other checks
- M4 anchor (monthly/weekly/daily): `results/summary/anchor_*.csv`, `anchor_gap.csv` (`scripts/anchor_gap.py`).
  FM advantage over classical is **not** larger on M4 than on the live tasks. No contamination claim is made.
- Seed variance: deep models (training seeds 1-3) and prompted LLMs (sampling seeds 0-2) in
  `results/summary/seed_variance.csv` (`scripts/seed_variance.py`).
- Gemma 4 E4B partial flu result (earliest-origin rows, not random): `results/summary/gemma_partial.csv`.

## Known limitations
- FRED has 8 origins (low power); EIA LLM subset has 16 rows (13 origins): feasibility only.
- ARIMA is restricted-search (`arima_fast`); full-search ARIMA never completed. It is now run on FRED, flu and Wikipedia
  (Wikipedia WQL 0.418, MASE 1.155). NOTE: it is tuned LightGBM (longest lag 336 days), not ARIMA, that cannot be scored on ~330 of
  the 6256 Wikipedia origin-series pairs, so the full-task Wikipedia significance uses the 5924 pairs common to all models); EIA ARIMA not run (hourly, season 168).
- Prompted LLMs use short contexts (104-168 values), 4-bit weights, batch size 1 (8 GB laptop GPU); results are not a ceiling for LLMs.
- GFLOP figures are estimates (nominal parameter counts, ~3.2 tokens per value) and ignore memory bandwidth, which understates LLM cost.
  The agent's FLOPs are inflated because the constrained decoding re-processes the prompt for the digit step (KV-cache reuse would halve it).
- Pretraining-data contamination of the foundation models/LLMs cannot be audited from here.
- GIFT-Eval not run (M4 used as the benchmark anchor).

## Seed-variance findings
- Deep models (training seeds 1-3): WQL spread 0.5-9.5% of the mean; largest on flu (tiny training set). Small next to model gaps.
- Prompted LLMs (sampling seeds 0-2, strided rows): spread 2.5-7% on flu/FRED for both LLMs, BUT Qwen on Wikipedia (40 rows) spans
  0.517-0.993 because ONE (origin, page) item (Andy_Dick, 2026-07-20) had a loss of 14.9 in seed 1 (37% of the run's total). Median
  item loss is stable (0.42 / 0.50 / 0.42). LLM mean-loss metrics are heavy-tailed: a single wild sampled forecast can dominate.
  Medians should be reported alongside means for LLMs; the LLM-vs-best conclusions hold either way (LLMs lose by about 2x or more).
- Foundation models are deterministic at inference, so they have no seed variance.
