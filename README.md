# Code and data for "Do large language models and agents out-forecast time-series foundation models?"

Reproducible code for an empirical comparison of statistical, gradient-boosting, deep, foundation-model
(TSFM), prompted-LLM and LLM-selector-agent forecasters on four post-cutoff public time series
(EIA hourly electricity demand, FRED-MD real-time vintages, CDC FluView, Wikipedia page views) plus an M4 anchor.

This repository contains only what is needed to **reproduce the results**: data acquisition, models,
evaluation, significance tests and the result tables (`results/summary`, `results/tuned`).
Manuscript, figure and table-typesetting code are not part of it.

## Contents
| Path | What |
|---|---|
| `src/data` | Fetchers/parsers for the four datasets (EIA needs a free API key in `EIA_API_KEY`) |
| `src/models` | Baselines, LightGBM, deep models, foundation models, prompted LLMs, selector agent |
| `src/eval` | Task definitions (origins, horizons, vintages), metrics (SQL/WQL, MASE, coverage) |
| `scripts` | Run scripts (`python -m scripts.<name>`), comparisons, significance tests, cost table |
| `scripts/runners` | The exact PowerShell/batch queues used for the long runs (absolute Windows paths) |
| `results/summary`, `results/tuned` | Final result tables and the 2025-tuned parameters |
| `docs/protocol.md` | Pre-registered protocol |
| `RESULTS_INDEX.md` | Which script produces which result file |
| `DATA.md` | Data package, sources, licences, checksums |

## Data
Raw data and per-forecast outputs are archived with this record as `data_raw.zip` and `forecasts.zip`
(checksums in `DATA.md`). Unzip both at the repository root so that `data/raw/...` and
`results/forecasts/...` exist. With them, every comparison/significance/cost script can be rerun
without refitting any model. To rebuild from sources instead, run `python -m scripts.fetch_all`
(the live endpoints return later data than our snapshot, so numbers will differ).

## Environment
Python 3.11; `pip install -r requirements.txt` (install the CUDA build of torch first, see the file header).
Moirai 2.0 needs its own environment (pinned torch 2.4.1 / uni2ts) and is run through `scripts/run_moirai.py`.
Reference hardware: one RTX 2000 Ada laptop GPU (8 GB), 15.6 GB RAM, Windows 11. LLMs are run in 4-bit (bitsandbytes).
Hugging Face model ids are in `src/models/llm.py` and `src/models/foundation.py`.

## Reproducing the results (order)
All commands from the repository root.
1. Data: unzip the data package (or `python -m scripts.fetch_all`; `python -m scripts.check_data`).
2. Tune baselines on 2025 only: `python -m scripts.tune_baselines`
3. Baselines: `python -m scripts.run_baselines`, then `python -m scripts.run_tuned <task>`
4. Foundation and deep models: `python -m scripts.run_foundation chronos_bolt_small chronos_bolt_base chronos2 timesfm25 nhits patchtst`
   Moirai: `python -m scripts.export_task fred flu wiki eia`, then in the Moirai env `python -I scripts/run_moirai.py fred flu wiki eia`
5. Prompted LLMs: `python -m scripts.run_llm qwen3_8b flu fred wiki eia` (same for `mistral7b`)
6. Selector agent: `python -m scripts.run_agent qwen3_8b` (same for `mistral7b`)
7. Controls and checks: `scripts.run_arima_wiki`, `scripts.seeds_deep`, `scripts.run_anchor`, `scripts.agent_sensitivity <llm>`
8. Analysis: `scripts.make_table`, `scripts.compare_llm`, `scripts.compare_agent`, `scripts.compare_ctx`,
   `scripts.significance`, `scripts.significance_llm`, `scripts.cost_table`, `scripts.seed_variance`, `scripts.anchor_gap`,
   `scripts.extra_analyses` (per-step loss, reliability, sharpness, selector choices; parses `results/summary/compare_agentc_log.txt`, the saved output of `scripts.compare_agent`)

Foundation-model inference is deterministic; sampled LLM outputs depend on seeds and GPU/driver details, so
rerunning step 5 reproduces the numbers only up to sampling noise (see seed-variance results). Rerunning
steps 8 on the archived forecasts reproduces the tables exactly.

## Citation
See `CITATION.cff`. Please cite the Zenodo record (DOI added on release) and the associated article.

## Licence
Code: MIT (`LICENSE`). Datasets keep their original terms (`DATA.md`).
