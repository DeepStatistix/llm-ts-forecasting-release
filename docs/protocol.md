# Protocol v2 (2026-11-15)

## Research question
Once strong, tuned classical and foundation-model baselines are required, do prompted LLMs
and LLM agents add anything for forecasting real public series? Evaluated on leak-free,
revision-aware data and scored on accuracy, calibration, cost and reproducibility together.

Positioning vs prior work: see docs/related_work_gap.md. We do NOT claim novelty for
live/post-cutoff evaluation of foundation models (LiveHouse-TS, Impermanent, TS-Arena),
contamination auditing (TSFMAudit) or macro vintages (MacroCast).

## Objectives
Primary
1. **Head-to-head across all families on identical leak-free data:** statistical, global
   LightGBM, deep (N-BEATS/NHITS, PatchTST), foundation models (Chronos, TimesFM, Moirai),
   prompted open LLMs, and an LLM agent. Baselines tuned, not defaults.
2. **Calibration:** 80% interval coverage, WQL/CRPS and interval sharpness for every
   family (point accuracy alone is insufficient).
3. **Cost and reproducibility:** wall-clock, energy/VRAM on one 8 GB GPU, and run-to-run
   variance (seeds, temperature) for stochastic models.
4. **Where each family wins:** stratify by seasonality, noise, intermittency, history
   length, horizon and revision size.
5. **Practitioner decision framework** derived from 1-4.

Secondary
6. **Text-context test** against covariate-aware baselines (FOMC/news for FRED, CDC
   narrative for flu, event notes for Wikipedia).
7. **Agent value:** agent vs best fixed pipeline: accuracy, tokens/time, failure rate.
8. **Anchor check:** same models on GIFT-Eval/M4 (possibly leaked) vs our live data, as a
   replication-level check for LLM/agent families (TSFM version already reported by
   LiveHouse-TS).

## Data (all public, post-cutoff origins from 2026-01)
EIA hourly demand (65 BAs, h=168), FRED-MD real-time vintages (19, h=12), CDC FluView
issue history (11 regions, h=4, history ends issue 202640), Wikipedia daily views in
log1p (300 pages, h=28). Overlap in kind with TS-Arena (energy) and LiveHouse-TS
(Wikipedia/macro) is acknowledged; differentiation is the model families, calibration,
cost, vintages and text context.

## Fairness rules
- Same histories, horizons, origins and metrics for every model.
- Baselines tuned on data BEFORE the evaluation origins only.
- Non-negative targets clipped at zero for all models; Wikipedia in log1p for all models.
- Headline = mean MASE/WQL, plus per-series medians; Diebold-Mariano / Model Confidence Set.
- Open models only for the main study; closed LLMs only as a labelled small illustration.
- Report negative results; all code, data snapshots and cutoffs documented.

## Status
Done: pipelines, harness, baselines (naive, snaive, ETS, Theta, ARIMA, MSTL, LightGBM).
Next: tune baselines properly, add deep models, foundation models, LLM and agent runs.
