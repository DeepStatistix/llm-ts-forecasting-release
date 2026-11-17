"""Restricted-search ARIMA on the Wikipedia task (it was skipped in the main runs for cost).
python -m scripts.run_arima_wiki  -> results/forecasts/wiki_full_arima_fast.parquet
(EIA ARIMA is not run: hourly series with season 168 are impractical for AutoARIMA; documented in RESULTS_INDEX.md.)"""
import time
from src.eval import metrics, run, tasks

if __name__ == "__main__":
    task = tasks.BUILDERS["wiki"]()
    t0 = time.time()
    fc = run.run_baseline(task, "arima_fast", tag="full")
    r = metrics.summarize(run.score(task, fc)).iloc[0]
    print(f"wiki arima_fast MASE={r.MASE:.3f} WQL={r.WQL:.3f} cov80={r.cov80:.3f} [{time.time() - t0:.0f}s]", flush=True)
