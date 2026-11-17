"""Tune baselines on PRE-EVALUATION data only (protocol fairness rule).

Validation tasks use origins in 2025 (FRED: vintages 2025-04..2025-12, truth from the
2025-12 vintage), so nothing from the 2026 evaluation window is touched.

Selects, per task, by validation WQL:
  * LightGBM: hyper-parameters and lag set (small grid)
  * seasonal period for ETS / Theta / ARIMA where it is ambiguous (FRED, flu)
and fits a post-hoc interval-width factor per (task, model) so validation 80% coverage ~ 0.80.
Writes results/tuned/params.json and results/tuned/validation.csv.

python -m scripts.tune_baselines [task ...]
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tuned"

FRED_KEYS = [f"2025-{m:02d}" for m in range(4, 13)]
LAG_SETS = {   # A = default; B = longer memory. All lags >= horizon, < context length
    "eia": {"A": [168, 336, 504, 672], "B": [168, 336, 504, 672, 840, 1008]},
    "wiki": {"A": [28, 35, 42, 56], "B": [28, 35, 42, 49, 56, 84, 336]},
    "flu": {"A": [4, 5, 6, 52], "B": [4, 5, 6, 7, 8, 52, 104]},
    "fred": {"A": [12, 13, 24], "B": [12, 13, 24, 36, 48]},
}
GRID = [dict(num_leaves=l, n_estimators=n, learning_rate=lr)
        for l, n, lr in [(15, 200, .05), (31, 200, .05), (63, 200, .05), (15, 400, .03),
                         (31, 400, .03), (31, 200, .1)]]
SEASON_CANDS = {"fred": [1, 12], "flu": [1, 52]}      # others are unambiguous
STAT = {"fred": ["ets", "theta", "arima_fast"], "flu": ["ets", "theta", "arima_fast"],
        "wiki": ["naive", "snaive", "ets", "theta", "mstl"],
        "eia": ["naive", "snaive", "mstl"]}
FOUND = ["chronos_bolt_small", "chronos_bolt_base", "chronos2"]


def val_task(name: str):
    from src.eval import tasks
    return {
        "eia": lambda: tasks.eia_task(first="2025-03-03", last="2025-11-24", every="28D"),
        "wiki": lambda: tasks.wiki_task(first="2025-03-03", last="2025-11-17", every="28D"),
        "flu": lambda: tasks.flu_task(first_issue=202520, last_issue=202548),
        "fred": lambda: tasks.fred_task(keys=FRED_KEYS),
    }[name]()


def width_factor(task, fc, run) -> float:
    """Factor c on interval half-widths whose validation 80% coverage is closest to 0.80."""
    s = run.score(task, fc)
    y, q10, q50, q90 = (s[k].to_numpy(float) for k in ("y_true", "q10", "q50", "q90"))
    nonneg = task.name in run.NONNEG
    best, bd = 1.0, 9
    for c in np.arange(0.4, 5.0, 0.05):
        lo, hi = q50 - c * (q50 - q10), q50 + c * (q90 - q50)
        if nonneg:
            lo, hi = np.maximum(lo, 0), np.maximum(hi, 0)
        d = abs(((y >= lo) & (y <= hi)).mean() - 0.8)
        if d < bd:
            best, bd = float(round(c, 2)), d
    return best


def main(only: list[str]) -> None:
    from src.eval import metrics, run

    OUT.mkdir(parents=True, exist_ok=True)
    params = run.tuned() or {"lgbm": {}, "season": {}, "width": {}}
    rows = []

    def evaluate(task, model, name):
        fc = run.run_baseline(task, model, tag="val")
        r = metrics.summarize(run.score(task, fc)).iloc[0]
        rows.append(dict(task=name, model=model, MASE=r.MASE, WQL=r.WQL, cov80=r.cov80))
        print(f"  {model:34s} MASE={r.MASE:.3f} WQL={r.WQL:.3f} cov80={r.cov80:.3f}", flush=True)
        return fc, r

    for name in (only or ["fred", "flu", "wiki", "eia"]):
        task = val_task(name)
        print(f"\n== VAL {name}: series-origins={task.n_series_origins()}", flush=True)
        fcs = {}

        # ---- LightGBM grid (written to a throw-away params entry per config)
        best = (9, None)
        for lag_key, cfg in itertools.product(LAG_SETS[name], GRID):
            t0 = time.time()
            tag = f"lgbm_{lag_key}_{cfg['num_leaves']}_{cfg['n_estimators']}_{cfg['learning_rate']}"
            params.setdefault("lgbm", {})[name] = {"lags": LAG_SETS[name][lag_key], "params": cfg}
            (OUT / "params.json").write_text(json.dumps(params, indent=1))
            cache = ROOT / "results" / "forecasts" / f"{name}_val_{tag}.parquet"
            try:
                if cache.exists():
                    fc = pd.read_parquet(cache)
                else:
                    fc = run.run_baseline(task, "lgbm_tuned", tag="val_tmp")
                    fc.to_parquet(cache, index=False)
                    (ROOT / "results" / "forecasts" / f"{name}_val_tmp_lgbm_tuned.parquet").unlink()
                r = metrics.summarize(run.score(task, fc)).iloc[0]
                rows.append(dict(task=name, model=tag, MASE=r.MASE, WQL=r.WQL, cov80=r.cov80))
                print(f"  {tag:34s} MASE={r.MASE:.3f} WQL={r.WQL:.3f} cov80={r.cov80:.3f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
                if r.WQL < best[0]:
                    best = (r.WQL, (lag_key, cfg), fc)
            except Exception as e:
                print(f"  {tag} FAILED {e!r}", flush=True)
                traceback.print_exc(limit=2)
        lag_key, cfg = best[1]
        params["lgbm"][name] = {"lags": LAG_SETS[name][lag_key], "params": cfg, "lag_set": lag_key}
        fcs["lgbm_tuned"] = best[2]
        print(f"  -> LightGBM best: lags {lag_key}, {cfg}", flush=True)

        # ---- statistical models; pick season where ambiguous
        params.setdefault("season", {}).setdefault(name, {})
        for m in STAT[name]:
            cands = SEASON_CANDS.get(name) if m in ("ets", "theta", "arima_fast") else None
            names = [f"{m}@{s}" for s in cands] if cands else [m]
            res = {}
            for mm in names:
                try:
                    res[mm] = evaluate(task, mm, name)
                except Exception as e:
                    print(f"  {mm} FAILED {e!r}", flush=True)
                    traceback.print_exc(limit=2)
            if res:
                pick = min(res, key=lambda k: res[k][1].WQL)
                params["season"][name][m] = pick
                fcs[pick] = res[pick][0]
        for m in FOUND:
            try:
                fcs[m] = evaluate(task, m, name)[0]
            except Exception as e:
                print(f"  {m} FAILED {e!r}", flush=True)

        # ---- width calibration for every selected model
        params.setdefault("width", {})[name] = {}
        for m, fc in fcs.items():
            params["width"][name][m] = width_factor(task, fc, run)
        print("  width factors:", params["width"][name], flush=True)
        (OUT / "params.json").write_text(json.dumps(params, indent=1))
        pd.DataFrame(rows).to_csv(OUT / "validation.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
