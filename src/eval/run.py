"""Run models over a Task and score them. Results are cached per (task, model, origin)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.eval import metrics
from src.eval.tasks import Task
from src.models import baselines, foundation, lgbm

ROOT = Path(__file__).resolve().parents[2]
NONNEG = {"eia", "wiki", "flu"}   # tasks whose target is non-negative
TUNED = ROOT / "results" / "tuned" / "params.json"   # written by scripts/tune_baselines.py


def tuned() -> dict:
    return json.loads(TUNED.read_text()) if TUNED.exists() else {}


SEASONS = {"eia": [24, 168], "wiki": [7], "flu": [52], "fred": [12]}   # for MSTL


def run_baseline(task: Task, model: str, tag: str = "full") -> pd.DataFrame:
    cache = ROOT / "results" / "forecasts" / f"{task.name}_{tag}_{model}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # One batched call over all origins: unique_id = "<origin>||<series>". This avoids
    # restarting the multiprocessing pool per origin (which dominated runtime).
    origin_of = {str(o): o for o in task.histories}
    if foundation.is_foundation(model):
        fc, cost = foundation.forecast_task(model, task)
        foundation.save_cost(cache.with_suffix(".cost.json"), cost)
    elif model == "moirai2":   # produced by scripts/run_moirai.py in its own env
        raw = ROOT / "results" / "moirai_out" / f"{task.name}.parquet"
        fc = pd.read_parquet(raw)
        key = fc["unique_id"].str.split(r"\|\|", n=1, expand=True)
        fc["origin"] = key[0].map(origin_of)
        fc["unique_id"] = key[1]
        (cache.with_suffix(".cost.json")).write_text(raw.with_suffix(".cost.json").read_text())
    elif model in ("nhits", "patchtst"):   # deep global models, trained once pre-evaluation
        from src.models import deep
        fc, cost = deep.forecast_task(model, task)
        foundation.save_cost(cache.with_suffix(".cost.json"), cost)
    elif model in ("lgbm", "lgbm_tuned"):   # global model trained separately per origin
        cfg = tuned().get("lgbm", {}).get(task.name, {}) if model == "lgbm_tuned" else {}
        lags = cfg.get("lags") or lgbm.LAGS[task.name]
        frames = []
        for o, h in task.histories.items():
            f = lgbm.forecast(h, task.horizon, task.freq, lags, params=cfg.get("params"))
            f["origin"] = o
            frames.append(f)
        fc = pd.concat(frames, ignore_index=True)
    else:   # statsforecast models
        batch = pd.concat(
            [h.assign(unique_id=f"{o}||" + h["unique_id"].astype(str)) for o, h in
             ((str(k), v) for k, v in task.histories.items())], ignore_index=True)
        base = model.split("@")[0]                      # "ets@1" = ETS with season 1
        season = int(model.split("@")[1]) if "@" in model else task.season
        fc = baselines.forecast(base, batch, task.horizon, task.freq, season,
                                seasons=SEASONS.get(task.name))
        key = fc["unique_id"].str.split(r"\|\|", n=1, expand=True)
        fc["origin"] = key[0].map(origin_of)
        fc["unique_id"] = key[1]
    fc["model"] = model
    fc = apply_floor(task, fc)
    fc.to_parquet(cache, index=False)
    return fc


def apply_floor(task: Task, fc: pd.DataFrame) -> pd.DataFrame:
    if task.name in NONNEG:   # views, demand and wILI cannot be negative
        for c in ("q10", "q50", "q90"):
            fc[c] = fc[c].clip(lower=0)
    return fc


def widen(task: Task, fc: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Post-hoc interval recalibration: scale q10/q90 distances from q50 by `factor`."""
    fc = fc.copy()
    fc["q10"] = fc["q50"] - factor * (fc["q50"] - fc["q10"])
    fc["q90"] = fc["q50"] + factor * (fc["q90"] - fc["q50"])
    return apply_floor(task, fc)


def score(task: Task, fc: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts to truth by (unique_id, origin, ds) and return the scored frame."""
    fc = fc.copy()
    fc["ds"] = pd.to_datetime(fc["ds"])
    t = task.truth.copy()
    t["ds"] = pd.to_datetime(t["ds"])
    return t.merge(fc, on=["unique_id", "origin", "ds"], how="inner")
