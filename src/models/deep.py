"""Deep global forecasters (NHITS, PatchTST) via neuralforecast, quantile (MQLoss) heads.

Protocol: ONE model per (task, architecture), trained only on data available before the
first evaluation origin (no 2026 data), then used to predict every 2026 origin without
retraining. Early stopping uses the last `val_size` steps of the training window.
Training budget is deliberately modest and fixed per task (documented here); this is
a reasonable-effort deep baseline, not a hyper-parameter search.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
import logging
for _n in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS, PatchTST

from src.data import eia_demand
from src.eval.tasks import RAW

Q = [0.1, 0.5, 0.9]
CUTOFF = "2025-12-31 23:00"

# input window, training steps, validation steps (all in the task's own frequency)
CFG = {
    "eia": dict(input_size=672, max_steps=1500, val_size=168 * 4),
    "wiki": dict(input_size=140, max_steps=1500, val_size=28 * 4),
    "flu": dict(input_size=104, max_steps=800, val_size=12),
    "fred": dict(input_size=48, max_steps=800, val_size=24),
}
NAMES = {"nhits": "nhits", "patchtst": "patchtst"}


def is_deep(name: str) -> bool:
    return name in NAMES


def train_frame(task) -> pd.DataFrame:
    """Training data strictly before the first evaluation origin, long format."""
    if task.name == "eia":
        df = pd.read_parquet(next((RAW / "eia").glob("demand_*.parquet")))
        wide = eia_demand.clean_panel(df).loc[:CUTOFF]
        wide = wide.interpolate(limit=24, limit_direction="both").dropna(axis=1, how="any")
        return wide.rename_axis("ds").reset_index().melt("ds", var_name="unique_id", value_name="y")
    if task.name == "wiki":
        w = pd.read_parquet(next((RAW / "wikipedia").glob("views_*.parquet"))).loc[:"2025-12-31"]
        parts = []
        for uid in w.columns:
            s = w[uid]
            fv = s.first_valid_index()
            if fv is None:
                continue
            s = np.log1p(s.loc[fv:].fillna(0))
            if len(s) < CFG['wiki']['input_size'] + CFG['wiki']['val_size'] + 28 + 10:   # too short to train on
                continue
            parts.append(pd.DataFrame({"unique_id": uid, "ds": s.index, "y": s.to_numpy()}))
        return pd.concat(parts, ignore_index=True)
    # flu / fred: histories at the earliest origin are the pre-evaluation data
    first = sorted(task.histories)[0]
    return task.histories[first][["unique_id", "ds", "y"]].copy()


def _model(arch: str, h: int, cfg: dict, seed: int = 1):
    common = dict(h=h, input_size=cfg["input_size"], max_steps=cfg["max_steps"],
                  loss=MQLoss(quantiles=Q), scaler_type="standard", random_seed=seed,
                  val_check_steps=100, early_stop_patience_steps=5, batch_size=32,
                  windows_batch_size=512, accelerator="gpu", devices=1,
                  enable_progress_bar=False, logger=False, enable_checkpointing=False)
    if arch == "nhits":
        return NHITS(**common)
    return PatchTST(**{**common, "patch_len": 16, "stride": 8, "windows_batch_size": 256})


def forecast_task(arch: str, task, seed: int = 1):
    """Train once on pre-evaluation data, predict every origin. Returns (forecast_df, cost)."""
    cfg = CFG[task.name]
    tr = train_frame(task)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    nf = NeuralForecast(models=[_model(arch, task.horizon, cfg, seed)], freq=task.freq)
    nf.fit(df=tr, val_size=cfg["val_size"])
    t_train = time.time() - t0

    t1 = time.time()
    origin_of = {str(o): o for o in task.histories}
    batch = pd.concat([h.assign(unique_id=f"{o}||" + h["unique_id"].astype(str))
                       for o, h in ((str(k), v) for k, v in task.histories.items())],
                      ignore_index=True)
    pred = nf.predict(df=batch[["unique_id", "ds", "y"]])
    t_pred = time.time() - t1
    name = type(nf.models[0]).__name__
    def pick(*needles):
        hit = [c for c in pred.columns if c.startswith(name) and any(n in c for n in needles)]
        if len(hit) != 1:
            raise RuntimeError(f"cannot map {needles} in columns {list(pred.columns)}")
        return hit[0]
    ql = {"q10": pick("-lo-80"), "q50": pick("-median"), "q90": pick("-hi-80")}
    out = pd.DataFrame({"unique_id": pred["unique_id"], "ds": pred["ds"],
                        "q10": pred[ql["q10"]], "q50": pred[ql["q50"]], "q90": pred[ql["q90"]]})
    out[["q10", "q50", "q90"]] = np.sort(out[["q10", "q50", "q90"]].to_numpy(), axis=1)
    key = out["unique_id"].str.split(r"\|\|", n=1, expand=True)
    out["origin"] = key[0].map(origin_of)
    out["unique_id"] = key[1]
    cost = {"model": arch, "train_seconds": round(t_train, 1), "predict_seconds": round(t_pred, 1),
            "seconds": round(t_train + t_pred, 1), "n_series": int(batch.unique_id.nunique()),
            "train_series": int(tr.unique_id.nunique()), "seed": seed,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    return out, cost
