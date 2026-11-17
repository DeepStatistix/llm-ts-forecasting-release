"""Time series foundation models (zero-shot). Currently: Chronos-Bolt and Chronos-2.

Every call records wall-clock seconds and peak GPU memory in a sidecar JSON next to the
cached forecasts (cost objective). Weights are cached on E: so C: does not fill up.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", r"E:\hf_cache")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = {
    "chronos_bolt_small": "amazon/chronos-bolt-small",
    "chronos_bolt_base": "amazon/chronos-bolt-base",
    "chronos2": "amazon/chronos-2",
    "timesfm25": "google/timesfm-2.5-200m-pytorch",
}
QUANTILES = [0.1, 0.5, 0.9]
_PIPE: dict = {}


def is_foundation(name: str) -> bool:
    return name in REPO


def _pipeline(name: str):
    if name not in _PIPE:
        from chronos import BaseChronosPipeline
        _PIPE[name] = BaseChronosPipeline.from_pretrained(
            REPO[name], device_map="cuda", torch_dtype=torch.float32)
    return _PIPE[name]


def _timesfm(task, max_ctx: int):
    """TimesFM 2.5 must be compiled for a fixed context/horizon, so it is built per task."""
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO["timesfm25"])
    ctx = int(np.ceil(max_ctx / 32) * 32)
    hor = int(np.ceil(task.horizon / 128) * 128)
    m.compile(timesfm.ForecastConfig(
        max_context=ctx, max_horizon=hor, normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=False, fix_quantile_crossing=True,
        per_core_batch_size=32))
    return m


def _predict_timesfm(m, ctx: list[torch.Tensor], horizon: int) -> np.ndarray:
    _, q = m.forecast(horizon=horizon, inputs=[c.numpy() for c in ctx])   # (n, h, 10): mean, q10..q90
    return q[:, :horizon, [1, 5, 9]]


def _predict_batch(pipe, ctx: list[torch.Tensor], horizon: int) -> np.ndarray:
    """Returns array (n, horizon, 3) of q10/q50/q90."""
    q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=QUANTILES)
    if isinstance(q, list):                      # Chronos-2: list of (n_var, h, q)
        q = torch.stack([x[0] for x in q])
    return q.float().cpu().numpy()


def forecast_task(name: str, task, batch_size: int = 64, max_context: int | None = None):
    """Forecast every (origin, series) in `task`. Returns (forecast_df, cost_dict)."""
    max_ctx = max(len(g) for h in task.histories.values() for _, g in h.groupby("unique_id"))
    pipe = _timesfm(task, min(max_ctx, max_context or max_ctx)) if name == "timesfm25"         else _pipeline(name)
    off = pd.tseries.frequencies.to_offset(task.freq)
    items = []   # (origin, uid, last_ds, tensor)
    for origin, h in task.histories.items():
        for uid, g in h.groupby("unique_id", sort=False):
            y = g["y"].to_numpy(np.float32)
            if max_context:
                y = y[-max_context:]
            items.append((origin, uid, g["ds"].iloc[-1], torch.from_numpy(y)))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        q = (_predict_timesfm if name == "timesfm25" else _predict_batch)(
            pipe, [c[3] for c in chunk], task.horizon)
        for (origin, uid, last, _), qq in zip(chunk, q):
            qq = np.sort(qq, axis=1)
            ds = [pd.Timestamp(last) + off * j for j in range(1, task.horizon + 1)]
            rows.append(pd.DataFrame({"unique_id": uid, "ds": ds, "origin": origin,
                                      "q10": qq[:, 0], "q50": qq[:, 1], "q90": qq[:, 2]}))
    fc = pd.concat(rows, ignore_index=True)
    cost = {"model": name, "seconds": round(time.time() - t0, 1), "n_series": len(items),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "sec_per_series": round((time.time() - t0) / max(len(items), 1), 4)}
    return fc, cost


def save_cost(path: Path, cost: dict) -> None:
    path.write_text(json.dumps(cost, indent=2))
