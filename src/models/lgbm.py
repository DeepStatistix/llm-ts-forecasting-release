"""Global LightGBM forecaster with quantile heads.

One model per forecast origin, trained across all series in that origin's context
window. Features: lags that are >= the forecast horizon (so no recursion, no future
leakage), calendar features, and the series level. Targets are scaled by the series'
mean absolute level so one model serves all series.
Quantiles 0.1 / 0.5 / 0.9 come from three quantile-objective models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# lags per task, all >= that task's horizon
LAGS = {"eia": [168, 336, 504, 672], "wiki": [28, 35, 42, 56],
        "flu": [4, 5, 6, 52], "fred": [12, 13, 24]}


def _cal(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    if freq == "h":
        return np.c_[idx.hour, idx.dayofweek, idx.month]
    if freq == "D":
        return np.c_[idx.dayofweek, idx.month, idx.dayofyear]
    if freq.startswith("W"):
        return np.c_[idx.isocalendar().week.to_numpy()]
    return np.c_[idx.month]


DEFAULT = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20)


def forecast(hist: pd.DataFrame, horizon: int, freq: str, lags: list[int],
             params: dict | None = None, seed: int = 0) -> pd.DataFrame:
    hp = {**DEFAULT, **(params or {})}
    assert min(lags) >= horizon, "lags must be >= horizon (direct forecasting)"
    off = pd.tseries.frequencies.to_offset(freq)
    maxlag = max(lags)
    Xtr, ytr, Xte, meta = [], [], [], []
    for uid, g in hist.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(float)
        ds = pd.DatetimeIndex(g["ds"])
        if len(y) <= maxlag + 8:
            continue
        level = float(np.mean(np.abs(y))) or 1.0
        yn = y / level
        t = np.arange(maxlag, len(y))
        lagmat = np.stack([yn[t - k] for k in lags], axis=1)
        Xtr.append(np.c_[lagmat, _cal(ds[t], freq), np.full(len(t), np.log1p(level))])
        ytr.append(yn[t])
        fut = pd.DatetimeIndex([ds[-1] + off * j for j in range(1, horizon + 1)])
        idx = np.arange(len(y), len(y) + horizon)
        lagf = np.stack([yn[idx - k] for k in lags], axis=1)
        Xte.append(np.c_[lagf, _cal(fut, freq), np.full(horizon, np.log1p(level))])
        meta.append((uid, fut, level))
    if not Xtr:
        return pd.DataFrame(columns=["unique_id", "ds", "q10", "q50", "q90"])
    X, Y = np.vstack(Xtr), np.concatenate(ytr)
    preds = {}
    for name, a in (("q10", .1), ("q50", .5), ("q90", .9)):
        m = LGBMRegressor(objective="quantile", alpha=a, n_estimators=hp["n_estimators"],
                          learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"],
                          min_child_samples=hp["min_child_samples"], subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                          n_jobs=4, random_state=seed, verbose=-1)
        m.fit(X, Y)
        preds[name] = m.predict(np.vstack(Xte))
    out, p = [], 0
    for uid, fut, level in meta:
        sl = slice(p, p + horizon)
        q = np.sort(np.c_[preds["q10"][sl], preds["q50"][sl], preds["q90"][sl]], axis=1) * level
        out.append(pd.DataFrame({"unique_id": uid, "ds": fut,
                                 "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]}))
        p += horizon
    return pd.concat(out, ignore_index=True)
