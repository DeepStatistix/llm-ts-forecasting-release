"""Forecast accuracy metrics. All take long frames with columns:
unique_id, origin, step, y_true, q10, q50, q90, scale
where `scale` is the in-sample MAE of the (seasonal) naive forecast for that series/origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mase(df: pd.DataFrame) -> float:
    e = (df.y_true - df.q50).abs() / df.scale
    return float(e.mean())


def smape(df: pd.DataFrame) -> float:
    den = (df.y_true.abs() + df.q50.abs())
    ok = den > 0
    return float((200 * (df.y_true - df.q50).abs()[ok] / den[ok]).mean())


def _pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def wql(df: pd.DataFrame) -> float:
    """Scaled quantile loss averaged over tau in {0.1, 0.5, 0.9} (a coarse CRPS proxy)."""
    loss = sum(_pinball(df.y_true, df[c], t) for c, t in (("q10", .1), ("q50", .5), ("q90", .9))) / 3
    return float((loss / df.scale).mean())


def coverage80(df: pd.DataFrame) -> float:
    return float(((df.y_true >= df.q10) & (df.y_true <= df.q90)).mean())


def _per_series(g: pd.DataFrame) -> tuple[float, float]:
    """Median across series of each series' mean MASE / WQL (robust to a few bad series)."""
    a = g.groupby("unique_id").apply(mase, include_groups=False)
    b = g.groupby("unique_id").apply(wql, include_groups=False)
    return float(a.median()), float(b.median())


def summarize(df: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    """Means (headline) plus per-series medians (robust). sMAPE is not meaningful for
    series centred near zero (FRED transformed series) -- ignore it there."""
    rows = []
    for key, g in df.groupby(list(by)):
        mm, wm = _per_series(g)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "MASE": mase(g), "MASE_med": mm, "WQL": wql(g), "WQL_med": wm,
                     "sMAPE": smape(g), "cov80": coverage80(g), "n": len(g)})
    return pd.DataFrame(rows)
