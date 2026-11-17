"""Classical baselines via statsforecast. Every model returns the same columns:
unique_id, ds, q10, q50, q90 (80% central interval from the model's own intervals)."""
from __future__ import annotations

import os

# One BLAS thread per worker process. Must be set before numpy is imported; workers
# inherit it. Without this, many workers each spawn many threads and exhaust memory
# (Windows "paging file too small").
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (MSTL, AutoARIMA, AutoETS, AutoTheta, Naive,
                                  SeasonalNaive)

warnings.filterwarnings("ignore")


def _make(name: str, season: int, seasons: list[int] | None = None):
    """`season` is the main period; `seasons` lists all periods for MSTL (e.g. [24, 168])."""
    seasons = seasons or [season]
    return {
        "naive": lambda: Naive(),
        "snaive": lambda: SeasonalNaive(season_length=season),
        "ets": lambda: AutoETS(season_length=season),
        "theta": lambda: AutoTheta(season_length=season),
        # very long periods make seasonal ARIMA impractical: fall back to non-seasonal
        "arima": lambda: AutoARIMA(season_length=season if season <= 24 else 1),
        # cheaper search: approximation + small orders (full search did not finish in >45 min)
        "arima_fast": lambda: AutoARIMA(season_length=season if season <= 24 else 1,
                                        approximation=True, max_p=3, max_q=3, max_P=1, max_Q=1,
                                        max_d=1, max_D=1, nmodels=30, alias="AutoARIMA"),
        "mstl": lambda: MSTL(season_length=seasons if len(seasons) > 1 else seasons[0]),
    }[name]()


def forecast(name: str, hist: pd.DataFrame, horizon: int, freq: str, season: int,
             n_jobs: int = 4, seasons: list[int] | None = None) -> pd.DataFrame:
    m = _make(name, season, seasons)
    sf = StatsForecast(models=[m], freq=freq, n_jobs=n_jobs)
    out = sf.forecast(df=hist[["unique_id", "ds", "y"]], h=horizon, level=[80]).reset_index()
    alias = m.alias if hasattr(m, "alias") else type(m).__name__
    return pd.DataFrame({
        "unique_id": out["unique_id"], "ds": out["ds"],
        "q10": out[f"{alias}-lo-80"], "q50": out[alias], "q90": out[f"{alias}-hi-80"]})
