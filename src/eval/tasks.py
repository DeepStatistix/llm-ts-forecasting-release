"""Forecasting tasks: one builder per dataset.

A Task holds, for each forecast origin, the history available *at that time* and the
realised truth. Models only ever see `histories`; scoring uses `truth` + `scales`.

Design choices (documented here):
- Evaluation origins start in 2026-01, after the documented training cutoffs of the
  open models we test (to be re-verified per model).
- FRED-MD and CDC use real-time vintages for histories. Truth is the latest available
  value (final vintage), the usual real-time evaluation convention.
- MASE scale = in-sample MAE of the seasonal-naive forecast with period `mase_m`
  (EIA 24, Wikipedia 7, FRED 1, flu 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import cdc_flu, eia_demand, fred_md

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


@dataclass
class Task:
    name: str
    freq: str
    season: int                 # seasonal period for models
    horizon: int
    mase_m: int
    # origin label -> long frame [unique_id, ds, y]
    histories: dict = field(default_factory=dict)
    # long frame [unique_id, origin, step, ds, y_true, scale]
    truth: pd.DataFrame = None

    def n_series_origins(self) -> int:
        return sum(h.unique_id.nunique() for h in self.histories.values())


def _scale(y: np.ndarray, m: int) -> float:
    y = y[~np.isnan(y)]
    if len(y) <= m:
        return np.nan
    s = np.abs(y[m:] - y[:-m]).mean()
    return s if s > 0 else np.nan


def _collect(name, freq, season, horizon, mase_m, items):
    """items: iterable of (origin, uid, hist Series(index=ds), future Series(index=ds))."""
    hist: dict = {}
    rows = []
    for origin, uid, h, fut in items:
        sc = _scale(h.to_numpy(float), mase_m)
        if not np.isfinite(sc):
            continue
        hist.setdefault(origin, []).append(
            pd.DataFrame({"unique_id": uid, "ds": h.index, "y": h.to_numpy(float)}))
        for k, (ds, y) in enumerate(fut.items(), start=1):
            if np.isfinite(y):
                rows.append((uid, origin, k, ds, float(y), sc))
    histories = {o: pd.concat(v, ignore_index=True) for o, v in hist.items()}
    truth = pd.DataFrame(rows, columns=["unique_id", "origin", "step", "ds", "y_true", "scale"])
    return Task(name, freq, season, horizon, mase_m, histories, truth)


# ------------------------------------------------------------------ EIA
def eia_task(first="2026-01-05", last="2026-10-26", every="14D", context=1344, horizon=168,
             max_series=None) -> Task:
    df = pd.read_parquet(next((RAW / "eia").glob("demand_*.parquet")))
    wide = eia_demand.clean_panel(df)
    if max_series:
        wide = wide.iloc[:, :max_series]
    items = []
    for o in pd.date_range(first, last, freq=every):
        for uid in wide.columns:
            h = wide[uid].loc[o - pd.Timedelta(hours=context): o - pd.Timedelta(hours=1)]
            if h.isna().mean() > 0.05:
                continue
            h = h.interpolate(limit_direction="both")
            fut = wide[uid].loc[o: o + pd.Timedelta(hours=horizon - 1)]
            if len(fut) < horizon:
                continue
            items.append((o, uid, h, fut))
    return _collect("eia", "h", 24, horizon, 24, items)


# ------------------------------------------------------------------ Wikipedia
def wiki_task(first="2026-01-05", last="2026-10-12", every="14D", context=365, horizon=28,
              min_hist=120, max_series=None) -> Task:
    w = pd.read_parquet(next((RAW / "wikipedia").glob("views_*.parquet")))
    if max_series:
        w = w.iloc[:, :max_series]
    items = []
    for o in pd.date_range(first, last, freq=every):
        for uid in w.columns:
            h = w[uid].loc[o - pd.Timedelta(days=context): o - pd.Timedelta(days=1)]
            first_valid = h.first_valid_index()
            if first_valid is None:
                continue
            h = h.loc[first_valid:].fillna(0)   # API omits zero-view days
            if len(h) < min_hist:
                continue
            fut = w[uid].loc[o: o + pd.Timedelta(days=horizon - 1)]
            if len(fut) < horizon:
                continue
            # Target is log1p(views) for ALL models: view counts are heavy-tailed and
            # raw-scale analytical intervals (e.g. ETS) explode on spiky pages.
            items.append((o, uid, np.log1p(h), np.log1p(fut.fillna(0))))
    return _collect("wiki", "D", 7, horizon, 7, items)


# ------------------------------------------------------------------ CDC flu
def _epiweek_dates(epiweeks) -> dict:
    """Contiguous epiweek -> weekly date index (sequential Sundays)."""
    eps = sorted(set(epiweeks))
    start = pd.Timestamp("2009-12-27")  # Sunday starting MMWR week 201001 (approx.)
    return {e: start + pd.Timedelta(weeks=i) for i, e in enumerate(eps)}


def flu_task(first_issue=202601, last_issue=202636, horizon=4, context_weeks=312,
             regions=None) -> Task:
    hist_df = cdc_flu.drop_invalid(pd.read_parquet(
        next((RAW / "cdc_flu").glob("fluview_history_*.parquet"))))
    old = pd.read_parquet(RAW / "cdc_flu" / "fluview_latest.parquet")
    old = old[old.epiweek < hist_df.epiweek.min()][["region", "epiweek", "issue", "wili"]]
    full = pd.concat([old, hist_df[["region", "epiweek", "issue", "wili"]]], ignore_index=True)
    dates = _epiweek_dates(full.epiweek.unique())
    final = (full.sort_values("issue").groupby(["region", "epiweek"], as_index=False).last())
    items = []
    issues = sorted(i for i in hist_df.issue.unique() if first_issue <= i <= last_issue)
    for iss in issues:
        known = full[full.issue <= iss]
        known = known.sort_values("issue").groupby(["region", "epiweek"], as_index=False).last()
        for reg in (regions or sorted(known.region.unique())):
            k = known[known.region == reg].sort_values("epiweek")
            h = pd.Series(k.wili.to_numpy(), index=[dates[e] for e in k.epiweek]).iloc[-context_weeks:]
            last_ds = h.index[-1]
            f = final[final.region == reg].set_index("epiweek").wili
            fut_idx = [last_ds + pd.Timedelta(weeks=s) for s in range(1, horizon + 1)]
            inv = {v: e for e, v in dates.items()}
            fut = pd.Series([f.get(inv.get(d), np.nan) for d in fut_idx], index=fut_idx)
            items.append((iss, reg, h, fut))
    return _collect("flu", "W-SUN", 52, horizon, 1, items)


# ------------------------------------------------------------------ FRED-MD
def _tcode(x: pd.Series, code: int) -> pd.Series:
    x = x.astype(float)
    if code == 1: return x
    if code == 2: return x.diff()
    if code == 3: return x.diff().diff()
    if code == 4: return np.log(x.where(x > 0))
    if code == 5: return np.log(x.where(x > 0)).diff()
    if code == 6: return np.log(x.where(x > 0)).diff().diff()
    if code == 7: return (x / x.shift(1) - 1).diff()
    raise ValueError(code)


def _vintage_defective(d: pd.DataFrame, final: pd.DataFrame, cols, max_bad: int = 10) -> bool:
    """A vintage file is defective if many series disagree materially with the final vintage
    over their last 12 shared dates (ordinary revisions are tiny). The 2026-08 file has 44
    misaligned columns (e.g. S&P-500 levels under the VIX label); every other vintage has 0."""
    bad = 0
    for c in cols:
        if c not in d.columns:
            continue
        b = pd.concat([d[c], final[c]], axis=1, join="inner").dropna().tail(12)
        if len(b) < 6:
            continue
        rel = (b.iloc[:, 0] - b.iloc[:, 1]).abs() / b.iloc[:, 1].abs().clip(lower=1e-9)
        bad += int((rel > 0.05).mean() > 0.5)
    return bad > max_bad


def fred_task(horizon=12, min_obs=120, max_series=None, keys=None, first_origin="2026-01") -> Task:
    """keys: vintage keys to use (last = truth vintage). Default: all. Validation uses
    2025-04..2025-12 so tuning never touches 2026 data."""
    files = sorted((RAW / "fred_md").glob("*-md.csv"))
    vint = {f.stem[:7]: fred_md.load_vintage(f) for f in files
            if keys is None or f.stem[:7] in keys}
    final_key = max(vint)
    final = vint[final_key]
    tc = final.attrs["tcodes"]
    cols = [c for c in final.columns if c in tc][: max_series or None]
    final_t = {c: _tcode(final[c], int(tc[c])).dropna() for c in cols}
    items, corrupt = [], []
    for key, d in vint.items():
        if key == final_key or (keys is None and first_origin and key < first_origin):
            continue    # evaluation origins start in 2026-01 (2025 vintages are used for tuning)
        if _vintage_defective(d, final, cols):
            corrupt.append(key)
            continue
        for c in cols:
            if c not in d.columns:
                continue
            h = _tcode(d[c], int(tc[c])).dropna()
            if len(h) < min_obs:
                continue
            last_ds = h.index[-1]
            fut_idx = [last_ds + pd.DateOffset(months=s) for s in range(1, horizon + 1)]
            fut = pd.Series([final_t[c].get(i, np.nan) for i in fut_idx], index=fut_idx)
            if fut.notna().sum() == 0:
                continue
            items.append((key, c, h, fut))
    if corrupt:
        print(f"fred_task: dropped defective vintage file(s): {corrupt}", flush=True)
    return _collect("fred", "MS", 12, horizon, 1, items)


BUILDERS = {"eia": eia_task, "wiki": wiki_task, "flu": flu_task, "fred": fred_task}


# ------------------------------------------------------------------ M4 anchor (public benchmark)
# M4 has no calendar dates: a synthetic date index of the right frequency is used (models only
# need the spacing). Context is truncated to the last `context` values. One origin ("m4"),
# MASE scale = in-sample seasonal-naive MAE of the (truncated) history, as in M4 with m below.
M4 = {  # group: (freq, model season, horizon, mase_m, context, n_series)
    "monthly": ("MS", 12, 18, 12, 360, 500),
    "weekly": ("W", 52, 13, 1, 400, 359),
    "daily": ("D", 7, 14, 1, 365, 500),
}


def m4_task(group: str, seed: int = 0) -> Task:
    freq, season, horizon, mase_m, context, n = M4[group]
    d = RAW / "m4"
    tr = pd.read_csv(d / f"{group.capitalize()}-train.csv").set_index("V1")
    te = pd.read_csv(d / f"{group.capitalize()}-test.csv").set_index("V1")
    ids = list(tr.index)
    if n < len(ids):
        ids = sorted(np.random.default_rng(seed).choice(ids, n, replace=False))
    items = []
    for uid in ids:
        y = tr.loc[uid].dropna().to_numpy(float)[-context:]
        f = te.loc[uid].dropna().to_numpy(float)[:horizon]
        if len(f) < horizon or len(y) < 2 * max(mase_m, 2):
            continue
        idx = pd.date_range("2000-01-01", periods=len(y) + horizon, freq=freq)
        items.append(("m4", uid, pd.Series(y, index=idx[:len(y)]), pd.Series(f, index=idx[len(y):])))
    return _collect(f"m4_{group}", freq, season, horizon, mase_m, items)


ANCHORS = {f"m4_{g}": (lambda g=g: m4_task(g)) for g in M4}
BUILDERS.update(ANCHORS)
