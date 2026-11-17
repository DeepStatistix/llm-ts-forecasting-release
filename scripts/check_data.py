"""Integrity checks for the four fetched datasets. Writes results/data_quality.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import fred_md

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
out: list[str] = []


def h(t):
    out.append(f"\n## {t}\n")


def p(t=""):
    out.append(str(t))


# ---------------- EIA ----------------
h("EIA hourly demand")
eia = pd.read_parquet(next((RAW / "eia").glob("demand_*.parquet")))
p(f"rows={len(eia):,}  respondents={eia.respondent.nunique()}  "
  f"{eia.period.min()} -> {eia.period.max()}")
dups = eia.duplicated(["respondent", "period"]).sum()
p(f"duplicate (respondent, hour) rows: {dups}")
rows = []
for r, g in eia.groupby("respondent"):
    g = g.sort_values("period")
    full = pd.date_range(g.period.min(), g.period.max(), freq="h")
    miss = len(full) - g.period.nunique()
    v = g.value
    med = v.median()
    rows.append(dict(resp=r, first=g.period.min().date(), last=g.period.max().date(),
                     n=len(g), missing_hours=miss, missing_pct=round(100 * miss / len(full), 2),
                     nan=int(v.isna().sum()), zero=int((v == 0).sum()), neg=int((v < 0).sum()),
                     spikes_gt10x_median=int((v > 10 * med).sum()) if med > 0 else -1))
q = pd.DataFrame(rows).set_index("resp")
p(f"series with data ending before 2026-11-01: {int((pd.to_datetime(q['last']) < '2026-11-01').sum())}")
p(f"series starting after 2023-01-02: {int((pd.to_datetime(q['first']) > '2023-01-02').sum())}")
p(f"series with >1% missing hours: {int((q.missing_pct > 1).sum())}; "
  f"with any neg: {int((q.neg > 0).sum())}; with zeros: {int((q.zero > 0).sum())}; "
  f"with spikes: {int((q.spikes_gt10x_median > 0).sum())}")
p("\nWorst 10 by missing %:\n" + q.sort_values("missing_pct", ascending=False).head(10).to_string())
q.to_csv(ROOT / "results" / "eia_quality_by_series.csv")

# ---------------- Wikipedia ----------------
h("Wikipedia page views")
wiki = pd.read_parquet(next((RAW / "wikipedia").glob("views_*.parquet")))
p(f"pages={wiki.shape[1]}  days={wiki.shape[0]}  {wiki.index.min().date()} -> {wiki.index.max().date()}")
nan_frac = wiki.isna().mean()
p(f"pages with any NaN days: {int((nan_frac > 0).sum())}; with >5% NaN: {int((nan_frac > .05).sum())}")
p(f"pages with zero-view days: {int((wiki == 0).any().sum())}")
p(f"constant (std=0) pages: {int((wiki.std() == 0).sum())}")
p(f"median of page mean views: {wiki.mean().median():,.0f}; min page mean: {wiki.mean().min():,.0f}")
try:
    from src.data import wikipedia_views as wv
    sample = set(wv.sample_pages())
    dropped = sorted(sample - set(wiki.columns))
    p(f"sampled pages not fetched: {len(dropped)}  e.g. {dropped[:8]}")
except Exception as e:  # network optional
    p(f"(drop-out check skipped: {e!r})")

# ---------------- CDC ----------------
h("CDC FluView history")
cdc = pd.read_parquet(next((RAW / "cdc_flu").glob("fluview_history_*.parquet")))
p(f"rows={len(cdc):,} regions={cdc.region.nunique()} epiweeks {cdc.epiweek.min()}-{cdc.epiweek.max()} "
  f"issues {cdc.issue.min()}-{cdc.issue.max()}")
p(f"duplicate (region, epiweek, issue): {cdc.duplicated(['region', 'epiweek', 'issue']).sum()}")
p(f"issue < epiweek (impossible): {(cdc.issue < cdc.epiweek).sum()}")
p(f"null wili: {cdc.wili.isna().sum()}; null ili: {cdc.ili.isna().sum()}")
n = cdc.groupby(["region", "epiweek"]).size()
p(f"revisions per week: median {int(n.median())}, max {int(n.max())}")
fr = cdc.sort_values("issue").groupby(["region", "epiweek"]).agg(first=("wili", "first"), last=("wili", "last"))
rel = ((fr["last"] - fr["first"]).abs() / fr["last"].replace(0, np.nan)).dropna()
p(f"first-release vs final |relative revision| in wILI: median {rel.median():.1%}, 90th pct {rel.quantile(.9):.1%}")
p("NOTE: the Delphi API no longer serves issues after 2026-09-22, so history ends at issue 202640.")

# ---------------- FRED-MD ----------------
h("FRED-MD vintages")
for f in sorted((RAW / "fred_md").glob("*-md.csv")):
    ok = fred_md._is_valid(f.read_text(encoding="utf-8", errors="replace"))
    if not ok:
        p(f"{f.name}: INVALID"); continue
    d = fred_md.load_vintage(f)
    p(f"{f.name}: series={d.shape[1]} last_obs={d.index.max().date()} rows={d.shape[0]}")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "data_quality.md").write_text("# Data quality report\n" + "\n".join(out), encoding="utf-8")
print("\n".join(out))
