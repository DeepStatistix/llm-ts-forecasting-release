"""Fetch all four live datasets. Resumable: reruns skip anything already downloaded.

Run from repo root:  python -m scripts.fetch_all
Progress is logged to results/fetch_log.txt.
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path

import pandas as pd

from src.data import cdc_flu, eia_demand, fred_md, wikipedia_views

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "fetch_log.txt"
LOG.parent.mkdir(exist_ok=True)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def step(name, fn):
    log(f"START {name}")
    try:
        out = fn()
        log(f"DONE  {name}: {out}")
    except Exception as e:  # keep going so one source can't block the rest
        log(f"FAIL  {name}: {e!r}")
        log(traceback.format_exc(limit=3))


def do_eia():
    resp = eia_demand.list_respondents()["respondent"].tolist()
    df = eia_demand.fetch_panel(resp, "2023-01-01T00", "2026-11-10T00")
    return f"{df['respondent'].nunique()} balancing authorities, {len(df):,} rows, " \
           f"{df['period'].min()} -> {df['period'].max()}"


def do_flu():
    d = cdc_flu.fetch_history()
    return f"{d.shape}, regions={d['region'].nunique()}, epiweeks {d['epiweek'].min()}-{d['epiweek'].max()}"


def do_wiki():
    pages = wikipedia_views.sample_pages()
    df = wikipedia_views.fetch_panel(pages, "20230101", "20261110")
    return f"{df.shape[1]} pages x {df.shape[0]} days"


def do_fred():
    # reuse any validated vintages fetched earlier into the probe folder
    probe = ROOT / "data" / "raw" / "_probe"
    if probe.exists():
        for f in probe.glob("*.csv"):
            tgt = fred_md.RAW_DIR / f"{f.stem}-md.csv"
            if not tgt.exists() and fred_md._is_valid(f.read_text(encoding="utf-8", errors="replace")):
                fred_md.RAW_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, tgt)
    res = fred_md.fetch_range("2025-01", "2026-10")
    missing = [v for v, p in res.items() if p is None]
    return f"{len(res) - len(missing)}/{len(res)} vintages; missing={missing}"


if __name__ == "__main__":
    step("EIA electricity demand", do_eia)
    step("Wikipedia page views", do_wiki)
    step("CDC FluView history", do_flu)
    step("FRED-MD vintages", do_fred)
    log("ALL STEPS FINISHED")
