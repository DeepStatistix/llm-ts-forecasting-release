"""Fetch the FRED-MD vintages still missing, using a browser-style User-Agent
(approved by the project owner after the honest client name was blocked by the host's
bot filter). Polite: one request at a time, long pauses, validated content, hard stop
after repeated blocks.

Run from repo root:  python -m scripts.fetch_fred_missing
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from src.data import fred_md

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
PAUSE = 20          # seconds between files
MAX_BLOCKS = 4      # stop entirely after this many consecutive blocked responses


def main() -> None:
    fred_md.RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [str(m) for m in pd.period_range("2025-01", "2026-10", freq="M")]
    todo = [v for v in wanted if not (fred_md.RAW_DIR / f"{v}-md.csv").exists()]
    print(f"missing vintages: {todo}", flush=True)
    blocks = 0
    for v in todo:
        url = fred_md.BASE.format(vintage=v)
        try:
            r = requests.get(url, headers={"User-Agent": BROWSER_UA,
                                           "Accept": "text/csv,text/plain,*/*"}, timeout=90)
            ok = r.status_code == 200 and fred_md._is_valid(r.text)
            note = f"http={r.status_code} bytes={len(r.content)}"
        except requests.RequestException as e:
            ok, note = False, repr(e)
        if ok:
            (fred_md.RAW_DIR / f"{v}-md.csv").write_text(r.text, encoding="utf-8")
            blocks = 0
            print(f"{v} OK ({note})", flush=True)
        else:
            blocks += 1
            print(f"{v} BLOCKED/INVALID ({note}) [{blocks}/{MAX_BLOCKS}]", flush=True)
            if blocks >= MAX_BLOCKS:
                print("Stopping: host keeps refusing. Not escalating.", flush=True)
                break
        time.sleep(PAUSE)
    have = sorted(p.name for p in fred_md.RAW_DIR.glob("*-md.csv"))
    print(f"now have {len(have)} vintages: {have}")


if __name__ == "__main__":
    main()
