"""Evaluate tuned baselines (+ post-hoc width calibration) on the 2026 evaluation tasks.

Models and width factors come from results/tuned/params.json (fit on 2025 data only).
Writes results/summary/tuned_<task>.csv with raw and calibrated ("+cal") rows.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
import pandas as pd


def main(only: list[str]) -> None:
    from src.eval import metrics, run, tasks
    out = Path(__file__).resolve().parents[1] / "results" / "summary"
    P = run.tuned()
    for name in (only or ["fred", "flu", "wiki", "eia"]):
        task = tasks.BUILDERS[name]()
        print(f"\n== {name}", flush=True)
        rows = []
        for m, factor in P["width"][name].items():
            t0 = time.time()
            try:
                fc = run.run_baseline(task, m, tag="full")
                for label, f in ((m, fc), (m + "+cal", run.widen(task, fc, factor))):
                    r = metrics.summarize(run.score(task, f)).iloc[0]
                    r["model"] = label
                    rows.append(r)
                    print(f"  {label:26s} MASE={r.MASE:.3f} WQL={r.WQL:.3f} (med {r.WQL_med:.3f}) "
                          f"cov80={r.cov80:.3f}", flush=True)
                print(f"    [{time.time() - t0:.0f}s]", flush=True)
            except Exception as e:
                print(f"  {m} FAILED {e!r}", flush=True); traceback.print_exc(limit=2)
        pd.DataFrame(rows).to_csv(out / f"tuned_{name}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
