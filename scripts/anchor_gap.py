"""Benchmark-vs-live gap: how much better is the best foundation model than the best classical model
on the M4 anchor tasks versus the post-cutoff live tasks?

Ratio = (loss of best FM) / (loss of best classical), each model picked on the SAME rows by mean scaled
quantile loss; < 1 means foundation models win. Unit of analysis: series (M4, one origin) or origin
(live tasks, as in significance.py). 95% CI by bootstrap (circular blocks for live, iid for M4).
Classical = naive, snaive, ets*, theta*, arima*, mstl. Foundation = chronos*, timesfm25, moirai2.
No contamination claim is made: only that the FM advantage differs between benchmark and live data.

python -m scripts.anchor_gap   ->  results/summary/anchor_gap.csv
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scripts.significance import FC, OUT, BLOCK, FM, B, pinball

LIVE = ["flu", "fred", "wiki", "eia"]
ANCH = ["m4_monthly", "m4_weekly", "m4_daily"]


def is_classical(m: str) -> bool:
    return m in ("naive", "snaive", "mstl") or m.startswith(("ets", "theta", "arima"))


def unit_losses(name: str) -> pd.DataFrame:
    from src.eval import run, tasks
    t = tasks.BUILDERS[name]()
    out = {}
    for p in sorted(FC.glob(f"{name}_full_*.parquet")):
        m = p.stem.split("_full_", 1)[1]
        s = run.score(t, pd.read_parquet(p).assign(model=m))
        s["loss"] = sum(pinball(s.y_true, s[c], tau) for c, tau in
                        (("q10", .1), ("q50", .5), ("q90", .9))) / 3 / s.scale
        g = s.groupby(["origin", "unique_id"]).loss.mean()
        out[m] = g.groupby(level=0).mean() if name in LIVE else g.droplevel(0)
    return pd.DataFrame(out).dropna()


def main():
    rng = np.random.default_rng(0)
    rows = []
    for name in LIVE + ANCH:
        P = unit_losses(name)
        fm = [m for m in P if m in FM]
        cl = [m for m in P if is_classical(m)]
        bf, bc = P[fm].mean().idxmin(), P[cl].mean().idxmin()
        a, b = P[bf].to_numpy(), P[bc].to_numpy()
        n, blk = len(a), (BLOCK[name] if name in LIVE else 1)
        nb = int(np.ceil(n / blk))
        idx = (rng.integers(0, n, (B, nb))[:, :, None] + np.arange(blk)) % n
        idx = idx.reshape(B, -1)[:, :n]
        ratios = a[idx].mean(1) / b[idx].mean(1)
        rows.append(dict(task=name, kind="live" if name in LIVE else "anchor", units=n, best_fm=bf, best_classical=bc,
                         ratio=a.mean() / b.mean(), ci_lo=np.percentile(ratios, 2.5), ci_hi=np.percentile(ratios, 97.5)))
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "anchor_gap.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
