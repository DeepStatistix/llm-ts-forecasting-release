"""Paired significance tests of every model against the best model, per task.

Loss: scaled quantile loss (mean over tau 0.1/0.5/0.9 and horizon steps) per (origin, series).
Series inside one origin are strongly cross-correlated, so the unit of analysis is the ORIGIN:
d_t = mean over series of (loss_model - loss_ref) at origin t. A moving-block bootstrap over
origins (block length covers overlapping forecast windows) gives a two-sided p-value for
E[d]=0; Holm correction across all models compared against the reference.
Also reports the best foundation model vs the best non-foundation model, with a CI on the ratio.

python -m scripts.significance
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FC, OUT = ROOT / "results" / "forecasts", ROOT / "results" / "summary"
BLOCK = {"eia": 1, "wiki": 2, "flu": 4, "fred": 2}   # fred: only 8 origins, so a longer block is invalid
FM = {"chronos_bolt_small", "chronos_bolt_base", "chronos2", "timesfm25", "moirai2"}
B, SEED = 10000, 0


def pinball(y, q, t):
    d = y - q
    return np.maximum(t * d, (t - 1) * d)


def losses(task, name) -> dict:
    from src.eval import run, tasks
    t = tasks.BUILDERS[name]()
    out = {}
    for p in sorted(FC.glob(f"{name}_full_*.parquet")):
        m = p.stem.split("_full_", 1)[1]
        s = run.score(t, pd.read_parquet(p))
        l = sum(pinball(s.y_true, s[c], tau) for c, tau in (("q10", .1), ("q50", .5), ("q90", .9))) / 3
        s["loss"] = l / s.scale
        out[m] = s.groupby(["origin", "unique_id"]).loss.mean()
    return out


def boot_p(d: np.ndarray, block: int, rng) -> tuple[float, tuple[float, float]]:
    n = len(d)
    nb = math.ceil(n / block)
    idx = (rng.integers(0, n, (B, nb))[:, :, None] + np.arange(block)) % n   # circular blocks
    means = d[idx.reshape(B, -1)[:, :n]].mean(1)
    obs = d.mean()
    c = means - obs                                     # centred bootstrap distribution
    p = float((np.abs(c) >= abs(obs)).mean())
    return max(p, 1 / B), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def holm(p: pd.Series) -> pd.Series:
    o = p.sort_values(); m = len(o); adj, run = [], 0.0
    for i, v in enumerate(o):
        run = max(run, min(1.0, (m - i) * v)); adj.append(run)
    return pd.Series(adj, index=o.index).reindex(p.index)


def main(names):
    rng = np.random.default_rng(SEED)
    for name in names or ["fred", "flu", "wiki", "eia"]:
        L = losses(None, name)
        common = None
        for s in L.values():
            common = s.index if common is None else common.intersection(s.index)
        per_origin = pd.DataFrame({m: s.loc[common].groupby(level="origin").mean() for m, s in L.items()})
        base = per_origin.drop(columns=[c for c in per_origin if "+cal" in c])
        ref = base.mean().idxmin()
        rows = []
        for m in base:
            if m == ref:
                continue
            d = (base[m] - base[ref]).to_numpy()
            p, ci = boot_p(d, BLOCK[name], rng)
            rows.append(dict(model=m, loss=base[m].mean(), ratio_to_best=base[m].mean() / base[ref].mean(),
                             diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1], p=p,
                             origins_worse=float((d > 0).mean())))
        r = pd.DataFrame(rows).set_index("model")
        r["p_holm"] = holm(r.p)
        r["sig_worse"] = r.p_holm < 0.05
        r = r.sort_values("loss")
        r.insert(0, "ref", ref)
        r.to_csv(OUT / f"significance_{name}.csv")
        # headline: best foundation model vs best non-foundation model
        fm = [m for m in base if m in FM]; nf = [m for m in base if m not in FM]
        bf, bn = base[fm].mean().idxmin(), base[nf].mean().idxmin()
        d = (base[bf] - base[bn]).to_numpy()
        p, ci = boot_p(d, BLOCK[name], rng)
        ratio = base[bf].mean() / base[bn].mean()
        print(f"\n== {name}: {len(per_origin)} origins, {len(common)} series-origins; best overall: {ref}")
        print(f"   best foundation {bf} vs best non-foundation {bn}: loss ratio {ratio:.3f} "
              f"(diff {d.mean():+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], p={p:.4f}, "
              f"better in {(d < 0).mean():.0%} of origins)")
        print(r[["loss", "ratio_to_best", "p_holm", "sig_worse", "origins_worse"]].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
