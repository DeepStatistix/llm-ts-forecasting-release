"""Significance on the LLM subset: every model (classical, ML, deep, foundation, prompted LLMs, agents)
scored on exactly the rows the LLMs ran on (same fixed seeded subset per task).

Unit of analysis = origin (mean scaled quantile loss over the rows of that origin); circular
moving-block bootstrap (B=10000); Holm correction.
Pre-specified contrasts (no peeking at results to choose them):
  A. each CATEGORY's best model vs the overall best model          (Holm over categories)
  B. best prompted LLM vs the trivial baselines (naive, seasonal naive)
  C. agent vs best prompted LLM, and vs best classical model
Categories: classical (naive, snaive, ets*, theta*, arima*, mstl), ml (lgbm*), deep (nhits, patchtst),
foundation (chronos*, timesfm25, moirai2), prompted_llm, agent. "+cal" post-hoc variants are excluded.

python -m scripts.significance_llm   ->  results/summary/significance_llm_<task>.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scripts.significance import FC, OUT, BLOCK, boot_p, holm, pinball

LLMS = ["qwen3_8b", "mistral7b"]


def category(m: str) -> str:
    if m.startswith("prompted_"):
        return "prompted_llm"
    if m.startswith("agent_"):
        return "agent"
    if m.startswith(("chronos", "timesfm", "moirai")):
        return "foundation"
    if m in ("nhits", "patchtst"):
        return "deep"
    if m.startswith("lgbm"):
        return "ml"
    return "classical"


def main(names):
    from src.eval import run, tasks
    rng = np.random.default_rng(0)
    for name in names:
        t = tasks.BUILDERS[name]()
        files = {f"prompted_{m}": FC / f"{name}_llm_{m}.parquet" for m in LLMS}
        files.update({f"agent_{m}": FC / f"{name}_agentc_{m}.parquet" for m in LLMS})
        files = {k: p for k, p in files.items() if p.exists()}
        keep = None
        for p in files.values():
            d = pd.read_parquet(p, columns=["origin", "unique_id"])
            ks = set(zip(d.origin.astype(str), d.unique_id))
            keep = ks if keep is None else keep & ks
        for p in sorted(FC.glob(f"{name}_full_*.parquet")):
            m = p.stem.split("_full_", 1)[1]
            if "+cal" not in m:
                files[m] = p
        per_origin = {}
        for m, p in files.items():
            d = pd.read_parquet(p)
            d = d[[k in keep for k in zip(d.origin.astype(str), d.unique_id)]]
            s = run.score(t, d.assign(model=m))
            s["loss"] = sum(pinball(s.y_true, s[c], tau) for c, tau in
                            (("q10", .1), ("q50", .5), ("q90", .9))) / 3 / s.scale
            per_origin[m] = s.groupby(["origin", "unique_id"]).loss.mean().groupby(level=0).mean()
        P = pd.DataFrame(per_origin).dropna()
        cats = {m: category(m) for m in P}
        best_in = {c: min((m for m in P if cats[m] == c), key=lambda m: P[m].mean())
                   for c in set(cats.values())}
        ref = min(best_in.values(), key=lambda m: P[m].mean())
        rows = []

        def contrast(label, a, b, family):
            d = (P[a] - P[b]).to_numpy()
            p, ci = boot_p(d, BLOCK[name], rng)
            rows.append(dict(family=family, contrast=label, a=a, b=b, loss_a=P[a].mean(), loss_b=P[b].mean(),
                             ratio=P[a].mean() / P[b].mean(), diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1], p=p,
                             a_better_frac=float((d < 0).mean())))

        for c, m in best_in.items():                                   # A
            if m != ref:
                contrast(f"best {c} vs overall best", m, ref, "A")
        for trivial in ("naive", "snaive"):                             # B
            for m in (k for k in P if cats[k] == "prompted_llm"):
                contrast(f"{m} vs {trivial}", m, trivial, "B")
        for a in (k for k in P if cats[k] == "agent"):                  # C
            for m in (k for k in P if cats[k] == "prompted_llm" and k.split("_", 1)[1] == a.split("_", 1)[1]):
                contrast(f"{a} vs {m}", a, m, "C")
            contrast(f"{a} vs best classical", a, best_in["classical"], "C")
        r = pd.DataFrame(rows)
        r["p_holm"] = r.groupby("family").p.transform(lambda s: holm(s))
        r["a_significantly_better"] = (r.p_holm < 0.05) & (r["diff"] < 0)
        r["a_significantly_worse"] = (r.p_holm < 0.05) & (r["diff"] > 0)
        r.to_csv(OUT / f"significance_llm_{name}.csv", index=False)
        print(f"\n== {name}: {len(keep)} rows, {len(P)} origins, overall best on subset: {ref}")
        print(r[["family", "contrast", "ratio", "p_holm", "a_significantly_better", "a_significantly_worse",
                 "a_better_frac"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["flu", "fred", "wiki", "eia"])
