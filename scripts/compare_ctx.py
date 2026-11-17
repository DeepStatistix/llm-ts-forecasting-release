"""Does text context help?  Paired comparison of each LLM setting WITH vs WITHOUT text context,
on exactly the same rows. Context = series description + forecast-window calendar (origin-time
information only; see src/models/context.py).

Settings: prompted LLM (llmctx vs llm) and agent (agentcctx vs agentc, constrained decoding), per LLM and task.
Unit of analysis = origin; circular block bootstrap as elsewhere.  Negative diff = context helps.
Also reports the best classical / foundation model on the same rows for scale.

python -m scripts.compare_ctx   ->  results/summary/ctx_compare.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scripts.significance import FC, OUT, BLOCK, boot_p, pinball

LLMS = ["qwen3_8b", "mistral7b"]


def main(names):
    from src.eval import run, tasks
    rng = np.random.default_rng(0)
    rows = []
    for name in names:
        t = tasks.BUILDERS[name]()
        for kind, plain, ctx in (("prompted", "llm", "llmctx"), ("agent", "agentc", "agentcctx")):
            for m in LLMS:
                pp, cp = FC / f"{name}_{plain}_{m}.parquet", FC / f"{name}_{ctx}_{m}.parquet"
                if not (pp.exists() and cp.exists()):
                    continue
                a, b = pd.read_parquet(pp), pd.read_parquet(cp)
                keys = set(zip(b.origin.astype(str), b.unique_id)) & set(zip(a.origin.astype(str), a.unique_id))
                loss = {}
                for lab, d in (("plain", a), ("ctx", b)):
                    d = d[[k in keys for k in zip(d.origin.astype(str), d.unique_id)]]
                    s = run.score(t, d.assign(model=lab))
                    s["loss"] = sum(pinball(s.y_true, s[c], tau) for c, tau in
                                    (("q10", .1), ("q50", .5), ("q90", .9))) / 3 / s.scale
                    loss[lab] = s
                po = {k: v.groupby(["origin", "unique_id"]).loss.mean().groupby(level=0).mean() for k, v in loss.items()}
                d = (po["ctx"] - po["plain"]).dropna().to_numpy()
                p, ci = boot_p(d, BLOCK[name], rng)
                r = dict(task=name, setting=kind, llm=m, rows=len(keys), origins=len(d), wql_plain=po["plain"].mean(),
                         wql_ctx=po["ctx"].mean(), ratio_ctx_over_plain=po["ctx"].mean() / po["plain"].mean(),
                         diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1], p=p, ctx_better_frac=float((d < 0).mean()),
                         fallback_plain=float(a.get("fallback", pd.Series([0])).mean()) if "fallback" in a else np.nan,
                         fallback_ctx=float(b["fallback"].mean()) if "fallback" in b else np.nan)
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ctx_compare.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["flu", "fred", "wiki", "eia"])
