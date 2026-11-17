"""Gemma 4 E4B ran only partially (flu; too slow on the 8 GB card: ~80 s/item, VRAM spill). Report what exists, on the
rows it finished, next to Qwen3-8B, Mistral-7B and the best classical / foundation models on the SAME rows.

python -m scripts.gemma_partial  ->  results/summary/gemma_partial.csv
"""
import pandas as pd
from scripts.significance import FC, OUT, pinball
from src.eval import metrics, run, tasks

if __name__ == "__main__":
    t = tasks.BUILDERS["flu"]()
    g = pd.read_parquet(FC / "flu_llm_gemma4_e4b.parquet")
    keys = set(zip(g.origin.astype(str), g.unique_id))
    models = {"gemma4_e4b": g}
    for m in ("qwen3_8b", "mistral7b"):
        models[m] = pd.read_parquet(FC / f"flu_llm_{m}.parquet")
    for m in ("naive", "snaive", "theta", "chronos_bolt_small", "chronos2", "timesfm25"):
        models[m] = pd.read_parquet(FC / f"flu_full_{m}.parquet")
    scored = []
    for m, d in models.items():
        d = d[[k in keys for k in zip(d.origin.astype(str), d.unique_id)]]
        scored.append(run.score(t, d.assign(model=m)))
    s = metrics.summarize(pd.concat(scored)).sort_values("WQL")
    s["rows"] = len(keys)
    cost = pd.read_json(FC / "flu_llm_gemma4_e4b.cost.json", typ="series") if (FC / "flu_llm_gemma4_e4b.cost.json").exists() else None
    s.to_csv(OUT / "gemma_partial.csv", index=False)
    print(f"Gemma 4 E4B rows finished: {len(keys)} of 396 (flu)")
    print(s[["model", "MASE", "WQL", "cov80"]].round(3).to_string(index=False))
