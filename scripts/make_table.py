"""Combine baseline + foundation summaries into results/summary/all_models.csv and print."""
from pathlib import Path
import pandas as pd
S = Path(__file__).resolve().parents[1] / "results" / "summary"
fr = []
for t in ["fred", "flu", "wiki", "eia"]:
    for kind in ["baselines", "foundation", "tuned"]:
        p = S / f"{kind}_{t}.csv"
        if p.exists():
            d = pd.read_csv(p); d.insert(0, "task", t); fr.append(d)
df = pd.concat(fr, ignore_index=True).drop_duplicates(["task", "model"], keep="last")
df.to_csv(S / "all_models.csv", index=False)
cols = ["task", "model", "MASE", "MASE_med", "WQL", "WQL_med", "cov80"]
for t, g in df.groupby("task", sort=False):
    print(f"\n== {t}"); print(g.sort_values("WQL")[cols[1:]].round(3).to_string(index=False))
