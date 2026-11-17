"""Export every (origin, series) history of a task to parquet for models in other envs.
python -m scripts.export_task fred flu wiki eia  ->  results/moirai_in/<task>.parquet (+ .json meta)"""
import json, sys
from pathlib import Path
from src.eval import tasks

OUT = Path(__file__).resolve().parents[1] / "results" / "moirai_in"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(tasks.BUILDERS):
        t = tasks.BUILDERS[name]()
        b = []
        for o, h in t.histories.items():
            b.append(h.assign(unique_id=f"{o}||" + h["unique_id"].astype(str)))
        import pandas as pd
        pd.concat(b, ignore_index=True)[["unique_id", "ds", "y"]].to_parquet(OUT / f"{name}.parquet")
        (OUT / f"{name}.json").write_text(json.dumps({"horizon": t.horizon, "freq": t.freq}))
        print(name, t.n_series_origins(), flush=True)
