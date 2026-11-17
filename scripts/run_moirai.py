"""Moirai 2.0 inference. Runs in the separate env E:\envs\tsllm_moirai (pinned torch 2.4.1).

python scripts/run_moirai.py fred flu wiki eia
Reads  results/moirai_in/<task>.parquet (unique_id = "origin||series"), writes
       results/moirai_out/<task>.parquet + <task>.cost.json. Standard columns are rebuilt
by src/eval/run.py (origin mapping, non-negative floor).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
IN, OUT = ROOT / "results" / "moirai_in", ROOT / "results" / "moirai_out"
REPO = "Salesforce/moirai-2.0-R-small"
BATCH = 64
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # Moirai 2.0 quantile heads


def main(names):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    OUT.mkdir(exist_ok=True)
    module = Moirai2Module.from_pretrained(REPO)
    for name in names:
        df = pd.read_parquet(IN / f"{name}.parquet")
        meta = json.loads((IN / f"{name}.json").read_text())
        h, off = meta["horizon"], pd.tseries.frequencies.to_offset(meta["freq"])
        items = [(uid, g["ds"].iloc[-1], g["y"].to_numpy(np.float32))
                 for uid, g in df.groupby("unique_id", sort=False)]
        ctx = int(max(len(i[2]) for i in items))
        model = Moirai2Forecast(module=module, prediction_length=h, context_length=ctx,
                                target_dim=1, feat_dynamic_real_dim=0,
                                past_feat_dynamic_real_dim=0).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        t0, rows = time.time(), []
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            with torch.no_grad():
                q = model.predict([c[2][:, None] for c in chunk])      # (b, nq, h)
            q = np.asarray(q)
            for (uid, last, _), qq in zip(chunk, q):
                ds = [pd.Timestamp(last) + off * j for j in range(1, h + 1)]
                qs = np.sort(qq[[0, 4, 8], :h].T, axis=1)               # q10, q50, q90
                rows.append(pd.DataFrame({"unique_id": uid, "ds": ds,
                                          "q10": qs[:, 0], "q50": qs[:, 1], "q90": qs[:, 2]}))
        out = pd.concat(rows, ignore_index=True)
        out.to_parquet(OUT / f"{name}.parquet", index=False)
        sec = time.time() - t0
        (OUT / f"{name}.cost.json").write_text(json.dumps({
            "model": "moirai2", "seconds": round(sec, 1), "n_series": len(items),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "sec_per_series": round(sec / len(items), 4), "context_length": ctx}))
        print(name, len(items), f"{sec:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["flu", "fred", "wiki", "eia"])
