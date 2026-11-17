"""Prompted-LLM forecasts on a fixed seeded subset per task (see src/models/llm.py).

python -m scripts.run_llm <model> [task ...] [--n N]     (--n overrides the subset size; smoke tests)
Writes results/forecasts/<task>_llm_<model>.parquet (resumable) + .cost.json.
"""
import json, sys
from pathlib import Path
from src.eval import run, tasks, metrics
from src.models import llm

if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else None
    if n is not None:
        del args[args.index("--n"):args.index("--n") + 2]
    ctx_n = int(args[args.index("--ctx") + 1]) if "--ctx" in args else None   # text-context run on N rows
    if ctx_n is not None:
        del args[args.index("--ctx"):args.index("--ctx") + 2]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else None     # sampling-seed repeat
    if seed is not None:
        del args[args.index("--seed"):args.index("--seed") + 2]
    rows_n = int(args[args.index("--rows") + 1]) if "--rows" in args else None    # strided subset size
    if rows_n is not None:
        del args[args.index("--rows"):args.index("--rows") + 2]
    model, names = args[0], args[1:] or ["flu", "fred", "wiki", "eia"]
    tag = f"llmseed{seed}" if seed is not None else ("llmctx" if ctx_n else ("llmsmoke" if n else "llm"))
    for name in names:
        task = tasks.BUILDERS[name]()
        items = llm.subset(task, n or llm.CFG[name]["n_eval"])
        if ctx_n or rows_n:   # evenly strided subset OF the main run's rows, so the main forecasts already exist
            k = ctx_n or rows_n
            items = items[::max(1, len(items) // k)][:k]
        path = run.ROOT / "results" / "forecasts" / f"{name}_{tag}_{model}.parquet"
        fc, cost = llm.forecast_items(model, task, items, path, with_ctx=bool(ctx_n), seed=seed)
        fc = run.apply_floor(task, fc.assign(model=model))
        fc.to_parquet(path, index=False)
        path.with_suffix(".cost.json").write_text(json.dumps(cost, indent=1))
        r = metrics.summarize(run.score(task, fc)).iloc[0]
        print(f"== {name} {model}: n={len(items)} MASE={r.MASE:.3f} WQL={r.WQL:.3f} cov80={r.cov80:.3f} "
              f"valid={cost['valid_sample_rate']} fallback={cost['fallback_rate']} "
              f"{cost['sec_per_item']}s/item vram={cost['peak_vram_gb']}GB", flush=True)
