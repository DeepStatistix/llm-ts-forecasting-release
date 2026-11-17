"""Hardware-independent + measured cost table for every model.

Per model and task:
  params            nominal parameter count (fixed per model)
  ctx_len           mean history length actually fed in (LLM: prompt context `ctx`)
  gen_tokens/item   LLM only: measured generated tokens per (origin, series)
  gflops/forecast   ESTIMATE: 2 * params * tokens processed
                    foundation/deep: tokens = ctx_len / patch_len (one forward pass)
                    LLM: prompt tokens (est. 3.2 per context value) + generated tokens, x1 (KV cache
                         shares the prompt across samples; generated tokens already counts all samples)
  sec/forecast      MEASURED wall-clock on this laptop (RTX 2000 Ada 8 GB), inference only
  peak_vram_gb      MEASURED
FLOP estimates ignore memory bandwidth (the real LLM bottleneck), so they UNDERSTATE LLM slowness.

python -m scripts.cost_table  ->  results/summary/cost_table.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "results" / "forecasts"

# nominal parameter counts (millions) and patch length; check against model cards before publishing
PARAMS_M = {"chronos_bolt_small": 48, "chronos_bolt_base": 205, "chronos2": 120, "timesfm25": 200,
            "moirai2": 11, "nhits": None, "patchtst": None,
            "qwen3_8b": 8200, "mistral7b": 7250, "gemma4_e4b": 4500}
PATCH = {"chronos_bolt_small": 16, "chronos_bolt_base": 16, "chronos2": 16, "timesfm25": 32,
         "moirai2": 16}
TOK_PER_VALUE = 3.2   # "123, " is ~3-4 tokens with these tokenizers (estimate)


def main() -> None:
    from src.eval import tasks
    from src.models import llm

    rows = []
    for name in ["fred", "flu", "wiki", "eia"]:
        t = tasks.BUILDERS[name]()
        n_items = t.n_series_origins()
        mean_len = float(pd.Series([len(g) for h in t.histories.values()
                                    for _, g in h.groupby("unique_id")]).mean())
        for f in sorted(FC.glob(f"{name}_full_*.cost.json")) + sorted(FC.glob(f"{name}_llm_*.cost.json")) + sorted(FC.glob(f"{name}_agentc_*.cost.json")):
            c = json.loads(f.read_text())
            m = c["model"]
            is_agent = "_agentc_" in f.name           # one constrained LLM call (2 prompt passes) per item
            is_llm = "_llm_" in f.name or is_agent
            n = c.get("n_items_run", c.get("n_items", c.get("n_series", n_items)))
            gen = (2 if is_agent else c["generated_tokens"] / max(n, 1)) if is_llm else None
            p = PARAMS_M.get(m)
            if is_llm:
                ctx = llm.CFG[name]["ctx"]
                if is_agent:   # prompt ~650 tokens (estimate), processed twice (letter step, digit step)
                    gfl = 2 * p * 1e6 * (2 * 650 + gen) / 1e9
                    m = "agent_" + m
                else:
                    gfl = 2 * p * 1e6 * (ctx * TOK_PER_VALUE + gen) / 1e9
                sec = c["sec_per_item"]
            else:
                ctx = mean_len
                gfl = 2 * p * 1e6 * (ctx / PATCH[m]) / 1e9 if p and m in PATCH else None
                sec = c.get("sec_per_series", c.get("predict_seconds", c["seconds"]) / max(n, 1))
            rows.append(dict(task=name, model=m, params_M=p, ctx_len=round(ctx),
                             gen_tokens_per_item=None if gen is None else round(gen),
                             gflops_per_forecast=None if gfl is None else round(gfl, 1),
                             sec_per_forecast=round(sec, 4), peak_vram_gb=c.get("peak_vram_gb"),
                             train_seconds=c.get("train_seconds"),
                             valid_sample_rate=c.get("valid_sample_rate"),
                             fallback_rate=c.get("fallback_rate")))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "summary" / "cost_table.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
