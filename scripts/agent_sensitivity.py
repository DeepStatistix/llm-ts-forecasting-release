"""Prompt-wording sensitivity of the agent (instruction paraphrases 0-3) on fixed small subsets,
plus seed variance of the candidate-order shuffle. Same rows for every variant.

python -m scripts.agent_sensitivity <llm>   ->  results/summary/agent_sensitivity_<llm>.csv
"""
import sys
import numpy as np, pandas as pd
from scripts.significance import pinball
from src.eval import metrics, run, tasks
from src.models import agent, llm

N = {"flu": 150, "fred": 120, "wiki": 60}
if __name__ == "__main__":
    model = sys.argv[1]
    rows = []
    for name, n in N.items():
        task = tasks.BUILDERS[name]()
        items = llm.subset(task, n, seed=5)          # a different fixed subset from the main run
        for variant, seed in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2)]:
            fc, ch, cost = agent.run_agent(model, task, items, run.ROOT, seed=seed, variant=variant)
            fc = run.apply_floor(task, fc.assign(model="x"))
            r = metrics.summarize(run.score(task, fc)).iloc[0]
            pick = ch.chosen.value_counts(normalize=True).round(2).to_dict()
            rows.append(dict(task=name, variant=variant, shuffle_seed=seed, MASE=r.MASE, WQL=r.WQL,
                             cov80=r.cov80, fallback=cost["fallback_rate"], mean_width=ch.width.mean(), **{f"p_{k}": v for k, v in pick.items()}))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(run.ROOT / "results" / "summary" / f"agent_sensitivity_constrained_{model}.csv", index=False)
