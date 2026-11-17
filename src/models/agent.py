"""Light tool-augmented LLM selector ("agent").

One greedy LLM call per (origin, series). Code runs the tools: diagnostics of the history and a
few cached candidate forecasts (naive, seasonal naive, a classical model, Chronos-Bolt-small).
The LLM sees a short text summary and answers with a candidate letter and an interval-width
digit (e.g. "B3"). Nothing but the history and the candidates' forecasts is in the prompt;
candidate order is shuffled per item (seeded) to remove position bias.
Parse failure -> the classical candidate at width 2, flagged `fallback`.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import torch

from src.models import llm

CAND = {"flu": ["naive", "snaive", "theta", "chronos_bolt_small"],
        "fred": ["naive", "snaive", "theta", "chronos_bolt_small"],
        "wiki": ["naive", "snaive", "theta", "chronos_bolt_small"],
        "eia": ["naive", "snaive", "mstl", "chronos_bolt_small"]}
SEASON = {"flu": 52, "fred": 12, "wiki": 7, "eia": 168}
NREC = {"flu": 26, "fred": 24, "wiki": 28, "eia": 48}      # recent values shown
WIDTH = {"1": 0.8, "2": 1.0, "3": 1.3, "4": 1.6}
LETTERS = "ABCD"
# instruction-wording variants for the prompt-sensitivity test (variant 0 = the main prompt)
INSTR = [
    "Pick the candidate most likely to be the most accurate, and an interval width: "
    "1=narrow (0.8x), 2=standard (1.0x), 3=wide (1.3x), 4=very wide (1.6x). "
    "Answer with exactly one letter and one digit, for example B3.",
    "Which candidate forecast do you trust most for this series? Also rate how uncertain the future is: "
    "1=low uncertainty (0.8x), 2=normal (1.0x), 3=high (1.3x), 4=very high (1.6x). "
    "Reply with a single letter followed by a single digit, such as B3.",
    "Act as a careful forecaster. Choose the single best candidate and the interval width multiplier "
    "(1: 0.8x, 2: 1.0x, 3: 1.3x, 4: 1.6x) that would give well-calibrated 80% intervals. "
    "Output only a letter and a digit, e.g. B3.",
    "Select one candidate (A-D) and a width code (1-4: 0.8x, 1.0x, 1.3x, 1.6x) so that the forecast is as "
    "accurate as possible. Respond in the form B3 and nothing else.",
]


def _sig(x: float) -> str:
    return f"{x:.3g}"


def load_candidates(task_name: str, root) -> dict:
    """{model: {(str(origin), uid): (h,3) array of q10,q50,q90 ordered by ds}}"""
    out = {}
    for m in CAND[task_name]:
        d = pd.read_parquet(root / "results" / "forecasts" / f"{task_name}_full_{m}.parquet")
        d = d.sort_values(["origin", "unique_id", "ds"])
        out[m] = {(str(o), u): g[["q10", "q50", "q90"]].to_numpy()
                  for (o, u), g in d.groupby(["origin", "unique_id"], sort=False)}
    return out


def diagnostics(y: np.ndarray, m: int) -> str:
    lvl = float(np.mean(np.abs(y[-m:])) if len(y) >= m else np.mean(np.abs(y))) or 1.0
    k = min(len(y), max(8, m // 2 if m > 12 else 12))
    slope = np.polyfit(np.arange(k), y[-k:], 1)[0] / lvl * 100
    vol = np.std(np.diff(y[-k:])) / lvl
    if len(y) >= 2 * m:
        a, b = y[:-m] - y[:-m].mean(), y[m:] - y[m:].mean()
        sa = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
    else:
        sa = float("nan")
    return (f"last={_sig(y[-1])}, mean of last {k}={_sig(y[-k:].mean())}, trend={slope:+.2f}% of level per step, "
            f"volatility={vol:.2f}, seasonal autocorrelation at lag {m}={sa:.2f}")


def build_prompt(tok, task, y, cands_arr: list, qwen_think_off: bool, variant: int = 0, extra: str = "") -> str:
    h = task.horizon
    m = SEASON[task.name]
    lines = []
    for L, a in zip(LETTERS, cands_arr):
        mid = a[:, 1]
        hw = float((a[-1, 2] - a[-1, 0]) / 2)
        lines.append(f"{L}: median at step 1={_sig(mid[0])}, step {h // 2}={_sig(mid[h // 2 - 1])}, "
                     f"step {h}={_sig(mid[-1])}; 80% interval half-width at step {h}={_sig(hw)}")
    recent = ", ".join(_sig(v) for v in y[-NREC[task.name]:])
    text = (f"You choose among candidate forecasts for a {llm.FREQ_TXT[task.name]} time series (horizon {h} steps).\n"
            + (f"Context: {extra}\n" if extra else "") +
            f"Recent values (oldest to newest): {recent}\n"
            f"Diagnostics: {diagnostics(y, m)}\n"
            f"Candidate forecasts:\n" + "\n".join(lines) + "\n"
            + INSTR[variant])
    kw = {"enable_thinking": False} if qwen_think_off else {}
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, **kw) + "Answer: "   # prefilled reply


def _first_new_id(tok, base_ids: list, text: str) -> int:
    """Id of the first token that `text` adds after `base_ids` (robust to BPE/sentencepiece merges)."""
    full = tok(text, add_special_tokens=False).input_ids
    k = 0
    while k < min(len(base_ids), len(full)) and base_ids[k] == full[k]:
        k += 1
    return full[k]


@torch.no_grad()
def choose_constrained(tok, model, prompt: str):
    """Greedy choice restricted to the legal answers: a letter A-D, then a digit 1-4.
    Needs no free-text parsing and cannot fail on format (the LLM only ranks the legal options).
    The prompt ends in 'Answer:'; options are ' A'..' D' then '1'..'4'. Returns (letter_index, digit_char)."""
    p = prompt.rstrip(" ")
    base = tok(p, add_special_tokens=False).input_ids
    ids = [_first_new_id(tok, base, p + " " + L) for L in LETTERS]
    x = torch.tensor([base], device="cuda")
    lg = model(input_ids=x, logits_to_keep=1).logits[0, -1]
    li = int(torch.argmax(lg[ids]))
    p2 = p + " " + LETTERS[li]
    base2 = tok(p2, add_special_tokens=False).input_ids
    dig = list(WIDTH)
    ids2 = [_first_new_id(tok, base2, p2 + d) for d in dig]
    lg2 = model(input_ids=torch.tensor([base2], device="cuda"), logits_to_keep=1).logits[0, -1]
    return li, dig[int(torch.argmax(lg2[ids2]))]


def run_agent(name: str, task, items: list, root, seed: int = 0, variant: int = 0, with_ctx: bool = False,
              decode: str = "constrained"):
    """Returns (agent_fc, choices_df, cost). agent_fc has q10/q50/q90 like every other model."""
    cands = load_candidates(task.name, root)
    tok, model = llm.load(name)
    think_off = name.startswith("qwen3")
    rng = np.random.default_rng(seed)
    hist = {o: {u: g for u, g in df.groupby("unique_id", sort=False)} for o, df in task.histories.items()}
    off = pd.tseries.frequencies.to_offset(task.freq)
    h = task.horizon
    names = CAND[task.name]
    frames, rows, n_fb, n_tok = [], [], 0, 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i, (o, u) in enumerate(items):
        g = hist[o][u]
        y = g["y"].to_numpy(float)
        perm = rng.permutation(len(names))
        arrs = [cands[names[j]][(str(o), u)] for j in perm]
        extra = ""
        if with_ctx:
            from src.models.context import context_text
            extra = context_text(task.name, u, [pd.Timestamp(g["ds"].iloc[-1]) + off * k for k in range(1, h + 1)])
        prompt = build_prompt(tok, task, y, arrs, think_off, variant, extra)
        if decode == "constrained":
            li, dg = choose_constrained(tok, model, prompt)
            out, n_tok = f"{LETTERS[li]}{dg}", n_tok + 2
        else:   # free-text generation + regex parse (used for the first agent runs; kept for reference)
            enc = tok([prompt], return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=6, pad_token_id=tok.pad_token_id)
            out = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            n_tok += gen.shape[1] - enc.input_ids.shape[1]
        mt = re.search(r"([A-D])\s*([1-4])", out)
        if mt:
            chosen, w, fb = int(perm[LETTERS.index(mt.group(1))]), WIDTH[mt.group(2)], False
        else:
            chosen, w, fb = names.index(next(n for n in names if n in ("theta", "mstl"))), 1.0, True
            n_fb += 1
        a = cands[names[chosen]][(str(o), u)].copy()
        a[:, 0] = a[:, 1] - w * (a[:, 1] - a[:, 0])
        a[:, 2] = a[:, 1] + w * (a[:, 2] - a[:, 1])
        ds = [pd.Timestamp(g["ds"].iloc[-1]) + off * k for k in range(1, h + 1)]
        frames.append(pd.DataFrame({"unique_id": u, "ds": ds, "origin": o, "q10": a[:, 0], "q50": a[:, 1],
                                    "q90": a[:, 2], "fallback": fb}))
        rows.append(dict(origin=o, unique_id=u, chosen=names[chosen], width=w, fallback=fb, raw=out))
        if (i + 1) % 50 == 0:
            print(f"    agent {name} {task.name}: {i + 1}/{len(items)} [{time.time() - t0:.0f}s]", flush=True)
    sec = time.time() - t0
    cost = {"model": name, "seconds": round(sec, 1), "n_items": len(items), "sec_per_item": round(sec / len(items), 2),
            "generated_tokens_per_item": round(n_tok / len(items), 1), "fallback_rate": round(n_fb / len(items), 3),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "llm_calls_per_item": 1}
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows), cost
