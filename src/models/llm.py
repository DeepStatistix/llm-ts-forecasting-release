"""Prompted open LLM forecasting (LLMTime-style numeric text), 4-bit, sampled quantiles.

For each (origin, series): rescale the last `ctx` values to integers (context 5th..95th
percentile -> 0..100), write them as "v1, v2, ...", ask the model to continue with exactly h
integers, draw N samples, parse, invert the rescaling, and take the 10/50/90th percentiles per
step. Samples that cannot be parsed to >= h integers are invalid; with < MIN_VALID valid samples
the forecast falls back to the last observed value (naive) and is flagged `fallback=True`.
No weights are trained; nothing from the future is in the prompt.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", r"E:\hf_cache")

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = {"qwen3_8b": "Qwen/Qwen3-8B", "gemma4_e4b": "google/gemma-4-E4B-it",
        "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen35_9b": "Qwen/Qwen3.5-9B",
        "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3"}
N_SAMPLES, MIN_VALID, TEMP, TOP_P = 8, 3, 0.8, 0.95
# context length used in the prompt, prompts per generate call, evaluation-subset size
# bs=1: the 4-bit 8B model already uses ~7.5 of the 8 GB, batching spills into shared memory
# (89 s/item instead of ~5 s). sc = samples per generate call (EIA's long output needs chunking).
CFG = {"flu": dict(ctx=104, bs=1, sc=8, n_eval=396), "fred": dict(ctx=60, bs=1, sc=8, n_eval=300),
       "wiki": dict(ctx=112, bs=1, sc=8, n_eval=120), "eia": dict(ctx=168, bs=1, sc=4, n_eval=16)}
FREQ_TXT = {"flu": "weekly", "fred": "monthly", "wiki": "daily", "eia": "hourly"}
_M: dict = {}


def is_llm(name: str) -> bool:
    return name in REPO


def load(name: str):
    if name in _M:
        return _M[name]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(REPO[name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(REPO[name], quantization_config=q,
                                                     device_map="cuda", dtype=torch.bfloat16)
    except (ValueError, KeyError):   # multimodal checkpoints (e.g. Gemma 4)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(REPO[name], quantization_config=q,
                                                            device_map="cuda", dtype=torch.bfloat16)
    model.eval()
    _M[name] = (tok, model)
    return _M[name]


def subset(task, n: int, seed: int = 0) -> list[tuple]:
    """Deterministic random subset of (origin, unique_id) shared by all models in the LLM table."""
    keys = sorted({(o, u) for o, h in task.histories.items() for u in h.unique_id.unique()},
                  key=lambda k: (str(k[0]), str(k[1])))
    if n >= len(keys):
        return keys
    idx = np.random.default_rng(seed).choice(len(keys), n, replace=False)
    return [keys[i] for i in sorted(idx)]


def _scale(y: np.ndarray):
    lo, hi = np.percentile(y, 5), np.percentile(y, 95)
    if hi - lo < 1e-9:
        sd = y.std()
        lo, hi = y.mean() - 2 * sd, y.mean() + 2 * sd
    if hi - lo < 1e-9:
        lo, hi = y.mean() - 1.0, y.mean() + 1.0
    return float(lo), float(hi - lo)


def _prompt(tok, task_name: str, ints: np.ndarray, h: int, qwen_think_off: bool, extra: str = "") -> str:
    text = (f"You are a time series forecasting model. The following is a {FREQ_TXT[task_name]} "
            f"time series, rescaled to integers. Continue it with the next {h} values. "
            f"Output exactly {h} integers separated by commas and nothing else.\n\n"
            + (f"Context: {extra}\n\n" if extra else "") +
            f"Series: {', '.join(map(str, ints))}")
    kw = {"enable_thinking": False} if qwen_think_off else {}
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, **kw)


def _parse(s: str, h: int):
    v = [int(x) for x in re.findall(r"-?\d+", s)]
    return v[:h] if len(v) >= h else None


def forecast_items(name: str, task, items: list[tuple], out_path: Path | None = None, with_ctx: bool = False,
                   seed: int | None = None):
    """Forecast the given (origin, uid) items. Resumable via `out_path` parts. Returns (df, cost).
    `seed` fixes the sampling RNG (the main runs were unseeded; seed variance runs use seeds 1, 2)."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cfg, h = CFG[task.name], task.horizon
    tok, model = load(name)
    think_off = name.startswith("qwen3")
    off = pd.tseries.frequencies.to_offset(task.freq)
    done = pd.read_parquet(out_path) if out_path is not None and out_path.exists() else None
    done_keys = set() if done is None else set(zip(done.origin.astype(str), done.unique_id.astype(str)))
    todo = [k for k in items if (str(k[0]), str(k[1])) not in done_keys]
    sc = cfg["sc"]
    hist = {o: {u: g for u, g in df.groupby("unique_id", sort=False)} for o, df in task.histories.items()}
    torch.cuda.reset_peak_memory_stats()
    t0, n_valid, n_fb, n_tok = time.time(), 0, 0, 0
    frames = [] if done is None else [done]
    new = []
    for i in range(0, len(todo), cfg["bs"]):
        chunk = todo[i:i + cfg["bs"]]
        meta, prompts = [], []
        for o, u in chunk:
            g = hist[o][u]
            y = g["y"].to_numpy(float)[-cfg["ctx"]:]
            lo, rng = _scale(y)
            ints = np.rint((y - lo) / rng * 100).astype(int)
            extra = ""
            if with_ctx:   # text context: description + calendar of the forecast window (origin-time info only)
                from src.models.context import context_text
                extra = context_text(task.name, u, [pd.Timestamp(g["ds"].iloc[-1]) + off * k for k in range(1, h + 1)])
            prompts.append(_prompt(tok, task.name, ints, h, think_off, extra))
            meta.append((o, u, g["ds"].iloc[-1], lo, rng, y[-1]))
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        texts = []
        while len(texts) < N_SAMPLES:
            k = min(sc, N_SAMPLES - len(texts))
            try:
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P,
                                         num_return_sequences=k, max_new_tokens=h * 5 + 12,
                                         pad_token_id=tok.pad_token_id)
            except (torch.OutOfMemoryError, RuntimeError) as e:   # near the 8 GB limit: halve chunk
                if "out of memory" not in str(e).lower() or sc == 1:
                    raise
                sc = max(1, sc // 2)
                torch.cuda.empty_cache()
                print(f"    OOM -> samples per call reduced to {sc}", flush=True)
                continue
            texts += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            n_tok += int((gen[:, enc.input_ids.shape[1]:] != tok.pad_token_id).sum())
        for j, (o, u, last, lo, rng, y_last) in enumerate(meta):
            samples = [p for p in (_parse(t, h) for t in texts[j * N_SAMPLES:(j + 1) * N_SAMPLES])
                       if p is not None]
            n_valid += len(samples)
            ds = [pd.Timestamp(last) + off * k for k in range(1, h + 1)]
            if len(samples) >= MIN_VALID:
                arr = lo + np.array(samples, float) / 100 * rng
                q = np.percentile(arr, [10, 50, 90], axis=0)
                fb = False
            else:
                q = np.tile(y_last, (3, h)).astype(float)
                fb = True
                n_fb += 1
            new.append(pd.DataFrame({"unique_id": u, "ds": ds, "origin": o, "q10": q[0], "q50": q[1],
                                     "q90": q[2], "fallback": fb, "n_valid": len(samples)}))
        if out_path is not None and (len(new) >= 20 or i + cfg["bs"] >= len(todo)):
            frames.append(pd.concat(new, ignore_index=True)); new = []
            pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
            print(f"    {name} {task.name}: {i + len(chunk)}/{len(todo)} "
                  f"[{time.time() - t0:.0f}s] fallback so far {n_fb}", flush=True)
    if new:
        frames.append(pd.concat(new, ignore_index=True))
    fc = pd.concat(frames, ignore_index=True)
    sec = time.time() - t0
    cost = {"model": name, "seconds": round(sec, 1), "n_items_run": len(todo),
            "n_total": len(items), "valid_sample_rate": round(n_valid / max(len(todo) * N_SAMPLES, 1), 3),
            "fallback_rate": round(n_fb / max(len(todo), 1), 3), "generated_tokens": n_tok,
            "sec_per_item": round(sec / max(len(todo), 1), 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "n_samples": N_SAMPLES, "temperature": TEMP}
    return fc, cost
