import time, torch
from src.models import llm
tok, model = llm.load("qwen3_8b")
print("loaded GB", torch.cuda.memory_allocated()/1e9, flush=True)
p = llm._prompt(tok, "wiki", list(range(112)), 28, True)
enc = tok([p], return_tensors="pt", add_special_tokens=False).to("cuda")
print("prompt tokens", enc.input_ids.shape[1], flush=True)
for sc in (8,):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.time()
    g = model.generate(**enc, do_sample=True, temperature=.8, top_p=.95, num_return_sequences=sc,
                       max_new_tokens=100, min_new_tokens=100, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt=time.time()-t
    print(f"sc={sc}: {dt:.1f}s for 100 steps = {dt/100*1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

p = llm._prompt(tok, "eia", list(range(168)), 168, True)
enc = tok([p], return_tensors="pt", add_special_tokens=False).to("cuda")
print("eia prompt tokens", enc.input_ids.shape[1], flush=True)
torch.cuda.reset_peak_memory_stats(); t=time.time()
g = model.generate(**enc, do_sample=True, temperature=.8, top_p=.95, num_return_sequences=4,
                   max_new_tokens=300, min_new_tokens=300, pad_token_id=tok.pad_token_id)
torch.cuda.synchronize(); dt=time.time()-t
print(f"eia sc=4: {dt:.1f}s for 300 steps = {dt/300*1000:.0f} ms/step, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
