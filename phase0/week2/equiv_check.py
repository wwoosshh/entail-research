"""Teacher-forced equivalence of sdpa vs triton_decode attention paths (bf16 and int4), eager, static cache."""
import json, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import bench_decode_attn_swap as S
from bench_llm_decode import build_model, quantize_int4
print(S.register_impl())
model, cfg, src, _ = build_model()
res = {"bf16": S.equivalence_check(model, cfg.vocab_size, B=2, prompt_len=64, max_len=128, steps=8)}
print("bf16:", res["bf16"], flush=True)
quantize_int4(model)
res["int4"] = S.equivalence_check(model, cfg.vocab_size, B=2, prompt_len=64, max_len=128, steps=8)
print("int4:", res["int4"], flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "results", "equivalence_teacher_forced.json"), "w"), indent=1)
print("saved")
