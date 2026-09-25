"""M5.1: what the KV container contract costs per update, apart from a model's run-to-run noise.

  core      kv_contract.grew through guarded, as the transformers adapter calls it: 36 layers x 64 steps, no model
  adapter   Cache.update on a real DynamicCache on the GPU (bf16, 8 KV heads x 128), 36 layers x 64 steps: not
            wrapped, wrapped with the mode off, wrapped with the mode load. The difference is what entail adds.
Medians of 7 runs each. Writes testbed/results/m51/overhead.json. Run in ~/venvs/gpu: python testbed/m51_overhead.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from entail import core, kv_contract  # noqa: E402
from entail.adapters import cache_contract  # noqa: E402

L, STEPS, RUNS = 36, 64, 7


class Owner:
    pass


def core_path():
    owner, length = Owner(), [0] * L
    kv_contract.reset()
    t0 = time.perf_counter()
    for s in range(STEPS):
        added = 5 if s == 0 else 1
        for layer in range(L):
            before = length[layer]
            length[layer] += added
            kv_contract.guarded("container:bench", "bench", kv_contract.grew, "container:bench", "bench", owner,
                                "bench", layer, before, length[layer], added, None)
    return (time.perf_counter() - t0) / (L * STEPS) * 1e6


def on_cache():
    k1 = torch.zeros(1, 8, 1, 128, device="cuda", dtype=torch.bfloat16)
    k5 = torch.zeros(1, 8, 5, 128, device="cuda", dtype=torch.bfloat16)
    cache = DynamicCache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(STEPS):
        k = k5 if s == 0 else k1
        for layer in range(L):
            cache.update(k, k, layer)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (L * STEPS) * 1e6


def median(fn):
    fn()   # warm-up: imports, allocator
    return round(sorted(fn() for _ in range(RUNS))[RUNS // 2], 2)


def main():
    core.set_mode("load")
    res = {"layers": L, "steps": STEPS, "runs": RUNS, "torch": torch.__version__,
           "core_us_per_update": median(core_path)}
    core.set_mode("off")
    res["not_wrapped_us"] = median(on_cache)
    cache_contract.install()
    res["wrapped_off_us"] = median(on_cache)
    core.set_mode("load")
    kv_contract.reset()
    res["wrapped_load_us"] = median(on_cache)
    res["refused"] = kv_contract.stats(cache_contract.BOUNDARY)["refused"]
    core.set_mode("off")
    cache_contract.uninstall()
    res["entail_adds_us"] = round(res["wrapped_load_us"] - res["not_wrapped_us"], 2)
    res["per_decode_of_this_size_ms"] = round(res["entail_adds_us"] * L * STEPS / 1e3, 1)
    print(json.dumps(res), flush=True)
    out = os.path.join(HERE, "results", "m51", "overhead.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
