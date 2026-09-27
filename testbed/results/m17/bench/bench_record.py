"""Where the per-request cost of the cache-key adapter goes: the record append on /mnt/c (9P) vs ext4, and the
decision path itself with the record on ext4."""
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/ai_compiler/entail"))
from entail import cache_key_contract, core, load, record  # noqa: E402

N = 200


def bench(label, fn):
    fn()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    print(f"{label:55s} {(time.perf_counter() - t0) / N * 1e3:8.3f} ms/call")


line = '{"pid": 1, "timing": "container:vllm.request.block_hashes", "ms": 0.1}\n'
bench("append on /mnt/c (project folder)", lambda: record._append(os.path.expanduser("~/ai_compiler/entail_logs/_bench.jsonl"), line))
bench("append on ext4 (/tmp)", lambda: record._append("/tmp/_bench.jsonl", line))
os.remove(os.path.expanduser("~/ai_compiler/entail_logs/_bench.jsonl"))

os.environ["ENTAIL_RECORD"] = "/tmp/_bench_record.jsonl"
core.set_mode("load")
present = {"prompt_token_ids": True, "prompt_embeds": False, "prompt_is_token_ids": False, "mm_features": False,
           "lora_request": False, "cache_salt": False}
bench("cache_key_contract.check PASS path, record on ext4",
      lambda: cache_key_contract.check("container:vllm.request.block_hashes", "vllm.block_hashes", "vllm", present,
                                       "vllm Request r", {}, owner=("prompt_token_ids",)))
bench("load.safely(no-op), record on ext4",
      lambda: load.safely("container:vllm.request.block_hashes", "vllm.block_hashes", "Coverage", lambda: None))
os.environ["ENTAIL_RECORD"] = os.path.expanduser("~/ai_compiler/entail_logs/_bench2.jsonl")
bench("load.safely(no-op), record on /mnt/c",
      lambda: load.safely("container:vllm.request.block_hashes", "vllm.block_hashes", "Coverage", lambda: None))
os.remove(os.path.expanduser("~/ai_compiler/entail_logs/_bench2.jsonl"))
