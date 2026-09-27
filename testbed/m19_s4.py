"""M19 L4, S4 with the whole library: one process per state, vLLM 0.30.0 with its default settings (CUDA graphs,
torch.compile), Qwen3-4B bf16, offline LLM.generate with the engine core in this process
(VLLM_ENABLE_V1_MULTIPROCESSING=0, the harder case: host time is not overlapped by a separate engine process).

m55_graph.py switches one adapter (the per-step KV contract) inside one process; every other adapter stayed installed
in both of its states. Since M19 L3 some adapters sit on per-call paths (the Triton launch hook, the wrapped
functions), so this measures what a user gets: the state is chosen per process by testbed/m19_s4.sh -
  on       ENTAIL=load and the start-up hook (every adapter, as installed)
  off      no hook, entail not imported
  control  the same as off, timed as its own state: how far two identical states differ
Per batch size (1, 8, 32): 64-token prompts, 128 new tokens, ignore_eos, greedy; WARM warm-up runs, then REPS timed
runs; the process writes its seconds and tokens to M19_S4_OUT/<state>_<round>.json. m19_s4_summary.py takes the
median per process and the median over rounds of on/off and control/off.
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
MODEL = os.path.expanduser("~/models/Qwen3-4B")
BATCHES = (1, 8, 32)
PROMPT, NEW, WARM, REPS = 64, 128, 2, 5


def main():
    import torch
    from vllm import LLM, SamplingParams

    state, rnd, out_dir = os.environ["M19_S4_STATE"], os.environ["M19_S4_ROUND"], os.environ["M19_S4_OUT"]
    entail_on = "entail" in sys.modules
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    load = time.perf_counter() - t0
    cfg = llm.llm_engine.vllm_config
    res = {"state": state, "round": rnd, "entail_imported": entail_on, "load_seconds": load,
           "vllm": __import__("vllm").__version__, "enforce_eager": cfg.model_config.enforce_eager,
           "cudagraph_mode": str(getattr(cfg.compilation_config, "cudagraph_mode", None)),
           "async_scheduling": getattr(cfg.scheduler_config, "async_scheduling", None), "batches": {}}
    if entail_on:
        from entail import core
        res["entail_mode"] = core.mode()
    params = SamplingParams(max_tokens=NEW, ignore_eos=True, temperature=0)

    def run(prompts):
        torch.cuda.synchronize()
        t = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t, [list(o.outputs[0].token_ids) for o in outs]

    for b in BATCHES:
        prompts = [{"prompt_token_ids": [(1000 + 37 * i + 11 * j) % 30000 for j in range(PROMPT)]} for i in range(b)]
        for _ in range(WARM):
            _, toks = run(prompts)
        secs = [run(prompts)[0] for _ in range(REPS)]
        res["batches"][str(b)] = {"seconds": secs, "median": statistics.median(secs), "tokens": toks}
        print(f"{state} r{rnd} B={b:3d} median {statistics.median(secs):.4f}s", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{state}_{rnd}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
