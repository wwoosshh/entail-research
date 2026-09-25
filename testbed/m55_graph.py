"""M5.5, S4 on the CUDA graph path: the always-on mode against off, vLLM 0.30.0 with its default settings (CUDA graphs,
torch.compile), Qwen3-4B bf16, offline LLM.generate.

What runs per step with ENTAIL=load in vLLM is the KV contract at KVCacheManager.allocate_slots (adapters/
vllm_cache_contract.py, host side, once per request per step); the other vLLM adapters decide while the model is
built. One engine, in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so the check can be switched between runs:
  on       the adapter installed (as sitecustomize put it), mode load
  off      the adapter uninstalled (allocate_slots is vLLM's own again), mode off
  control  the same as off, timed as a separate state: how far two identical states differ
The engine core in this process is the harder case for the check: its host time is not overlapped by a separate
engine process. Per batch size (1, 8, 32): 64-token prompts, 128 new tokens, ignore_eos, greedy; two warm-up runs per
state, then ROUNDS rounds, each timing on, off and control in a rotated order. Reported: the median over rounds of
on/off and control/off, the checks made while on, and whether the tokens on and off are the same.
vLLM's default schedules asynchronously, which can hide host time behind the GPU; M55_SYNC=1 turns that off (the
harder case) and writes graph_sync.json instead.
Writes testbed/results/m55/graph.json. Run: testbed/m55_graph.sh (sets ENTAIL=load and the start-up hook).
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
MODEL = os.path.expanduser("~/models/Qwen3-4B")
BATCHES = (1, 8, 32)
PROMPT, NEW, ROUNDS, WARM = 64, 128, 8, 2


def main():
    import torch
    from vllm import LLM, SamplingParams

    from entail import core, kv_contract
    from entail.adapters import vllm_cache_contract as cc

    assert core.mode() == "load", "run with ENTAIL=load and the start-up hook (m55_graph.sh)"
    sync = os.environ.get("M55_SYNC") == "1"
    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.85, seed=0,
              **({"async_scheduling": False} if sync else {}))
    cfg = llm.llm_engine.vllm_config
    res = {"vllm": __import__("vllm").__version__, "torch": torch.__version__,
           "enforce_eager": cfg.model_config.enforce_eager,
           "cudagraph_mode": str(getattr(cfg.compilation_config, "cudagraph_mode", None)),
           "compilation_mode": str(getattr(cfg.compilation_config, "mode", getattr(cfg.compilation_config, "level",
                                                                                     None))),
           "async_scheduling": getattr(cfg.scheduler_config, "async_scheduling", None),
           "multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
           "adapter_installed_at_start": cc._ORIG is not None,
           "prompt_tokens": PROMPT, "new_tokens": NEW, "rounds": ROUNDS, "batches": {}}
    params = SamplingParams(max_tokens=NEW, ignore_eos=True, temperature=0)

    def state(name):
        if name == "on":
            cc.install()
            core.set_mode("load")
        else:
            cc.uninstall()
            core.set_mode("off")

    def run(prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, [list(o.outputs[0].token_ids) for o in outs]

    names = ("on", "off", "control")
    for b in BATCHES:
        prompts = [{"prompt_token_ids": [(1000 + 37 * i + 11 * j) % 30000 for j in range(PROMPT)]} for i in range(b)]
        tokens = {}
        for name in names:
            state(name)
            for _ in range(WARM):
                _, tokens[name] = run(prompts)
        times = {n: [] for n in names}
        cc.reset()
        for r in range(ROUNDS):
            for name in names[r % 3:] + names[:r % 3]:
                state(name)
                t, toks = run(prompts)
                times[name].append(t)
                assert toks == tokens[name]
        st = cc.stats()
        ratio = [on / off for on, off in zip(times["on"], times["off"])]
        control = [c / off for c, off in zip(times["control"], times["off"])]
        res["batches"][str(b)] = {
            "seconds": times, "on_over_off": ratio, "control_over_off": control,
            "median_on_over_off": statistics.median(ratio), "median_control_over_off": statistics.median(control),
            "checks_while_on": st.get("checks"), "broken_while_on": st.get("broken"),
            "refused_while_on": st.get("refused"), "same_tokens_on_off": tokens["on"] == tokens["off"]}
        print(f"B={b:3d}  on/off {statistics.median(ratio):.4f}  control/off {statistics.median(control):.4f}  "
              f"off {statistics.median(times['off']):.3f}s  checks {st.get('checks')}  "
              f"same tokens {tokens['on'] == tokens['off']}", flush=True)
    state("off")
    res["kv_contract_stats"] = {k: v for k, v in kv_contract.stats(cc.BOUNDARY).items()} \
        if hasattr(kv_contract, "stats") else None
    out = os.path.join(RESULTS, "m55", "graph_sync.json" if sync else "graph.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print({k: res[k] for k in ("enforce_eager", "cudagraph_mode", "compilation_mode", "async_scheduling",
                               "adapter_installed_at_start")})
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
