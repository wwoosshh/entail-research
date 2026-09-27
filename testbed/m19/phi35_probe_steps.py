"""M19 L4: which of the path check's probe requests kills SGLang 0.5.20 on Phi-3.5-mini-instruct (CUDA illegal memory
access)? The requests of entail/adapters/sglang_paths.read_choice sent one at a time with entail off, the engine built as
the E2 harness builds it. Research tool. Run in ~/venvs/sglang with PYTHONPATH=<entail> (only path_contract's texts).
python testbed/m19/phi35_probe_steps.py <model> [steps: alone,forced,batched,cache,forced_plain,input_lp]"""
import sys
import time

import sglang as sgl
from entail import path_contract as pc


def main():
    model = sys.argv[1]
    steps = (sys.argv[2] if len(sys.argv) > 2 else "alone,forced,batched,cache").split(",")
    eng = sgl.Engine(model_path=model, mem_fraction_static=0.8, context_length=2048, log_level="error",
                     disable_cuda_graph=True, disable_radix_cache=True, random_seed=0)
    sp = {"temperature": 0.0, "max_new_tokens": pc.PROBE_TOKENS, "ignore_eos": True}
    lp = {"return_logprob": True, "top_logprobs_num": pc.TOP}
    tok = eng.tokenizer_manager.tokenizer
    ids = {p["id"]: list(tok.encode(p["text"])) for p in pc.PROBES}
    print("prompt lengths", {k: len(v) for k, v in ids.items()}, flush=True)


    def step(name, fn):
        t = time.perf_counter()
        try:
            out = fn()
            print(f"OK   {name} ({time.perf_counter() - t:.2f}s)", flush=True)
            return out
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            raise SystemExit(1)


    gen = {}
    for s in steps:
        if s.startswith("len:"):
            n = int(s[4:])
            step(f"plain generate, first {n} tokens of story_a",
                 lambda: eng.generate(input_ids=ids["story_a"][:n], sampling_params=sp))
        if s == "text":
            step("plain generate, story_a as text", lambda: eng.generate(pc.PROBES[0]["text"], sampling_params=sp))
        if s == "plain":
            step("plain generate, no logprobs", lambda: eng.generate(input_ids=ids["story_a"], sampling_params=sp))
        if s == "alone":
            for p in pc.PROBES:
                step(f"flush before {p['id']}", eng.flush_cache)
                o = step(f"alone {p['id']} (return_logprob, top {pc.TOP})",
                         lambda: eng.generate(input_ids=ids[p["id"]], sampling_params=sp, **lp))
                gen[p["id"]] = [int(t) for _, t, *_ in o["meta_info"]["output_token_logprobs"]]
        if s == "input_lp":
            n0 = len(ids["story_a"])
            step("input logprobs from n0-1, no generated tail",
                 lambda: eng.generate(input_ids=ids["story_a"], sampling_params=dict(sp, max_new_tokens=1),
                                      logprob_start_len=n0 - 1, **lp))
        if s == "forced":
            for pid, toks in gen.items():
                step(f"flush before forced {pid}", eng.flush_cache)
                n0 = len(ids[pid])
                step(f"forced {pid} (prompt+{len(toks)} generated, logprob_start_len={n0 - 1})",
                     lambda: eng.generate(input_ids=ids[pid] + toks, sampling_params=dict(sp, max_new_tokens=1),
                                          logprob_start_len=n0 - 1, **lp))
        if s == "batched":
            step("flush before batch", eng.flush_cache)
            step("batched all probes", lambda: eng.generate(input_ids=[ids[p["id"]] for p in pc.PROBES],
                                                            sampling_params=sp, **lp))
        if s == "cache":
            step("cache target", lambda: eng.generate(input_ids=ids[pc.TARGET], sampling_params=sp, **lp))
            step("flush at end", eng.flush_cache)
    print("DONE", flush=True)
    eng.shutdown()


if __name__ == "__main__":
    main()
