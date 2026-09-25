"""M8.3: the speed of Qwen3-4B's decode step written with entail's front end, against the week-4 paths.

ROADMAP M8.3 asks for the narrow path within 2% of week 4 (phase0/week4/WEEK4_NOTES.md 6: int4 batch 8, one decode
step at position 512, FlexAttention 9.25 ms, the hand kernel 9.39 ms). The week-4 harness is used as it is
(phase0/week3/graph_ab.py, ab_common.py): each variant is compiled with torch.compile (mode default), captured into
its own CUDA graph, checked against an eager reference, and the graphs are replayed interleaved (20 rounds x 10
steps). The same session measures, on the same model and cache:
  week4_triton   transformers' step with the hand kernel (attention "triton_decode", phase0/week2)
  week4_flex     transformers' step with FlexAttention (attention "maskmod_decode", block mask built once)
  entail_triton  the front-end program (entail/frontend/qwen3.py), attention lowered to the hand kernel
  entail_flex    the same program, attention lowered to FlexAttention (its block mask made by arithmetic, in the graph)
and the GPU kernel time of one replay of each, by kind (week 4's kernel_breakdown).
Writes testbed/results/m83/speed.json. Run in ~/venvs/gpu: python testbed/m83_speed.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHASE0 = os.path.join(ROOT, "phase0")
sys.path[:0] = [os.path.join(ROOT, "entail"), os.path.join(PHASE0, "week3"), PHASE0, os.path.join(PHASE0, "week2")]

import torch  # noqa: E402

import ab_attention as A  # noqa: E402  (registers maskmod_decode; its block mask)
import bench_decode_attn_swap as S  # noqa: E402  (registers triton_decode; VALID)
from ab_common import sync  # noqa: E402
from bench_llm_decode import make_static_cache, quantize_int4  # noqa: E402
from graph_ab import build_and_check, capture, time_graphs  # noqa: E402

sys.path.insert(0, os.path.join(PHASE0, "week4"))
from e4_fact_ablation import kernel_breakdown  # noqa: E402  (GPU kernel time of one replay, by kind)

from entail.frontend import qwen3  # noqa: E402

MODEL = os.path.expanduser("~/models/Qwen3-4B")
OUT = os.path.join(HERE, "results", "m83", "speed.json")
B, L = 8, 512
MAX_LEN = L + 128
ROUNDS, STEPS, WARMUP_S = 20, 10, 3.0


def main():
    import transformers
    from transformers import AutoModelForCausalLM

    transformers.utils.logging.disable_progress_bar()
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__,
           "transformers": transformers.__version__, "batch": B, "position": L, "slots": MAX_LEN,
           "rounds": ROUNDS, "steps": STEPS, "week4_ms": {"week4_flex": 9.25, "week4_triton": 9.39}}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    quantize_int4(model)
    S.register_impl()
    A.register_maskmod()
    g = torch.Generator(device="cuda").manual_seed(1000 + B)
    ids = torch.randint(0, model.config.vocab_size, (B, L), device="cuda", generator=g)
    S.set_impl(model, "sdpa")
    cache = make_static_cache(model, B, MAX_LEN)
    with torch.no_grad():
        lg = model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache, use_cache=True,
                   return_dict=False)[0]
    tok = lg[:, -1].argmax(-1, keepdim=True).contiguous()
    pos = torch.tensor([L], device="cuda")
    S.VALID.zero_()
    S.VALID[:B].fill_(L + 1)
    A.FLEX["bm"] = A.build_block_mask(B, MAX_LEN)
    lens = [layer.cumulative_length for layer in cache.layers]

    def rewind():   # transformers 5.x writes at its own counter; every step decodes position L again
        torch._foreach_zero_(lens)
        torch._foreach_add_(lens, L)

    week4 = {"week4_triton": "triton_decode", "week4_flex": "maskmod_decode"}
    eager = {}
    for name, impl in week4.items():
        S.set_impl(model, impl)
        rewind()
        with torch.no_grad():
            eager[name] = model(tok, cache_position=pos, past_key_values=cache, use_cache=True,
                                return_dict=False)[0][:, -1].float().clone()

    def decode_one(t, p):
        return model(t, cache_position=p, past_key_values=cache, use_cache=True, return_dict=False)[0][:, -1]

    compiled = torch.compile(decode_one)
    setups = {name: (lambda impl=impl: S.set_impl(model, impl)) for name, impl in week4.items()}
    graphs, res["variants"] = build_and_check(setups, compiled, (tok, pos), eager, "week4_triton",
                                              lambda n, names: True, pre=rewind)
    positions = torch.tensor([L], device="cuda")
    until = torch.full((B,), L, device="cuda", dtype=torch.int64)
    for name, backend in (("entail_triton", "triton"), ("entail_flex", "flex")):
        v = {}
        program = qwen3.trace_decode(model.config, B, MAX_LEN, attention=backend, quantized=True, model_path=MODEL)
        values = qwen3.tensors(model, cache, tok, positions, until)
        qwen3.check_layouts(program, values)
        step = program.bind(**values)
        fn = torch.compile(lambda step=step: step()["logits"])
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        sync()
        v["compile_s"] = round(time.perf_counter() - t0, 1)
        graph, out = capture(fn)
        graph.replay()
        sync()
        o = out.float().clone()
        ref = eager["week4_" + backend]
        v["max_abs_diff_vs_week4_eager_same_kernel"] = (o - ref).abs().max().item()
        v["mean_abs_diff_vs_week4_eager_same_kernel"] = (o - ref).abs().mean().item()
        v["same_next_token"] = int((o.argmax(-1) == ref.argmax(-1)).sum().item())
        v["nodes"] = len(program.graph.nodes)
        graphs[name] = graph
        res["variants"][name] = v
        print(f"    {name}: {v}", flush=True)
    res["timing"] = time_graphs(graphs, "week4_triton", ROUNDS, STEPS, WARMUP_S)
    timing = res["timing"]["variants"]
    for name, rec in timing.items():
        print(f"  {name}: median {rec['median_ms']:.3f} ms" +
              (f", x{rec['ratio_vs_base_median']:.4f} vs week4_triton" if "ratio_vs_base_median" in rec else ""),
              flush=True)
    t = {n: r["median_ms"] for n, r in timing.items()}
    res["kernels_ms"] = {name: kernel_breakdown(graph.replay) for name, graph in graphs.items()}
    for name, k in res["kernels_ms"].items():
        print(f"  kernels {name}: {k}", flush=True)
    res["entail_vs_week4_same_session"] = {"triton": t["entail_triton"] / t["week4_triton"],
                                           "flex": t["entail_flex"] / t["week4_flex"]}
    res["entail_vs_week4_notes"] = {"triton": t["entail_triton"] / 9.39, "flex": t["entail_flex"] / 9.25}
    print(json.dumps({k: res[k] for k in ("entail_vs_week4_same_session", "entail_vs_week4_notes")}), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
