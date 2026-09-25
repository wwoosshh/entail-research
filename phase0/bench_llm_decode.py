"""Workload B: single-token decode step of a ~4B LLM (Qwen3-4B architecture, random weights).

Modes:
  eager            : DynamicCache, plain PyTorch
  compile_default  : StaticCache + torch.compile(decode step)
  compile_graphs   : StaticCache + torch.compile(mode="reduce-overhead") = CUDA graphs
Optionally repeats with torchao int4 weight-only quantization (the realistic consumer setting).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import env_info, run_mode, save_results  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B"


def build_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        src = "hub-config"
    except Exception as e:
        from transformers import LlamaConfig
        cfg = LlamaConfig(vocab_size=151936, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                          num_attention_heads=32, num_key_value_heads=8, max_position_embeddings=40960,
                          rms_norm_eps=1e-6, tie_word_embeddings=True)
        src = f"fallback-llama-config ({type(e).__name__})"
    torch.manual_seed(0)
    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(torch.bfloat16).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LLM built from {src}: {n_params/1e9:.2f}B params, {time.perf_counter()-t0:.1f}s", flush=True)
    return model, cfg, src, n_params


def make_static_cache(model, B, max_len):
    from transformers import StaticCache
    try:
        return StaticCache(config=model.config, max_cache_len=max_len)
    except TypeError:
        return StaticCache(config=model.config, max_batch_size=B, max_cache_len=max_len,
                           device="cuda", dtype=torch.bfloat16)


def eager_step_factory(model, B, prompt_len, vocab):
    ids = torch.randint(0, vocab, (B, prompt_len), device="cuda")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    state = {"pkv": out.past_key_values, "tok": out.logits[:, -1].argmax(-1, keepdim=True)}

    def step():
        o = model(state["tok"], past_key_values=state["pkv"], use_cache=True)
        state["pkv"] = o.past_key_values
        state["tok"] = o.logits[:, -1].argmax(-1, keepdim=True)
    return step


def compiled_step_factory(model, B, prompt_len, vocab, mode, max_len):
    ids = torch.randint(0, vocab, (B, prompt_len), device="cuda")
    cache = make_static_cache(model, B, max_len)
    with torch.no_grad():
        logits = model(ids, cache_position=torch.arange(prompt_len, device="cuda"),
                       past_key_values=cache, use_cache=True, return_dict=False)[0]
    state = {"tok": logits[:, -1].argmax(-1, keepdim=True),
             "pos": torch.tensor([prompt_len], device="cuda")}

    def decode_one(tok, pos):
        lg = model(tok, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)[0]
        return lg[:, -1].argmax(-1, keepdim=True)

    cstep = torch.compile(decode_one, mode=("reduce-overhead" if mode == "compile_graphs" else "default"))

    def step():
        state["tok"] = cstep(state["tok"], state["pos"]).clone()
        state["pos"] = state["pos"] + 1
    return step


def quantize_int4(model):
    """torchao int4 weight-only. torchao 0.18's default packing format needs the external 'mslk'
    kernel library; 'tile_packed_to_4d' uses PyTorch's built-in tinygemm (_weight_int4pack_mm),
    which works on sm80+ and was verified on this sm_89 card (see env/logs, torchao_int4_probe)."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    quantize_(model, Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d"))
    return model


def sweep(model, tag, cfg, args, results):
    vocab = cfg.vocab_size
    max_len = args.prompt_len + 128
    for mode in args.modes:
        torch._dynamo.reset()
        for B in args.batches:
            def make_fn(mode=mode, B=B):
                if mode == "eager":
                    return eager_step_factory(model, B, args.prompt_len, vocab)
                return compiled_step_factory(model, B, args.prompt_len, vocab, mode, max_len)
            results.append(run_mode(
                f"llm_decode{tag}/{mode}/B{B}", make_fn, iters=32, profile_steps=4,
                extra={"workload": "llm_decode" + tag, "mode": mode, "batch": B,
                       "prompt_len": args.prompt_len, "dtype": "bf16" if not tag else "int4wo+bf16"},
            ))
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--modes", nargs="+", default=["eager", "compile_default", "compile_graphs"])
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--int4", action="store_true", help="also run torchao int4 weight-only variant")
    ap.add_argument("--skip-bf16", action="store_true", help="skip the bf16 sweep (int4-only rerun)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "llm_decode.json"))
    args = ap.parse_args()

    model, cfg, src, n_params = build_model()
    results = []
    if not args.skip_bf16:
        sweep(model, "", cfg, args, results)
    quant_status = "skipped"
    if args.int4:
        torch._dynamo.reset()
        try:
            t0 = time.perf_counter()
            quantize_int4(model)
            torch.cuda.synchronize()
            quant_status = f"ok ({time.perf_counter()-t0:.1f}s)"
            print("int4 quantization", quant_status, flush=True)
            sweep(model, "_int4", cfg, args, results)
        except Exception as e:
            quant_status = f"error: {type(e).__name__}: {str(e)[:300]}"
            print("int4 quantization", quant_status, flush=True)
    save_results(args.out, {"env": env_info(), "model_source": src, "model_id": MODEL_ID, "params": n_params,
                            "int4": quant_status, "results": results})


if __name__ == "__main__":
    main()
