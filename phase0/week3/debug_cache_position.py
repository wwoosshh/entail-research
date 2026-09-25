"""Does the model honour the cache_position we pass, or does the StaticCache advance its own counter?

Prefill L tokens, then call the decode step 3 times with the SAME cache_position=L and report after each call:
which cache slots of layer 0 are non-zero, cache.get_seq_length(), and the cache_position / position values
the first attention layer actually received.
"""
import inspect
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.dirname(HERE)
sys.path[:0] = [HERE, PHASE0, os.path.join(PHASE0, "week2")]
import bench_decode_attn_swap as S  # noqa: E402
from bench_llm_decode import make_static_cache  # noqa: E402
from numerics_experiment import load  # noqa: E402


def layer0_keys(cache):
    return cache.layers[0].keys if hasattr(cache, "layers") else cache.key_cache[0]


def main():
    import transformers
    print("transformers", transformers.__version__, flush=True)
    _, model = load()
    fwd_params = list(inspect.signature(model.forward).parameters)
    print("model.forward parameters:", fwd_params, flush=True)
    seen = {}

    def pre_hook(mod, args, kwargs):
        seen["kwargs"] = sorted(kwargs)
        cp = kwargs.get("cache_position")
        seen["cache_position"] = None if cp is None else cp.tolist()
        pe = kwargs.get("position_embeddings")
        if pe is not None:
            seen["cos_row0_first4"] = [round(x, 4) for x in pe[0][0, -1, :4].float().tolist()]

    model.model.layers[0].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    S.set_impl(model, "sdpa")
    L, max_len = 8, 16
    cache = make_static_cache(model, 1, max_len)
    ids = torch.randint(0, model.config.vocab_size, (1, L), device="cuda")
    with torch.no_grad():
        model(ids, cache_position=torch.arange(L, device="cuda"), past_key_values=cache, use_cache=True)
    k = layer0_keys(cache)
    print(f"after prefill: non-zero slots {k[0, 0].abs().sum(-1).ne(0).nonzero().flatten().tolist()} "
          f"| get_seq_length {cache.get_seq_length()} | seen cache_position {seen.get('cache_position')}", flush=True)
    tok = torch.tensor([[42]], device="cuda")
    for i in range(3):
        with torch.no_grad():
            model(tok, cache_position=torch.tensor([L], device="cuda"), past_key_values=cache, use_cache=True)
        k = layer0_keys(cache)
        print(f"decode call {i} with cache_position=[{L}]: non-zero slots "
              f"{k[0, 0].abs().sum(-1).ne(0).nonzero().flatten().tolist()} | get_seq_length {cache.get_seq_length()} "
              f"| layer-0 saw cache_position {seen.get('cache_position')} | rope cos[:4] {seen.get('cos_row0_first4')}",
              flush=True)
    print("attention kwargs keys:", seen.get("kwargs"), flush=True)
    attrs = {a: getattr(cache.layers[0], a) for a in dir(cache.layers[0])
             if not a.startswith("__") and not callable(getattr(cache.layers[0], a))
             and not isinstance(getattr(cache.layers[0], a), torch.Tensor)} if hasattr(cache, "layers") else {}
    print("static layer non-tensor attributes:", attrs, flush=True)


if __name__ == "__main__":
    main()
