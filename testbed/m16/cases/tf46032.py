"""M16 case, huggingface/transformers#46032 (testbed/M16_PROTOCOL.md 5): Mamba2Mixer with use_cache and a cached
state assumes one token per call, so feeding the rest of a sequence in one chunk silently gives results that differ
from token-by-token processing (the CPU path takes only the first token; the parallel path ignores the cached state).
The report's own script: a tiny random Mamba2Config on CPU, no checkpoint. Fixed in 5.13.0 (PR #46084), so this runs
in the transformers 5.12.1 venv.
Run in ~/venvs/tf5121: python testbed/m16/cases/tf46032.py <out.json>
"""
import json
import os
import sys


def main():
    import torch
    import transformers
    from transformers import Mamba2Config
    from transformers.cache_utils import DynamicCache
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Model

    torch.manual_seed(0)
    config = Mamba2Config(hidden_size=64, num_hidden_layers=2, state_size=16, num_heads=8, head_dim=16, n_groups=1,
                          expand=2, conv_kernel=4, chunk_size=8, vocab_size=2, use_bias=True, use_conv_bias=True)
    model = Mamba2Model(config).eval()
    B, L = 1, 8
    inputs_embeds = torch.randn(B, L, config.hidden_size)
    with torch.no_grad():
        cache_tbt = DynamicCache(config=config)
        outs = []
        for t in range(L):
            out = model(inputs_embeds=inputs_embeds[:, t:t + 1, :], cache_params=cache_tbt, use_cache=True)
            outs.append(out.last_hidden_state)
        result_tbt = torch.cat(outs, dim=1)
        cache_chunk = DynamicCache(config=config)
        out_first = model(inputs_embeds=inputs_embeds[:, :1, :], cache_params=cache_chunk, use_cache=True)
        err = None
        try:
            out_rest = model(inputs_embeds=inputs_embeds[:, 1:, :], cache_params=cache_chunk, use_cache=True)
            rest = out_rest.last_hidden_state
        except Exception as e:  # noqa: BLE001
            err, rest = f"{type(e).__name__}: {e}"[:300], None
        full = model(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "tbt_vs_full_max_abs_err": float((result_tbt - full).abs().max()), "chunk_shape": None, "error": err,
           "chunk_vs_tbt_max_abs_err": None}
    if rest is not None:
        row["chunk_shape"] = list(rest.shape)
        if rest.shape == result_tbt[:, 1:, :].shape:
            row["chunk_vs_tbt_max_abs_err"] = float((result_tbt[:, 1:, :] - rest).abs().max())
    row["reproduced"] = bool(err is None and (row["chunk_shape"] != [B, L - 1, config.hidden_size]
                                              or (row["chunk_vs_tbt_max_abs_err"] or 0.0) > 1e-3))
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
