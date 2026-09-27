"""M18.6 replay 3 case, huggingface/transformers#47328 (testbed/M16_PROTOCOL.md 8): Qwen2.5-Omni's Token2Wav DiT
builds its rotary cos/sin half-split and applies them with an interleaved rotate, which is not a rotation: the
attention score of R_m q and R_{m+d} k then depends on m, not only on d. The report's measurement (the spread of
<R_m q, R_{m+5} k> over m; ~0 for a valid RoPE), made here with the library's own objects: the DiT rotary module
built from Qwen2_5OmniDiTConfig and the apply function DiTAttention calls. Reproduced when the spread is not ~0.
Run in a transformers venv (5.12.1 has the defect; the fix is in 5.15.0 and later):
  python testbed/m18/replay3/cases/tf47328.py <out.json>
"""
import json
import os
import sys


def main():
    import torch
    import transformers
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as m
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniDiTConfig

    torch.manual_seed(0)
    cfg = Qwen2_5OmniDiTConfig()
    rope = m.Qwen2_5OmniDiTRotaryEmbedding(cfg)
    apply = m.DiTAttention.forward.__globals__["apply_rotary_pos_emb"]
    dim = int(rope.inv_freq.numel() * 2)
    pos = torch.arange(0, 64)[None, :]
    cos, sin = rope(torch.zeros(1, 64, dim), pos)                     # [1, 64, dim]
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    def score(mq, mk):
        qe, _ = apply(q, q, cos[:, mq:mq + 1], sin[:, mq:mq + 1])
        _, ke = apply(k, k, cos[:, mk:mk + 1], sin[:, mk:mk + 1])
        return float((qe * ke).sum())

    vals = [score(p, p + 5) for p in range(40)]
    spread = max(vals) - min(vals)
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__, "dim": dim,
           "spread": spread, "apply_function": f"{apply.__module__}.{apply.__qualname__}", "reproduced": spread > 1e-3}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
