"""M17.6 case, huggingface/transformers#47475 (testbed/M16_PROTOCOL.md 7): in Zamba2's pure-PyTorch path a token
at position P changes the logits at positions before P (the inter-chunk state recurrence reduces over the wrong
axis), which violates causality. The report's script, unchanged: Zamba2-1.2B in float32 on CPU, a random
512-token sequence, the token at 400 changed, and the logits of positions 0..399 compared. Reproduced when they
differ (the report: clean up to 255, contaminated from 256).
Run in ~/venvs/gpu (CPU, transformers 5.17.0): python testbed/m17/replay2/cases/tf47475.py <out.json>
"""
import json
import os
import sys


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-1.2B", dtype=torch.float32).eval()
    torch.manual_seed(0)                 # seed after loading (loading consumes RNG)
    T, P = 512, 400
    ids = torch.randint(5, 1000, (1, T))
    ids2 = ids.clone()
    ids2[0, P] = (ids2[0, P] + 7) % 1000 + 5
    with torch.no_grad():
        a = m(ids, use_cache=False).logits
        b = m(ids2, use_cache=False).logits
    diff = (a[:, :P] - b[:, :P]).abs().max(dim=-1).values[0]
    per_pos = {str(p): float(diff[p]) for p in (0, 100, 200, 255, 256, 300, 399)}
    first_bad = next((p for p in range(P) if diff[p] > 1e-6), None)
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "model": "Zyphra/Zamba2-1.2B", "mamba_kernels": bool(getattr(m.config, "use_mamba_kernels", False)),
           "max_abs_diff_before_P": float(diff.max()), "per_position": per_pos, "first_contaminated_position": first_bad}
    row["reproduced"] = row["max_abs_diff_before_P"] > 1e-6
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
