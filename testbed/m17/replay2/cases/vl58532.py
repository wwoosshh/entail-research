"""M17.6 case, vllm-project/vllm#58532 (testbed/M16_PROTOCOL.md 7): the legacy Triton fused-MoE path indexes
per-output-channel weight scales with the per-tensor branch when the activation scales are static per-tensor
(per_act_token_quant=False selects both layouts), so the routed experts' channel scales are ignored. The probe
the fix PR (#58570) adds as a test, run against the installed vLLM through its public fused_experts with an INT8
W8A8 config of static activation scales and per-channel weight scales: multiplying expert 3's W13 channel scales
by 4 (expert 3 is routed on every token) must change the output; multiplying expert 0's (never routed) must not.
Reproduced when the routed perturbation leaves the output unchanged.
Run in ~/venvs/vllm (0.30.0): python testbed/m17/replay2/cases/vl58532.py <out.json>
"""
import inspect
import json
import os
import sys
import traceback


def main():
    import torch
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config

    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__}
    try:
        torch.manual_seed(0)
        dev = torch.device("cuda")
        num_tokens, num_experts, hidden, inter, top_k = 8, 4, 64, 64, 2
        x = torch.randn((num_tokens, hidden), device=dev, dtype=torch.bfloat16)
        w1 = torch.randint(-8, 8, (num_experts, 2 * inter, hidden), device=dev, dtype=torch.int8)
        w2 = torch.randint(-8, 8, (num_experts, hidden, inter), device=dev, dtype=torch.int8)
        w1_scale = torch.linspace(0.02, 0.08, num_experts * 2 * inter, device=dev).view(num_experts, 2 * inter, 1)
        w2_scale = torch.linspace(0.02, 0.08, num_experts * hidden, device=dev).view(num_experts, hidden, 1)
        a1 = torch.tensor(0.1, device=dev)
        a2 = torch.tensor(0.1, device=dev)
        topk_ids = torch.tensor([[3, 1]] * num_tokens, device=dev, dtype=torch.int32)
        topk_w = torch.ones((num_tokens, top_k), device=dev, dtype=torch.bfloat16)
        params = inspect.signature(int8_w8a8_moe_quant_config).parameters
        row["config_takes_per_out_ch_quant"] = "per_out_ch_quant" in params

        def cfg(w1s):
            kw = dict(w1_scale=w1s, w2_scale=w2_scale.clone(), a1_scale=a1, a2_scale=a2, per_act_token_quant=False)
            if "per_out_ch_quant" in params:
                kw["per_out_ch_quant"] = True
            return int8_w8a8_moe_quant_config(**kw)

        with set_current_vllm_config(VllmConfig()):
            base = fused_experts(x, w1, w2, topk_w, topk_ids, quant_config=cfg(w1_scale.clone()))
            routed = w1_scale.clone()
            routed[3] *= 4
            out_routed = fused_experts(x, w1, w2, topk_w, topk_ids, quant_config=cfg(routed))
            unrouted = w1_scale.clone()
            unrouted[0] *= 4
            out_unrouted = fused_experts(x, w1, w2, topk_w, topk_ids, quant_config=cfg(unrouted))
        torch.cuda.synchronize()
        row["routed_perturbation_changes_output"] = bool(not torch.equal(base, out_routed))
        row["routed_max_abs_change"] = float((base - out_routed).abs().max())
        row["unrouted_perturbation_changes_output"] = bool(not torch.equal(base, out_unrouted))
        row["reproduced"] = not row["routed_perturbation_changes_output"]
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        row["trace_tail"] = traceback.format_exc().strip().splitlines()[-3:]
        row["reproduced"] = None
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
