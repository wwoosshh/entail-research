"""M18.6 replay 3 case, huggingface/diffusers#14213 (testbed/M16_PROTOCOL.md 8): EulerAncestralDiscreteScheduler and
KDPM2AncestralDiscreteScheduler give all-NaN samples with beta_schedule="squaredcos_cap_v2" at low step counts. The
report's reproduction, unchanged (with a seed): reproduced when the ancestral outputs are not finite while the
linear control is.
Run in ~/venvs/gpu (diffusers 0.40.0): python testbed/m18/replay3/cases/df14213.py <out.json>
"""
import json
import os
import sys


def main():
    import diffusers
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, KDPM2AncestralDiscreteScheduler

    torch.manual_seed(0)
    ea = EulerAncestralDiscreteScheduler(beta_schedule="squaredcos_cap_v2")
    ea.set_timesteps(4)
    out = ea.step(torch.zeros(1, 3, 8, 8), ea.timesteps[0], torch.randn(1, 3, 8, 8)).prev_sample
    k = KDPM2AncestralDiscreteScheduler(beta_schedule="squaredcos_cap_v2")
    k.set_timesteps(4)
    lin = EulerAncestralDiscreteScheduler(beta_schedule="linear")
    lin.set_timesteps(4)
    o2 = lin.step(torch.zeros(1, 3, 8, 8), lin.timesteps[0], torch.randn(1, 3, 8, 8)).prev_sample
    row = {"entail": os.environ.get("ENTAIL", "off"), "diffusers": diffusers.__version__,
           "euler_ancestral_nan": int(torch.isnan(out).sum()), "euler_ancestral_finite": bool(torch.isfinite(out).all()),
           "kdpm2_ancestral_timesteps_finite": bool(torch.isfinite(k.timesteps.float()).all()),
           "linear_finite": bool(torch.isfinite(o2).all())}
    row["reproduced"] = (not row["euler_ancestral_finite"] or not row["kdpm2_ancestral_timesteps_finite"]) \
        and row["linear_finite"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
