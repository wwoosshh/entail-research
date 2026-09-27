"""M16 case, huggingface/diffusers#13411 (testbed/M16_PROTOCOL.md 5): LTXEulerAncestralRFScheduler.set_timesteps
accepts a sigma schedule that is not non-increasing, and step() then runs the ancestral decomposition with
sigma_down outside [0, 1] and alpha_down < 0: the sample enters the next latent with a negative coefficient
(sign-inverted), finite and silent. The report's own script (scheduler only, CPU; the pipeline the scheduler serves
is a video model far over this card), plus the same step on the valid schedule [0.8, 0.2, 0.0] as the control.
The report's headline "ratio 1.45x" assumes denoised = 0; with model_output = 0 the denoised estimate equals the
sample, so the norm ratio is not the discriminating number - the sign of the sample's coefficient is (cosine
between the output and the sample: positive on a valid schedule, negative here). Closed as not_planned, so 0.40.0
behaves as reported.
Run in ~/venvs/gpu: python testbed/m16/cases/df13411.py <out.json>
"""
import json
import os
import sys


def one_step(sigmas):
    import torch
    from diffusers import LTXEulerAncestralRFScheduler

    scheduler = LTXEulerAncestralRFScheduler()
    err = None
    try:
        scheduler.set_timesteps(sigmas=sigmas)
    except Exception as e:  # noqa: BLE001
        return {"sigmas": sigmas, "set_timesteps_error": f"{type(e).__name__}: {e}"[:300]}
    torch.manual_seed(0)
    sample = torch.randn(1, 4, 8, 8)
    model_output = torch.zeros_like(sample)
    out = scheduler.step(model_output, scheduler.timesteps[0], sample).prev_sample
    cos = float(torch.nn.functional.cosine_similarity(out.flatten(), sample.flatten(), dim=0))
    return {"sigmas": sigmas, "set_timesteps_error": err, "input_norm": float(sample.norm()),
            "output_norm": float(out.norm()), "ratio": float(out.norm() / sample.norm()),
            "cosine_output_vs_sample": cos, "finite": bool(torch.isfinite(out).all())}


def main():
    import diffusers

    bad = one_step([0.2, 0.8, 0.5, 0.0])
    good = one_step([0.8, 0.2, 0.0])
    row = {"entail": os.environ.get("ENTAIL", "off"), "diffusers": diffusers.__version__, "non_monotone": bad,
           "control_valid_schedule": good,
           "reproduced": bool(bad.get("set_timesteps_error") is None and bad.get("finite")
                              and bad.get("cosine_output_vs_sample", 1.0) < 0.0
                              and good.get("cosine_output_vs_sample", -1.0) > 0.0)}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
