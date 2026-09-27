"""M17.6 case, huggingface/diffusers#13425 (testbed/M16_PROTOCOL.md 7): rescale_noise_cfg divides by the guided
noise's standard deviation without a guard; a zero-variance noise_cfg gives NaN or inf, which the diffusion loop
carries on silently. The report's snippet through the library's own helper (the Stable Diffusion pipeline's copy,
which 36 pipelines and guiders share). Reproduced when the result holds a non-finite value.
Run in ~/venvs/gpu: python testbed/m17/replay2/cases/df13425.py <out.json>
"""
import json
import os
import sys


def main():
    import diffusers
    import torch
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

    torch.manual_seed(0)
    noise_cfg = torch.zeros(1, 4, 64, 64)                  # std = 0
    noise_pred_text = torch.randn_like(noise_cfg)
    out = rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.7)
    finite = bool(torch.isfinite(out).all())
    row = {"entail": os.environ.get("ENTAIL", "off"), "diffusers": diffusers.__version__,
           "noise_cfg_std": float(noise_cfg.std()), "output_finite": finite,
           "nan_count": int(torch.isnan(out).sum()), "inf_count": int(torch.isinf(out).sum()),
           "reproduced": not finite}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
