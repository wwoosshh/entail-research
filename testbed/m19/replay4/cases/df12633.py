"""M19 L4 replay 4 case, huggingface/diffusers#12633 (testbed/M16_PROTOCOL.md 9): DDIMScheduler.step() computes the
previous timestep as `t - num_train_timesteps // num_inference_steps`, which is not the schedule set_timesteps() made
when timestep_spacing='linspace'. The report's reproduction, made checkable: the timestep the step moves to (read back
from the alpha it uses) against the schedule's next timestep. Reproduced when a step's previous timestep is not the
next one in scheduler.timesteps. No model: the scheduler is built from its defaults with timestep_spacing='linspace'.
Run in ~/venvs/gpu (diffusers 0.40.0): python testbed/m19/replay4/cases/df12633.py <out.json>
"""
import json
import os
import sys


def main():
    import diffusers
    import torch
    from diffusers import DDIMScheduler

    steps = 10
    s = DDIMScheduler(timestep_spacing="linspace")
    s.set_timesteps(steps)
    ts = [int(t) for t in s.timesteps]
    stride = s.config.num_train_timesteps // steps
    used = [t - stride for t in ts]                            # what step() uses (the report's formula)
    expected = ts[1:] + [-1]                                   # the schedule's next timestep (-1: final_alpha_cumprod)
    # the step's own arithmetic: prev_sample with a zero model output and eta 0 is x * sqrt(a_prev / a_t), so the
    # alpha step() used can be read back and compared with the schedule's
    mism = []
    for i, t in enumerate(ts[:-1]):
        x = torch.ones(1, 1, 2, 2)
        out = s.step(torch.zeros_like(x), t, x, eta=0.0).prev_sample
        a_t = s.alphas_cumprod[t]
        ratio = float(out.flatten()[0]) ** 2 * float(a_t)      # = a_prev the step used
        a_sched = float(s.alphas_cumprod[expected[i]]) if expected[i] >= 0 else float(s.final_alpha_cumprod)
        if abs(ratio - a_sched) > 1e-4 * max(1.0, a_sched):
            mism.append({"t": t, "schedule_next": expected[i], "step_uses": used[i], "alpha_used": ratio,
                         "alpha_schedule": a_sched})
    row = {"entail": os.environ.get("ENTAIL", "off"), "diffusers": diffusers.__version__, "timesteps": ts,
           "stride_used_by_step": stride, "steps_off_schedule": len(mism), "first": mism[:2]}
    row["reproduced"] = bool(mism)
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
