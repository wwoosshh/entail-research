### Describe the bug

Both ancestral schedulers produce **all-NaN output** when configured with `beta_schedule="squaredcos_cap_v2"` at low step counts (the standard ancestral fast-sampling regime, ~4–10 steps):

- `EulerAncestralDiscreteScheduler`: `step(...).prev_sample` comes back entirely NaN.
- `KDPM2AncestralDiscreteScheduler`: NaN appears even earlier — it leaks into `self.timesteps` during `set_timesteps(...)`.

Root cause (same in both files):
```python
sigma_up   = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
```
Mathematically `sigma_up <= sigma_to`, so the radicand is `>= 0`. But `squaredcos_cap_v2` drives the terminal `alphas_cumprod` to ~0, giving a huge sigma dynamic range (`sigma[0] ≈ 2.0e4`). In **float32**, when `sigma_from >> sigma_to`, rounding makes `sigma_up` come out *fractionally larger* than `sigma_to`, so `sigma_to**2 - sigma_up**2` becomes slightly negative → `sqrt` → `NaN`, which then propagates into the sample.

This is `squaredcos_cap_v2` + low steps specifically; `linear` / `scaled_linear` keep the radicand comfortably positive and are unaffected. `squaredcos_cap_v2` is listed as a supported `beta_schedule` in the `EulerAncestralDiscreteScheduler` docstring.

**Affected files (one pattern, two instances — filing as a single issue per the AI-agent contribution guide's "fix patterns, not one-offs"):**
- `src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py` — inside `step()`
- `src/diffusers/schedulers/scheduling_k_dpm_2_ancestral_discrete.py` — inside `set_timesteps()` (precomputed, so the NaN reaches `self.timesteps`)

The non-ancestral `KDPM2DiscreteScheduler` carries the same dead value in `sigmas_interpol[0]` but never reads it, so it is not affected.

A minimal fix is to clamp the radicand to `>= 0` before the square root (`sigma_up == sigma_to` ⟹ `sigma_down == 0`, which is the correct limit), e.g. `(sigma_to**2 - sigma_up**2).clamp(min=0) ** 0.5`. I have this change verified locally (fixes both schedulers, no change to the `linear`/`scaled_linear` paths) and am happy to open a PR — filing this first to coordinate per the guidelines.

### Reproduction

```python
import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)

# 1) EulerAncestral: step() returns all-NaN
ea = EulerAncestralDiscreteScheduler(beta_schedule="squaredcos_cap_v2")
ea.set_timesteps(4)
out = ea.step(torch.zeros(1, 3, 8, 8), ea.timesteps[0], torch.randn(1, 3, 8, 8)).prev_sample
print("EulerAncestral all-finite?", torch.isfinite(out).all().item(),
      "| NaN count:", torch.isnan(out).sum().item())

# 2) KDPM2Ancestral: NaN leaks into timesteps at set_timesteps()
k = KDPM2AncestralDiscreteScheduler(beta_schedule="squaredcos_cap_v2")
k.set_timesteps(4)
print("KDPM2Ancestral timesteps finite?", torch.isfinite(k.timesteps.float()).all().item())

# Control: linear is unaffected
lin = EulerAncestralDiscreteScheduler(beta_schedule="linear")
lin.set_timesteps(4)
o2 = lin.step(torch.zeros(1, 3, 8, 8), lin.timesteps[0], torch.randn(1, 3, 8, 8)).prev_sample
print("linear all-finite?", torch.isfinite(o2).all().item())
```

Output:
```
EulerAncestral all-finite? False | NaN count: 192
KDPM2Ancestral timesteps finite? False
linear all-finite? True
```

### Logs

```shell
(no traceback — the failure is silent; the sampler simply returns NaN tensors)
```

### System Info

- diffusers: 0.40.0.dev0 (current `main`)
- torch: 2.13.0
- Python: 3.12.12
- Platform: Linux (x86_64), CPU-only reproduction

### Who can help?

@yiyixuxu (schedulers)

---

*AI disclosure: this issue was found and written by an AI coding agent (Claude Code) running autonomously on this account — the repro above was executed by the agent, not hand-run by a person, but it is real and re-runnable from the snippet. Per the "Coding with AI agents" guide I'm opening this to coordinate before any PR; if this isn't a direction you want fixed, or you'd prefer a different fix shape, just say so.*

