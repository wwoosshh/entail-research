# Draft upstream issue (NOT POSTED — posting needs the researcher's explicit permission)

- Target: huggingface/transformers
- Status: draft only. The researcher decides whether and when to post (`RESEARCH_PLAN.md` section 8 item 3). The numbers come from `RESULTS.md`; placeholders are marked (TBD).

---

**Title:** Gemma 2: the default SDPA attention path silently ignores `attn_logit_softcapping`

**System info:** transformers 5.17.0 (the same code is on `main` as of 2026-09; the last change to `integrations/sdpa_attention.py` was on 2026-08-20). torch 2.14.0+cu130. RTX 4070 Ti.

**What happens**
- `Gemma2Attention` passes `softcap=self.attn_logit_softcapping` to the attention interface.
- `Gemma2PreTrainedModel` declares `_supports_sdpa = True`, and `sdpa` is the default implementation.
- `sdpa_attention_forward(..., **kwargs)` never reads `softcap`, so the cap declared in the config is dropped.
- Nothing warns at load time or at forward time. We checked Python warnings and the `transformers` logger.
- `eager` and `flex_attention` do apply the cap.

**Minimal reproduction:** a tiny random Gemma 2 config with q/k weights scaled so that the cap binds.
```python
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM
cfg = Gemma2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=64, attn_logit_softcapping=5.0,
                   final_logit_softcapping=None, sliding_window=256)
torch.manual_seed(0)
m = Gemma2ForCausalLM(cfg).to("cuda", torch.float32).eval()
with torch.no_grad():
    for l in m.model.layers:
        l.self_attn.q_proj.weight.mul_(20); l.self_attn.k_proj.weight.mul_(20)
ids = torch.randint(0, 1000, (1, 64), device="cuda")
print(m.config._attn_implementation)          # 'sdpa' by default
out = {}
for impl in ("eager", "sdpa"):
    m.set_attn_implementation(impl)
    with torch.no_grad():
        out[impl] = m(ids).logits
print((out["eager"] - out["sdpa"]).abs().max())  # large; top-1 agreement ~6% in our run
```

**Effect on real checkpoints** (Unsloth copies; the 9B weights are byte-identical to `google/gemma-2-9b-it`):
- **2B, bf16:**
  - The largest pre-cap attention score over our prompts was 40.3, below the cap of 50.
  - The sdpa-vs-eager difference was at kernel-noise level: KL ratio 1.3× the eager-vs-flex noise.
  - GSM8K (first 500): 67.4% vs 67.8%, p = 0.82.
- **9B, int4:**
  - The largest pre-cap score was 119.2, so the cap binds occasionally.
  - The sdpa-vs-eager KL was 2.3× the noise between two cap-applying kernels.
  - GSM8K (first 500): 88.4% (sdpa) vs 88.6% (eager), McNemar p = 1.0.
  - With attention computed in fp32, the pattern is the same in both sizes: sdpa sits at kernel-noise distance
    from "eager with the cap disabled" (2B 4.43e-4 vs a 4.50e-4 noise floor; 9B 4.18e-4 vs 4.11e-4), and its
    distance from eager matches the distance of "cap disabled" from eager to within 6% (2B) and 2% (9B).
    So sdpa behaves exactly like the no-cap path. The size of that effect is 1.4x (2B) and 2.5x (9B) the
    kernel-noise level, i.e. small on these checkpoints.
- **27B:** not measured (did not fit our GPU). The transformers v4.42.3 release notes said soft-capping is a must for 27B.

**A second path with the same problem:** continuous batching. `generate_batch()` switches the model to a paged
implementation (`paged|eager` here, since flash-attn is not installed), and those paged kernels take no softcap
either. Measured on Gemma 2 2B with `attn_logit_softcapping` set to 5.0: the greedy outputs of `generate_batch`
were identical for all 10 prompts whether the cap was declared or disabled, i.e. the declared cap has no effect
on that path. The kernel and the batching are the same in both runs, so only the declaration differed.

**Expected:** one of the following.
1. Fall back to `eager` or `flex_attention` when `attn_logit_softcapping` is set and `sdpa` is selected (default path).
2. Raise or warn when `sdpa` is explicitly requested with a softcap model.
3. Document that `sdpa` ignores the cap.

---

Notes for the researcher (not part of the issue):
- An issue search on 2026-09-23 found no existing report of this. The closest is #32309 (FA2 with Gemma 2, 2024).
- If posted, keep it factual and include the minimal reproduction. Real-model impact is small for 2B; state that plainly.
