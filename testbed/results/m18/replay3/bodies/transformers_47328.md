### System Info
- `transformers`: main (reproduced at `ab1771c`); bug present since the v5 RoPE refactor (#39847, merged 2025-10-17)
- Code-level bug, weight-independent — reproducible on CPU, no GPU or pretrained weights needed

### Who can help?
@eustlb @ebezzam

### Description
`Qwen2_5OmniDiTRotaryEmbedding.forward` builds the rotary `cos`/`sin` with a **half-split** layout:
```python
# modeling_qwen2_5_omni.py:2503
emb = torch.cat((freqs, freqs), dim=-1)   # [f0, f1, …, f0, f1, …]
```
but the Token2Wav DiT applies an **interleaved** rotate function, `rotate_half_codec`, which pairs adjacent channels `(2i, 2i+1)`:
```python
# modeling_qwen2_5_omni.py:2940
def rotate_half_codec(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)
```
A half-split cos/sin combined with an interleaved rotate is **not a valid rotation**: RoPE's defining property `⟨R_m·q, R_n·k⟩` depending only on the relative offset `m − n` is destroyed. This is applied to head 0 of every DiT attention layer (`query[:, :1], key[:, :1] = apply_rotary_pos_emb(...)`), silently degrading the predicted mel-spectrogram and thus the synthesized audio for `return_audio=True`.

This is a **regression**. The DiT rotary originally used an interleaved layout (`torch.stack((freqs, freqs), dim=-1).reshape(...)`, i.e. `repeat_interleave`), which matches `rotate_half_codec`. PR #39847 ("🚨 [v5] Refactor RoPE for layer types") made the class inherit `LlamaRotaryEmbedding`, replacing `stack` with `cat` while leaving `rotate_half_codec` interleaved. The original add PR #36752 (integration-tested against `Qwen/Qwen2.5-Omni-7B`) used the interleaved layout, so the released weights expect it. (The text model's mrope at line ~1380 correctly uses `cat`, because it uses the half-split `rotate_half` — that one is fine.)

### Reproduction (CPU, no weights)
```python
import torch
dim, base = 16, 10000.0
positions = torch.arange(0, 64, dtype=torch.float)
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
freqs = positions[:, None] * inv_freq[None, :]

def rotate_half_codec(x):  # verbatim from modeling_qwen2_5_omni.py
    x = x.reshape(*x.shape[:-1], -1, 2); x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1); return x.reshape(*x.shape[:-2], -1)
def apply(q, cos, sin): return q * cos + rotate_half_codec(q) * sin

def spread(emb, delta=5):  # max drift of <R_m q, R_{m+delta} k> over absolute m; ~0 == valid RoPE
    cos, sin = emb.cos(), emb.sin(); q = torch.randn(dim); k = torch.randn(dim); vals = []
    for m in range(40):
        vals.append((apply(q, cos[m], sin[m]) * apply(k, cos[m+delta], sin[m+delta])).sum().item())
    v = torch.tensor(vals); return (v.max() - v.min()).item()

print("current (cat):            ", spread(torch.cat((freqs, freqs), dim=-1)))            # ~5.1  (broken)
print("fixed   (repeat_interleave):", spread(torch.repeat_interleave(freqs, 2, dim=-1)))  # ~1e-6 (valid)
```

### Expected behavior
The DiT rotary cos/sin layout should match its interleaved rotate function, so RoPE is translation-invariant and consistent with the released checkpoint.

### Suggested fix (one line, checkpoint-faithful)
```diff
-            emb = torch.cat((freqs, freqs), dim=-1)
+            emb = torch.repeat_interleave(freqs, 2, dim=-1)
```
(in the modular source, override `forward` on `Qwen2_5OmniDiTRotaryEmbedding` instead of inheriting Llama's, so the generated forward emits the interleaved layout). I have this fix ready and am happy to open a PR if you agree with the direction.

> Disclosure: the root-cause analysis, regression bisect, and reproduction in this report were prepared with AI assistance and independently verified by me.

