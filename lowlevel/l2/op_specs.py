"""Op specs for ops.py (M19 L2/L4): how to call each operator, deterministic inputs, the operator's definition in plain
PyTorch and the forms its interface allows. A spec never says which form, input or size shows a defect; forms cover
the interface's range. Inputs whose names start with "_" are opaque to the generic checks (packed layouts, index
structures, a memory pool). Specs named after a report carry that report's operator, not its trigger; the two vllm
controls are healthy ops.
"""
E2M1 =[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def vllm_rms_norm():
    """Control: vLLM's RMSNorm custom op."""
    import torch
    from vllm import _custom_ops as ops

    def inputs():
        torch.manual_seed(0)
        return {"x": torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16),
                "w": torch.rand(1024, device="cuda", dtype=torch.bfloat16) + 0.5}

    def call(x):
        out = torch.empty(x["x"].shape, dtype=x["x"].dtype, device="cuda")
        ops.rms_norm(out, x["x"], x["w"], 1e-6)
        return out

    def definition(x):
        v = x["x"].double()
        return v / torch.sqrt((v * v).mean(-1, keepdim=True) + 1e-6) * x["w"].double()

    return {"name": "vllm.rms_norm", "about": "control: RMSNorm custom op (vLLM 0.30.0)", "inputs": inputs,
            "call": call, "definition": definition, "tol": 2e-2}


def vllm_silu_and_mul():
    """Control: vLLM's SiLU-and-multiply custom op."""
    import torch
    from vllm import _custom_ops  # noqa: F401 - loads torch.ops._C

    def inputs():
        torch.manual_seed(0)
        return {"x": torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)}

    def call(x):
        d = x["x"].shape[-1] // 2
        out = torch.empty(*x["x"].shape[:-1], d, dtype=x["x"].dtype, device="cuda")
        torch.ops._C.silu_and_mul(out, x["x"])
        return out

    def definition(x):
        v = x["x"].double()
        d = v.shape[-1] // 2
        return torch.nn.functional.silu(v[..., :d]) * v[..., d:]

    return {"name": "vllm.silu_and_mul", "about": "control: SiLU-and-mul custom op (vLLM 0.30.0)", "inputs": inputs,
            "call": call, "definition": definition, "tol": 2e-2}


def vllm_softcap():
    """vllm#56578's operator: apply_softcap, the Triton device function vLLM's Triton attention uses for logit
    softcapping (cap * tanh(s / cap)). A one-line Triton kernel hands it values; forms: caps in use (50 for Gemma-2
    attention, 30 for its final logits, 5)."""
    import torch
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.ops.triton_attention_helpers import apply_softcap

    @triton.jit
    def _probe(s_ptr, out_ptr, n, cap, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        s = tl.load(s_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, apply_softcap(s, cap), mask=mask)

    def inputs():
        torch.manual_seed(0)
        return {"s": torch.randn(4096, device="cuda") * 3.0}

    def call(x, cap):
        s = x["s"].contiguous()
        out = torch.empty_like(s)
        _probe[(triton.cdiv(s.numel(), 1024),)](s, out, s.numel(), cap, BLOCK=1024)
        return out

    def definition(x, cap):
        return cap * torch.tanh(x["s"].double() / cap)

    return {"name": "vllm.softcap", "about": "apply_softcap Triton helper (vLLM 0.30.0); vllm#56578's operator",
            "inputs": inputs, "call": call, "definition": definition, "tol": 1e-5,
            "forms": [{"cap": 50.0}, {"cap": 30.0}, {"cap": 5.0}],
            "skip": {"layout": ["s"]}}  # a device function works on values; the probe owns the addressing


def vllm_w8a8_block_mm():
    """vllm#52576's operator: w8a8_triton_block_scaled_mm, FP8 activations with a scale per token and 128-group, FP8
    weights with a scale per 128 x 128 block. Forms: the library's own launch configuration, then BLOCK_SIZE_K over
    the powers of two a Triton config can hold (the config lookup is pointed at the requested one)."""
    import torch
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    M, N, K, gn, gk = 64, 256, 1024, 128, 128

    def inputs():
        torch.manual_seed(0)
        return {"A": (torch.randn(M, K, device="cuda") * 0.5).to(torch.float8_e4m3fn),
                "B": (torch.randn(N, K, device="cuda") * 0.5).to(torch.float8_e4m3fn),
                "As": torch.rand(M, K // gk, device="cuda") * 0.9 + 0.1,
                "Bs": torch.rand(N // gn, K // gk, device="cuda") * 0.9 + 0.1}

    def call(x, block_k=None):
        orig = fp8_utils.get_w8a8_block_fp8_configs
        if block_k:
            cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": 8,
                   "num_warps": 4, "num_stages": 3}
            fp8_utils.get_w8a8_block_fp8_configs = lambda *a, **k: {M: cfg}
        try:
            return fp8_utils.w8a8_triton_block_scaled_mm(x["A"], x["B"], x["As"], x["Bs"], [gn, gk], torch.float32)
        finally:
            fp8_utils.get_w8a8_block_fp8_configs = orig

    def definition(x, block_k=None):
        a = x["A"].double() * x["As"].double().repeat_interleave(gk, 1)
        b = x["B"].double() * x["Bs"].double().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
        return a @ b.T

    return {"name": "vllm.w8a8_block_mm", "about": "w8a8_triton_block_scaled_mm (vLLM 0.30.0); vllm#52576's operator",
            "inputs": inputs, "call": call, "definition": definition, "tol": 1e-3,
            "forms": [{}] + [{"block_k": b} for b in (16, 32, 64, 128, 256, 512)]}


def vllm_moe_int8():
    """vllm#58532's operator: fused_experts with an INT8 W8A8 config: static per-tensor activation scales and
    per-output-channel weight scales. Random top-2 routing over 4 experts."""
    import inspect

    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config

    T, E, H, I, K = 8, 4, 64, 64, 2
    takes_per_out = "per_out_ch_quant" in inspect.signature(int8_w8a8_moe_quant_config).parameters

    def inputs():
        torch.manual_seed(0)
        return {"x": torch.randn(T, H, device="cuda", dtype=torch.bfloat16),
                "w1": torch.randint(-8, 8, (E, 2 * I, H), device="cuda", dtype=torch.int8),
                "w2": torch.randint(-8, 8, (E, H, I), device="cuda", dtype=torch.int8),
                "w1_scale": torch.linspace(0.02, 0.08, E * 2 * I, device="cuda").view(E, 2 * I, 1),
                "w2_scale": torch.linspace(0.02, 0.08, E * H, device="cuda").view(E, H, 1),
                "a1_scale": torch.tensor(0.1, device="cuda"), "a2_scale": torch.tensor(0.1, device="cuda"),
                "topk_w": (torch.rand(T, K, device="cuda") + 0.5).to(torch.bfloat16),
                "topk_ids": torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)]).to(torch.int32)}

    def call(x):
        kw = dict(w1_scale=x["w1_scale"], w2_scale=x["w2_scale"], a1_scale=x["a1_scale"], a2_scale=x["a2_scale"],
                  per_act_token_quant=False)
        if takes_per_out:
            kw["per_out_ch_quant"] = True
        with set_current_vllm_config(VllmConfig()):
            return fused_experts(x["x"], x["w1"], x["w2"], x["topk_w"], x["topk_ids"],
                                 quant_config=int8_w8a8_moe_quant_config(**kw))

    def definition(x):
        def q8(v, s):
            return torch.clamp(torch.round(v / s), -128, 127) * s
        a1, a2 = x["a1_scale"].double(), x["a2_scale"].double()
        out = torch.zeros(T, H, dtype=torch.float64, device="cuda")
        for t in range(T):
            for j in range(K):
                e = int(x["topk_ids"][t, j])
                h = q8(x["x"][t].double(), a1) @ (x["w1"][e].double() * x["w1_scale"][e].double()).T
                act = torch.nn.functional.silu(h[:I]) * h[I:]
                y = q8(act, a2) @ (x["w2"][e].double() * x["w2_scale"][e].double()).T
                out[t] += x["topk_w"][t, j].double() * y
        return out

    return {"name": "vllm.moe_int8", "about": "fused_experts, INT8 W8A8, static activation scales, per-channel weight "
            "scales (vLLM 0.30.0); vllm#58532's operator", "inputs": inputs, "call": call, "definition": definition,
            "tol": 5e-2, "indices": ["topk_ids"]}


def vllm_moe_marlin_nvfp4():
    """vllm#48895's operator: moe_wna16_marlin_gemm on NVFP4 MoE weights at gpt-oss-20b's MoE shapes (E 32, top-4,
    N = K = 2880 padded to 2944, group 16; random codes and scales, the report's setup). Forms: the op's
    mul_topk_weights option on, or off with the routing weights applied outside (the same product by the op's
    interface)."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        _nvfp4_compute_scale_factor, nvfp4_marlin_process_global_scale, nvfp4_marlin_process_scales)
    from vllm.scalar_type import scalar_types

    E, N, K_RAW, K, M, TOPK, BM, G = 32, 2880, 2880, 2944, 32, 4, 16, 16
    dt = torch.bfloat16

    def inputs():
        torch.manual_seed(0)
        dev = "cuda"
        lut = torch.tensor(E2M1, device=dev)
        codes = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        codes[..., K_RAW // 2:] = 0
        scale_f = torch.rand(E, N, K // G, device=dev) * 3 + 0.5
        scale_f[..., K_RAW // G:] = 0
        scales_fp8 = scale_f.to(torch.float8_e4m3fn)
        gscale = torch.rand(E, device=dev) * 0.002 + 1e-4
        w = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1).reshape(E, N, K)
        w = (w * scales_fp8.float().repeat_interleave(G, dim=2) * gscale.view(E, 1, 1)).to(torch.bfloat16)
        csf = _nvfp4_compute_scale_factor(scales_fp8.to(dt), dt)
        qw, ms = [], []
        for e in range(E):
            q = codes[e].view(torch.int32).T.contiguous()
            qw.append(ops.gptq_marlin_repack(b_q_weight=q, size_k=K, size_n=N, num_bits=4, is_a_8bit=False))
            s = marlin_permute_scales(s=scales_fp8[e].to(dt).T, size_k=K, size_n=N, group_size=G, is_a_8bit=False)
            s, _ = nvfp4_marlin_process_scales(s, scale_factor=csf, a_dtype=dt)
            ms.append(s)
        router = torch.randn(M, E, device=dev)
        topk_w, topk_ids = torch.topk(torch.softmax(router, -1), TOPK, dim=-1)
        sorted_ids, expert_ids, num_post_pad = moe_align_block_size(topk_ids.to(torch.int32), BM, E)
        row_expert = torch.full((M * TOPK,), -1, dtype=torch.long, device=dev)
        for b in range(len(expert_ids)):
            if expert_ids[b] < 0:
                continue
            blk = sorted_ids[b * BM:(b + 1) * BM]
            row_expert[blk[blk < M * TOPK].long()] = expert_ids[b].long()
        act = torch.randn(M * TOPK, K, device=dev, dtype=dt) * 0.5
        act[:, K_RAW:] = 0
        return {"act": act, "topk_w": topk_w.to(dt), "_w": w, "_w_marlin": torch.stack(qw), "_s_marlin": torch.stack(ms),
                "_g_marlin": nvfp4_marlin_process_global_scale(gscale.float(), dt) / csf, "_sorted_ids": sorted_ids,
                "_expert_ids": expert_ids, "_num_post_pad": num_post_pad, "_row_expert": row_expert}

    def call(x, mul_topk_weights):
        c = torch.zeros(M * TOPK, N, device="cuda", dtype=dt)
        out = ops.moe_wna16_marlin_gemm(
            x["act"], c, x["_w_marlin"], None, x["_s_marlin"], None, x["_g_marlin"], None,
            torch.zeros(1024, dtype=torch.int, device="cuda"), x["_sorted_ids"], x["_expert_ids"], x["_num_post_pad"],
            x["topk_w"], moe_block_size=BM, top_k=1, mul_topk_weights=mul_topk_weights,
            b_q_type=scalar_types.float4_e2m1f, size_m=M * TOPK, size_n=N, size_k=K,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        if not mul_topk_weights:
            out = out * x["topk_w"].reshape(-1, 1).to(out.dtype)
        return out

    def definition(x, mul_topk_weights):
        ref = torch.zeros(M * TOPK, N, dtype=torch.float64, device="cuda")
        tw = x["topk_w"].double().reshape(-1)
        rows = x["_row_expert"]
        for e in torch.unique(rows[rows >= 0]).tolist():
            r = (rows == e).nonzero().flatten()
            ref[r] = (x["act"][r].double() @ x["_w"][e].double().T) * tw[r].unsqueeze(1)
        return ref

    return {"name": "vllm.moe_marlin_nvfp4", "about": "moe_wna16_marlin_gemm, NVFP4 weights (vLLM 0.30.0); "
            "vllm#48895's operator", "inputs": inputs, "call": call, "definition": definition, "tol": 2e-2,
            "forms": [{"mul_topk_weights": True}, {"mul_topk_weights": False}], "clone": False}


def torch_flex_paged():
    """vllm#50427's operator: FlexAttention over a paged KV pool (torch 2.13, as vLLM's FlexAttention backend uses
    it): pages of 16 rows, 8 KV heads of 128, the pool inside a larger allocation holding other data (as a serving
    engine's cache sits among other buffers). K and V are constants, so every page gives the same attention output.
    Forms: where in the pool the pages sit (block ids across the pool's range)."""
    import torch
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    BLOCK, NKVH, HD, QH, NUM_BLOCKS, Q_LEN, PER_ROW = 16, 8, 128, 32, 68370, 512, 64
    KC, VC = 0.125, 0.75
    flex = torch.compile(flex_attention, fullgraph=True)

    def inputs():
        rows = NUM_BLOCKS * BLOCK
        below = int(5 * 1024 ** 3) // 2
        pool = torch.empty(below + rows * 2 * NKVH * HD, dtype=torch.bfloat16, device="cuda")
        pool[:below].fill_(-8.0)
        kv = pool[below:].view(rows, 2, NKVH, HD)
        kv[:, 0].fill_(KC)
        kv[:, 1].fill_(VC)
        return {"_pool": pool, "_kv": kv}

    def call(x, block_id):
        kv = x["_kv"]
        k = kv[:, 0].permute(1, 0, 2).unsqueeze(0)
        v = kv[:, 1].permute(1, 0, 2).unsqueeze(0)
        q = torch.full((Q_LEN, QH, HD), KC, dtype=torch.bfloat16, device="cuda").permute(1, 0, 2).unsqueeze(0)
        nq = Q_LEN // BLOCK
        bm = BlockMask.from_kv_blocks(
            kv_num_blocks=torch.full((nq,), PER_ROW, dtype=torch.int32, device="cuda")[None, None],
            kv_indices=torch.full((nq, PER_ROW), block_id, dtype=torch.int32, device="cuda")[None, None],
            full_kv_num_blocks=None, full_kv_indices=None, BLOCK_SIZE=(BLOCK, BLOCK),
            mask_mod=lambda b, h, qi, kvi: qi >= 0, seq_lengths=(Q_LEN, NUM_BLOCKS * BLOCK), compute_q_blocks=False)
        return flex(q, k, v, None, bm, HD ** -0.5, enable_gqa=True,
                    kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": BLOCK, "BLOCK_N": BLOCK})

    def definition(x, block_id):
        return torch.full((1, QH, Q_LEN, HD), VC, dtype=torch.float64)

    return {"name": "torch.flex_paged", "about": "FlexAttention over a paged KV pool (torch 2.13); vllm#50427's "
            "operator", "inputs": inputs, "call": call, "definition": definition, "tol": 1e-2, "clone": False,
            "forms": [{"block_id": b} for b in (0, 1024, 32767, 65535, 65536, 68000)],
            "checks": ["definition", "forms"]}


def sglang_gdn_gating():
    """sglang#21843's operator: fused_gdn_gating, the Gated DeltaNet gate g = -exp(A_log) * softplus(a + dt_bias)
    and beta = sigmoid(b) (SGLang 0.5.20)."""
    import torch
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    def inputs():
        torch.manual_seed(0)
        return {"A_log": torch.randn(16, device="cuda"), "a": torch.randn(64, 16, device="cuda"),
                "b": torch.randn(64, 16, device="cuda"), "dt_bias": torch.randn(16, device="cuda")}

    def call(x):
        return fused_gdn_gating(x["A_log"], x["a"], x["b"], x["dt_bias"])

    def definition(x):
        g = -torch.exp(x["A_log"].double()) * torch.nn.functional.softplus(x["a"].double() + x["dt_bias"].double())
        return (g, torch.sigmoid(x["b"].double()))

    return {"name": "sglang.gdn_gating", "about": "fused_gdn_gating (SGLang 0.5.20); sglang#21843's operator",
            "inputs": inputs, "call": call, "definition": definition, "tol": 1e-4}


def tf_switch_router():
    """transformers#48293's operator: SwitchTransformersTop1Router of google/switch-base-8 (transformers 5.17.0). By
    its documentation it returns the top-1 probability, the one-hot expert mask with each expert's capacity enforced
    (tokens in order), and the router logits. Forms: the configured capacity, then capacities 1, 2 and 4."""
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersTop1Router

    model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8").eval()
    router = next(m for m in model.modules() if isinstance(m, SwitchTransformersTop1Router))
    cap0, n_exp = router.expert_capacity, router.num_experts

    def inputs():
        torch.manual_seed(0)
        return {"hidden": torch.randn(2, 8, model.config.d_model)}

    def call(x, capacity=None):
        router.expert_capacity = cap0 if capacity is None else capacity
        with torch.no_grad():
            return router(x["hidden"])

    def definition(x, capacity=None):
        cap = cap0 if capacity is None else capacity
        with torch.no_grad():
            logits = router.classifier(x["hidden"].to(router.dtype)).double()
        probs = torch.softmax(logits, -1)
        onehot = torch.nn.functional.one_hot(probs.argmax(-1), n_exp).double()
        mask = onehot * (torch.cumsum(onehot, dim=1) <= cap)
        return (probs.max(-1).values.unsqueeze(-1), mask.unsqueeze(2), logits)

    return {"name": "tf.switch_router", "about": "SwitchTransformersTop1Router (transformers 5.17.0); "
            "transformers#48293's operator", "inputs": inputs, "call": call, "definition": definition, "tol": 1e-4,
            "forms": [{}, {"capacity": 1}, {"capacity": 2}, {"capacity": 4}]}


def tf_whisper_features():
    """transformers#47885's operator: WhisperFeatureExtractor (log-mel features for openai/whisper-tiny, transformers
    5.17.0). No definition here: only the checks that need none."""
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def inputs():
        rng = np.random.default_rng(0)
        return {"audio": torch.from_numpy((rng.standard_normal(16000) * 0.05).astype(np.float32))}

    def call(x):
        return torch.from_numpy(fe(x["audio"].numpy(), sampling_rate=16000, return_tensors="np")["input_features"])

    return {"name": "tf.whisper_features", "about": "WhisperFeatureExtractor (transformers 5.17.0); "
            "transformers#47885's operator", "inputs": inputs, "call": call, "tol": 1e-4,
            "checks": ["magnitude", "degenerate", "poison"]}


def diffusers_rescale_noise_cfg():
    """diffusers#13425's operator: rescale_noise_cfg, the guidance rescale shared by the Stable Diffusion pipelines
    (diffusers 0.40.0): noise_cfg rescaled to noise_pred_text's standard deviation, mixed by guidance_rescale."""
    import torch
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

    def inputs():
        torch.manual_seed(0)
        return {"noise_cfg": torch.randn(1, 4, 64, 64), "noise_pred_text": torch.randn(1, 4, 64, 64)}

    def call(x, guidance_rescale=0.7):
        return rescale_noise_cfg(x["noise_cfg"], x["noise_pred_text"], guidance_rescale=guidance_rescale)

    def definition(x, guidance_rescale=0.7):
        c, t = x["noise_cfg"].double(), x["noise_pred_text"].double()
        dims = list(range(1, c.ndim))
        rescaled = c * (t.std(dim=dims, keepdim=True) / c.std(dim=dims, keepdim=True))
        return guidance_rescale * rescaled + (1 - guidance_rescale) * c

    return {"name": "diffusers.rescale_noise_cfg", "about": "rescale_noise_cfg (diffusers 0.40.0); diffusers#13425's "
            "operator", "inputs": inputs, "call": call, "definition": definition, "tol": 1e-5,
            "forms": [{"guidance_rescale": 0.7}, {"guidance_rescale": 0.0}, {"guidance_rescale": 1.0}]}


SPECS = {
    "vllm.rms_norm": vllm_rms_norm, "vllm.silu_and_mul": vllm_silu_and_mul, "vllm.softcap": vllm_softcap,
    "vllm.w8a8_block_mm": vllm_w8a8_block_mm, "vllm.moe_int8": vllm_moe_int8,
    "vllm.moe_marlin_nvfp4": vllm_moe_marlin_nvfp4, "torch.flex_paged": torch_flex_paged,
    "sglang.gdn_gating": sglang_gdn_gating, "tf.switch_router": tf_switch_router,
    "tf.whisper_features": tf_whisper_features, "diffusers.rescale_noise_cfg": diffusers_rescale_noise_cfg,
}
