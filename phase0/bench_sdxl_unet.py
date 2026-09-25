"""Workload A: one SDXL UNet denoising step (fp16) on RTX 4070 Ti.

Weights are randomly initialized: kernel structure and timing do not depend on weight values,
so no 7 GB download is needed for structural profiling. Modes: eager, compile_default,
compile_graphs (torch.compile mode="reduce-overhead" = CUDA graphs).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import env_info, run_mode, save_results  # noqa: E402

SDXL_UNET_CONFIG_FALLBACK = {
    "act_fn": "silu", "addition_embed_type": "text_time", "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256, "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280], "center_input_sample": False, "class_embed_type": None,
    "class_embeddings_concat": False, "conv_in_kernel": 3, "conv_out_kernel": 3,
    "cross_attention_dim": 2048, "cross_attention_norm": None,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "downsample_padding": 1, "dual_cross_attention": False, "encoder_hid_dim": None,
    "encoder_hid_dim_type": None, "flip_sin_to_cos": True, "freq_shift": 0, "in_channels": 4,
    "layers_per_block": 2, "mid_block_only_cross_attention": None, "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn", "norm_eps": 1e-05, "norm_num_groups": 32,
    "num_attention_heads": None, "num_class_embeds": None, "only_cross_attention": False,
    "out_channels": 4, "projection_class_embeddings_input_dim": 2816, "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False, "resnet_time_scale_shift": "default", "sample_size": 128,
    "time_cond_proj_dim": None, "time_embedding_act_fn": None, "time_embedding_dim": None,
    "time_embedding_type": "positional", "timestep_post_act": None,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "upcast_attention": None, "use_linear_projection": True,
}


def build_unet():
    from diffusers import UNet2DConditionModel
    try:
        cfg = UNet2DConditionModel.load_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
        src = "hub-config"
    except Exception as e:
        cfg = SDXL_UNET_CONFIG_FALLBACK
        src = f"fallback-config ({type(e).__name__})"
    torch.manual_seed(0)
    prev = torch.get_default_dtype()
    t0 = time.perf_counter()
    try:
        torch.set_default_dtype(torch.float16)
        unet = UNet2DConditionModel.from_config(cfg)
    except Exception:
        torch.set_default_dtype(prev)
        unet = UNet2DConditionModel.from_config(cfg)
    finally:
        torch.set_default_dtype(prev)
    unet = unet.half().cuda().eval()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"UNet built from {src}: {n_params/1e9:.2f}B params, {time.perf_counter()-t0:.1f}s", flush=True)
    return unet, src, n_params


def make_inputs(B, latent=128):
    d = dict(device="cuda", dtype=torch.float16)
    return dict(
        sample=torch.randn(B, 4, latent, latent, **d),
        timestep=torch.tensor(500, device="cuda"),
        encoder_hidden_states=torch.randn(B, 77, 2048, **d),
        added_cond_kwargs=dict(text_embeds=torch.randn(B, 1280, **d), time_ids=torch.randn(B, 6, **d)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--modes", nargs="+", default=["eager", "compile_default", "compile_graphs"])
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--latent", type=int, default=128, help="latent H=W; 128 corresponds to 1024x1024 images")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sdxl_unet.json"))
    args = ap.parse_args()

    unet, src, n_params = build_unet()
    results = []
    for mode in args.modes:
        torch._dynamo.reset()
        for B in args.batches:
            inputs = make_inputs(B, args.latent)

            def make_fn(mode=mode, inputs=inputs):
                if mode == "eager":
                    return lambda: unet(**inputs).sample
                cm = torch.compile(unet, mode=("reduce-overhead" if mode == "compile_graphs" else "default"))
                return lambda: cm(**inputs).sample

            results.append(run_mode(
                f"sdxl_unet/{mode}/B{B}", make_fn, iters=args.iters,
                profile_steps=2,
                extra={"workload": "sdxl_unet", "mode": mode, "batch": B, "latent": args.latent, "dtype": "fp16"},
            ))
            del inputs
            torch.cuda.empty_cache()

    save_results(args.out, {"env": env_info(), "model_source": src, "params": n_params, "results": results})


if __name__ == "__main__":
    main()
