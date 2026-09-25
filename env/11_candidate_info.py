"""Metadata only, nothing downloaded: size, licence and gating of the candidate models for the sweep.

Run: python env/11_candidate_info.py   (writes env/candidate_models.json)
"""
import json
import os

from huggingface_hub import HfApi

CANDIDATES = {
    # repo: why it is on the list
    "unsloth/gemma-3-1b-it": "sweep: new facts (rope_local_base_freq, sliding_window_pattern, query_pre_attn_scalar)",
    "unsloth/Llama-3.2-3B-Instruct": "sweep: rope_scaling type llama3 (factor, low/high_freq_factor)",
    "Qwen/Qwen2.5-3B-Instruct": "sweep: sliding_window declared but use_sliding_window=false",
    "microsoft/Phi-3.5-mini-instruct": "sweep: rope_scaling type longrope (short/long factors)",
    "Qwen/Qwen2.5-3B-Instruct-AWQ": "deep arm: AWQ int4 checkpoint, packed weights and Marlin repack",
    "Qwen/Qwen3-4B-FP8": "deep arm: FP8 checkpoint with block scales (scale-format path)",
}


def main():
    api = HfApi()
    out = {}
    for repo, why in CANDIDATES.items():
        try:
            info = api.model_info(repo, files_metadata=True)
            weights = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
            card = info.card_data.to_dict() if info.card_data else {}
            out[repo] = {"why": why, "revision": info.sha, "gated": info.gated,
                         "license": card.get("license"), "weights_gb": round(weights / 1e9, 2)}
        except Exception as e:
            out[repo] = {"why": why, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        r = out[repo]
        print(f"{repo:36} {r.get('weights_gb', '?'):>6} GB  gated={r.get('gated')}  licence={r.get('license')}"
              f"{'  ERROR ' + r['error'] if 'error' in r else ''}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_models.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
