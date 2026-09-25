"""M6.3, fd-vae: a pinned manifest that declares a single-file SDXL checkpoint's latent scale.

A single-file checkpoint states no latent scale (LIBRARY_DESIGN.md 11, M6.1 (2)); its user can declare it the way a
.d.ts declares an untyped library (4.3). waiIllustriousSDXL v160 is an SDXL model, so its latents are SDXL's: the
scaling_factor of the SDXL VAE config the pipeline itself is built with (~/sdxl_local_config/vae/config.json, SDXL
base 1.0's value). The manifest is keyed by the checkpoint's SHA-256 and written to testbed/results/m63/manifests
(ENTAIL_MANIFESTS); nothing is written next to the checkpoint.
Run in WSL ~/venvs/gpu: python testbed/m63_vae_manifest.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))
from entail import manifest  # noqa: E402
from entail.facts import Certainty, Fact, LatentScale, Source  # noqa: E402

CKPT = "/mnt/c/Users/<user>/Desktop/ComfyUI/ComfyUI-new/models/checkpoints/waiIllustriousSDXL_v160.safetensors"
VAE_CONFIG = os.path.expanduser("~/sdxl_local_config/vae/config.json")


def main():
    scale = json.load(open(VAE_CONFIG, encoding="utf-8"))["scaling_factor"]
    fact = Fact("LatentScale", LatentScale(float(scale)), Source("manifest", "m63 fd-vae#LatentScale"),
                Certainty.DECLARED)
    m = manifest.pin(manifest.Manifest(manifest.sha256_of(CKPT), (fact,), file=os.path.basename(CKPT),
                                       fingerprint=manifest.quick_fingerprint(CKPT)))
    m = type(m)(**{**m.__dict__, "evidence": {"LatentScale": [
        f"an SDXL model: its latents are SDXL's, scaling_factor {scale} as in {VAE_CONFIG} (SDXL base 1.0)"]}}) \
        if hasattr(m, "evidence") else m
    out_dir = os.path.join(HERE, "results", "m63", "manifests")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{m.sha256}.json")
    manifest.save(m, path)
    print("wrote", path, "declares", fact.value)


if __name__ == "__main__":
    main()
