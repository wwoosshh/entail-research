"""M15.7: every fused-MoE kernel config SGLang ships for block-quantized FP8 weights, read for the one relation the
kernel needs (KernelConfig.tile_k vs the quantization block: `offs_ks = k_start // group_k` steps the scale once per
tile, so a tile wider than the block skips scales; entail rule tile_over_block). Reads the installed sglang package's
config folder(s); writes a JSON with the counts and every violating entry. Read-only; no GPU.

Usage: python testbed/m15/moe_config_sweep.py testbed/results/m15/moe_config_sweep.json
"""
import glob
import json
import os
import re
import sys


def block_shape_of(name):
    m = re.search(r"block_shape=\[(\d+),\s*(\d+)\]", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main(out):
    import sglang

    root = os.path.dirname(sglang.__file__)
    files = sorted(set(glob.glob(os.path.join(root, "**", "*moe*", "**", "*.json"), recursive=True)))
    files = [f for f in files if "config" in f.lower()]
    result = {"sglang": getattr(sglang, "__version__", "?"), "root": root, "files_seen": len(files),
              "block_fp8_files": 0, "entries": 0, "violations": [], "tile_k_values": {}}
    for f in files:
        bs = block_shape_of(os.path.basename(f))
        if bs is None:
            continue
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        result["block_fp8_files"] += 1
        block_k = bs[1]
        for m_key, cfg in d.items():
            if not isinstance(cfg, dict) or "BLOCK_SIZE_K" not in cfg:
                continue
            result["entries"] += 1
            tk = int(cfg["BLOCK_SIZE_K"])
            result["tile_k_values"][str(tk)] = result["tile_k_values"].get(str(tk), 0) + 1
            if tk > block_k or block_k % tk != 0:
                result["violations"].append({"file": os.path.relpath(f, root), "M": m_key, "block_shape": list(bs),
                                             "BLOCK_SIZE_K": tk, "BLOCK_SIZE_N": cfg.get("BLOCK_SIZE_N")})
    result["violating_files"] = sorted({v["file"] for v in result["violations"]})
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1)
    print(f"sglang {result['sglang']}: {result['block_fp8_files']} block-FP8 config files, {result['entries']} entries, "
          f"{len(result['violations'])} entries with a K tile over the block in {len(result['violating_files'])} files")
    for v in result["violations"]:
        print("  ", v["file"], "M=" + str(v["M"]), "BLOCK_SIZE_K=" + str(v["BLOCK_SIZE_K"]), "block", v["block_shape"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "moe_config_sweep.json")
