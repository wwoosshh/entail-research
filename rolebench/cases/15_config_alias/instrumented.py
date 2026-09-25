"""Detection check for #15 with entail (W arm: unknown config keys are an error at load time).

The consumer's known keys are the fields of the installed LlamaConfig, so the check needs no hand-made list.
"""
import json
import os
import sys

import entail as rc
from transformers import LlamaConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

KNOWN = set(LlamaConfig().to_dict()) | {"rope_scaling"}


def _checked_load(ctx, map_alias):
    cfg = json.load(open(os.path.join(ctx["dir"], "config.json")))
    if map_alias and "rope_scale" in cfg:  # the fixed loader maps the alias to the recognized field first
        cfg["rope_scaling"] = {"rope_type": "linear", "factor": float(cfg.pop("rope_scale"))}
    rc.check_config_keys(cfg, KNOWN, "transformers LlamaConfig")
    return cfg


ARMS = {
    "W": {"mode": "load", "checks": "unknown config keys are an error",
          "defect": lambda ctx: _checked_load(ctx, False), "fixed": lambda ctx: _checked_load(ctx, True)},
}
