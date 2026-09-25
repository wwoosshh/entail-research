"""Detection check for #6: a custom mask declares which model properties it implements; the kernel's effective
capability is what the mask declares (the kernel convention: a custom mask replaces the built-in window)."""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402

PROPS = rc.ModelProps(sliding_window=case.W)


def _kernel_checked(mask_implements):
    caps = rc.KernelCaps(sliding_window="sliding_window" in mask_implements)
    rc.check_props(PROPS, caps, f"kernel with a custom mask implementing {sorted(mask_implements)}")


ARMS = {"W": {"mode": "load", "checks": "model property vs effective kernel capability",
              "defect": lambda ctx: _kernel_checked({"causal"}),
              "fixed": lambda ctx: _kernel_checked({"causal", "sliding_window"})}}
