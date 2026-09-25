"""Detection check for #12: after a restore, the decode position must equal the declared valid KV length."""
import os
import sys

import entail as rc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case  # noqa: E402


def _decode_checked(ctx, session, pos):
    valid = session["kv_count"]  # the restore declares how many KV entries are valid
    rc.require(pos == valid, "decode after session restore",
               f"position {pos} does not follow the valid KV length {valid} (Valid.length)", at="load")
    return case._decode(ctx, session["kv"], session["tokens"][-1], pos)


def _defect(ctx):
    s = case._saved_session(ctx)
    _decode_checked(ctx, s, len(s["tokens"]))


def _fixed(ctx):
    s = case._saved_session(ctx)
    _decode_checked(ctx, s, s["kv_count"])


ARMS = {"W": {"mode": "load", "checks": "declared valid length vs resume position",
              "defect": _defect, "fixed": _fixed}}
