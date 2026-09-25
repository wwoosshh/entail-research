"""Detection check for #7 with entail (W arm: a declared tie must match the checkpoint tensors at load time)."""
import entail as rc


def _defect(ctx):
    sd = ctx["state"]
    # the trusting loader asserts its assumption: config says tied, so the checkpoint head must equal the embedding
    rc.check_tied(True, sd["model.embed_tokens.weight"], sd["lm_head.weight"], "loader (declared tie)")


def _fixed(ctx):
    sd = ctx["state"]
    tied = bool((sd["model.embed_tokens.weight"] == sd["lm_head.weight"]).all())
    # the fixed loader decides from the tensors and declares what it actually did
    rc.check_tied(tied, sd["model.embed_tokens.weight"], sd["lm_head.weight"], "loader (tie decided from tensors)")


ARMS = {"W": {"mode": "load", "checks": "declared tie vs checkpoint tensors", "defect": _defect, "fixed": _fixed}}
