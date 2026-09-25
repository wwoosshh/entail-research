"""Detection check for #4: reduction-state tags (spmd_types notation), simulated in one process.

The all_reduce itself is not needed to check the contract: it turns P into R. A consumer that reduces again
declares that it takes P.
"""
import entail as rc


@rc.boundary(name="dp_gather_partial (reduces its input)", y=rc.Reduction("P"))
def reduce_again(*, y):
    return y


@rc.boundary(name="consumer of a replicated value", y=rc.Reduction("R"))
def use_replicated(*, y):
    return y


def _after_all_reduce(ctx):
    y = ctx["x"][:, :32] @ ctx["w"][:, :32].T
    rc.tag(y, rc.Reduction("P"))
    y = y.clone()  # stands for the all_reduce output buffer
    return rc.tag(y, rc.Reduction("R"))


def _defect(ctx):
    reduce_again(y=_after_all_reduce(ctx))


def _fixed(ctx):
    use_replicated(y=_after_all_reduce(ctx))


ARMS = {"W": {"mode": "debug", "checks": "R/P tag at the second reduction", "defect": _defect, "fixed": _fixed}}
