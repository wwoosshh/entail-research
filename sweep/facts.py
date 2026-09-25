"""Which declared facts can be swept, and how to make each one bind or go away.

A fact belongs here only if it can be taken off and put back (PROTOCOL.md section 3): the sweep decides whether
a backend read it by comparing a run where it binds with a run where it is gone. Facts that cannot be removed
from a config - reduction state, position frame - are not sweepable and go straight to the deep trace.

Each entry says:
  applies(cfg)  does this model declare the fact at all
  bind(cfg)     set it to a value that must change the output if honoured
  remove(cfg)   take it away, changing nothing else
`cfg` is a plain dict read from config.json; both functions edit it in place.
"""


def _text(cfg):
    """Where the per-layer fields live: some configs nest them under text_config."""
    return cfg.get("text_config", cfg)


def _has(cfg, key):
    return _text(cfg).get(key) is not None


FACTS = {}


def fact(name, kind, why):
    def deco(fns):
        FACTS[name] = {"name": name, "kind": kind, "why": why, **fns}
        return fns

    return deco


fact("attn_logit_softcapping", "PROPERTY", "attention logits are capped before softmax")({
    "applies": lambda cfg: _has(cfg, "attn_logit_softcapping"),
    "bind": lambda cfg: _text(cfg).__setitem__("attn_logit_softcapping", 5.0),
    "remove": lambda cfg: _text(cfg).pop("attn_logit_softcapping", None),
})

fact("final_logit_softcapping", "PROPERTY", "output logits are capped before sampling")({
    "applies": lambda cfg: _has(cfg, "final_logit_softcapping"),
    # 2.0 it is, and this one is measured: at 5.0 and at 10.0 the value does not bind at all (every backend,
    # including the reference ones, reports "dropped" because the cap never changes an argmax). At 2.0 it binds
    # - torch_native and flex_attention change on 6 of 6 prompts - but the squashed logits make near-ties, so
    # some backends stop reproducing their own control run. Token identity cannot judge this fact everywhere;
    # see PROTOCOL.md revision 2.
    "bind": lambda cfg: _text(cfg).__setitem__("final_logit_softcapping", 2.0),
    "remove": lambda cfg: _text(cfg).pop("final_logit_softcapping", None),
})

fact("sliding_window", "PROPERTY", "attention only sees the last N tokens on windowed layers")({
    "applies": lambda cfg: _has(cfg, "sliding_window"),
    "bind": lambda cfg: _text(cfg).__setitem__("sliding_window", 16),
    "remove": lambda cfg: _text(cfg).pop("sliding_window", None),
})

fact("rope_theta", "FRAME", "the base of the rotary position encoding")({
    # transformers 5.17 moves this into rope_parameters; edit both spellings so the sweep works either way
    "applies": lambda cfg: _has(cfg, "rope_theta") or _text(cfg).get("rope_parameters") is not None,
    "bind": lambda cfg: _set_rope(cfg, 1000.0),
    "remove": lambda cfg: _set_rope(cfg, 1000000.0),
})


def _set_rope(cfg, value):
    """transformers 5.17 reads rope_parameters, not rope_theta.

    Setting only `rope_theta` looked like it worked and did nothing: the resolved config still held the
    checkpoint's 10000.0, so the "bound" and "removed" runs were the same config and every backend looked like
    it had dropped the fact - including eager, the reference path. That is what PROTOCOL.md section 4 rule 1 is
    for, and it caught this on the first sweep.
    """
    t = _text(cfg)
    params = dict(t["rope_parameters"]) if isinstance(t.get("rope_parameters"), dict) else {"rope_type": "default"}
    params["rope_theta"] = value
    t["rope_parameters"] = params
    t["rope_theta"] = value


def sweepable(cfg):
    """The facts this model actually declares."""
    return [f for f in FACTS.values() if f["applies"](cfg)]
