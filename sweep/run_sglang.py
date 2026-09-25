"""Sweep one (fact, backend) pair in SGLang, by the rules in PROTOCOL.md.

One pair per process on purpose: when an SGLang engine fails to start it takes its whole process group with it,
so a loop inside one process would lose every later pair. audits-style shell loop drives this.

The fact is perturbed by building two model directories whose weights are symlinks to the real checkpoint and
whose config.json differs only in that fact: one where it binds, one where it is gone.

Usage: python sweep/run_sglang.py <model-dir> <fact> <backend>
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facts import FACTS  # noqa: E402

PROMPTS = ["Explain in three sentences why the sky is blue.", "Write a haiku about a quiet library.",
           "List five differences between TCP and UDP.", "What is the capital of Australia?",
           "Solve for x: 3x + 7 = 25.", "Describe photosynthesis simply."]
N_NEW = 24
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def model_copy(model_dir, cfg):
    """A model directory that shares the weights and carries a different config."""
    d = tempfile.mkdtemp(prefix="sweep_sglang_")
    for name in os.listdir(model_dir):
        if name == "config.json":
            continue
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            os.symlink(src, os.path.join(d, name))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return d


def decode(model_path, backend, runs, one_at_a_time=False):
    """Start an engine on this directory, decode the prompts `runs` times, shut down.

    `one_at_a_time` sends each prompt as its own request. A server batches whatever arrives together, and the
    batch a prompt lands in changes its numerics, so two identical calls need not give identical tokens. That
    is what made the triton pairs void on the first sweep; sending one prompt per request removes the variable.
    """
    import sglang as sgl

    engine = sgl.Engine(model_path=model_path, attention_backend=backend, mem_fraction_static=0.7,
                        max_total_tokens=4096, log_level="error", disable_cuda_graph=True)
    try:
        out = []
        for _ in range(runs):
            if one_at_a_time:
                out.append([engine.generate(p, {"max_new_tokens": N_NEW, "temperature": 0})["text"]
                            for p in PROMPTS])
            else:
                res = engine.generate(PROMPTS, {"max_new_tokens": N_NEW, "temperature": 0})
                out.append([r["text"] for r in res])
        return out
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass


def score(model_path, backend):
    """Log-probability of a fixed set of tokens under this configuration (teacher forcing, no generation).

    Token identity cannot judge a fact whose binding value squashes the logits: the argmax then flips on
    ordinary noise and even the control run stops reproducing (PROTOCOL.md revision 2). Scoring the *same*
    tokens removes the divergence entirely - there is no sequence to diverge - and gives a distance instead of
    a yes/no. This is the comparator the earlier SGLang survey used.
    """
    import sglang as sgl

    engine = sgl.Engine(model_path=model_path, attention_backend=backend, mem_fraction_static=0.7,
                        max_total_tokens=4096, log_level="error", disable_cuda_graph=True)
    try:
        # logprob_start_len=0 is required: without it the input tokens come back with no log-probabilities
        res = engine.generate(PROMPTS, {"max_new_tokens": 1, "temperature": 0}, return_logprob=True,
                              logprob_start_len=0)
        out = []
        for r in res:
            lps = r["meta_info"].get("input_token_logprobs") or []
            out.append([t[0] for t in lps if t and t[0] is not None])
        return out
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass


def distance(x, y):
    """Mean absolute log-probability difference over the tokens both runs scored."""
    pairs = [(a, b) for xs, ys in zip(x, y) for a, b in zip(xs, ys)]
    return sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None


def main():
    model_dir, fact_name, backend = os.path.expanduser(sys.argv[1]), sys.argv[2], sys.argv[3]
    one = "--one-at-a-time" in sys.argv
    use_logprob = "--logprob" in sys.argv
    name = os.path.basename(model_dir.rstrip("/"))
    fact = FACTS[fact_name]
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        base = json.load(f)
    if not fact["applies"](base):
        print(f"SKIP {fact_name}: {name} does not declare it", flush=True)
        return
    bound_cfg = copy.deepcopy(base)
    fact["bind"](bound_cfg)
    gone_cfg = copy.deepcopy(base)
    fact["bind"](gone_cfg)
    fact["remove"](gone_cfg)

    row = {"fact": fact_name, "kind": fact["kind"], "model": name, "engine": "sglang", "backend": backend,
           "one_at_a_time": one, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    bound_dir = gone_dir = None
    t0 = time.time()
    try:
        bound_dir = model_copy(model_dir, bound_cfg)
        gone_dir = model_copy(model_dir, gone_cfg)
        if use_logprob:
            a = score(bound_dir, backend)
            a2 = score(bound_dir, backend)
            b = score(gone_dir, backend)
            noise, signal = distance(a, a2), distance(a, b)
            row.update(noise=noise, signal=signal,
                       ratio=(signal / noise) if noise else None)
            # the control gives the yardstick: a difference the repeat cannot tell apart is not a difference
            if signal is None or noise is None:
                row.update(verdict="void", why="no scored tokens")
            elif signal <= max(noise, 1e-6):
                row.update(verdict="dropped")
            elif signal >= 3 * max(noise, 1e-6):
                row.update(verdict="honoured")
            else:
                row.update(verdict="void", why=f"between the bands: signal {signal:.4g}, noise {noise:.4g}")
        else:
            a, a2 = decode(bound_dir, backend, runs=2, one_at_a_time=one)
            (b,) = decode(gone_dir, backend, runs=1, one_at_a_time=one)
            if a != a2:
                row.update(verdict="void", why="the control run did not reproduce")
            else:
                row.update(verdict="dropped" if a == b else "honoured",
                           differing_prompts=sum(1 for x, y in zip(a, b) if x != y))
    except Exception as e:
        row.update(verdict="void", why=f"{type(e).__name__}: {str(e)[:160]}")
    finally:
        for d in (bound_dir, gone_dir):
            if d:
                shutil.rmtree(d, ignore_errors=True)
    row["seconds"] = round(time.time() - t0, 1)
    os.makedirs(RESULTS, exist_ok=True)
    suffix = ("_logprob" if use_logprob else "") + ("_single" if one else "")
    with open(os.path.join(RESULTS, f"sglang_{name}_{fact_name}_{backend}{suffix}.json"), "w",
              encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    extra = (f"signal={row['signal']:.4g} noise={row['noise']:.4g}"
             if row.get("signal") is not None and row.get("noise") is not None
             else f"differs={row.get('differing_prompts')}")
    print(f"RESULT {fact_name} {backend} {row['verdict']} {extra} {row['seconds']}s {row.get('why', '')}",
          flush=True)


if __name__ == "__main__":
    main()
