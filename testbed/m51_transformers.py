"""M5.1 on a real model: the transformers KV container contract after the rules moved to the core.

The same runs as entail/audits/CACHE_CONTRACT.md measured before the move (Qwen3-4B, 64 greedy tokens), plus the two
things the move changed:
  dynamic   contract off and on, eight pairs in alternating order after warming up both, with an off-vs-off
            control: time (medians), identical output, what was checked. Every generate in load mode reuses the process and its books: before M5.1 the second
            one was refused (the layers of the first cache were still compared with the new one).
  static    cache_implementation="static": before M5.1 a check inside the update cost 8x-48x and changed the output,
            because generate compiles that path. Now nothing runs inside the captured region; the cache is checked
            once after the request (check_cache).
  seeded    a restore that loses one token from every layer between two steps: off it runs on, on it is refused.
Writes testbed/results/m51/transformers.json. Run in ~/venvs/gpu: python testbed/m51_transformers.py
"""
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "entail"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from entail import core, kv_contract, load  # noqa: E402
from entail.adapters import cache_contract  # noqa: E402

transformers.utils.logging.disable_progress_bar()
MODEL = os.path.expanduser("~/models/Qwen3-4B")
OUT = os.path.join(RESULTS, "m51", "transformers.json")
N_NEW = 64


def timed(model, ids, cache_implementation=None):
    kw = {"cache_implementation": cache_implementation} if cache_implementation else {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, return_dict_in_generate=True, **kw)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def healthy(model, ids, cache_implementation=None, repeats=8):
    """Pairs of off and on in alternating order, after both were warmed up, and an off-vs-off control measured the
    same way: the spread of the control is the noise the multiplier has to be read against. (The mode is a global
    torch.compile guards on, so the first run after switching recompiles the static path once; that is not what a
    check costs.) Every load-mode generate reuses the same process and books: before M5.1 the second was refused."""
    for mode in ("off", "load", "off", "load"):
        core.set_mode(mode)
        timed(model, ids, cache_implementation)
    kv_contract.reset()
    first = len(load.LEDGER.decisions)
    times, outs = {"off": [], "load": [], "off_control": []}, {"off": [], "load": [], "off_control": []}
    for i in range(repeats):
        for mode in (("off", "load") if i % 2 == 0 else ("load", "off")) + ("off_control",):
            core.set_mode("off" if mode == "off_control" else mode)
            t, out = timed(model, ids, cache_implementation)
            times[mode].append(t)
            outs[mode].append(out)
    core.set_mode("load")
    after = {}
    if cache_implementation == "static":
        last = outs["load"][-1]
        n = int(last.sequences.shape[1]) - 1   # the last token is generated, not written into the cache
        after["check_cache_layers"] = cache_contract.check_cache(last.past_key_values, n)
        after["flushed_on_device"] = cache_contract.flush()
    core.set_mode("off")
    med = {m: sorted(v)[len(v) // 2] for m, v in times.items()}
    ref = outs["off"][0].sequences
    return {"cache": cache_implementation or "dynamic", "pairs": repeats,
            "seconds": {m: [round(t, 3) for t in v] for m, v in times.items()},
            "median": {m: round(v, 3) for m, v in med.items()},
            "multiplier": round(med["load"] / med["off"], 3),
            "control_multiplier": round(med["off_control"] / med["off"], 3),
            "same_output": all(torch.equal(ref, o.sequences) for m in outs for o in outs[m]),
            "ledger_entries": len(load.LEDGER.decisions) - first, **after,
            "update_boundary": kv_contract.stats(cache_contract.BOUNDARY),
            "after_request": kv_contract.stats(cache_contract.AFTER_REQUEST)}


def lost_token(model, ids, steps=6):
    """Decode step by step and drop the last token from every layer at step 2, as a bad restore would."""
    with torch.no_grad():
        res = model(ids, use_cache=True)
    cache, step, text = res.past_key_values, res.logits[:, -1:].argmax(-1), []
    for i in range(steps):
        if i == 2:
            for layer in cache.layers:
                layer.keys = layer.keys[..., :-1, :].contiguous()
                layer.values = layer.values[..., :-1, :].contiguous()
        with torch.no_grad():
            res = model(step, past_key_values=cache, use_cache=True)
        step = res.logits[:, -1:].argmax(-1)
        text.append(int(step))
    return text


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODEL, "new_tokens": N_NEW,
           "torch": torch.__version__, "transformers": transformers.__version__}
    cache_contract.install()
    res["dynamic"] = healthy(model, ids)
    print("dynamic:", json.dumps(res["dynamic"]), flush=True)
    res["static"] = healthy(model, ids, "static")
    print("static: ", json.dumps(res["static"]), flush=True)

    core.set_mode("off")
    res["seeded_off"] = {"finished": True, "tokens": lost_token(model, ids)}
    core.set_mode("load")
    kv_contract.reset()
    try:
        with redirect_stdout(io.StringIO()):
            lost_token(model, ids)
        res["seeded_on"] = {"refused": False}
    except core.RoleError as e:
        res["seeded_on"] = {"refused": True, "error": str(e)[:600]}
    core.set_mode("off")
    print("seeded: off", res["seeded_off"], "| on", json.dumps(res["seeded_on"])[:300], flush=True)
    cache_contract.uninstall()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
