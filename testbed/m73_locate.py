"""M7.3: S8, locating planted defects (ROADMAP M7.3; testbed/PROBLEMS.md 4; LIBRARY_DESIGN.md 8, 12).

Each scenario runs in a process of its own with entail on (entail.enable), its record in results/m73/logs/<name>/.
The scenario says where the fault lies with `entail.locate()` in its own process; the parent says it again from the
record files with `entail locate` (python -m entail.cli locate), as a developer would after the run. Both answers are
compared with where the defect was planted.

  reference          entail not installed. The outputs the other runs are compared with: Qwen3-4B (sdpa), the same
                     model given yarn RoPE scaling done right, the same decode step by step, gemma-2-2b-it (eager,
                     which honours its softcap)
  loc-boundary
    rope_observe     Qwen3-4B; the user restates rope_scaling (yarn, factor 4) after the config is built, which drops
                     rope_theta in transformers 5.17; observe policy (ENTAIL_POLICY=refuse): broken, the run goes on -
                     and transformers then fails to build the yarn rotary embedding (TypeError: its base is None), so
                     here the defect is loud (vLLM falls back to 10000 silently, M3.5). A failed run is a wrong output
    rope_debug       the same in debug mode: refused, the run stops at the write
    config_typo      Qwen3-4B from a copy of its folder whose config.json gives the yarn scaling under a misspelt key
                     (rope_scalling): transformers keeps the key as a plain attribute and builds the model without the
                     scaling, silently (rolebench 15's mechanism on a real model); default policy: broken, the run
                     goes on, and the output is not the one the scaling asked for
    kv_seeded        Qwen3-4B; a restore drops the last token from every layer between two decode steps (M5.1's
                     seed), KV container contract installed, default policy: broken, the run goes on
    code_rb01        rolebench 01 with producer signatures (M4.3), debug mode: refused at the second reader
    code_rb04        rolebench 04 the same: a replicated value reduced again
    op_transpose     rolebench 01's buffer transposed between its producer and its reader (a planted plumbing op),
                     debug mode inside operation-level propagation: the reader names the operation
  loc-inside         (every boundary keeps its meaning; a kernel computes wrong) diagnosis mode: debug, operation-level
                     propagation, and three layers compared with references on the same inputs - attention (an
                     explicit float32 attention), the MLP (float32 recomputation) and RMSNorm (float32 formula)
    inside_attention the "sdpa" attention function computes with its softmax scale 15% too large
    inside_mlp       the MLP computes GELU where the model's SiLU belongs
    healthy          nothing planted: the same diagnosis finds nothing, and no layer differs (no false narrowing)
  loc-unchecked
    unchecked        gemma-2-2b-it (softcap 50, sliding window 4096) given an attention implementation the capability
                     table does not know ("planted": its scale twice what it should be; 15% left its output as it
                     was): the attention boundary cannot be checked, so it and the layers beside it must stay suspect
  cost               Qwen3-4B, 64 greedy tokens, eager and sdpa: off, debug (every check), debug with propagation,
                     debug with propagation and the three layers watched (S4: the diagnosis site at most 2x)

Run in ~/venvs/gpu: python testbed/m73_locate.py  (all), or python testbed/m73_locate.py <scenario>
Writes testbed/results/m73/<scenario>.json, SUMMARY.json and SUMMARY.md.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENTAIL_DIR = os.path.join(ROOT, "entail")
# M9.1: TESTBED_RESULTS moves the results (a re-run on the final code keeps each milestone's own results)
RESULTS = os.path.abspath(os.environ.get("TESTBED_RESULTS") or os.path.join(HERE, "results"))
OUT = os.path.join(RESULTS, "m73")
QWEN = os.path.expanduser("~/models/Qwen3-4B")
GEMMA = os.path.expanduser("~/models/gemma-2-2b-it")
QUESTION = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many "
            "clips did Natalia sell altogether in April and May? Explain step by step.")
N_NEW = 32
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
SCALE_ERROR = 1.15
# gemma-2-2b-it kept its 32 greedy tokens with the scale 15% off (832 calls of the planted function: every layer at
# every step; results/m73/unchecked_scale_x1.15.json), so the unchecked case plants a scale twice as large
UNCHECKED_SCALE_ERROR = 2.0

# scenario -> (entail mode, policy, what must be said)
SCENARIOS = {
    "rope_observe": ("load", "refuse", {"kind": "boundary", "at": "load:transformers.config.rope_scaling"}),
    "rope_debug": ("debug", "refuse", {"kind": "boundary", "at": "load:transformers.config.rope_scaling"}),
    "config_typo": ("load", "resolve", {"kind": "boundary", "at": "load:transformers.config"}),
    "kv_seeded": ("load", "resolve", {"kind": "boundary", "at": "container:transformers.cache_update"}),
    "code_rb01": ("debug", "resolve", {"kind": "boundary", "at": "boundary:second-path dequantize"}),
    "code_rb04": ("debug", "resolve", {"kind": "boundary", "at": "boundary:dp_gather_partial (reduces its input)"}),
    "op_transpose": ("debug", "resolve", {"kind": "op", "at": "boundary:second-path dequantize", "op": "t"}),
    "inside_attention": ("debug", "resolve", {"kind": "inside", "layer": "attention"}),
    "inside_mlp": ("debug", "resolve", {"kind": "inside", "layer": "mlp"}),
    "healthy": ("debug", "resolve", {"kind": "healthy"}),
    "unchecked": ("load", "resolve", {"kind": "unchecked", "at": "load:transformers.attention"}),
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --- in a scenario's process -----------------------------------------------------------------------------------------

def _torch():
    import torch
    import transformers

    transformers.utils.logging.disable_progress_bar()
    return torch, transformers


def prompt_ids(tok, text, device="cuda"):
    msgs = [{"role": "user", "content": text}]
    kw = {"enable_thinking": False} if "qwen" in tok.name_or_path.lower() else {}
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def generate(model, ids, n=N_NEW):
    torch, _ = _torch()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return out[0, ids.shape[1]:].tolist()


def stepwise(model, ids, drop_at=None, steps=12):
    """Decode token by token; at step `drop_at` a restore loses the last token of every layer (M5.1's seed)."""
    torch, _ = _torch()
    with torch.no_grad():
        res = model(ids, use_cache=True)
    cache, step, text = res.past_key_values, res.logits[:, -1:].argmax(-1), []
    for i in range(steps):
        if i == drop_at:
            for layer in cache.layers:
                layer.keys = layer.keys[..., :-1, :].contiguous()
                layer.values = layer.values[..., :-1, :].contiguous()
        with torch.no_grad():
            res = model(step, past_key_values=cache, use_cache=True)
        step = res.logits[:, -1:].argmax(-1)
        text.append(int(step))
    return text


def manual_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, softcap=None,
                     sliding_window=None, scale_error=1.0, **kwargs):
    """Attention written out in float32, for any mask transformers hands over (None: causal; boolean: keep; float:
    added). The reference for the attention layer, and - with scale_error - a planted kernel."""
    torch, _ = _torch()
    q, k, v = query.float(), key.float(), value.float()
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    scale = (scaling if scaling is not None else q.shape[-1] ** -0.5) * scale_error
    s = (q @ k.transpose(-1, -2)) * scale
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    lq, lk = s.shape[-2], s.shape[-1]
    if attention_mask is None:
        i = torch.arange(lq, device=s.device)[:, None] + (lk - lq)
        j = torch.arange(lk, device=s.device)[None, :]
        keep = j <= i
        if sliding_window:
            keep = keep & (j > i - sliding_window)
        s = s.masked_fill(~keep, float("-inf"))
    elif attention_mask.dtype == torch.bool:
        s = s.masked_fill(~attention_mask[..., :lk], float("-inf"))
    else:
        s = s + attention_mask[..., :lk].float()
    p = torch.softmax(s, dim=-1)
    return (p @ v).transpose(1, 2).contiguous().to(query.dtype), None


def mlp_reference(self, x):
    torch, _ = _torch()
    f = torch.nn.functional
    x32 = x.float()
    gate = f.linear(x32, self.gate_proj.weight.float())
    up = f.linear(x32, self.up_proj.weight.float())
    return f.linear(f.silu(gate) * up, self.down_proj.weight.float()).to(x.dtype)


def rmsnorm_reference(self, hidden_states):
    torch, _ = _torch()
    h = hidden_states.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return (self.weight.float() * h).to(hidden_states.dtype)


def watched_layers():
    """The diagnosis: the three layers compared with their references (diagnose.watch), as one context."""
    import contextlib

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3 import modeling_qwen3 as q3

    from entail import diagnose

    stack = contextlib.ExitStack()
    stack.enter_context(diagnose.watch(ALL_ATTENTION_FUNCTIONS, "sdpa", manual_attention, label="attention"))
    stack.enter_context(diagnose.watch(q3.Qwen3MLP, "forward", mlp_reference, label="mlp"))
    stack.enter_context(diagnose.watch(q3.Qwen3RMSNorm, "forward", rmsnorm_reference, label="rmsnorm"))
    return stack


def reference_run():
    torch, transformers = _torch()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = {"when": now(), "torch": torch.__version__, "transformers": transformers.__version__}
    tok = AutoTokenizer.from_pretrained(QWEN)
    ids = prompt_ids(tok, QUESTION)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda").eval()
    out["qwen"] = generate(model, ids)
    out["qwen_text"] = tok.decode(out["qwen"])
    out["qwen_stepwise"] = stepwise(model, ids)
    del model
    torch.cuda.empty_cache()
    cfg = AutoConfig.from_pretrained(QWEN)
    right = dict(cfg.rope_parameters)
    right.update(YARN)   # the user's yarn scaling, with the model's rope_theta kept (the new name, written whole)
    cfg.rope_parameters = right
    model = AutoModelForCausalLM.from_pretrained(QWEN, config=cfg, dtype=torch.bfloat16, device_map="cuda").eval()
    out["qwen_yarn_rope_parameters"] = {k: v for k, v in dict(model.config.rope_parameters).items()}
    out["qwen_yarn"] = generate(model, ids)
    out["qwen_yarn_text"] = tok.decode(out["qwen_yarn"])
    del model
    torch.cuda.empty_cache()
    gtok = AutoTokenizer.from_pretrained(GEMMA)
    gids = prompt_ids(gtok, QUESTION)
    model = AutoModelForCausalLM.from_pretrained(GEMMA, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    out["gemma"] = generate(model, gids)
    out["gemma_text"] = gtok.decode(out["gemma"])
    return out


def reference():
    with open(os.path.join(OUT, "reference.json"), encoding="utf-8") as f:
        return json.load(f)


def scenario_run(name):
    """One planted defect, entail already on in this process (the parent set ENTAIL, ENTAIL_POLICY, the log folder)."""
    import entail

    entail.enable(os.environ["ENTAIL"], os.environ.get("ENTAIL_POLICY", "resolve"))
    from entail import core, diagnose

    out = {"when": now(), "mode": core.mode(), "policy": core.policy()}
    ref = reference()
    wrong = None
    try:
        if name.startswith("rope"):
            wrong = rope_scenario(out, ref)
        elif name == "kv_seeded":
            wrong = kv_scenario(out, ref)
        elif name == "config_typo":
            wrong = typo_scenario(out, ref)
        elif name.startswith("code_") or name == "op_transpose":
            wrong = code_scenario(name, out)
        elif name in ("inside_attention", "inside_mlp", "healthy"):
            wrong = inside_scenario(name, out, ref)
        elif name == "unchecked":
            wrong = unchecked_scenario(out, ref)
    except core.RoleError as e:
        out["stopped"] = str(e)[:1200]
    except Exception as e:  # noqa: BLE001 - the run failed: that is a wrong result, and where it lies is the question
        out["failed"] = f"{type(e).__name__}: {str(e)[:600]}"
        wrong = True
    out["output_wrong"] = wrong
    found = diagnose.locate(output_wrong=wrong, say=True)
    out["located"] = found.to_json()
    out["lines"] = found.lines()
    return out


def rope_scenario(out, ref):
    torch, _ = _torch()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(QWEN)
    cfg.rope_scaling = dict(YARN)   # restated after the config was built: the old name
    out["rope_parameters_after_the_write"] = {k: v for k, v in dict(cfg.rope_parameters).items()}
    tok = AutoTokenizer.from_pretrained(QWEN)
    ids = prompt_ids(tok, QUESTION)
    model = AutoModelForCausalLM.from_pretrained(QWEN, config=cfg, dtype=torch.bfloat16, device_map="cuda").eval()
    out["tokens"] = generate(model, ids)
    out["text"] = tok.decode(out["tokens"])
    return out["tokens"] != ref["qwen_yarn"]


def typo_scenario(out, ref):
    import shutil
    import tempfile

    torch, _ = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    folder = tempfile.mkdtemp(prefix="m73_typo_")
    try:
        for f in os.listdir(QWEN):
            if f != "config.json":
                os.symlink(os.path.join(QWEN, f), os.path.join(folder, f))
        with open(os.path.join(QWEN, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["rope_scalling"] = dict(YARN)   # the user's yarn scaling, under a misspelt key
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        out["planted"] = "config.json gives the yarn scaling under a misspelt key, rope_scalling"
        tok = AutoTokenizer.from_pretrained(QWEN)
        ids = prompt_ids(tok, QUESTION)
        model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16, device_map="cuda").eval()
        out["rope_parameters_in_the_model"] = {k: v for k, v in dict(model.config.rope_parameters).items()}
        out["tokens"] = generate(model, ids)
        out["text"] = tok.decode(out["tokens"])
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return out["tokens"] != ref["qwen_yarn"]


def kv_scenario(out, ref):
    torch, _ = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from entail.adapters import cache_contract

    tok = AutoTokenizer.from_pretrained(QWEN)
    ids = prompt_ids(tok, QUESTION)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda").eval()
    cache_contract.install()
    out["tokens"] = stepwise(model, ids, drop_at=2)
    out["text"] = tok.decode(out["tokens"])
    return out["tokens"] != ref["qwen_stepwise"]


def code_scenario(name, out):
    """Rolebench cases with the producer signatures of M4.3 (testbed/m43_signed.py), debug mode."""
    sys.path.insert(0, HERE)
    import m43_signed as m43

    from entail import diagnose

    folder, signed = {"code_rb01": ("01_reorder_layout", m43.signed_01), "op_transpose": ("01_reorder_layout",
                                                                                          m43.signed_01),
                      "code_rb04": ("04_double_reduce", m43.signed_04)}[name]
    case = m43.load_case(folder)
    setup, defect, fixed = signed(case)
    ctx = setup()
    if name == "op_transpose":
        out["planted"] = "the reordered buffer handed on transposed (.t()) between its producer and its reader"
        from entail.boundaries import boundary
        from entail.facts import Layout

        interleaved = Layout("q8_0", packing="interleaved")
        fresh = boundary("fresh copy of the weight buffer", returns=interleaved)(lambda *, data: data.clone())
        read = boundary("second-path dequantize", takes={"buf": interleaved})(lambda *, buf: case.read_interleaved(buf))
        with diagnose.propagating():
            data = fresh(data=ctx["data"])
            read(buf=data.t().contiguous().t())   # a round trip that is the same bytes, and a transpose on the way
        return None
    defect(ctx)
    return None


def inside_scenario(name, out, ref):
    torch, _ = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3 import modeling_qwen3 as q3

    from entail import diagnose
    from entail.adapters import cache_contract

    tok = AutoTokenizer.from_pretrained(QWEN)
    ids = prompt_ids(tok, QUESTION)
    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda").eval()
    cache_contract.install()
    if name == "inside_attention":
        real = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def planted_sdpa(module, query, key, value, attention_mask, scaling=None, **kwargs):
            scaling = (scaling if scaling is not None else query.shape[-1] ** -0.5) * SCALE_ERROR
            return real(module, query, key, value, attention_mask, scaling=scaling, **kwargs)

        ALL_ATTENTION_FUNCTIONS["sdpa"] = planted_sdpa
        out["planted"] = f"the sdpa attention function computes with its softmax scale x{SCALE_ERROR}"
    elif name == "inside_mlp":
        def planted_mlp(self, x):
            return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x)) * self.up_proj(x))

        q3.Qwen3MLP.forward = planted_mlp
        out["planted"] = "the MLP computes GELU where the model's SiLU belongs"
    with diagnose.propagating(), watched_layers():
        out["tokens"] = generate(model, ids)
    out["text"] = tok.decode(out["tokens"])
    from entail import load

    out["layers"] = load.LEDGER.layers
    return out["tokens"] != ref["qwen"]


def unchecked_scenario(out, ref):
    torch, _ = _torch()
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    calls = [0]

    def planted(module, query, key, value, attention_mask, **kwargs):
        calls[0] += 1
        return manual_attention(module, query, key, value, attention_mask, scale_error=UNCHECKED_SCALE_ERROR, **kwargs)

    AttentionInterface.register("planted", planted)
    AttentionMaskInterface.register("planted", sdpa_mask)
    out["planted"] = (f"an attention implementation the capability table does not know, its scale "
                      f"x{UNCHECKED_SCALE_ERROR}")
    tok = AutoTokenizer.from_pretrained(GEMMA)
    ids = prompt_ids(tok, QUESTION)
    model = AutoModelForCausalLM.from_pretrained(GEMMA, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="planted").eval()
    out["attn_implementation"] = model.config._attn_implementation
    out["tokens"] = generate(model, ids)
    out["text"] = tok.decode(out["tokens"])
    out["planted_calls"] = calls[0]
    return out["tokens"] != ref["gemma"]


def cost_run():
    """off / debug / debug + propagation / debug + propagation + the three layers watched; eager and sdpa."""
    torch, transformers = _torch()
    import entail

    entail.enable("debug")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from entail import core, diagnose, load
    from entail.adapters import cache_contract

    cache_contract.install()
    tok = AutoTokenizer.from_pretrained(QWEN)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
    out = {"when": now(), "new_tokens": 64, "reps": 5, "torch": torch.__version__,
           "transformers": transformers.__version__, "prompt": "The capital of France is"}
    arms = ["off", "debug", "debug_propagation", "diagnosis"]
    for impl in ("eager", "sdpa"):
        core.set_mode("off")
        model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.bfloat16, device_map="cuda",
                                                     attn_implementation=impl).eval()

        def once(arm):
            import contextlib

            core.set_mode("off" if arm == "off" else "debug")
            load.LEDGER.layers.clear()
            with contextlib.ExitStack() as stack:
                if arm in ("debug_propagation", "diagnosis"):
                    stack.enter_context(diagnose.propagating())
                if arm == "diagnosis":
                    stack.enter_context(watched_layers())
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    seq = model.generate(ids, max_new_tokens=64, do_sample=False)
                torch.cuda.synchronize()
                return time.perf_counter() - t0, seq[0].tolist()

        for arm in arms:   # warm-up
            once(arm)
        times, outs = {a: [] for a in arms}, {}
        for r in range(5):
            for arm in (arms if r % 2 == 0 else list(reversed(arms))):
                t, seq = once(arm)
                times[arm].append(round(t, 3))
                outs[arm] = seq
        base = statistics.median(times["off"])
        out[impl] = {"seconds": times, "median_x_off": {a: round(statistics.median(times[a]) / base, 3) for a in arms},
                     "same_tokens_as_off": {a: outs[a] == outs["off"] for a in arms}}
        core.set_mode("off")
        del model
        torch.cuda.empty_cache()
    return out


# --- the parent ------------------------------------------------------------------------------------------------------

def child(name, env_extra=None):
    env = dict(os.environ, PYTHONPATH=ENTAIL_DIR)
    for k in ("ENTAIL", "ENTAIL_POLICY", "ENTAIL_LOG_DIR", "ENTAIL_RECORD", "ENTAIL_MANIFESTS"):
        env.pop(k, None)
    env.update(env_extra or {})
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.abspath(__file__), name], env=env, capture_output=True, text=True)
    with open(os.path.join(OUT, f"{name}.stdout.txt"), "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n--- stderr ---\n" + p.stderr[-20000:])
    return p.returncode, round(time.time() - t0, 1)


def locate_from_records(folder, wrong):
    records = sorted(f for f in os.listdir(folder) if f.startswith("record-")) if os.path.isdir(folder) else []
    if not records:
        return None, []
    paths = [os.path.join(folder, r) for r in records]
    cmd = [sys.executable, "-m", "entail.cli", "locate", *paths, "--json"] + (["--wrong"] if wrong else [])
    p = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=ENTAIL_DIR), capture_output=True, text=True)
    try:
        return json.loads(p.stdout), paths
    except ValueError:
        return {"error": p.stdout[-2000:] + p.stderr[-2000:]}, paths


def judged(expect, found):
    """Whether a localization says what the planted defect requires."""
    if not found or "suspects" not in found:
        return False
    kind = expect["kind"]
    if kind == "boundary":
        return found["broken_at"] == expect["at"] and found["suspects"][0] == f"boundary {expect['at']}"
    if kind == "op":
        return any(expect["at"] in x and x.endswith(f"made untrue by {expect['op']}") for x in found["lost_by"])
    if kind == "inside":
        return found["all_intact"] and not found["broken"] and found["suspects"] == [f"inside {expect['layer']}"]
    if kind == "healthy":
        return found["all_intact"] and found["suspects"] == [] and all(" agrees with " in x for x in found["layers"])
    if kind == "unchecked":
        return (any(u.startswith(expect["at"] + " (") for u in found["unchecked"])
                and found["suspects"][0].startswith(f"boundary {expect['at']} and beside it"))
    return False


def main(only=None):
    """Every scenario (only: the ones named, merged into the SUMMARY.json already there)."""
    os.makedirs(OUT, exist_ok=True)
    summary_path = os.path.join(OUT, "SUMMARY.json")
    rows = {}
    if only and os.path.isfile(summary_path):
        rows = json.load(open(summary_path, encoding="utf-8")).get("scenarios", {})
    if not only or "reference" in only:
        code, secs = child("reference")
        rows["reference"] = {"exit": code, "seconds": secs}
    for name, (mode, policy, expect) in SCENARIOS.items():
        if only and name not in only:
            continue
        logs = os.path.join(OUT, "logs", name)
        if os.path.isdir(logs):
            for f in os.listdir(logs):
                os.remove(os.path.join(logs, f))
        code, secs = child(name, {"ENTAIL": mode, "ENTAIL_POLICY": policy, "ENTAIL_LOG_DIR": logs})
        path = os.path.join(OUT, f"{name}.json")
        mine = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
        here = mine.get("located")
        there, records = locate_from_records(logs, mine.get("output_wrong"))
        rows[name] = {"exit": code, "seconds": secs, "mode": mode, "policy": policy, "expect": expect,
                      "output_wrong": mine.get("output_wrong"), "stopped": bool(mine.get("stopped")),
                      "in_process": here, "from_records": there,
                      "records": [os.path.relpath(r, ROOT).replace(os.sep, "/") for r in records],
                      "located_in_process": judged(expect, here), "located_from_records": judged(expect, there)}
        print(f"{name:17} exit {code} wrong={mine.get('output_wrong')} in-process {rows[name]['located_in_process']} "
              f"records {rows[name]['located_from_records']}", flush=True)
    if not only or "cost" in only:
        code, secs = child("cost", {"ENTAIL_LOG_DIR": os.path.join(OUT, "logs", "cost")})
        rows["cost"] = {"exit": code, "seconds": secs}
    done = [n for n in SCENARIOS if n in rows]
    summary = {"when": now(), "scenarios": rows,
               "located": sum(1 for n in done if rows[n]["located_in_process"] and rows[n]["located_from_records"]),
               "of": len(done)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"located {summary['located']} of {summary['of']}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        main(set(sys.argv[2].split(",")))
    elif len(sys.argv) > 1:
        name = sys.argv[1]
        result = reference_run() if name == "reference" else cost_run() if name == "cost" else scenario_run(name)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    else:
        main()
