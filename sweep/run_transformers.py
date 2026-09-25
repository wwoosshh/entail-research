"""Sweep one engine (transformers) over (fact x backend) for one model, by the rules in PROTOCOL.md.

Three runs per pair: the fact binding (A), the same again (A', the control), and the fact removed (B). A == A'
must hold or the pair is void; then A == B means the backend never read the declaration.

Usage: python sweep/run_transformers.py <model-dir> [--out results/<name>.json]
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402

from facts import sweepable  # noqa: E402

transformers.utils.logging.disable_progress_bar()

BACKENDS = ["eager", "sdpa", "flex_attention", "paged|eager", "paged|sdpa"]
PROMPTS = ["Explain in three sentences why the sky is blue.", "Write a haiku about a quiet library.",
           "List five differences between TCP and UDP.", "What is the capital of Australia?",
           "Solve for x: 3x + 7 = 25.", "Describe photosynthesis simply."]
N_NEW = 24


def load(model_dir, raw_cfg, backend):
    """A fresh model with this config. Paged backends are reached through generate_batch, not from_pretrained."""
    cfg = AutoConfig.from_pretrained(model_dir, **{k: v for k, v in raw_cfg.items() if k != "architectures"})
    impl = backend.split("|")[-1] if backend.startswith("paged|") else backend
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation=impl).cuda().eval()
    return model


def decode(model, ids, backend):
    if backend.startswith("paged|"):
        gc = GenerationConfig(max_new_tokens=N_NEW, do_sample=False, eos_token_id=None)
        with torch.no_grad():
            res = model.generate_batch(inputs=ids, generation_config=gc)
        out = []
        for i in range(len(ids)):
            r = res[f"req_{i}"] if f"req_{i}" in res else list(res.values())[i]
            out.append(list(r.generated_tokens)[:N_NEW])
        return out
    seqs = []
    for x in ids:
        with torch.no_grad():
            g = model.generate(torch.tensor([x], device="cuda"), max_new_tokens=N_NEW, do_sample=False)
        seqs.append(g[0, len(x):].tolist())
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out")
    ap.add_argument("--fact", help="sweep only this fact (for re-running one row)")
    args = ap.parse_args()
    model_dir = os.path.expanduser(args.model)
    name = os.path.basename(model_dir.rstrip("/"))
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                        f"transformers_{name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        base_cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True), add_special_tokens=False).input_ids
           for p in PROMPTS] if tok.chat_template else [tok(p).input_ids for p in PROMPTS]

    facts = [f for f in sweepable(base_cfg) if not args.fact or f["name"] == args.fact]
    print(f"{name}: {len(facts)} sweepable fact(s): {[f['name'] for f in facts]}", flush=True)
    rows = []
    for f in facts:
        bound = copy.deepcopy(base_cfg)
        f["bind"](bound)
        gone = copy.deepcopy(base_cfg)
        f["bind"](gone)
        f["remove"](gone)
        for backend in BACKENDS:
            t0 = time.time()
            row = {"fact": f["name"], "kind": f["kind"], "model": name, "engine": "transformers",
                   "engine_version": transformers.__version__, "backend": backend}
            try:
                model = load(model_dir, bound, backend)
                a = decode(model, ids, backend)
                a2 = decode(model, ids, backend)
                del model
                torch.cuda.empty_cache()
                model = load(model_dir, gone, backend)
                b = decode(model, ids, backend)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                row.update(verdict="void", why=f"{type(e).__name__}: {str(e)[:160]}")
                rows.append(row)
                print(f"  {f['name']:26} {backend:14} void  {row['why']}"[:140], flush=True)
                continue
            if a != a2:
                row.update(verdict="void", why="the control run did not reproduce")
            else:
                row.update(verdict="dropped" if a == b else "honoured",
                           differing_prompts=sum(1 for x, y in zip(a, b) if x != y))
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"  {f['name']:26} {backend:14} {row['verdict']:8} "
                  f"differs={row.get('differing_prompts')}  {row['seconds']}s", flush=True)

    res = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol": "sweep/PROTOCOL.md", "model": model_dir,
           "prompts": len(PROMPTS), "new_tokens": N_NEW, "rows": rows,
           "dropped": sum(1 for r in rows if r["verdict"] == "dropped"),
           "honoured": sum(1 for r in rows if r["verdict"] == "honoured"),
           "void": sum(1 for r in rows if r["verdict"] == "void")}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\ndropped {res['dropped']}, honoured {res['honoured']}, void {res['void']} -> {out_path}")


if __name__ == "__main__":
    main()
