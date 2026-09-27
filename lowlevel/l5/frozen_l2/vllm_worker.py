"""L2 worker: run the probe prompts on one vLLM engine configuration and write what came out.

Runs inside a vLLM venv:  python lowlevel/l2/vllm_worker.py <spec.json> <out.json>
spec: {"model", "engine": {LLM kwargs}, "plans": ["alone", "batched", "cache", "supplied", "image"],
       "prompt_logprobs": bool, "max_tokens", "probes": [{"id", "text"}], "cache_pairs": [{"warm": id, "target": id}],
       "image_probes": [{"id", "image", "text"}]}
"supplied" gives each probe as the same ids and embedding tensor under three prompt_is_token_ids masks, cold and
right after each other one; "image" sends each image probe through the chat interface.
Every request is greedy (temperature 0) with ignore_eos, so every mode produces the same number of tokens; each
step keeps the chosen token and the top-5 log-probabilities. "alone" sends each prompt by itself, "batched" sends
them all at once; both skip reading the prefix cache. "cache" resets the prefix cache, runs the warm-up prompt,
then the target twice (a partial hit on the shared prefix, then a full hit on its own blocks).
"""
import json
import sys
import time


def _lp(d):
    """{token id: logprob} from vLLM's {token id: Logprob}."""
    return {str(k): float(v.logprob) for k, v in (d or {}).items()}


def record(out):
    o = out.outputs[0]
    return {"tokens": [int(t) for t in o.token_ids],
            "logprobs": [_lp(d) for d in (o.logprobs or [])],
            "prompt_tokens": [int(t) for t in (out.prompt_token_ids or [])],
            "prompt_logprobs": [(_lp(d) if d is not None else None) for d in (out.prompt_logprobs or [])],
            "cached_tokens": getattr(out, "num_cached_tokens", None)}


def _embedding_table(model):
    """The input embedding matrix as stored in the checkpoint (the tensor whose name ends in embed_tokens.weight)."""
    import glob
    import os

    from safetensors import safe_open

    for f in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        h = safe_open(f, "pt")
        for k in h.keys():
            if k.endswith("embed_tokens.weight"):
                return h.get_tensor(k)
    raise RuntimeError("no embed_tokens.weight in the checkpoint")


def _image_url(desc):
    """A plain generated picture as a data URL: solid, a square on a background, or vertical stripes."""
    import base64
    import io

    from PIL import Image, ImageDraw

    w, h = desc.get("size", [512, 512])
    if desc["kind"] == "solid":
        img = Image.new("RGB", (w, h), tuple(desc["color"]))
    elif desc["kind"] == "square":
        img = Image.new("RGB", (w, h), tuple(desc["background"]))
        ImageDraw.Draw(img).rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=tuple(desc["color"]))
    else:
        img = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(img)
        for i in range(8):
            d.rectangle([i * w // 8, 0, (i + 1) * w // 8, h], fill=tuple(desc["colors"][i % len(desc["colors"])]))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    from vllm import LLM, SamplingParams

    kw = {"model": spec["model"], "seed": 0, "gpu_memory_utilization": 0.8, "max_model_len": 2048,
          "max_num_seqs": 16, "enable_prefix_caching": True}
    kw.update(spec.get("engine", {}))
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    n = int(spec.get("max_tokens", 24))

    def sp(prompt_lp, skip):
        return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, logprobs=5,
                              prompt_logprobs=(5 if prompt_lp else None), skip_reading_prefix_cache=skip)

    probes = spec["probes"]
    texts = {p["id"]: p["text"] for p in probes}
    plp = bool(spec.get("prompt_logprobs", True))
    res = {"model": spec["model"], "engine": spec.get("engine", {}), "load_s": round(load_s, 2), "plans": {}}
    if "alone" in spec["plans"]:
        res["plans"]["alone"] = {p["id"]: record(llm.generate([p["text"]], sp(plp, True), use_tqdm=False)[0])
                                 for p in probes}
    if "forced" in spec["plans"] and "alone" in res["plans"]:
        # the decode path against the prefill path: the generated tokens fed back teacher-forced in one prefill (no
        # prefix cache, prompt log-probs), so each generated step's distribution is computed again from scratch
        res["plans"]["forced"] = {}
        for p in probes:
            a = res["plans"]["alone"][p["id"]]
            full = a["prompt_tokens"] + a["tokens"]
            o = llm.generate([{"prompt_token_ids": full}], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=5,
                                                                          skip_reading_prefix_cache=True),
                             use_tqdm=False)[0]
            n0 = len(a["prompt_tokens"])
            res["plans"]["forced"][p["id"]] = {"steps": [_lp(o.prompt_logprobs[n0 + i]) for i in range(len(a["tokens"]))]}
    if "batched" in spec["plans"]:
        outs = llm.generate([p["text"] for p in probes], sp(plp, True), use_tqdm=False)
        res["plans"]["batched"] = {p["id"]: record(o) for p, o in zip(probes, outs)}
    if "cache" in spec["plans"]:
        res["plans"]["cache"] = {}
        for pair in spec.get("cache_pairs", []):
            llm.reset_prefix_cache()
            llm.generate([texts[pair["warm"]]], sp(False, False), use_tqdm=False)
            partial = record(llm.generate([texts[pair["target"]]], sp(False, False), use_tqdm=False)[0])
            full = record(llm.generate([texts[pair["target"]]], sp(False, False), use_tqdm=False)[0])
            res["plans"]["cache"][pair["target"]] = {"partial": partial, "full": full, "warm": pair["warm"]}
    if "supplied" in spec["plans"]:
        # the same token ids and the same embedding tensor, differing only in which positions the engine reads as ids
        # (prompt_is_token_ids): requests that mean different things (the tensor holds another sequence's rows). As a
        # caller supplies them, positions that any request gives as embeddings carry the placeholder id 0. Each
        # request must come out the same with a cold prefix cache and right after any of the others
        table = _embedding_table(spec["model"])
        tok = llm.get_tokenizer()
        res["plans"]["supplied"] = {}
        for p in probes:
            text_ids = list(tok(p["text"]).input_ids)
            n = len(text_ids)
            emb = table[text_ids[1:] + text_ids[:1]]
            ids = [0] * (n - 1) + text_ids[-1:]
            masks = {"ids": [True] * n, "embeds": [False] * (n - 1) + [True],
                     "half": [False] * (n // 2) + [True] * (n - n // 2)}

            def req(m):
                return {"prompt_token_ids": ids, "prompt_embeds": emb.clone(), "prompt_is_token_ids": masks[m]}

            cold, warm = {}, {}
            for m in masks:
                llm.reset_prefix_cache()
                cold[m] = record(llm.generate([req(m)], sp(False, False), use_tqdm=False)[0])
            for a in masks:
                for b in masks:
                    if a != b:
                        llm.reset_prefix_cache()
                        llm.generate([req(a)], sp(False, False), use_tqdm=False)
                        warm[f"{b}_after_{a}"] = record(llm.generate([req(b)], sp(False, False), use_tqdm=False)[0])
            res["plans"]["supplied"][p["id"]] = {"cold": cold, "warm": warm}
    if "image" in spec["plans"]:
        res["plans"]["image"] = {}
        for p in spec.get("image_probes", []):
            msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _image_url(p["image"])}},
                                                 {"type": "text", "text": p["text"]}]}]
            o = llm.chat(msgs, sp(False, True), use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0]
            res["plans"]["image"][p["id"]] = record(o)
    json.dump(res, open(sys.argv[2], "w", encoding="utf-8"))
    print("RESULT", json.dumps({"model": spec["model"], "load_s": res["load_s"], "plans": list(res["plans"])}))


if __name__ == "__main__":
    main()
