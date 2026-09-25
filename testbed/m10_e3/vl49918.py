"""M10 E3, vllm-project/vllm#49918 (testbed/M10_PROTOCOL.md 3.3): with speculative decoding (ngram, K=7) and the default
CUDA graph mode, a prefill of exactly K+1 = 8 prompt tokens is dispatched as a uniform spec-decode batch; models with
recurrent state never write that request's state and produce garbage. The report confirmed GDN hybrids (Qwen3-Next,
Qwen3.6) and wrote "Mamba-family presumably likewise"; the recurrent-state model that fits here is
NVIDIA-Nemotron-3-Nano-4B-BF16 (a Mamba-2 hybrid). Raw completions (token ids, 32 greedy tokens) of an 8-token and
a 9-token prompt, with and without speculative decoding, each run in its own process.
Reproduced: the 8-token prompt's output with ngram differs from the one without, and the 9-token prompt's does not.
Run in ~/venvs/vllm: python testbed/m10_e3/vl49918.py <out.json> <model dir>
"""
import json
import os
import subprocess
import sys

TEXT = "Write one sentence about the quiet library near the old harbor and the ships"


def worker(mode, model, out):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    ids = AutoTokenizer.from_pretrained(model)(TEXT, add_special_tokens=False).input_ids
    kw = {}
    if mode == "spec":
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": 7, "prompt_lookup_max": 4,
                                    "prompt_lookup_min": 2}
    llm = LLM(model=model, max_model_len=1024, gpu_memory_utilization=0.85, seed=0, disable_log_stats=True, **kw)
    res = {}
    for n in (8, 9):
        o = llm.generate([TokensPrompt(prompt_token_ids=ids[:n])], SamplingParams(max_tokens=32, temperature=0),
                         use_tqdm=False)[0].outputs[0]
        res[str(n)] = {"ids": list(o.token_ids), "text": o.text}
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3], sys.argv[4])
    out, model = sys.argv[1], sys.argv[2]
    parts = {}
    for mode in ("plain", "spec"):
        p = f"{out}.{mode}.json"
        r = subprocess.run([sys.executable, __file__, "--worker", mode, model, p], capture_output=True, text=True)
        parts[mode] = json.load(open(p, encoding="utf-8")) if r.returncode == 0 and os.path.exists(p) else \
            {"error": (r.stderr or "")[-1500:]}
    ok = all("error" not in parts[m] for m in parts)
    row = {"entail": os.environ.get("ENTAIL", "off"), "model": os.path.basename(model.rstrip("/")), **parts}
    if ok:
        row["8_differs"] = parts["plain"]["8"]["ids"] != parts["spec"]["8"]["ids"]
        row["9_differs"] = parts["plain"]["9"]["ids"] != parts["spec"]["9"]["ids"]
    row["reproduced"] = bool(ok and row["8_differs"] and not row["9_differs"])
    json.dump(row, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps({"reproduced": row["reproduced"], "8_differs": row.get("8_differs"),
                                "9_differs": row.get("9_differs"),
                                "errors": {m: parts[m]["error"][-300:] for m in parts if "error" in parts[m]}},
                               ensure_ascii=False))


if __name__ == "__main__":
    main()
