"""M15.8 replay of the Llama 3 stop-id class (April 2024): the model's answers end with an id that the file the
engine reads does not list. Here: a copy of Llama-3.2-3B-Instruct (symlinked weights) whose generation_config.json
lists only 128001 (<|end_of_text|>), while config.json still declares [128001, 128008, 128009] and the chat answers
end with 128009 (<|eot_id|>). transformers' generate() reads generation_config.json alone, so without entail it runs
past the end of every answer; with entail on, the transformers_stops adapter adds the ends config.json declares.

Usage: python testbed/m15/stops_replay.py <model folder> <out.json> [max_new_tokens]
Writes: per prompt the tokens generated, whether 128009 appeared before the end, the ids the model's generation
config held after loading, and the entail record lines (from ENTAIL_RECORD, when set).
"""
import json
import os
import sys
import tempfile
import time

PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Name three primary colours.",
           "Write one sentence about the sea."]


def make_copy(src):
    """A folder that is the model with one file changed: every file symlinked, generation_config.json rewritten."""
    dst = tempfile.mkdtemp(prefix="stops_replay_")
    for name in os.listdir(src):
        if name == "generation_config.json":
            continue
        os.symlink(os.path.join(src, name), os.path.join(dst, name))
    gen = json.load(open(os.path.join(src, "generation_config.json"), encoding="utf-8"))
    gen["eos_token_id"] = 128001
    json.dump(gen, open(os.path.join(dst, "generation_config.json"), "w", encoding="utf-8"), indent=1)
    return dst


def main(src, out, max_new=160, as_is=False):
    """`as_is`: run the folder unchanged (a model whose own files disagree about the end, no edit needed)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    folder = src if as_is else make_copy(src)
    tok = AutoTokenizer.from_pretrained(folder)
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16, device_map="cuda")
    load_s = time.perf_counter() - t
    held = model.generation_config.eos_token_id
    rows = []
    for p in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to("cuda")
        with torch.no_grad():
            out_ids = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        new = out_ids[0, ids.shape[1]:].tolist()
        special = set(getattr(tok, "all_special_ids", []) or [])
        firsts = [i for i, t in enumerate(new) if t in special]
        first_eot = firsts[0] if firsts else None
        rows.append({"prompt": p, "generated": len(new), "first_eot_at": first_eot,
                     "first_special_id": new[first_eot] if first_eot is not None else None,
                     "ran_to_max": len(new) >= max_new, "text": tok.decode(new, skip_special_tokens=False)[:300]})
    result = {"model": src, "as_is": as_is, "entail": os.environ.get("ENTAIL", "off"), "held_eos_after_load": held,
              "load_s": load_s, "max_new_tokens": max_new, "eos_token": getattr(tok, "eos_token", None),
              "eos_token_id": getattr(tok, "eos_token_id", None), "rows": rows}
    rec = os.environ.get("ENTAIL_RECORD")
    if rec and os.path.isfile(rec):
        result["record"] = [json.loads(l) for l in open(rec, encoding="utf-8") if '"timing"' not in l]
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k != "record" and k != "rows"}),
          [(r["generated"], r["first_eot_at"], r["ran_to_max"]) for r in rows])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 160, as_is="asis" in sys.argv[4:])
