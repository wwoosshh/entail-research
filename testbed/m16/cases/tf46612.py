"""M16 case, huggingface/transformers#46612 (testbed/M16_PROTOCOL.md 5): beam search reorders only a cache stored
under past_key_values, so a Mamba model's cache_params is never reordered and the beams continue from another
beam's state. The report's script (state-spaces/mamba-130m-hf, CPU, greedy and 3 beams), with the same beam search
run without a cache as the reference (the fix PR's own check). Fixed in 5.13.0 (PR #46819), so this runs in the
transformers 5.12.1 venv.
Run in ~/venvs/tf5121: python testbed/m16/cases/tf46612.py <out.json>
"""
import json
import os
import sys

MODEL = "state-spaces/mamba-130m-hf"
N = 8


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    inputs = tok("Hello, my dog is cute", return_tensors="pt")
    with torch.no_grad():
        greedy = model.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=N)
        beam_cached = model.generate(**inputs, num_beams=3, num_return_sequences=3, do_sample=False, max_new_tokens=N)
        beam_nocache = model.generate(**inputs, num_beams=3, num_return_sequences=3, do_sample=False,
                                      max_new_tokens=N, use_cache=False)
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "greedy": tok.decode(greedy[0], skip_special_tokens=True),
           "beam_with_cache": tok.batch_decode(beam_cached, skip_special_tokens=True),
           "beam_without_cache": tok.batch_decode(beam_nocache, skip_special_tokens=True),
           "reproduced": beam_cached.tolist() != beam_nocache.tolist()}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
