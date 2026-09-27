# Draft (not posted): transformers 5's LlamaTokenizer rebuilds a legacy-export tokenizer.json as Metaspace, and the ids differ from the file in two ways

Status: draft, 2026-09-26, found by entail M18.1 (`testbed/results/m18/SUMMARY.md`); corrected after the M18.1
review (the first draft claimed "5.17 ignores legacy=true", which the measurement below refutes). Posting is the
researcher's decision ("상류보고는 일단 문서만"). No upstream issue found by `gh search issues` (three queries).

## What differs

`LlamaTokenizer` in transformers 5 (5.4.0 to 5.17.0) does not load a Llama-2-era tokenizer.json as it is: it
rebuilds the backend from the vocabulary and merges with `normalizer = None` and
`Metaspace(prepend_scheme=..., split=False)` (`models/llama/tokenization_llama.py`). Such a tokenizer.json was
exported in transformers 4's legacy mode (normalizer `Prepend "▁"` + `Replace " " -> "▁"`, no pre-tokenizer), and
transformers 4 used the file as it was whenever it was present. Two differences follow, measured on
TinyLlama/TinyLlama-1.1B-Chat-v1.0 (`legacy: false` declared) and hmellor/tiny-random-LlamaForCausalLM
(`legacy: true`), against tokenizer.json (tokenizers 0.23.2) and tokenizer.model (sentencepiece 0.2.2), which
agree with each other (`testbed/results/m18/llama_legacy/legacy_flag_check.json`):

1. **Flag-independent: text that starts with whitespace.** The file (and sentencepiece's `add_dummy_prefix`)
   prepend `▁` to the text, so `"   leading spaces and trailing   "` starts `▁▁▁ ▁leading` = `[1678, 8236, ...]`.
   Metaspace never doubles the `▁` there, under `legacy=True` ("always") and `legacy=False` ("first") alike:
   `▁▁ ▁leading` = `[259, 8236, ...]`. This is the difference behind every Llama-2-era folder in the static corpus
   of 300 popular folders (TinyLlama, CodeLlama-7b-hf, MiniCPM-SALA, EuroLLM-22B-Instruct, two tiny test folders):
   prompts that begin with indentation (code) get other ids than the file gives.
2. **Flag-dependent: text after a special token.** `legacy=True` prepends `▁` there ("always"), as the file does:
   `"</s>Hello there"` = `[2, 15043, 727]` on both. `legacy=False` does not ("first"): `[2, 10994, 727]`. TinyLlama
   declares `legacy: false`, so on 5.17.0 every chat prompt loses the `▁` (id 29871) after the user turn's `</s>`,
   which the file (and transformers 4's fast tokenizer, which ignored the flag without `from_slow`) kept. The flag
   is honoured; the change is that v5 applies it to a folder whose tokenizer.json was exported the other way.

## Effect of (2)

TinyLlama-1.1B-Chat-v1.0, bf16, greedy 48 tokens, 8 chat prompts
(`testbed/results/m18/llama_legacy/tinyllama_effect.json`): the ids differ for all 8 (one token, right after
`</s>`), and the greedy outputs differ for 7 of 8 - mostly wording; in the one substantive difference the file-side
ids gave the wrong description ("creates an infinite loop") and the engine-side ids the right one. Which tokenization
the model was fine-tuned with is not stated by the folder; transformers 4 users with the fast tokenizer got the
file's ids. The effect of (1) was not measured.

## Reproduce (transformers 5.17.0, tokenizers 0.23.2)

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer
f = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
t = AutoTokenizer.from_pretrained(f)                       # LlamaTokenizer, Metaspace(prepend_scheme="first")
r = Tokenizer.from_file(hf_hub_download(f, "tokenizer.json"))
for s in ("   leading spaces and trailing   ", "</s>Hello there"):
    print(t.encode(s, add_special_tokens=False), r.encode(s, add_special_tokens=False).ids)
# [259, 8236, 8162, 322, 25053, 1678]  [1678, 8236, 8162, 322, 25053, 1678]   <- (1), whatever legacy says
# [2, 10994, 727]                       [2, 15043, 727]                        <- (2), legacy=false
```

## Where entail says it

`ENTAIL=load` on any transformers load, or `entail check --model <folder> --engine transformers`: (1) is `broken` at
`load:transformers.tokenizer.ids` (rule `tokenizer_ids`: "1 of 10 probe texts encode differently; first: '   leading
spaces and trailing   ' ..."); (2) is noted on the same decision as a difference the folder's own `legacy=false`
explains ("1 further differ only after a declared token, as tokenizer_config.json's legacy=false would have it").
