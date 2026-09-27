# Draft (not posted): transformers 5.17.0 builds a Llama (Metaspace) pipeline for folders that declare `LlamaTokenizerFast` over a byte-level BPE tokenizer.json, and the spaces vanish

Status: draft, 2026-09-26, found by entail's M18.1 tokenizer check on the M10 static corpus (300 popular folders;
`testbed/results/m18/static/tokenizers.json`, verified in `testbed/results/m18/static/llamafast_check.json`).
Posting is the researcher's decision ("상류보고는 일단 문서만"). No upstream issue found by `gh search issues`
(four queries). The same class as transformers#46710 (DeepSeek-R1-Distill-Llama-8B, fixed in 5.13.0): the fix did
not reach these folders.

## What happens

The folders declare `"tokenizer_class": "LlamaTokenizerFast"` in tokenizer_config.json and ship a byte-level BPE
tokenizer.json (a `Split` regex pre-tokenizer, `Ġ`-style pieces). transformers 5.17.0's AutoTokenizer builds
`LlamaTokenizer`, which rebuilds the pipeline from the vocabulary and merges with
`Metaspace(prepend_scheme="always", split=False)` and no byte-level pre-tokenizer. The ids differ from
tokenizer.json on 9 of 10 probe texts, and the spaces are gone after decoding:

| folder | text | tokenizer.json (tokenizers 0.23.2) | transformers 5.17.0 | decode of the engine's ids |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | `How are you doing?` | `How Ġare Ġyou Ġdoing ?` | `How are y oud o ing ?` | `Howareyoudoing?` |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | `def fib(n):\n    return n` | `def Ġfib (n ):Ċ ĠĠĠ Ġreturn Ġn` | `de ff ib (n ): return n` | `deffib(n):returnn` |
| deepseek-ai/deepseek-coder-7b-instruct-v1.5 | `How are you doing?` | `How Ġare Ġyou Ġdoing ?` | `How are you doing ?` (no space pieces) | `Howareyoudoing?` |
| lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-MLX-4bit | as the first | as the first | as the first | `Howareyoudoing?` |

The tokens the engine produces are the byte-level vocabulary's entries without the `Ġ` space marker, so every
word boundary is lost. A model served through transformers' AutoTokenizer from these folders (vLLM and SGLang
build their tokenizers through it) reads a prompt with no spaces.

## Reproduce (transformers 5.17.0, tokenizers 0.23.2)

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

f = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
t = AutoTokenizer.from_pretrained(f)                       # LlamaTokenizer
r = Tokenizer.from_file(hf_hub_download(f, "tokenizer.json"))
s = "How are you doing?"
print(t.convert_ids_to_tokens(t.encode(s, add_special_tokens=False)))   # ['How', 'are', 'y', 'oud', 'o', 'ing', '?']
print([r.id_to_token(i) for i in r.encode(s, add_special_tokens=False).ids])   # ['How', 'Ġare', 'Ġyou', 'Ġdoing', '?']
print(t.decode(t.encode(s, add_special_tokens=False)))     # Howareyoudoing?
```

## Where entail says it

`entail check --model <folder> --engine transformers`, or any load with `ENTAIL=load`: `broken` at
`load:transformers.tokenizer.ids`, rule `tokenizer_ids`, "9 of 10 probe texts encode differently; first: 'How are
you doing?': LlamaTokenizer built ... gives [4340, 546, 88, 2950, 78, 287, 30], tokenizer.json gives
[4340, 525, 498, 3730, 30]".
