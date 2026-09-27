# 상류 보고 초안 — nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16: generation_config.json이 `<|im_end|>`를 끝으로 선언하지 않음 (미게시; 연구자 허락 뒤)

- 어디에: Hugging Face 모델 저장소 `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`의 Discussions (모델 파일의 문제이지 transformers의 버그가 아니다: `generate()`가 `generation_config.eos_token_id`만 보는 것은 설계다).
- 근거 파일: `testbed/results/m15/stops_nemotron_off.json`, `stops_nemotron_on.json`, `stops_nemotron_on.record.jsonl`; 재현 스크립트 `testbed/m15/stops_replay.py` (`... <folder> out.json 160 asis`).
- 게시 전 확인할 것: 저장소의 현재 `generation_config.json`이 여전히 `eos_token_id: 2`뿐인지(우리 사본은 2026-09-24 내려받음, `transformers_version 4.57.1`, `_from_model_config: true`); 모델 카드가 `eos_token_id=[2, 11]`을 손으로 넘기라고 안내하는지; 같은 보고가 이미 있는지. 있으면 게시하지 않는다.
- entail 언급 없음(홍보 아님).

---

**Title:** `generation_config.json` lists only `</s>` (2) as `eos_token_id`; chat turns end with `<|im_end|>` (11), so `transformers` `generate()` runs past every answer

**Body (English):**

`config.json` and `generation_config.json` (the latter written automatically, `"_from_model_config": true`, `transformers_version 4.57.1`) declare `eos_token_id: 2` (`</s>`). The tokenizer's `eos_token` in `tokenizer_config.json` is `<|im_end|>` (id 11), and the chat template ends every assistant turn with `<|im_end|>`.

`transformers` builds a model's stop set from `generation_config.json` alone (`model.generation_config.eos_token_id`, read by `generate()`); the tokenizer's `eos_token` is not consulted. So with the files as shipped, plain `generate()` after `apply_chat_template` does not stop at the end of the answer:

```python
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
out = model.generate(ids, max_new_tokens=160, do_sample=False)
```

Observed (transformers 5.17.0, greedy, three prompts): every generation ran to `max_new_tokens` (160). The answer was complete at token 45, 62 and 55 respectively, where the model emitted `<|im_end|>`, and it then kept emitting `<|im_end|>` until the limit (decoded tail: `...The capital of France is Paris.<|im_end|> <|im_end|> <|im_end|>...`).

With `eos_token_id=[2, 11]` in `generation_config.json` (or passed to `generate`), the same three generations stop right after the answer at 46, 63 and 56 tokens, with identical text up to that point.

Suggested fix: `generation_config.json` → `"eos_token_id": [2, 11]` (and the same in `config.json` if it is meant to carry the stop ids), so that every loader that reads the generation config stops at the turn end. vLLM and SGLang already stop, because both also match the tokenizer's `eos_token` (vLLM takes it as the primary eos; SGLang's scheduler checks `tokenizer.eos_token_id` for every request); plain `transformers` is the loader affected. The sibling repositories already declare both ids: NVIDIA-Nemotron-3-Nano-30B-A3B (BF16, FP8, NVFP4), Nemotron-3-Super-120B-A12B (BF16, NVFP4) and Nemotron-3.5-Lightning-30B-A3B (BF16, NVFP4) all ship `"eos_token_id": [2, 11]`, and Nemotron-Nano-9B-v2 `[2, 11, 12]`; only this 4B repository has `2` alone, which looks like an omission when the file was written from `config.json`.

Environment: transformers 5.17.0, torch 2.14.0+cu130, one RTX 4070 Ti; model files as downloaded on 2026-09-24.
