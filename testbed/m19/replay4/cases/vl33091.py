"""M19 L4 replay 4 case, vllm-project/vllm#33091 (testbed/M16_PROTOCOL.md 9): whisper-large-v3-turbo transcribes wrongly
with FlashAttention 2 and CUDA graphs (fine with FA3, or with graphs or compilation off; fixed by #33360: the
cross-attention builder overrode max_seq_len with the encoder lengths, so the graph was captured with max_seq_len 0).
The report's script: VLLM_FLASH_ATTN_VERSION=2 (this GPU's FlashAttention), enforce_eager=False, the mary_had_lamb
asset. Reproduced when "Mary had a little lamb" is not in the transcription (the second argument 'eager' runs the
control with enforce_eager=True).
Run: python testbed/m19/replay4/cases/vl33091.py <out.json> [eager]  (~/venvs/vllm0140: vLLM 0.14.0)
"""
import json
import os
import sys

os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")
MODEL = os.path.expanduser("~/models/replay4/openai__whisper-large-v3-turbo")


def main():
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset

    eager = len(sys.argv) > 2 and sys.argv[2] == "eager"
    llm = LLM(model=MODEL, tensor_parallel_size=1, enforce_eager=eager, gpu_memory_utilization=0.6)
    out = llm.generate([{"prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                         "multi_modal_data": {"audio": AudioAsset("mary_had_lamb").audio_and_sample_rate}}],
                       sampling_params=SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)
    text = out[0].outputs[0].text
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "enforce_eager": eager,
           "flash_attn_version": os.environ.get("VLLM_FLASH_ATTN_VERSION"), "text": text}
    row["reproduced"] = (not eager) and "Mary had a little lamb" not in text
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
