"""Reproducer: extract_hidden_states returns NaN on Qwen3.5 (hybrid attention).

Usage:
    CUDA_VISIBLE_DEVICES=5 python repro_nan_hidden_states.py
"""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import openai
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
PORT = 8321


def main():
    hs_dir = tempfile.mkdtemp()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    text = "Sample 0: " + "Explain neural networks. " * 50
    tokens = tokenizer.encode(text)[:500]

    # --- Reference: get real hidden states from transformers ---
    print("Computing reference hidden states with transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
    ).cuda()
    with torch.no_grad():
        out = model(
            torch.tensor([tokens], device="cuda"),
            output_hidden_states=True,
        )
    # Last hidden layer (layer 32)
    ref_hs = out.hidden_states[-1][0].cpu()  # [seq_len, hidden_size]
    print(f"Reference: shape={ref_hs.shape} "
          f"nan={ref_hs.isnan().any().item()} "
          f"min={ref_hs.min().item():.4f} max={ref_hs.max().item():.4f}")
    del model
    torch.cuda.empty_cache()

    # --- vLLM: get hidden states via extract_hidden_states ---
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
        "--speculative_config", json.dumps({
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": [32]}
            },
        }),
        "--kv_transfer_config", json.dumps({
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": hs_dir},
        }),
        "--port", str(PORT),
        "--max-model-len", "513",
        "--gpu-memory-utilization", "0.3",
        "--no-enable-chunked-prefill",
        "--disable-uvicorn-access-log",
    ]
    server = subprocess.Popen(cmd, stderr=subprocess.STDOUT, env=env)

    try:
        for _ in range(150):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5)
                break
            except Exception:
                time.sleep(2)

        client = openai.Client(base_url=f"http://localhost:{PORT}/v1", api_key="x")
        resp = client.completions.create(
            model=MODEL, prompt=tokens, max_tokens=1,
            extra_body={"return_token_ids": True}, timeout=120,
        )
        path = getattr(resp, "kv_transfer_params", {}).get("hidden_states_path")
        with open(path + ".lock") as lf:
            fcntl.flock(lf, fcntl.LOCK_SH)

        vllm_hs = load_file(path)["hidden_states"][:, -1]  # [seq_len, hidden_size]
        n = vllm_hs.shape[0]
        nan_tokens = sum(1 for j in range(n) if vllm_hs[j].isnan().any())

        print(f"\nvLLM:      shape={vllm_hs.shape} "
              f"nan_tokens={nan_tokens}/{n} ({nan_tokens/n*100:.0f}%)")

        print(f"\nExpected (from transformers): no NaN, valid float values")
        print(f"Got (from vLLM):              {nan_tokens}/{n} tokens are NaN")

        assert nan_tokens == 0, f"{nan_tokens}/{n} tokens contain NaN"
        print("PASS")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
