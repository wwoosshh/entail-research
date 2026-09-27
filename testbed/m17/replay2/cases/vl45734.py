"""M17.6 case, vllm-project/vllm#45734 (testbed/M16_PROTOCOL.md 7): ExampleHiddenStatesConnector returns NaN hidden
states on a hybrid-attention model (Qwen3.5: linear and full attention layers) - the hidden-states KV cache group's
block table is never filled, so the states are written to slot 0 and read from unwritten blocks. The reporter's
attached script (repro_nan_hidden_states.py), adapted to one 12 GB card: the transformers reference is computed
in a child process on the CPU (this process imports no torch before the server has stopped, so the whole card is
the server's), then vLLM is served with the extract_hidden_states method and the example connector, and the saved
hidden states are compared with the reference. Reproduced when vLLM's saved hidden states hold NaN while the
reference is finite.
Run in ~/venvs/vllm0230 (the reported version, pre-#45849): python testbed/m17/replay2/cases/vl45734.py <out.json>
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = "Qwen/Qwen3.5-4B"
PORT = 8321


def reference(path):
    """Child process: the transformers hidden states of the last layer on the CPU, saved to `path`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = "Sample 0: " + "Explain neural networks. " * 50
    tokens = tokenizer.encode(text)[:500]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(torch.tensor([tokens]), output_hidden_states=True)
    ref = out.hidden_states[-1][0].float()
    torch.save({"tokens": tokens, "hidden": ref, "layers": len(out.hidden_states) - 1}, path)
    print("REFERENCE", json.dumps({"shape": list(ref.shape), "nan": bool(ref.isnan().any()),
                                   "layers": len(out.hidden_states) - 1}))


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "reference":
        reference(sys.argv[1])
        return
    out_path = sys.argv[1]
    row = {"entail": os.environ.get("ENTAIL", "off"), "model": MODEL}
    ref_path = out_path + ".reference.pt"
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    env.pop("ENTAIL", None)                       # the reference is transformers on the CPU, not under test
    r = subprocess.run([sys.executable, __file__, ref_path, "reference"], capture_output=True, text=True, env=env)
    ref_line = next((ln for ln in r.stdout.splitlines() if ln.startswith("REFERENCE")), None)
    if not ref_line:
        row["error"] = "reference failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300]
        row["reproduced"] = None
        json.dump(row, open(out_path, "w", encoding="utf-8"), indent=1)
        print("RESULT", json.dumps(row))
        return
    row["reference"] = json.loads(ref_line[len("REFERENCE "):])
    n_layers = row["reference"]["layers"]
    hs_dir = tempfile.mkdtemp()
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + ":" + env.get("PATH", "")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL,
           "--speculative_config", json.dumps({"method": "extract_hidden_states", "num_speculative_tokens": 1,
                                                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [n_layers]}}}),
           "--kv_transfer_config", json.dumps({"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer",
                                               "kv_connector_extra_config": {"shared_storage_path": hs_dir}}),
           # this card shows 10.78 of 11.99 GiB free to any process at startup, so 0.89 is the most vLLM accepts;
           # the prompt is 205 tokens, so 256 for the model length and the batch keeps the profiling peak small
           "--port", str(PORT), "--max-model-len", "256", "--max-num-batched-tokens", "256",
           "--gpu-memory-utilization", "0.89", "--max-num-seqs", "1", "--enforce-eager",
           "--no-enable-chunked-prefill", "--disable-uvicorn-access-log"]
    log = open(out_path + ".server.log", "w")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    resp, err = None, None
    try:
        up = False
        for _ in range(300):
            if server.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5)
                up = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        if not up:
            err = "server did not come up (see .server.log)"
        else:
            import torch  # after the server is up: the GPU is already allocated

            tokens = torch.load(ref_path)["tokens"]
            body = {"model": MODEL, "prompt": tokens, "max_tokens": 1, "return_token_ids": True}
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as rr:
                resp = json.load(rr)
            time.sleep(3)
            # the connector's file goes with the server: copy it while the server is alive
            import shutil

            src = (resp.get("kv_transfer_params") or {}).get("hidden_states_path")
            if src and os.path.isfile(src):
                shutil.copy(src, out_path + ".hs.safetensors")
                resp["kv_transfer_params"]["hidden_states_path"] = out_path + ".hs.safetensors"
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    if err:
        row["error"] = err
        row["reproduced"] = None
    else:
        import torch
        from safetensors.torch import load_file

        ref = torch.load(ref_path)["hidden"]
        params = resp.get("kv_transfer_params") or {}
        path = params.get("hidden_states_path")
        row["hidden_states_path"] = path
        if path and os.path.isfile(path):
            hs = load_file(path)
            # the file holds the token ids beside the hidden states (tokens x 1 x hidden): take the widest tensor
            key = "hidden_states" if "hidden_states" in hs else max(hs, key=lambda k: hs[k].dim())
            t = hs[key].float().reshape(hs[key].shape[0], -1)
            row["vllm"] = {"keys": {k: list(v.shape) for k, v in hs.items()}, "key": key, "shape": list(t.shape),
                           "nan": bool(t.isnan().any()), "nan_fraction": float(t.isnan().float().mean())}
            if not row["vllm"]["nan"] and t.shape[-1] == ref.shape[-1]:
                n = min(t.shape[0], ref.shape[0])
                row["vllm"]["max_abs_diff_vs_reference"] = float((t[:n] - ref[:n]).abs().max())
            row["reproduced"] = bool(row["vllm"]["nan"] and not row["reference"]["nan"])
        else:
            row["error"] = f"no hidden states file: {params}"
            row["reproduced"] = None
    json.dump(row, open(out_path, "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
