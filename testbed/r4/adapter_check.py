"""R4/M14: the vLLM identity adapter on a real Request (read_choice, _hasher_env, _hex, handles) driving the core
rule, on the installed vLLM. No full engine, no GPU: a Request with a real block hasher, the session-update
statements, then the adapter's _decide. Confirms the stale key is caught and the repair restores the invariant.
Run: ENTAIL=load ~/venvs/vllm/bin/python testbed/r4/adapter_check.py testbed/results/r4/adapter_check.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/ai_compiler/entail"))


def main(out_path):
    os.environ.setdefault("ENTAIL", "load")
    os.environ["ENTAIL_LOG_DIR"] = "off"
    from entail import core, identity_contract, load
    from entail.adapters import vllm_identity
    core.set_mode("load")

    import vllm  # noqa: F401
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core import kv_cache_utils as kcu
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request
    try:
        from vllm.utils.hashing import sha256
    except ImportError:
        from vllm.utils import sha256
    import inspect

    init_none_hash(sha256)
    B = 4
    hasher = get_request_block_hasher(B, sha256)

    def make(prompt):
        params = inspect.signature(Request.__init__).parameters
        kw = {"request_id": "s", "prompt_token_ids": list(prompt), "sampling_params": SamplingParams(),
              "block_hasher": hasher}
        for name in ("pooling_params", "eos_token_id", "mm_features", "arrival_time", "lora_request",
                     "structured_output_request", "cache_salt", "priority", "trace_headers", "client_index"):
            if name in params:
                kw[name] = 0 if name in ("priority", "client_index") else None
        if "resumable" in params:
            kw["resumable"] = True
        return Request(**kw)

    # the bug: [1,2,3]+sampled 99 completes block0, then a session update truncates to [1,2,3] and appends 4
    req = make([1, 2, 3])
    req.append_output_token_ids(99)
    at = 3                                  # num_computed_tokens: token 99 was never computed
    req.num_computed_tokens = at
    del req._all_token_ids[at:]
    req._output_token_ids.clear()
    req.prompt_token_ids.extend([])
    req._all_token_ids.extend([4])
    req.prompt_token_ids.extend([4])
    req.update_block_hashes()               # what the scheduler runs; leaves block0 = hash of [1,2,3,99]

    stored_before = vllm_identity._hex(req.block_hashes[0])
    fresh_expected = vllm_identity._hex(kcu.hash_block_tokens(sha256, None, [1, 2, 3, 4]))
    old_expected = vllm_identity._hex(kcu.hash_block_tokens(sha256, None, [1, 2, 3, 99]))

    # what the adapter reads at the boundary, then decides
    read = vllm_identity.read_choice(req, at)
    n = len(load.LEDGER.decisions)
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        vllm_identity._decide(req, at)
    decisions = load.LEDGER.decisions[n:]

    res = {
        "vllm": vllm.__version__,
        "adapter_read": {"stored": read[0], "fresh": read[1], "block": read[2], "start": read[3]} if read else None,
        "stored_before_is_old_tokens": stored_before == old_expected,
        "decided": [(d.verdict.value, d.rule, d.handle, d.target) for d in decisions],
        "resolved": any(d.verdict.value == "resolved" for d in decisions),
        "key_after_repair": vllm_identity._hex(req.block_hashes[0]),
        "key_after_repair_is_new_tokens": vllm_identity._hex(req.block_hashes[0]) == fresh_expected,
    }
    # a healthy update makes no decision
    ok = make([1, 2, 3])
    ok.append_output_token_ids(7)
    ok.num_computed_tokens = 4
    del ok._all_token_ids[4:]
    ok._output_token_ids.clear()
    ok.prompt_token_ids.extend([7])
    ok._all_token_ids.extend([8, 9, 10, 11])
    ok.prompt_token_ids.extend([8, 9, 10, 11])
    ok.update_block_hashes()
    m = len(load.LEDGER.decisions)
    with redirect_stdout(io.StringIO()):
        vllm_identity._decide(ok, 4)
    res["healthy_update_decisions"] = len(load.LEDGER.decisions) - m

    res["ok"] = bool(res["stored_before_is_old_tokens"] and res["resolved"]
                     and res["key_after_repair_is_new_tokens"] and res["healthy_update_decisions"] == 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "testbed/results/r4/adapter_check.json")
