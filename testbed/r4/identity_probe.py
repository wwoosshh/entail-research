"""R4 (identity·epoch): the stale block-hash mechanism of vllm-project/vllm#49377 / #49449 on the installed vLLM,
and the cost of the invariant check that would catch it. Pure Python, no GPU (the E3 end-to-end run is
testbed/m10_e3/vl49449.py).

What is declared: Request.block_hashes[i] fingerprints the tokens of hash block i (chained over the prefix).
Who consumes it: KVCacheManager.get_computed_blocks (prefix-cache lookup by hash), KV events, offload.
The invariant: block_hashes[i] == hash of the tokens the request holds NOW for block i, for every i.
How it breaks: Scheduler._update_request_as_session truncates _all_token_ids past num_computed_tokens and calls
update_block_hashes(), which only appends; the hasher resumes from len(block_hashes) * B, so a hash chained over a
discarded sampled token stays, and every later hash chains from it.

Run: ~/venvs/vllm/bin/python testbed/r4/identity_probe.py testbed/results/r4/identity_probe.json
"""
import inspect
import json
import os
import sys
import time


def make_request(Request, SamplingParams, hasher, prompt):
    params = inspect.signature(Request.__init__).parameters
    kw = {"request_id": "session", "prompt_token_ids": list(prompt), "sampling_params": SamplingParams(),
          "block_hasher": hasher}
    for name, default in (("pooling_params", None), ("resumable", True), ("eos_token_id", None),
                          ("mm_features", None), ("arrival_time", None), ("lora_request", None),
                          ("structured_output_request", None), ("cache_salt", None), ("priority", 0),
                          ("trace_headers", None), ("client_index", 0)):
        if name in params:
            kw[name] = default if default is not None or params[name].default is inspect.Parameter.empty else \
                params[name].default
    return Request(**kw)


def session_update(session, new_tokens):
    """The statements Scheduler._update_request_as_session runs (vLLM 0.30: scheduler.py:1570-1611), on the request."""
    num_computed_tokens = session.num_computed_tokens
    kept = session._all_token_ids[session.num_prompt_tokens:num_computed_tokens]
    del session._all_token_ids[num_computed_tokens:]
    session._output_token_ids.clear()
    session.prompt_token_ids.extend(kept)
    session._all_token_ids.extend(new_tokens)
    session.prompt_token_ids.extend(new_tokens)
    session.update_block_hashes()
    session.num_prompt_tokens = len(session.prompt_token_ids)


def fresh_chain(kcu, hash_fn, request, block, n=None):
    """The hashes the request's CURRENT tokens give (what the stored hashes must equal)."""
    toks = list(request.all_token_ids)
    out, prev, mm = [], None, 0
    full = len(toks) // block if n is None else n
    for i in range(full):
        s, e = i * block, (i + 1) * block
        extra, mm = kcu.generate_block_hash_extra_keys(request, s, e, mm)
        h = kcu.hash_block_tokens(hash_fn, prev, toks[s:e], extra)
        out.append(h)
        prev = h
    return out


def identity_check(kcu, hash_fn, request, block):
    """The proposed rule: (stored hashes, first index whose stored hash is not the hash of the tokens held now)."""
    stored = list(request.block_hashes)
    fresh = fresh_chain(kcu, hash_fn, request, block, n=len(stored))
    for i, (a, b) in enumerate(zip(stored, fresh)):
        if a != b:
            return stored, i
    if len(stored) > len(request.all_token_ids) // block:
        return stored, len(request.all_token_ids) // block
    return stored, None


def repair(request, first_stale):
    """The proposed resolution: forget the hashes from the first stale one on; the hasher re-appends from there."""
    del request.block_hashes[first_stale:]
    request.update_block_hashes()


def main(out_path):
    import vllm
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core import kv_cache_utils as kcu
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request
    try:
        from vllm.utils.hashing import sha256
    except ImportError:  # older layout
        from vllm.utils import sha256

    init_none_hash(sha256)
    B = 4
    hasher = get_request_block_hasher(B, sha256)
    res = {"vllm": vllm.__version__, "hash_block_size": B}

    # 1. the mechanism (the issue's script, on this vLLM)
    req = make_request(Request, SamplingParams, hasher, [1, 2, 3])
    req.append_output_token_ids(99)                       # completes hash block [1, 2, 3, 99]
    stale_hash = req.block_hashes[0]
    req.num_computed_tokens = 3                           # 99 was sampled, never computed
    session_update(req, [4])                              # streaming chunk [4]
    res["tokens_after_update"] = list(req.all_token_ids)
    res["stored_hash_is_old_tokens"] = req.block_hashes[0] == kcu.hash_block_tokens(sha256, None, [1, 2, 3, 99])
    res["stored_hash_is_new_tokens"] = req.block_hashes[0] == kcu.hash_block_tokens(sha256, None, [1, 2, 3, 4])
    res["mechanism_reproduced"] = bool(req.block_hashes[0] == stale_hash and res["stored_hash_is_old_tokens"]
                                       and not res["stored_hash_is_new_tokens"])

    # 2. the consequence at the key level: another request whose prompt is the OLD tokens gets the same key
    other = make_request(Request, SamplingParams, hasher, [1, 2, 3, 99, 5])
    other.update_block_hashes()
    res["old_prompt_shares_the_key"] = other.block_hashes[0] == req.block_hashes[0]

    # 3. the invariant check catches it, and the repair restores the invariant
    stored, first = identity_check(kcu, sha256, req, B)
    res["check_first_stale_index"] = first
    repair(req, first)
    stored2, first2 = identity_check(kcu, sha256, req, B)
    res["after_repair_first_stale_index"] = first2
    res["after_repair_key_is_new_tokens"] = req.block_hashes[0] == kcu.hash_block_tokens(sha256, None, [1, 2, 3, 4])
    res["after_repair_old_prompt_shares_the_key"] = other.block_hashes[0] == req.block_hashes[0]

    # 4. a healthy update (no truncation past a hash boundary) is quiet
    ok = make_request(Request, SamplingParams, hasher, [1, 2, 3])
    ok.append_output_token_ids(7)                         # block [1,2,3,7] hashed
    ok.num_computed_tokens = 4                            # this time the sampled token WAS computed... (decode step)
    session_update(ok, [8, 9, 10, 11])                    # chunk completes block [8,9,10,11]
    _, first_ok = identity_check(kcu, sha256, ok, B)
    res["healthy_update_first_stale_index"] = first_ok

    # 5. cost of the check: a long session (1024 tokens, 256 hashed blocks), checking only the LAST stored block
    #    (the one a truncation can invalidate) versus checking the whole chain
    long_req = make_request(Request, SamplingParams, hasher, list(range(1000, 2024)))
    long_req.update_block_hashes()
    n_blocks = len(long_req.block_hashes)
    reps = 2000
    t0 = time.perf_counter()
    for _ in range(reps):
        i = n_blocks - 1
        s, e = i * B, (i + 1) * B
        prev = long_req.block_hashes[i - 1] if i else None
        extra, _mm = kcu.generate_block_hash_extra_keys(long_req, s, e, -1)
        _ = kcu.hash_block_tokens(sha256, prev, list(long_req.all_token_ids)[s:e], extra) == long_req.block_hashes[i]
    t_last = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(20):
        identity_check(kcu, sha256, long_req, B)
    t_all = (time.perf_counter() - t0) / 20
    res["cost_us"] = {"blocks": n_blocks, "check_last_block": round(t_last * 1e6, 1),
                      "check_whole_chain": round(t_all * 1e6, 1)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "testbed/results/r4/identity_probe.json")
