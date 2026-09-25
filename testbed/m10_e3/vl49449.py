"""M10 E3, vllm-project/vllm#49449 and #49377 (testbed/M10_PROTOCOL.md 3.3): a streaming-session update that replaces
the last generated token leaves the request's prefix-cache block hash of the old tokens; a later ordinary request
whose prompt matches the old tokens gets that cache entry and a different output than the same prompt recomputed.
The report's procedure (gist 514adfcf, read, not run), re-written for vLLM 0.30: SmolLM2-135M-Instruct, prefix
caching on, block size 16, P = [10]*15, A: P -> first token X -> streaming input [Y], then B = P + [X, 13] with the
default cache namespace (candidate) and with another cache_salt (reference, recomputed).
Reproduced: the candidate and the reference differ while the candidate hit 16 cached tokens.
Run in ~/venvs/vllm: python testbed/m10_e3/vl49449.py <out.json> <model dir>
"""
import asyncio
import inspect
import json
import os
import sys

P = [10] * 15
Z = 13


async def run(model):
    from vllm import TokensPrompt
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    def sp():
        return SamplingParams(max_tokens=1, temperature=0.0, output_kind=RequestOutputKind.DELTA)

    def tp(ids, salt=None):
        return TokensPrompt(prompt_token_ids=ids) if salt is None else TokensPrompt(prompt_token_ids=ids,
                                                                                    cache_salt=salt)

    def first(o):
        for c in getattr(o, "outputs", []) or []:
            if list(getattr(c, "token_ids", []) or []):
                return int(c.token_ids[0]), str(getattr(c, "text", ""))
        return None

    async def nxt(q):
        return q.get_nowait() or await q.get()

    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=model, tokenizer=model, enforce_eager=True, gpu_memory_utilization=0.5, max_model_len=128,
        max_num_seqs=2, max_num_batched_tokens=64, enable_prefix_caching=True, disable_log_stats=True, seed=0))
    out = {}
    try:
        seen, x, y = asyncio.Event(), None, None

        async def stream():
            nonlocal y
            yield StreamingInput(prompt=tp(list(P)), sampling_params=sp())
            await asyncio.wait_for(seen.wait(), timeout=60)
            y = 11 if x != 11 else 12
            yield StreamingInput(prompt=tp([y]), sampling_params=sp())

        q = await llm.add_request("a", stream(), sp())
        while True:
            o = await nxt(q)
            it = first(o)
            if it is not None and x is None:
                x = it[0]
                seen.set()
            if getattr(o, "finished", False):
                break
        q.close()

        async def b(rid, salt):
            q = await llm.add_request(rid, tp(P + [x, Z], salt), sp())
            res = None
            while True:
                o = await nxt(q)
                it = first(o)
                if it is not None and res is None:
                    res = {"token": it[0], "text": it[1], "cached_tokens": getattr(o, "num_cached_tokens", None)}
                if getattr(o, "finished", False):
                    break
            q.close()
            return res

        out = {"x": x, "y": y, "candidate": await b("b-candidate", None),
               "reference": await b("b-reference", "isolated-reference")}
    finally:
        r = llm.shutdown()
        if inspect.isawaitable(r):
            await r
    return out


def main():
    import vllm

    res = asyncio.run(run(sys.argv[2]))
    c, r = res.get("candidate") or {}, res.get("reference") or {}
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, **res,
           "reproduced": bool(c and r and c["token"] != r["token"] and c.get("cached_tokens") == 16)}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
