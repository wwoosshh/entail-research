"""M18.6 replay 3 case, vllm-project/vllm#49250 (testbed/M16_PROTOCOL.md 8): with kv_load_failure_policy="recompute",
a request whose promised synchronous KV load is rejected is recomputed into garbage (sync scheduling) or an offset
continuation (async), where a clean run gives the right answer. The report's setup: facebook/opt-125m, enforce_eager,
no prefix caching, the report's rogue connector (rogue_connector.py), greedy; the baseline has no connector.
Reproduced when a reject-then-recompute output differs from the baseline.
Run in ~/venvs/vllm (0.30.0): python testbed/m18/replay3/cases/vl49250.py <out.json>
"""
import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"    # the connector module is importable in this process
MODEL = "facebook/opt-125m"
PROMPT = ("The capital of France is Paris. The capital of Germany is Berlin. The capital of Spain is Madrid. "
          "The capital of Italy is")


def run(connector, async_scheduling):
    import torch
    from vllm import LLM, SamplingParams

    kw = dict(model=MODEL, enforce_eager=True, enable_prefix_caching=False, gpu_memory_utilization=0.3,
              async_scheduling=async_scheduling)
    if connector:
        kw["kv_transfer_config"] = {"kv_connector": "RogueSyncRejectConnector",
                                    "kv_connector_module_path": "rogue_connector", "kv_role": "kv_both",
                                    "kv_load_failure_policy": "recompute"}
    llm = LLM(**kw)
    o = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=8))[0].outputs[0]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {"text": o.text, "tokens": list(o.token_ids)}


def main():
    import vllm

    base = run(False, False)
    sync = run(True, False)
    asyn = run(True, True)
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "baseline": base,
           "reject_sync": sync, "reject_async": asyn,
           "sync_differs": sync["tokens"] != base["tokens"], "async_differs": asyn["tokens"] != base["tokens"]}
    row["reproduced"] = row["sync_differs"] or row["async_differs"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("RESULT", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
