"""Print compact tables from week-4 result files (E2 approaches, E4 ablation, E3 extra, E3b v2, E6, diagnosis)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def e2():
    print("== E2: batch sizes 1, 4, 8, 2 in order (first step includes any compile)")
    for a in ["auto", "dynamic_flag", "mark_dynamic", "unbacked", "unbacked_shared", "static"]:
        r = load(f"e2_{a}.json")
        if r is None:
            continue
        cells, stall = [], 0.0
        for b in r["per_batch"]:
            if "error" in b:
                cells.append(f"B{b['batch']}: ERROR {b['error'][:90]}")
                continue
            s = b["first_step_ms"] - b["steady_ms"]
            stall += s
            cells.append(f"B{b['batch']}: first {b['first_step_ms'] / 1e3:.1f}s steady {b['steady_ms']:.2f}ms "
                         f"compiles {b['compiles_during_batch']}")
        cm = r.get("compile_metrics_total", {})
        ci = next((v for k, v in cm.items() if k.endswith("compile_inner")), {})
        print(f"  {a:16s} precompile {r.get('precompile_s') or 0:.1f}s | serve-time stall {stall / 1e3:.1f}s | "
              f"graphs {r.get('unique_graphs')} | compile_inner n={ci.get('n')} {ci.get('s', 0):.1f}s")
        for c in cells:
            print("      ", c)


def e4():
    r = load("e4_fact_ablation.json")
    if r is None:
        return
    print("== E4: step ms (ratio vs sdpa) | GPU ms attention / copy+mask / gemm / total")
    for run in r["runs"]:
        t = run.get("timing", {})
        kb = t.get("kernel_breakdown_ms", {})
        print(f"  {run['dtype']} B={run['batch']}")
        for n, v in t.get("variants", {}).items():
            k = kb.get(n, {})
            print(f"    {n:18s} {v['median_ms']:7.2f} (x{v.get('ratio_vs_base_median', 1):.3f}) | "
                  f"{k.get('attention', 0):.2f} / {k.get('kv_copy_or_mask', 0):.2f} / {k.get('gemm', 0):.2f} / "
                  f"{k.get('total', 0):.2f}")
        bad = {n: v.get("excluded_because") or v.get("error") for n, v in run["variants"].items() if not v.get("timed")}
        if bad:
            print("    not timed:", bad)


def e3x():
    r = load("e3_rediscovery_extra.json")
    if r is None:
        return
    print("== E3 extra")
    for run in r["runs"]:
        print(f"  B={run['batch']} pairwise bitwise {run.get('flex_pairwise_bitwise_equal')}")
        print("    metadata same:", all(m["num_blocks_equal"] and m["used_indices_equal"]
                                        for m in run.get("declared_vs_rediscovered_metadata", [])))
        for n, v in run["variants"].items():
            print(f"    {n:26s}", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()
                                     if k in ("error", "max_abs_logit_diff_vs_sdpa_6steps", "top1_agree_vs_sdpa_6steps")})
        for n, v in run.get("timing", {}).get("variants", {}).items():
            print(f"    T {n:26s} {v['median_ms']:.2f} ms (x{v.get('ratio_vs_base_median', 1):.3f})")
        print("    meta", json.dumps(run.get("metadata_cost_alone")))


def e3b2():
    r = load("e3b_mask_scaling_v2.json")
    if r is None:
        return
    print("== E3b v2")
    for row in r["rows"]:
        print("  ", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()})


def other():
    for name in ("e6_one_spec.json", "diag_hf_flex_offset.json"):
        r = load(name)
        if r is not None:
            r = {k: v for k, v in r.items() if k != "note"}
            print(f"== {name}\n", json.dumps(r, ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    e2()
    e4()
    e3x()
    e3b2()
    other()
