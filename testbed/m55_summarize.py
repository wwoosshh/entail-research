"""Summarise the M5.5 measurements from their result files only (testbed/results/m55/).

  graph.json, graph_sync.json   testbed/m55_graph.py: S4 on vLLM's CUDA graph path (default settings; synchronous)
  rolebench.json                testbed/m55_rolebench.py: rolebench 09, 11, 12, 14 under each policy
  l05.json                      testbed/m55_l05.py: market case L05 simulated
  market_<tag>.json             testbed/m55_market_serve.py: market cases L07 and L13 simulated on a vLLM server
Writes SUMMARY.md next to them. Usage: python testbed/m55_summarize.py
"""
import json
import os
import statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "results", "m55")
MARKET = ("clean_off", "clean_on", "bug_off", "bug_on", "bug_strict")


def read(name):
    path = os.path.join(DIR, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def verdicts(decisions):
    c = Counter(d["verdict"] + (f" ({d['resolution'].split(' (')[0]})" if d.get("resolution") else "")
                for d in decisions)
    return ", ".join(f"{k}" + (f" x{n}" if n > 1 else "") for k, n in c.items()) or "none"


def graph(lines):
    runs = [(n, read(f)) for n, f in (("default (asynchronous scheduling)", "graph.json"),
                                       ("asynchronous scheduling off", "graph_sync.json"))]
    runs = [(n, r) for n, r in runs if r]
    if not runs:
        return
    r0 = runs[0][1]
    lines += ["## S4: the always-on mode on the CUDA graph path", "",
              f"vLLM {r0['vllm']}, torch {r0['torch']}, Qwen3-4B bf16, offline `LLM.generate`, engine core in the "
              f"process (VLLM_ENABLE_V1_MULTIPROCESSING={r0['multiprocessing']}). enforce_eager {r0['enforce_eager']}, "
              f"cudagraph_mode {r0['cudagraph_mode']}, compilation mode {r0['compilation_mode']}. "
              f"{r0['prompt_tokens']}-token prompts, {r0['new_tokens']} new tokens, greedy, ignore_eos; "
              f"{r0['rounds']} rounds per batch size, each timing on, off and control (= off) in a rotated order. "
              "Ratios are medians over rounds. Criterion: 1.02 or less.", ""]
    for name, r in runs:
        lines += [f"### {name} (async_scheduling {r['async_scheduling']})", "",
                  "| batch | on / off | control / off | off, median s | checks while on | broken | same tokens |",
                  "|---|---|---|---|---|---|---|"]
        for b, e in r["batches"].items():
            off = statistics.median(e["seconds"]["off"])
            lines.append(f"| {b} | {e['median_on_over_off']:.4f} | {e['median_control_over_off']:.4f} | {off:.3f} | "
                         f"{e['checks_while_on']} | {e['broken_while_on']} | {e['same_tokens_on_off']} |")
        lines.append("")


def rolebench(lines):
    r = read("rolebench.json")
    if not r:
        return
    lines += ["## S2: rolebench 09, 11, 12, 14", "",
              f"torch {r['torch']}, CUDA {r['cuda']}. The cases unchanged; the harness hooks where an adapter would "
              "(testbed/m55_rolebench.py). Policies: `off`; `on` the default; `strict` ENTAIL_ON_BROKEN=stop; "
              "`observe` ENTAIL_POLICY=refuse (nothing repaired). Each cell: the decisions recorded, then the "
              "output against the case's reference (within its tolerance), or `stopped`.", "",
              "| case | boundary | run | off | on | strict | observe |", "|---|---|---|---|---|---|---|"]
    for folder, e in r["cases"].items():
        if "skipped" in e:
            lines.append(f"| {folder} | | | skipped: {e['skipped']} | | | |")
            continue
        for p in ("defect", "fixed"):
            cells = []
            for pol in ("off", "on", "strict", "observe"):
                x = e[pol][p]
                out = "stopped" if x["stopped"] else ("right" if x["vs_reference"]["within"] else
                                                      f"wrong (max {x['vs_reference']['max_abs']:.3g})")
                cells.append(f"{verdicts(x['decisions'])}; {out}" if x["decisions"] else out)
            lines.append(f"| {folder} | `{e['boundary']}` | {p} | " + " | ".join(cells) + " |")
    lines.append("")
    for folder, e in r["cases"].items():
        if "skipped" in e:
            continue
        c = e["on"]["counts"]
        extra = f"; the harness loop gives the case's own outputs: {e['same_as_case']}" if "same_as_case" in e else ""
        lines.append(f"- {folder}, counted while `on` (defect and fixed runs): checks {c['checks']}, broken "
                     f"{c['broken']}, resolved {c['resolved']}{extra}")
    lines.append("")


def l05(lines):
    r = read("l05.json")
    if not r:
        return
    lines += ["## S2: market case L05 (simulated)", "",
              f"An Ollama-like server with a default context of {r['default_num_ctx']} tokens that keeps the first "
              f"{r['keep']} and the last of a longer prompt; the answer is right when the fact placed after the kept "
              "head reaches the model (testbed/m55_l05.py). Each cell: the decision, whether the answer is right, "
              "the tokens the model saw.", "",
              "| scenario | prompt | model context | off | on | strict |", "|---|---|---|---|---|---|"]
    for name, e in r["scenarios"].items():
        cells = []
        for pol in ("off", "on", "strict"):
            x = e[pol]
            if x["stopped"]:
                cells.append(f"{verdicts(x['decisions'])}; stopped")
            else:
                d = f"{verdicts(x['decisions'])}; " if x["decisions"] else ""
                cells.append(f"{d}{'right' if x['answer_right'] else 'wrong'}, saw {x['tokens_seen']}")
        ctx = f"{e['model_context']}" + (f", num_ctx {e['num_ctx']} set by the user" if "num_ctx" in e else "")
        lines.append(f"| {name} | {e['prompt_tokens']} | {ctx} | " + " | ".join(cells) + " |")
    lines.append("")


def market(lines):
    runs = {t: read(f"market_{t}.json") for t in MARKET}
    runs = {t: r for t, r in runs.items() if r}
    if not runs:
        return
    first = next(iter(runs.values()))
    lines += ["## S2: market cases L07 and L13 (simulated on a vLLM server)", "",
              "vLLM 0.30.0 serving Qwen3-4B with Qwen3's chat template changed to read `reasoning_effort` (a system "
              "line) and to keep every earlier turn's reasoning, given with `--chat-template` and declared in a pinned "
              f"manifest: `{first['declared']}`. L07 is planted in the server (ENTAIL_SEED=drop_effort: "
              "`reasoning_effort` no longer reaches the template); L13 is the client's (the earlier assistant turn "
              "sent without its reasoning). Each cell: HTTP status and the prompt's token count from the response's "
              "usage (testbed/m55_market_serve.py).", "",
              "| request | case | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
    names = next(list(r["requests"]) for r in runs.values() if r.get("requests"))
    for n in names:
        case = next(r["requests"][n]["case"] for r in runs.values() if r.get("requests"))
        cells = []
        for r in runs.values():
            x = (r.get("requests") or {}).get(n)
            cells.append("server did not start" if x is None else
                         f"{x['status']}" + (f", {x['prompt_tokens']} tokens" if x.get("prompt_tokens") else ""))
        lines.append(f"| {n} | {case} | " + " | ".join(cells) + " |")
    lines.append("")
    for t, r in runs.items():
        c = Counter(f"{d['boundary']} {d['verdict']}" for d in r["decisions"])
        lines.append(f"- `{t}` ({', '.join(f'{k}={v}' for k, v in r['env'].items())}): recorded "
                     + ("; ".join(f"{k} x{n}" for k, n in c.items()) or "nothing"))
    lines.append("")


def main():
    lines = ["# M5.5: measurements under the policy that reports and goes on", "",
             "Generated by `testbed/m55_summarize.py` from the files in this folder. RTX 4070 Ti (12 GB), WSL2.", ""]
    graph(lines)
    rolebench(lines)
    l05(lines)
    market(lines)
    out = os.path.join(DIR, "SUMMARY.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
