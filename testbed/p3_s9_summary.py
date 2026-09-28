"""S9 summary (ROADMAP product track P3): reads testbed/results/p3/s9*.json and writes summary.json there.

  python testbed/p3_s9_summary.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "p3")


def row(path):
    d = json.load(open(path, encoding="utf-8"))
    return {
        "run": os.path.basename(path)[:-5], "mode": d.get("mode"), "paths_s": d.get("paths_s"), "plant": d.get("plant") or None,
        "planted_calls": d["planted_calls"], "load_s": d["load_s"],
        "turned_off": [x["resolution"] for x in d["turned_off"]],
        "disagree": [p["paths"] for p in d["pairs"] if p["differs"]],
        "pairs": {p["paths"]: {"margin": p["margin"], "pdrift": p["pdrift"]} for p in d["pairs"]},
        "said": d["said"], "store": {k: v.get("status") for k, v in d["store"].items()},
        "answer": d["answer"], "bench": d.get("bench"),
    }


def main():
    rows = [row(p) for p in sorted(glob.glob(os.path.join(OUT, "s9*.json"))) if not p.endswith(".safe_paths.json")]
    print("engine starts in s9*:", len(rows))
    by = {r["run"]: r for r in rows}

    def recovered(cfg):
        first, second, third = (by[f"s9a_{cfg}.start{n}"] for n in (1, 2, 3))
        return {"start1_disagree": first["disagree"], "start1_said": first["said"],
                "start2_off": second["turned_off"], "start2_disagree": second["disagree"], "start2_said": second["said"],
                "start3_off": third["turned_off"], "start3_disagree": third["disagree"],
                "recovered": bool(first["disagree"]) and not second["disagree"] and not third["disagree"]}

    def cost(cfg):
        on, off = by[f"s9e_{cfg}_spec_on"], by[f"s9e_{cfg}_safe_path"]
        return {"spec_on_tok_s": on["bench"]["tok_s"], "safe_path_tok_s": off["bench"]["tok_s"],
                "ratio": round(off["bench"]["tok_s"] / on["bench"]["tok_s"], 3),
                "load_s": {"spec_on": on["load_s"], "safe_path": off["load_s"]},
                "spec_on_disagree": on["disagree"], "safe_path_disagree": off["disagree"]}

    def split(off_run, all_run, where):
        """Whether ENTAIL_SAFE=all told the planted fault's side right: it must show with nothing off, and then be
        gone (inside) or stay with 'the cause is outside them' said (outside)."""
        a, b = by[off_run], by[all_run]
        seen = bool(a["disagree"])
        stays = bool(b["disagree"])
        said_outside = any("the cause is outside them" in s for s in b["said"])
        told = "outside" if stays and said_outside else ("inside" if not stays else "stays, not said")
        return {"off": off_run, "all": all_run, "fault": where, "seen_with_nothing_off": seen,
                "planted_calls": {"off": a["planted_calls"], "all": b["planted_calls"]},
                "all_disagree": b["disagree"], "told": told, "right": seen and told == where}

    explicit_before = [split("s9b_inside_off", "s9b_inside_all", "inside"),
                       split("s9b_outside_off", "s9b_outside_all", "outside"),
                       split("s9c_inside_kernel_off", "s9c_inside_kernel_all", "inside"),
                       split("s9c_inside_op_off", "s9c_inside_op_all", "inside")]
    explicit_after = [split("s9c_inside_kernel_off", "s9d_inside_kernel_all", "inside"),
                      split("s9c_inside_op_off", "s9d_inside_op_all", "inside"),
                      split("s9b_outside_off", "s9d_outside_all", "outside")]
    clean = {r: {"disagree": by[r]["disagree"], "turned_off": by[r]["turned_off"]}
             for r in ("s9b_clean_off", "s9b_clean_all", "s9d_clean_all")}
    auto_planted = {r: {"disagree": by[r]["disagree"], "turned_off": by[r]["turned_off"], "said": by[r]["said"],
                        "store": by[r]["store"], "answer": by[r]["answer"]}
                    for r in ("s9b_inside_auto", "s9b_inside_auto2", "s9d_inside_kernel_auto1",
                              "s9d_inside_kernel_auto2")}
    as_met = {plant: {"on": {"disagree": by[f"s9f_{plant}_1off"]["disagree"], "answer": by[f"s9f_{plant}_1off"]["answer"]},
                      "all": {"disagree": by[f"s9f_{plant}_2all"]["disagree"], "said": by[f"s9f_{plant}_2all"]["said"],
                              "answer": by[f"s9f_{plant}_2all"]["answer"]}}
              for plant in ("inside_kernel", "outside")}
    check_cost = {cfg: {k: {"paths_s": by[f"s9g_{cfg}_{k}"]["paths_s"], "load_s": by[f"s9g_{cfg}_{k}"]["load_s"]}
                        for k in ("spec_on", "safe_path", "no_check")} for cfg in ("q35q", "nemotron")}
    live = {}
    for name in ("live_graphs_off", "live_graphs_all", "live_sglang_off", "live_sglang_all"):
        d = json.load(open(os.path.join(OUT, name + ".json"), encoding="utf-8"))
        live[name] = {"turned_off": [x if isinstance(x, str) else x["resolution"] for x in d["turned_off"]],
                      "ran_with": d.get("ran_with") or {"eager": d.get("engine_eager"),
                                                        "prefix_cache": d.get("engine_prefix_cache")},
                      "answer": d["answer"], "load_s": d["load_s"]}
    prereg = json.load(open(os.path.join(OUT, "prereg_s9b2.json"), encoding="utf-8"))
    summary = {
        "about": "S9 on vLLM 0.30.0 (ROADMAP product track P3, LIBRARY_DESIGN.md 8 and 13.6). Table before the fix: "
                 "entail product b6f1c7f^ (73b44a9); after: b6f1c7f; part f (the inside statement): bb5e80d.",
        "a_selective_safe_path": {"q35q": recovered("q35q"), "nemotron": recovered("nemotron")},
        "a_cost": {"q35q": cost("q35q"), "nemotron": cost("nemotron")},
        "b_explicit_before_fix": explicit_before,
        "b_explicit_after_fix": explicit_after,
        "b_right_after_fix": f"{sum(x['right'] for x in explicit_after)}/{len(explicit_after)}",
        "b_as_a_user_meets_it": as_met,
        "clean_starts": clean,
        "g_self_check_cost": check_cost,
        "h_preregistered": {"right": prereg["right"], "not_applicable": prereg["not_applicable"]},
        "live_checks": live,
        "auto_with_planted_fault": auto_planted,
        "runs": rows,
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    for k in ("a_selective_safe_path", "a_cost"):
        print(k, json.dumps(summary[k], ensure_ascii=False))
    for k in ("b_explicit_before_fix", "b_explicit_after_fix"):
        for x in summary[k]:
            print(k, x["all"], x["fault"], "->", x["told"], "right" if x["right"] else "WRONG", x["planted_calls"])
    print("b_right_after_fix", summary["b_right_after_fix"])
    print("clean", json.dumps(clean))


if __name__ == "__main__":
    main()
