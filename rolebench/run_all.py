"""Run every rolebench case in its own process, then write results/results.md (PROTOCOL.md section 7).

Usage: python run_all.py [case-dir-prefix ...]   e.g. python run_all.py 05 06
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(prefixes):
    cases = sorted(d for d in glob.glob(os.path.join(HERE, "cases", "*")) if os.path.isfile(os.path.join(d, "case.py")))
    if prefixes:
        cases = [c for c in cases if any(os.path.basename(c).startswith(p) for p in prefixes)]
    for c in cases:
        print("==", os.path.basename(c), flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, "run_case.py"), c], capture_output=True, text=True,
                           timeout=3600)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print("\n".join(tail), flush=True)
    write_table()


def write_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        if "meta" not in r:  # results/ also holds summary files such as g1_check.json
            continue
        m = r["meta"]
        d = r.get("defect_vs_reference") or {}
        f = r.get("fixed_vs_reference") or {}
        rows.append(f"| {r['case']} | {m.get('fact')} | {m.get('kind')} | {'예' if r.get('reproduced') else '아니오'} | "
                    f"{d.get('max_abs', float('nan')):.3g} | {f.get('max_abs', float('nan')):.3g} | "
                    f"{d.get('top1_agree', '—') if isinstance(d.get('top1_agree'), str) else round(d.get('top1_agree', float('nan')), 3)} | "
                    f"{'예' if r.get('defect_silent') else '아니오'} | {r.get('seconds')} |")
    head = ["# rolebench 결과", "", "`run_all.py`가 `results/*.json`에서 만든다. 판정 정의는 `PROTOCOL.md`.", "",
            "| 사례 | 사실 | 종류 | 재현 | 결함 판 최대 차이 | 수정 판 최대 차이 | 결함 판 최상위 토큰 일치 | 결함 판 조용함 | 초 |",
            "|---|---|---|---|---|---|---|---|---|"]
    with open(os.path.join(HERE, "results", "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(head + rows) + "\n")
    print("\n".join(head + rows))


if __name__ == "__main__":
    main(sys.argv[1:])
