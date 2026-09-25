"""M7.2: testbed/PROBLEMS.md -> testbed/problems.json, the test problems as entail's pytest plugin reads them
(`pytest --entail-problems testbed/problems.json`, fixture entail_problem; entail.testing.problems).

PROBLEMS.md stays the source: this reads its tables by their header names and writes one object per row with an ID -
id, fact, defect (where the defective version lives), fixed, expected, milestone, site, result. Rolebench rows point
at their case file; the others at the document the mechanism comes from.

Run: python testbed/m72_problems.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "PROBLEMS.md")
OUT = os.path.join(HERE, "problems.json")


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def plain(text):
    return re.sub(r"`([^`]*)`", r"\1", text).strip()


def tables(text):
    """(section title, header cells, rows of cells) for every table in the document."""
    section, header, rows, out = "", None, [], []
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        if line.startswith("|"):
            c = cells(line)
            if header is None:
                header = c
            elif set(line.replace("|", "").strip()) <= set("-: "):
                continue
            else:
                rows.append(c)
        elif header is not None:
            out.append((section, header, rows))
            header, rows = None, []
    if header is not None:
        out.append((section, header, rows))
    return out


def case_file(problem_id):
    number = problem_id.split("-", 1)[1]
    found = sorted(glob.glob(os.path.join(ROOT, "rolebench", "cases", f"{number}_*", "case.py")))
    return os.path.relpath(found[0], ROOT).replace(os.sep, "/") if found else None


def main():
    problems = []
    with open(SRC, encoding="utf-8") as f:
        text = f.read()
    for section, header, rows in tables(text):
        if not header or header[0] != "ID":
            continue
        col = {name: i for i, name in enumerate(header)}

        def get(row, *names):
            for n in names:
                if n in col and col[n] < len(row):
                    return plain(row[col[n]])
            return ""

        for row in rows:
            pid = plain(row[0])
            if not re.match(r"^(rb|fd|mk|loc)-", pid):
                continue
            defect, fixed = "", None
            if pid.startswith("rb-"):
                case = case_file(pid)
                defect, fixed = (f"{case}:defect", f"{case}:fixed") if case else ("", None)
            elif pid.startswith("fd-"):
                defect = get(row, "근거")
            elif pid.startswith("mk-"):
                defect = f"reinvestigation/market_incidents.md ({pid[3:]})"
            else:
                defect = get(row, "심는 곳")
            problems.append({"id": pid, "fact": get(row, "사실"), "defect": defect, "fixed": fixed,
                             "expected": get(row, "기대 판정", "기대 보고"), "milestone": get(row, "단계"),
                             "site": get(row, "자리"), "result": get(row, "결과"), "section": section})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "testbed/PROBLEMS.md", "made_by": "testbed/m72_problems.py", "problems": problems},
                  f, ensure_ascii=False, indent=1)
    print(f"{len(problems)} problems -> {os.path.relpath(OUT, ROOT)}")
    for p in problems:
        print(f"  {p['id']:14} {p['fact'][:28]:28} {p['expected'][:40]}")


if __name__ == "__main__":
    main()
