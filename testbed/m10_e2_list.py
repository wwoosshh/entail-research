"""M10 E2: the run list for the downloaded models (the local copies ran in the first batch, list_local.txt).
Reads results/m10/e2/downloads.json; writes results/m10/e2/list_downloaded.txt ("name<TAB>model dir" per line).
Run: python testbed/m10_e2_list.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "m10", "e2")


def main():
    rows = json.load(open(os.path.join(R, "downloads.json"), encoding="utf-8"))
    lines = [f"{r['id'].replace('/', '__')}\t{r['path']}" for r in rows if r["status"] == "downloaded"]
    with open(os.path.join(R, "list_downloaded.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(len(lines), "models;", [r["id"] for r in rows if r["status"] not in ("downloaded", "local copy")])


if __name__ == "__main__":
    main()
