"""Print and write the per-arm detection table from results/*.json (PROTOCOL.md section 4)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = ["| 사례 | 사실 | 재현 | 갈래 | 모드 | 검출 | 결함 판 오류 문구 | 수정 판 |", "|---|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        if "meta" not in r:  # skip summary files such as g1_check.json
            continue
        det = r.get("detection")
        if not det:
            rows.append(f"| {r['case']} | {r['meta'].get('fact')} | {'예' if r.get('reproduced') else '아니오'} | — | — | 검출 판 없음 | | |")
            continue
        for arm, v in det.items():
            msg = v["defect"].get("message") or v["defect"].get("other_error") or ""
            fix = "오류 없음" if not v["fixed"]["flagged"] and not v["fixed"].get("other_error") else \
                (v["fixed"].get("message") or v["fixed"].get("other_error") or "")[:60]
            rows.append(f"| {r['case']} | {r['meta'].get('fact')} | {'예' if r.get('reproduced') else '아니오'} | {arm} | "
                        f"{v['mode']} | {'예' if v['detected'] else '아니오'} | {msg[:110]} | {fix} |")
    text = "# rolebench 검출 표\n\n`detection_table.py`가 `results/*.json`에서 만든다.\n\n" + "\n".join(rows) + "\n"
    open(os.path.join(HERE, "results", "detection.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
