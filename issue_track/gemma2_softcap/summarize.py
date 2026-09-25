"""Applies the pass/fail rules in PROTOCOL.md section 5 to results/*.json and writes RESULTS.md."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def get(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    lines = ["# Gemma 2 softcap 측정 결과", "",
             "판정 기준은 `PROTOCOL.md` 5절이다(결과 전에 작성). 이 파일은 `summarize.py`가 결과 JSON에서 만든다.", ""]
    for size in ("2b", "9b"):
        e0, e1, e2, e3, e4 = (get(f"{e}_{size}.json") for e in ("e0", "e1", "e2", "e3", "e4"))
        if not any((e0, e1, e2, e3, e4)):
            continue
        lines += [f"## Gemma 2 {size.upper()}", ""]
        if e0:
            lines += [f"- 가중치: {e0['weights']}",
                      f"- E0 기본 경로: **{e0['default_attn_implementation']}**, softcap 경고·로그: "
                      f"**{'있음' if e0['softcap_mentioned'] else '없음'}** "
                      f"(경고 {len(e0['python_warnings'])}건, transformers 로그 {len(e0['transformers_log'])}건)"]
        verdict_silent = None
        if e1:
            s = e1["summary_pairs"]
            lines += ["", "### E1 로짓 비교 (teacher forcing)", "",
                      "| 비교 | 평균 KL | 최상위 토큰 불일치 | 로짓 최대 차이 | 위치 수 |", "|---|---|---|---|---|"]
            for k, v in s.items():
                lines.append(f"| {k} | {v['kl_mean']:.3e} | {v['top1_disagree']:.4f} | {v['max_abs_logit_diff']:.3f} | {v['positions']} |")
            mech = s.get("eager_nocap|sdpa")
            noise = s.get("eager|flex_attention")
            test = s.get("eager|sdpa")
            if mech:
                mech_ok = (1 - mech["top1_disagree"]) >= 0.995 and mech["kl_mean"] < 1e-3
                if e0:
                    verdict_silent = (e0["default_attn_implementation"] == "sdpa" and not e0["softcap_mentioned"] and mech_ok)
                lines += ["", f"- 기전 확인 (sdpa ≈ softcap 끈 eager: 일치율 ≥ 99.5%, KL < 1e-3): **{'성립' if mech_ok else '불성립'}**"]
            if noise and test:
                r_top = test["top1_disagree"] / max(noise["top1_disagree"], 1e-12)
                r_kl = test["kl_mean"] / max(noise["kl_mean"], 1e-30)
                impact = r_top > 3 or r_kl > 3
                lines += [f"- 출력 영향 (sdpa 대 eager가 잡음 대조군의 3배 초과): 불일치 비 {r_top:.1f}배, KL 비 {r_kl:.1f}배 → "
                          f"**{'영향 있음' if impact else '잡음 수준'}**"]
            sc = e1["pre_softcap_scores"]
            mx = max(r["max_abs"] for r in sc)
            worst = max(sc, key=lambda r: r["frac_gt_30"])
            lines += [f"- softcap 전 어텐션 점수: 모든 층의 최대 |점수| {mx:.1f}. |점수|>30 비율이 가장 큰 층은 "
                      f"{worst['layer']}번({worst['type']}) {worst['frac_gt_30']:.2e}, 같은 층 |점수|>50 비율 {worst['frac_gt_50']:.2e}"]
        if verdict_silent is not None:
            lines += [f"- **판정(원래 기준): 기본 경로가 softcap을 조용히 버린다 → {'확정' if verdict_silent else '확정 아님'}**"]
        e1b = get(f"e1fp32_{size}.json")
        if e1b:
            s = e1b["summary_pairs"]
            lines += ["", "### E1b 로짓 비교, 어텐션 fp32 (PROTOCOL 개정 1)", "",
                      "| 비교 | 평균 KL | 최상위 토큰 불일치 | 로짓 최대 차이 | 위치 수 |", "|---|---|---|---|---|"]
            for k, v in s.items():
                lines.append(f"| {k} | {v['kl_mean']:.3e} | {v['top1_disagree']:.4f} | {v['max_abs_logit_diff']:.3f} | {v['positions']} |")
            noise, mech, test = s["eager|flex_attention"], s["eager_nocap|sdpa"], s["eager|sdpa"]
            cond_a = mech["top1_disagree"] <= 2 * noise["top1_disagree"] + 1e-4
            cond_b = test["kl_mean"] >= 10 * noise["kl_mean"]
            lines += [f"- 조건 A (softcap 끈 eager 대 sdpa의 불일치 ≤ eager 대 flex의 2배): "
                      f"{mech['top1_disagree']:.4f} 대 {noise['top1_disagree']:.4f} → {'성립' if cond_a else '불성립'}",
                      f"- 조건 B (eager 대 sdpa의 KL ≥ eager 대 flex의 10배): {test['kl_mean']:.2e} 대 {noise['kl_mean']:.2e} → "
                      f"{'성립' if cond_b else '불성립'}"]
            if e0:
                ok = e0["default_attn_implementation"] == "sdpa" and not e0["softcap_mentioned"] and cond_a and cond_b
                lines += [f"- **판정(원래 개정 기준, 확인용): 기본 경로가 softcap을 조용히 버린다 → {'확정' if ok else '확정 아님'}**"]
            nocap = s["eager|eager_nocap"]
            pat_a = mech["kl_mean"] <= 2 * noise["kl_mean"]
            rel = abs(test["kl_mean"] - nocap["kl_mean"]) / max(nocap["kl_mean"], 1e-30)
            pat_b = rel <= 0.3
            lines += ["", "**패턴 검사 (PROTOCOL 개정 2)**",
                      f"- (가) softcap 끈 eager 대 sdpa의 KL ≤ 잡음의 2배: {mech['kl_mean']:.2e} 대 {noise['kl_mean']:.2e} → "
                      f"{'성립' if pat_a else '불성립'}",
                      f"- (나) eager 대 sdpa의 KL이 eager 대 (softcap 끈 eager)와 30% 이내: 차이 {rel * 100:.1f}% → "
                      f"{'성립' if pat_b else '불성립'}",
                      f"- 효과 크기: eager 대 sdpa의 KL ÷ 잡음 = {test['kl_mean'] / max(noise['kl_mean'], 1e-30):.1f}배",
                      f"- **판정(패턴 기준): 기본 경로가 softcap을 버린다 → "
                      f"{'확정' if (pat_a and pat_b and e0 and e0['default_attn_implementation'] == 'sdpa' and not e0['softcap_mentioned']) else '확정 아님'}**"]
        if e2:
            c = e2["compare_to_eager"]
            lines += ["", "### E2 탐욕 생성 (30개 × 256토큰, eager와 비교)", ""]
            for p, v in c.items():
                firsts = [x for x in v["first_divergence"] if x is not None]
                med = sorted(firsts)[len(firsts) // 2] if firsts else None
                lines.append(f"- {p}: 출력이 같은 비율 {v['identical_fraction']:.2f}, 갈라진 경우 처음 갈라진 위치의 중앙값 {fmt(med)}")
        if e3:
            lines += ["", f"### E3 GSM8K 앞 {e3['n']}문제", ""]
            for p, v in e3["paths"].items():
                lo, hi = v["wilson95"]
                lines.append(f"- {p}: 정답률 {v['accuracy']:.3f} (95% 구간 {lo:.3f}~{hi:.3f}), {v['seconds']}초")
            pr = e3["paired"]
            impact3 = abs(pr["accuracy_diff_eager_minus_sdpa"]) >= 0.02 and pr["mcnemar_exact_p"] < 0.05
            lines += [f"- 짝 비교: eager만 맞힘 {pr['only_eager_correct']}, sdpa만 맞힘 {pr['only_sdpa_correct']}, "
                      f"McNemar p={pr['mcnemar_exact_p']:.3g}, 차이 {pr['accuracy_diff_eager_minus_sdpa']*100:+.1f}%p → "
                      f"**{'과제 영향 있음' if impact3 else '이 표본 크기에서 확인되지 않음'}**"]
        if e4:
            lines += ["", f"### E4 vLLM {e4['vllm_version']} 대조", "", "| HF 경로 | vLLM과 최상위 토큰 일치 | 다음 토큰 로그확률 평균 차이 |", "|---|---|---|"]
            for p, v in e4["summary"].items():
                lines.append(f"| {p} | {v['top1_agree_vs_vllm']:.4f} | {v['mean_abs_dlogprob_vs_vllm']:.4f} |")
        lines.append("")
    open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
