"""entail/DESIGN.md section 7, measures (마) and (바), and the code-size half of (가).

  python generality_measure.py lines   code lines (no comments, docstrings, blank lines) of the adapters and the core
  python generality_measure.py scan    every model file of the user's ComfyUI: its own declarations, and what
                                       ComfyUI's rule (the v_pred / ztsnr keys) would set up; a mismatch is what the
                                       contract would act on
"""
import ast
import io
import json
import os
import subprocess
import sys
import tokenize

ENTAIL = r"<workspace>\entail"
MODELS = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new\models"
sys.path.insert(0, ENTAIL)


def code_lines(source):
    """Lines holding code: comments, docstrings and blank lines removed."""
    tree = ast.parse(source)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if ln not in doc_lines:
                lines.add(ln)
    return len(lines)


def lines():
    before = subprocess.run(["git", "-C", ENTAIL, "show", "HEAD:entail/adapters/comfyui.py"], capture_output=True,
                            text=True, encoding="utf-8").stdout
    out = {"comfyui.py at HEAD (0.3.0)": code_lines(before)}
    for rel in ("adapters/comfyui.py", "adapters/diffusers_adapter.py", "declared.py", "contract.py", "coverage.py",
                "behaviour.py", "ownership.py", "core.py", "facts.py"):
        p = os.path.join(ENTAIL, "entail", rel)
        if os.path.exists(p):
            out[rel] = code_lines(open(p, encoding="utf-8").read())
    uses = open(os.path.join(ENTAIL, "entail", "adapters", "comfyui.py"), encoding="utf-8").read()
    out["comfyui.py uses the core modules"] = [m for m in ("declared", "contract", "coverage", "behaviour", "ownership",
                                                           "Prediction") if m in uses]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def scan():
    from entail import declared
    from entail.facts import Prediction

    rows = []
    for sub in ("checkpoints", "diffusion_models", "loras"):
        for root, _, files in os.walk(os.path.join(MODELS, sub)):
            for f in sorted(files):
                if not f.endswith(".safetensors"):
                    continue
                p = os.path.join(root, f)
                keys, meta = declared.safetensors_header(p)
                d = declared.from_header(keys, meta)
                row = {"file": os.path.relpath(p, MODELS), "declared": str(d.get(Prediction)) if d.get(Prediction) else None,
                       "source": d.source(Prediction), "base": str(d.get("Base")) if d.get("Base") else None,
                       "conflicts": d.conflicts, "lora_modules": len(d.modules)}
                if sub != "loras":
                    # ComfyUI v0.34's rule for SD-family checkpoints (comfy/supported_models.py): v if the 'v_pred'
                    # key is there, zsnr if 'ztsnr' is; eps otherwise.
                    comfy = Prediction("v" if "v_pred" in keys else "eps", "ztsnr" in keys)
                    decl = d.get(Prediction)
                    row["comfyui_sets_up"] = str(comfy)
                    row["contract_acts"] = bool(decl is not None and (decl.kind != comfy.kind or
                                                                      (decl.zsnr is not None and decl.zsnr != comfy.zsnr)))
                rows.append(row)
    acts = [r for r in rows if r.get("contract_acts")]
    out = {"files": len(rows), "checkpoints": sum(1 for r in rows if "comfyui_sets_up" in r),
           "loras": sum(1 for r in rows if "comfyui_sets_up" not in r),
           "with_prediction_declared": sum(1 for r in rows if r["declared"]),
           "conflicting_statements": [r["file"] for r in rows if r["conflicts"]],
           "contract_would_act_on": [(r["file"], r["declared"], r["comfyui_sets_up"]) for r in acts], "rows": rows}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "generality_scan.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"lines": lines, "scan": scan}[sys.argv[1]]()
