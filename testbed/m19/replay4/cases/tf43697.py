"""M19 L4 replay 4 case, huggingface/transformers#43697 (testbed/M16_PROTOCOL.md 9): RTDetrV2ForObjectDetection gives
different logits and boxes on transformers 5.0.0 than on 4.57.6 for the same input (fixed in 5.1.0: the v5.0.0
tied-weights mapping direction kept the checkpoint's decoder head weights from loading). The reporter's fine-tuned
checkpoint is private; PekingU/rtdetr_v2_r18vd (saved before v5) stands in. The report's comparison: a fixed input,
the boxes on 4.57.6 written once as the reference (`ref` mode, ~/venvs/vllm0210), then every other version compared
with it. Reproduced when the boxes differ by more than 1e-3.
Run: python testbed/m19/replay4/cases/tf43697.py <out.json> [ref]
"""
import json
import os
import sys

MODEL = os.path.expanduser("~/models/replay4/PekingU__rtdetr_v2_r18vd")
REF = "<workspace>/testbed/results/m19/replay4/cases/tf43697_reference_4.57.6.json"


def main():
    import torch
    import transformers
    from transformers import RTDetrV2ForObjectDetection

    g = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(1, 3, 640, 640, generator=g, dtype=torch.float32)
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL).eval()
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    boxes, logits = out.pred_boxes[0], out.logits[0]
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__, "torch": torch.__version__}
    if len(sys.argv) > 2 and sys.argv[2] == "ref":
        json.dump({"transformers": transformers.__version__, "boxes": boxes.tolist(), "logits_head": logits[:5].tolist()},
                  open(REF, "w"))
        row["wrote_reference"] = REF
        row["reproduced"] = False
    else:
        ref = json.load(open(REF))
        rb = torch.tensor(ref["boxes"])
        diff = float((boxes - rb).abs().max())
        row.update({"reference": ref["transformers"], "max_box_diff": diff, "boxes_head": boxes[:3].tolist(),
                    "reference_boxes_head": ref["boxes"][:3]})
        row["reproduced"] = diff > 1e-3
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
