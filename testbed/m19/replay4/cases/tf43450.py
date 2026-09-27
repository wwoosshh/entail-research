"""M19 L4 replay 4 case, huggingface/transformers#43450 (testbed/M16_PROTOCOL.md 9): AutoVideoProcessor takes a batched
B x T x C x H x W tensor as one video and returns 1 x 1 x B x T x C x H x W. The report's script: reproduced when the
output shape is not (4, 5, 3, 256, 256).
Run: python testbed/m19/replay4/cases/tf43450.py <out.json>  (~/venvs/gpu: transformers 5.17.0; ~/venvs/vllm0210: 4.57.6)
"""
import json
import os
import sys


def main():
    import torch
    import transformers
    from transformers import AutoVideoProcessor

    processor = AutoVideoProcessor.from_pretrained("qubvel-hf/vjepa2-vitl-fpc16-256-ssv2")
    out = processor(torch.zeros(4, 5, 3, 100, 100), return_tensors="pt", do_sample_frames=False).pixel_values_videos
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__,
           "output_shape": list(out.shape), "expected_shape": [4, 5, 3, 256, 256]}
    row["reproduced"] = row["output_shape"] != row["expected_shape"]
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
