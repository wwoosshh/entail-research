"""M10 E3, huggingface/diffusers#14569 (testbed/M10_PROTOCOL.md 3.3): DDPMScheduler with variance_type
"fixed_large_log" takes the square root of a log variance and returns NaN. The issue's own scheduler-only
reproduction, and a DDPM pipeline (google/ddpm-cifar10-32, 20 steps, CPU) loaded through diffusers so that entail's
diffusers adapter is in the path when it is on.
Run in ~/venvs/gpu: python testbed/m10_e3/df14569.py <out.json>
"""
import json
import os
import sys


def main():
    import torch
    from diffusers import DDPMParallelScheduler, DDPMPipeline, DDPMScheduler

    issue = {}
    for cls in (DDPMScheduler, DDPMParallelScheduler):
        s = cls(variance_type="fixed_large_log")
        sample = torch.zeros((1, 2, 2, 2))
        out = s.step(torch.zeros_like(sample), 500, sample, generator=torch.Generator().manual_seed(0)).prev_sample
        issue[cls.__name__] = int(torch.isnan(out).sum())
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config, variance_type="fixed_large_log")
    img = pipe(batch_size=1, num_inference_steps=20, generator=torch.Generator().manual_seed(0),
               output_type="np").images
    nan_pixels = int(torch.isnan(torch.from_numpy(img)).sum())
    row = {"entail": os.environ.get("ENTAIL", "off"), "issue_repro_nan_counts": issue,
           "pipeline_nan_pixels": nan_pixels, "pipeline_pixels": int(img.size),
           "reproduced": nan_pixels > 0 or any(issue.values())}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
