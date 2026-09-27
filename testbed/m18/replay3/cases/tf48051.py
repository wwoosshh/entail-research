"""M18.6 replay 3 case, huggingface/transformers#48051 (testbed/M16_PROTOCOL.md 8): video_utils.convert_to_rgb on a
channels-last RGBA numpy video blends with the alpha channel as the foreground. The report's snippet, unchanged:
reproduced when the output shape is not (2, 3, 1, 1) (and the pixel values are not the white-blended colours).
Run in a transformers venv (the fix is in 5.16.0 and later): python testbed/m18/replay3/cases/tf48051.py <out.json>
"""
import json
import os
import sys


def main():
    import numpy as np
    import transformers
    from transformers.video_utils import convert_to_rgb

    video = np.array([[[[255, 0, 0, 128]]], [[[0, 255, 0, 64]]]], dtype=np.uint8)
    rgb = convert_to_rgb(video, input_data_format="channels_last")
    shape = list(np.asarray(rgb).shape)
    row = {"entail": os.environ.get("ENTAIL", "off"), "transformers": transformers.__version__, "shape": shape,
           "values": np.asarray(rgb).reshape(-1).tolist()[:12], "reproduced": shape != [2, 3, 1, 1]}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
