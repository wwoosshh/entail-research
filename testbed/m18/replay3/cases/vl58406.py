"""M18.6 replay 3 case, vllm-project/vllm#58406 (testbed/M16_PROTOCOL.md 8): engine and request mm_processor_kwargs
split across flat and modality-scoped keys lose configured values in the shallow merge and the video overlay. The
report's snippet, unchanged: reproduced when the effective video settings are not {'fps': 2, 'size':
{'shortest_edge': 200, 'longest_edge': 1800}}.
Run in ~/venvs/vllm (0.30.0): python testbed/m18/replay3/cases/vl58406.py <out.json>
"""
import json
import os
import sys


def main():
    import vllm
    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing.context import overlay_modality_mm_kwargs

    config = MultiModalConfig(
        mm_processor_kwargs={"size": {"shortest_edge": 100, "longest_edge": 1000},
                             "videos_kwargs": {"fps": 2, "size": {"longest_edge": 1200}}},
        mm_device_do_normalize=False)
    request = {"size": {"longest_edge": 1800}, "videos_kwargs": {"size": {"shortest_edge": 200}}}
    merged = config.merge_mm_processor_kwargs(request)
    video = overlay_modality_mm_kwargs(merged, "video")
    effective = {"fps": video.get("fps"), "size": video.get("size")}
    expected = {"fps": 2, "size": {"shortest_edge": 200, "longest_edge": 1800}}
    row = {"entail": os.environ.get("ENTAIL", "off"), "vllm": vllm.__version__, "merged": merged,
           "effective_video": effective, "expected_video": expected, "reproduced": effective != expected}
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("RESULT", json.dumps(row, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
