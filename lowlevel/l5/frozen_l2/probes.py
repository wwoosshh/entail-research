"""L2 probe prompts: plain text, fixed, written for this study. Long enough to span several KV blocks and several
prefill chunks, with a repeated list (so n-gram speculation proposes and accepts tokens) and two prompts that share a
long prefix (so the second can hit the prefix cache)."""

STORY = (
    "The harbour town woke slowly on the first cold morning of the season. Fishing boats knocked against the pier while "
    "the tide pulled at their ropes, and the smell of salt and diesel drifted up the narrow streets. At the bakery on "
    "the corner, Mara lit the ovens before dawn, as her grandmother had done for forty years, and set out the long "
    "wooden trays that still carried the marks of a thousand loaves. The first customers were always the same: the "
    "harbour master, who wanted two rolls and the weather report, the teacher from the school on the hill, who bought "
    "a loaf for the staff room, and old Tomas, who never bought anything but stayed to talk about the ships he had "
    "sailed on as a young man. That morning Tomas arrived early and did not sit down. He had seen something in the "
    "water beyond the breakwater, he said, a shape that was not a boat and not a whale, moving against the current. "
)
PROBES = [
    {"id": "story_a", "text": STORY + "Mara wiped her hands on her apron and asked him what he thought it was. Tomas"},
    {"id": "story_b", "text": STORY + "The harbour master laughed and said the old man had been dreaming again, but"},
    {"id": "code", "text": (
        "def merge_intervals(intervals):\n    \"\"\"Merge overlapping [start, end] intervals and return them sorted.\"\"\"\n"
        "    if not intervals:\n        return []\n    intervals = sorted(intervals, key=lambda iv: iv[0])\n"
        "    merged = [list(intervals[0])]\n    for start, end in intervals[1:]:\n        last = merged[-1]\n"
        "        if start <= last[1]:\n            last[1] = max(last[1], end)\n        else:\n")},
    {"id": "numbers", "text": (
        "Quarterly report. Revenue in the first quarter was 1,284 thousand, in the second 1,391 thousand, and in the "
        "third 1,457 thousand. Costs were 902, 951 and 1,010 thousand. The operating margin in the third quarter was")},
    {"id": "repeat", "text": (
        "apple, banana, cherry, date, elderberry, fig, grape; apple, banana, cherry, date, elderberry, fig, grape; "
        "apple, banana, cherry, date, elderberry, fig, grape; apple, banana, cherry, date,")},
    {"id": "short", "text": "The capital of France is"},
]
CACHE_PAIRS = [{"warm": "story_a", "target": "story_b"}]

# image probes for multimodal models (run_vllm.py with L2_IMAGES=1): plain generated pictures, one question each
IMAGE_PROBES = [
    {"id": "red", "image": {"kind": "solid", "color": [220, 30, 30], "size": [512, 512]},
     "text": "What color is this image?"},
    {"id": "square", "image": {"kind": "square", "color": [30, 60, 220], "background": [255, 255, 255], "size": [512, 512]},
     "text": "Describe the shape in this picture."},
    {"id": "stripes", "image": {"kind": "stripes", "colors": [[20, 160, 60], [240, 220, 40]], "size": [448, 448]},
     "text": "What colors do you see, and how are they arranged?"},
]
