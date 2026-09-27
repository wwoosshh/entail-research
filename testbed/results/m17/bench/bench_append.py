"""Variants of the record append on the 9P mount: open/append/close per line (today), a handle kept open with a
flush per line, a handle kept open with Python's buffer, and os.write on a kept fd."""
import os
import time

N = 200
path = os.path.expanduser("~/ai_compiler/entail_logs/_bench_variants.jsonl")
line = '{"pid": 1, "timing": "container:vllm.request.block_hashes", "ms": 0.1}\n'


def bench(label, fn, done=None):
    fn()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    if done:
        done()
    print(f"{label:55s} {(time.perf_counter() - t0) / N * 1e3:8.3f} ms/call")


def a():
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


bench("open/append/close per line", a)

fh = open(path, "a", encoding="utf-8")
bench("kept open, write + flush per line", lambda: (fh.write(line), fh.flush()), fh.close)
fh = open(path, "a", encoding="utf-8")
bench("kept open, buffered write (flush at close)", lambda: fh.write(line), fh.close)
fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
bench("kept fd, os.write per line", lambda: os.write(fd, line.encode()), lambda: os.close(fd))
os.remove(path)
