"""M3 setup: a copy of NoobAI-XL-Vpred without its 'v_pred' and 'ztsnr' marker keys - what a merge or conversion
that drops them leaves behind. Both markers are zero-length tensors, so only the header changes; every weight byte
is copied unchanged. The original file is not touched.
"""
import json
import os
import struct

SRC = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new\models\checkpoints\NoobAI-XL-Vpred-v1.0.safetensors"
DST_DIR = r"C:\Users\<user>\Desktop\ComfyUI\ComfyUI-new\models\checkpoints\entail_test"
DST = os.path.join(DST_DIR, "NoobAI-XL-Vpred-v1.0-no-marker.safetensors")
DROP = ("v_pred", "ztsnr")


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    with open(SRC, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        for k in DROP:
            start, end = header[k]["data_offsets"]
            assert start == end, f"{k} is not zero-length; dropping it would leave a hole"
        new = {k: v for k, v in header.items() if k not in DROP}
        raw = json.dumps(new, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)  # keep the data aligned as safetensors writers do
        with open(DST, "wb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            while True:
                chunk = f.read(1 << 26)
                if not chunk:
                    break
                out.write(chunk)
    print("wrote", DST, os.path.getsize(DST), "bytes; source", os.path.getsize(SRC), "bytes")


if __name__ == "__main__":
    main()
