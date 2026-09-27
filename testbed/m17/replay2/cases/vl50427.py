"""M17.6 case, vllm-project/vllm#50427 (testbed/M16_PROTOCOL.md 7): FlexAttention's paged K/V element offset is
computed in int32, so past 65,536 blocks (num_kv_heads 8, head_size 128, block 16) the address wraps ~4 GiB below
the K/V base: an illegal memory access, or - when that memory is mapped - silently wrong attention output. The
report's deterministic standalone trigger (no vLLM, no weights), unchanged in its arithmetic: block id 65535 must
give the value constant 0.75; block id 65536 with 5 GiB mapped below the base reads that memory (-8.0). The
crashing mode (65536 with nothing mapped below) is not run. Reproduced when the second mean is not 0.75.
Run in ~/venvs/vllm (torch 2.13): python testbed/m17/replay2/cases/vl50427.py <out.json>
"""
import gc
import json
import os
import sys

BLOCK, NKVH, HD, QH = 16, 8, 128, 32
NUM_BLOCKS, Q_LEN, PER_ROW = 68370, 512, 64
K_CONST, V_CONST, BELOW = 0.125, 0.75, -8.0


def trigger(block_id, below_gib):
    import torch
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    dev, rows = "cuda", NUM_BLOCKS * BLOCK
    kv_elems = rows * 2 * NKVH * HD
    if below_gib:
        below = int(below_gib * 1024 ** 3) // 2
        big = torch.empty(below + kv_elems, dtype=torch.bfloat16, device=dev)
        big[:below].fill_(BELOW)
        kv = big[below:].view(rows, 2, NKVH, HD)
    else:
        big = None
        kv = torch.empty(rows, 2, NKVH, HD, dtype=torch.bfloat16, device=dev)
    kv[:, 0].fill_(K_CONST)
    kv[:, 1].fill_(V_CONST)
    k = kv[:, 0].permute(1, 0, 2).unsqueeze(0)
    v = kv[:, 1].permute(1, 0, 2).unsqueeze(0)
    q = (torch.full((Q_LEN, QH, HD), K_CONST, dtype=torch.bfloat16, device=dev).permute(1, 0, 2).unsqueeze(0))
    nq = Q_LEN // BLOCK
    bm = BlockMask.from_kv_blocks(
        kv_num_blocks=torch.full((nq,), PER_ROW, dtype=torch.int32, device=dev)[None, None],
        kv_indices=torch.full((nq, PER_ROW), block_id, dtype=torch.int32, device=dev)[None, None],
        full_kv_num_blocks=None, full_kv_indices=None, BLOCK_SIZE=(BLOCK, BLOCK),
        mask_mod=lambda b, h, qi, kvi: qi >= 0, seq_lengths=(Q_LEN, rows), compute_q_blocks=False)
    out = torch.compile(flex_attention, fullgraph=True)(
        q, k, v, None, bm, HD ** -0.5, enable_gqa=True,
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": BLOCK, "BLOCK_N": BLOCK})
    torch.cuda.synchronize()
    mean = float(out.float().mean())
    del out, q, k, v, kv, big, bm
    gc.collect()
    torch.cuda.empty_cache()
    return mean


def main():
    import torch

    row = {"entail": os.environ.get("ENTAIL", "off"), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "expected": V_CONST, "bound_blocks": 2 ** 31 // (BLOCK * 2 * NKVH * HD)}
    row["mean_block_65535"] = trigger(65535, 0.0)
    row["mean_block_65536_5gib_below"] = trigger(65536, 5.0)
    row["reproduced"] = bool(abs(row["mean_block_65535"] - V_CONST) < 1e-3
                             and abs(row["mean_block_65536_5gib_below"] - V_CONST) > 1e-3)
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
