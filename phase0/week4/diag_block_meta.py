"""Why do declared and rediscovered block metadata differ in E3? (CPU only, a few ms)

Decode case: one query row, 640 key slots, keys 0..512 valid (p = 512, valid length 513).
Declared: blocks 0-3 are entirely valid (full), block 4 is partly valid.
Rediscovered: create_block_mask evaluates the predicate on a grid padded to 128 query rows.
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask

VALID = torch.tensor([513])


def valid_len_mask(b, h, q_idx, kv_idx):
    return kv_idx < VALID[b]


bm = create_block_mask(valid_len_mask, B=1, H=None, Q_LEN=1, KV_LEN=640, device="cpu")
print("rediscovered: partial blocks", bm.kv_num_blocks.flatten().tolist(), "indices",
      bm.kv_indices.flatten().tolist())
print("rediscovered: full blocks   ", bm.full_kv_num_blocks.flatten().tolist() if bm.full_kv_num_blocks is not None
      else None, "indices", bm.full_kv_indices.flatten().tolist() if bm.full_kv_indices is not None else None)
v = 513
print("declared:     partial blocks", [int(v % 128 != 0)], "index", [v // 128])
print("declared:     full blocks   ", [v // 128], "indices", list(range(v // 128)))
