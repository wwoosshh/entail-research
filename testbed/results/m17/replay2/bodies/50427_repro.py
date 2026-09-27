# Sample code to reproduce the problem
#
# (a) Are you exposed? Compare the bound against the num_gpu_blocks vLLM logs.
#     Run any FlexAttention-backed model and look for
#       "GPU KV cache size: N tokens"  /  "num_gpu_blocks=..."
#     (attribute names below are for vLLM 0.23.x, V1 engine)
from vllm import LLM

llm = LLM(model="<your model>", block_size=16, enforce_eager=True,
          gpu_memory_utilization=0.85, max_num_seqs=128)

cfg = llm.llm_engine.vllm_config
mc, cc = cfg.model_config, cfg.cache_config
nkvh = mc.get_num_kv_heads(cfg.parallel_config)   # per-rank, so this scales with TP
hd = mc.get_head_size()
kv_packing = 2                                    # (num_blocks, 2, block_size, nkvh, hd)
bound = 2**31 // (cc.block_size * kv_packing * nkvh * hd)
print("num_gpu_blocks =", cc.num_gpu_blocks, " safe bound =", bound,
      " EXPOSED" if cc.num_gpu_blocks > bound else " ok")

# (b) Deterministic standalone trigger (no vLLM, no weights). It reproduces the
#     kernel-level overflow directly and flips between the two modes one block id
#     apart, so it is the quickest way to confirm the mechanism on any box.
#
#       python repro.py 65535     -> PASS,  mean 0.75   (correct)
#       python repro.py 65536     -> illegal memory access
#       python repro.py 65536 5   -> PASS,  mean -8.0   (silent corruption)
import sys
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

BLOCK, NKVH, HD, QH = 16, 8, 128, 32
NUM_BLOCKS, Q_LEN, PER_ROW = 68370, 512, 64
K_CONST, V_CONST, BELOW = 0.125, 0.75, -8.0

block_id = int(sys.argv[1])
below_gib = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
dev, rows = "cuda", NUM_BLOCKS * BLOCK
kv_elems = rows * 2 * NKVH * HD

if below_gib:  # put mapped memory below the K/V base
    below = int(below_gib * 1024**3) // 2
    big = torch.empty(below + kv_elems, dtype=torch.bfloat16, device=dev)
    big[:below].fill_(BELOW)
    kv = big[below:].view(rows, 2, NKVH, HD)
else:
    kv = torch.empty(rows, 2, NKVH, HD, dtype=torch.bfloat16, device=dev)
kv[:, 0].fill_(K_CONST)
kv[:, 1].fill_(V_CONST)

# strided views into one interleaved buffer: row stride = 2*NKVH*HD
k = kv[:, 0].permute(1, 0, 2).unsqueeze(0)
v = kv[:, 1].permute(1, 0, 2).unsqueeze(0)
q = (torch.full((Q_LEN, QH, HD), K_CONST, dtype=torch.bfloat16, device=dev)
     .permute(1, 0, 2).unsqueeze(0))

nq = Q_LEN // BLOCK
bm = BlockMask.from_kv_blocks(
    kv_num_blocks=torch.full((nq,), PER_ROW, dtype=torch.int32, device=dev)[None, None],
    kv_indices=torch.full((nq, PER_ROW), block_id, dtype=torch.int32, device=dev)[None, None],
    full_kv_num_blocks=None, full_kv_indices=None,
    BLOCK_SIZE=(BLOCK, BLOCK), mask_mod=lambda b, h, qi, kvi: qi >= 0,
    seq_lengths=(Q_LEN, rows), compute_q_blocks=False)

out = torch.compile(flex_attention, fullgraph=True)(
    q, k, v, None, bm, HD ** -0.5, enable_gqa=True,
    kernel_options={"FORCE_USE_FLEX_ATTENTION": True, "BLOCK_M": BLOCK, "BLOCK_N": BLOCK})
torch.cuda.synchronize()
print(f"block_id={block_id} below_gib={below_gib} "
      f"mean={float(out.float().mean()):.6f} expected={V_CONST}")
