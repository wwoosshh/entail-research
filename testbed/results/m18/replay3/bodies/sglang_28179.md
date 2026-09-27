## Summary
The `repetition_penalty` sampling parameter is accepted by the API but completely ignored during inference. This occurs because `BatchedRepetitionPenalizer` is not implemented or registered in SGLang's `penaltylib` orchestrator, causing the penalty to never be applied to logits.

## What Happens (User-facing symptoms)
**Scenario A - Repetition loops in generation:**
When setting `repetition_penalty > 1.0` (e.g., `1.15`) via the API, the model occasionally gets stuck in a repetition loop where it generates the same content indefinitely. The penalty has zero effect on the output distribution.

**Scenario B - Silent parameter drop:**
Users configure `repetition_penalty` expecting standard HuggingFace-style behavior (multiplicative logit adjustment), but the generation behaves identically to `repetition_penalty=1.0`. No warning or error is raised.

## Reproduction Steps
This issue is sporadic and cannot be reliably reproduced, but it occurs frequently enough to be a concern. The following setup has been observed to trigger it:

1. Start an SGLang server with DFlash speculative decoding:
   ```bash
   python -m sglang.launch_server --model-path <your-model> --speculative-algo dflash --port 30000
   ```
2. Send a completion request with `repetition_penalty > 1.0`:
   ```bash
   curl http://localhost:30000/v1/completions -H "Content-Type: application/json" -d '{"model": "<your-model>", "prompt": "Hello, ", "repetition_penalty": 1.15}'
   ```
3. Observe that the output sometimes contains repeated tokens/phrases identical to a request with `repetition_penalty: 1.0`.

## Observed Behavior (Logs)
The following logs show a request stuck in a repetition loop with 100% accept rate and continuously increasing token count:

```
[2026-06-14 15:28:57] Decode batch, #running-req: 1, #full token: 47360, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 290.40, #queue-req: 0
[2026-06-14 15:28:58] Decode batch, #running-req: 1, #full token: 47680, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 290.22, #queue-req: 0
[2026-06-14 15:28:59] Decode batch, #running-req: 1, #full token: 48000, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 290.38, #queue-req: 0
[2026-06-14 15:29:00] Decode batch, #running-req: 1, #full token: 48320, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 289.19, #queue-req: 0
[2026-06-14 15:29:01] Decode batch, #running-req: 1, #full token: 48640, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 289.10, #queue-req: 0
[2026-06-14 15:29:02] Decode batch, #running-req: 1, #full token: 48960, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 288.85, #queue-req: 0
[2026-06-14 15:29:03] Decode batch, #running-req: 1, #full token: 49280, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 289.00, #queue-req: 0
[2026-06-14 15:29:05] Decode batch, #running-req: 1, #full token: 49600, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 260.42, #queue-req: 0
[2026-06-14 15:29:06] Decode batch, #running-req: 1, #full token: 49920, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 287.98, #queue-req: 0
[2026-06-14 15:29:07] Decode batch, #running-req: 1, #full token: 50240, accept len: 8.00, accept rate: 1.00, gen throughput (token/s): 287.31, #queue-req: 0
```

Key observations:
- `accept rate: 1.00` (100%) - draft model predictions are fully accepted
- `#full token` continuously increases (47360 to 50240+)
- `#running-req: 1` - request never terminates
- `gen throughput` remains stable (~290 token/s)
- Model is stuck in a repetition loop despite `repetition_penalty` being set

## Root Cause Analysis
1. In `sglang/srt/sampling/sampling_batch_info.py`, the `SamplingBatchInfo` class initializes the penalty orchestrator:
   ```python
   penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
       vocab_size=vocab_size,
       batch=batch,
       penalizers={
           penaltylib.BatchedFrequencyPenalizer,
           penaltylib.BatchedMinNewTokensPenalizer,
           penaltylib.BatchedPresencePenalizer,
       },
   )
   ```
2. `BatchedRepetitionPenalizer` is **missing** from the `penalizers` set.
3. The `SamplingParams` class accepts and validates `repetition_penalty`, but since no penalizer implements it, the value is never read or applied during the `_apply_penalties()` phase.
4. This is a gap in SGLang's sampling pipeline compared to HuggingFace/vLLM, which fully support `repetition_penalty`.

## Impact
- Users cannot control repetition behavior via the standard `repetition_penalty` parameter.
- Models prone to looping (especially under high temperature or long context) cannot be stabilized.
- Breaks compatibility with HuggingFace `generation_config` and other frameworks that rely on this parameter.
- Particularly problematic with speculative decoding (DFlash/Eagle) where repetition loops can cause infinite generation.

## Proposed Fix
1. Implement `BatchedRepetitionPenalizer` in `sglang/srt/sampling/penaltylib/repetition_penalty.py` following the HuggingFace formula:
   - If `logit > 0`: `logit /= repetition_penalty`
   - If `logit < 0`: `logit *= repetition_penalty`
   - If `logit == 0`: unchanged
2. Register it in `penaltylib/__init__.py`.
3. Add `penaltylib.BatchedRepetitionPenalizer` to the `penalizers` set in `SamplingBatchInfo.__init__`.
4. Add unit tests to verify logit modification matches HF behavior.

## Environment
- **Model:** Qwen3.6-27B (FP8)
- **Speculative Decoding:** DFlash (with Mamba hybrid layers)
- **GPU:** RTX PRO 6000 Blackwell (96GB)
- **Python Version:** 3.11
- **SGLang Version:** Both main branch and release versions tested
- **Relevant Files:**
  - `sglang/srt/sampling/sampling_batch_info.py`
  - `sglang/srt/sampling/penaltylib/__init__.py`
  - `sglang/srt/sampling/sampling_params.py`

