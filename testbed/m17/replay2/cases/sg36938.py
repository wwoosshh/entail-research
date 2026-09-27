"""M17.6 case, sgl-project/sglang#36938 (testbed/M16_PROTOCOL.md 7): in process_batch_result_prefill the branch
that skips a retracted (or finished-in-a-mixed-batch) request does not advance the input-logprob cursor, so the
next request is served the retracted request's prompt logprobs. The regression test from the fix PR (#36939),
driven here as a script on CPU against the installed SGLang: two prefill requests, the first retracted, a flat
logprob array [-1, -2, -30, -31, -32]. Expected for the active request [None, -30.0, -31.0]; reproduced when it
gets the retracted request's values instead.
Run in ~/venvs/sglang: python testbed/m17/replay2/cases/sg36938.py <out.json>
"""
import json
import os
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import Mock, patch


class _PrefillReq:
    def __init__(self, *, rid, origin_input_ids, is_retracted=False):
        from sglang.srt.managers.schedule_batch import ReqLogprob

        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.logprob_start_len = 0
        self.is_retracted = is_retracted
        self.inflight_middle_chunks = 0
        self.return_logprob = True
        self.logprob = ReqLogprob(top_logprobs_num=0, token_ids_logprob=None, output_token_logprobs_val=[],
                                  output_token_logprobs_idx=[])
        self.multi_item_delimiter_indices = None
        self.return_flat_raw_top_logprobs = False
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.temp_input_token_ids_logprobs_val = None
        self.temp_input_token_ids_logprobs_idx = None
        self.hidden_states = []
        self.output_ids = []
        self.time_stats = Mock()
        self.return_hidden_states = False
        self.return_sampling_mask = False
        self.grammar = None
        self.require_reasoning = False
        self.customized_info = None
        self.beam_group = None

    def finished(self):
        return False

    def update_finish_state(self):
        return None


def make_processor():
    from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
    from sglang.srt.managers.scheduler_components.logprob_result_processor import SchedulerLogprobResultProcessor

    try:
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args()
        override.install()
    except Exception:  # noqa: BLE001 - older layouts have no context to override
        pass
    metrics_reporter = Mock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    model_config = SimpleNamespace(think_end_ids=None, vocab_size=1000)
    return SchedulerBatchResultProcessor(
        is_generation=True, disaggregation_mode=None, enable_overlap=False, enable_overlap_mlx=False,
        model_config=model_config, token_to_kv_pool_allocator=Mock(), tree_cache=None, hisparse_coordinator=None,
        req_to_token_pool=None, decode_offload_manager=None, metrics_collector=None,
        metrics_reporter=metrics_reporter, draft_worker=None, model_worker=Mock(),
        logprob_result_processor=SchedulerLogprobResultProcessor(model_config=model_config),
        output_streamer=Mock(), beam_coordinator=Mock(), abort_request=lambda *a, **k: None)


def main():
    import sglang

    row = {"entail": os.environ.get("ENTAIL", "off"), "sglang": sglang.__version__}
    try:
        import torch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        retracted = _PrefillReq(rid="retracted", origin_input_ids=[101, 102], is_retracted=True)
        active = _PrefillReq(rid="active", origin_input_ids=[201, 202, 203])
        input_token_logprobs = torch.tensor([-1.0, -2.0, -30.0, -31.0, -32.0])
        batch = SimpleNamespace(reqs=[retracted, active], decoding_reqs=[], return_logprob=True,
                                return_hidden_states=False, return_hidden_states_mode=CaptureHiddenMode.NULL,
                                spec_info=None, prefill_stats=None, dp_cooperation_info=None)
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None, input_token_logprobs=input_token_logprobs,
                                                next_token_logprobs=torch.tensor([-0.5, -0.6])),
            next_token_ids=torch.tensor([7, 8]), extend_input_len_per_req=[2, 3],
            extend_logprob_start_len_per_req=[0, 0])
        processor = make_processor()
        with patch("sglang.srt.managers.scheduler_components.batch_result_processor.maybe_cache_unfinished_req"), \
             patch("sglang.srt.managers.scheduler_components.batch_result_processor.get_memory",
                   return_value=SimpleNamespace(enable_hisparse=False)):
            processor.process_batch_result_prefill(batch, result)
        got = list(active.logprob.input_token_logprobs_val)
        row.update({"expected": [None, -30.0, -31.0], "actual": got,
                    "retracted_block": [None, -1.0, -2.0]})
        row["reproduced"] = got != [None, -30.0, -31.0]
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        row["trace"] = traceback.format_exc().strip().splitlines()[-3:]
        row["reproduced"] = None
    json.dump(row, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
