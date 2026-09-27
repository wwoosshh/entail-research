"""The connector of vllm-project/vllm#49250's report, as written there: it promises a block-aligned prefix
synchronously, then rejects the load (loads nothing and reports every promised block as failed)."""
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata


@dataclass
class RogueMetadata(KVConnectorMetadata):
    reqs: list = field(default_factory=list)


class RogueSyncRejectConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._need_load = {}
        self._load_errors = set()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        n = len(request.prompt_token_ids or [])
        matched = ((n - 1) // self._block_size) * self._block_size
        new = matched - num_computed_tokens
        return (new, False) if new > 0 else (0, False)   # SYNC load

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self._need_load[request.request_id] = True

    def build_connector_meta(self, scheduler_output):
        meta = RogueMetadata()
        for r in scheduler_output.scheduled_new_reqs:
            if r.req_id in self._need_load:
                meta.reqs.append((r.req_id, list(r.block_ids[0])))
        self._need_load.clear()
        return meta

    def start_load_kv(self, forward_context, **kwargs):     # REJECT: load nothing
        meta = self._get_connector_metadata()
        for _rid, block_ids in meta.reqs:
            self._load_errors.update(block_ids)

    def get_block_ids_with_load_errors(self):
        errs, self._load_errors = self._load_errors, set()
        return errs

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        return
