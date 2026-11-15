from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class SDPAMetadata(BaseAttnMetadata):
    qo_indptr: List[int]
    kv_indices: List[torch.Tensor]
    last_indices: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.last_indices[:bs]


class SDPABackend(BaseAttnBackend):
    """Eager PyTorch attention over the existing paged KV cache, including CPU and MPS."""

    def __init__(self, config: ModelConfig) -> None:
        self.kvcache = get_global_ctx().kv_cache
        self.scale = config.head_dim**-0.5

    def prepare_metadata(self, batch: Batch) -> None:
        page_table = get_global_ctx().page_table
        qo_indptr = [0]
        kv_indices = []
        for req in batch.padded_reqs:
            qo_indptr.append(qo_indptr[-1] + req.extend_len)
            # The global page table contains token locations, regardless of page size.
            kv_indices.append(page_table[req.table_idx, : req.device_len].long())
        batch.attn_metadata = SDPAMetadata(
            qo_indptr=qo_indptr,
            kv_indices=kv_indices,
            last_indices=torch.tensor(qo_indptr[1:], device=self.kvcache.device, dtype=torch.int64)
            - 1,
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, SDPAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        k_cache = self.kvcache.k_cache(layer_id).flatten(0, 1)
        v_cache = self.kvcache.v_cache(layer_id).flatten(0, 1)
        outputs = []
        for i, indices in enumerate(metadata.kv_indices):
            start, end = metadata.qo_indptr[i : i + 2]
            query = q[start:end].transpose(0, 1).unsqueeze(0)
            key = k_cache[indices].transpose(0, 1).unsqueeze(0)
            value = v_cache[indices].transpose(0, 1).unsqueeze(0)

            # Expand GQA/MQA heads explicitly so this also works on MPS.
            repeats = query.size(1) // key.size(1)
            if repeats != 1:
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)

            # Cached queries start after the prefix; is_causal=True aligns at the top left.
            cached_len = indices.numel() - (end - start)
            query_positions = torch.arange(cached_len, indices.numel(), device=q.device)
            key_positions = torch.arange(indices.numel(), device=q.device)
            mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0, scale=self.scale
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)
