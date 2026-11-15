from __future__ import annotations

import minisgl.core as core
import minisgl.distributed.info as tp_info
import pytest
import torch
from minisgl.attention import create_attention_backend
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.models.config import ModelConfig
from transformers import LlamaConfig


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    monkeypatch.setattr(tp_info, "_TP_INFO", None)
    tp_info.set_tp_info(0, 1)


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return torch.device(request.param)


def make_backend(device, num_q_heads, num_kv_heads, page_size, dtype):
    config = ModelConfig.from_hf(
        LlamaConfig(
            hidden_size=num_q_heads * 8,
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            num_hidden_layers=2,
        )
    )
    ctx = core.Context(page_size=page_size)
    ctx.kv_cache = MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_layers=2,
        head_dim=8,
        num_pages=16,
        page_size=page_size,
        dtype=dtype,
        device=device,
    )
    for layer in range(2):
        ctx.kv_cache.k_cache(layer).fill_(float("nan"))
        ctx.kv_cache.v_cache(layer).fill_(float("nan"))
    core.set_global_ctx(ctx)
    return ctx, create_attention_backend("sdpa", config)


def make_req(table_idx, cached_len, seq_len):
    return core.Req(
        input_ids=torch.arange(seq_len, dtype=torch.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=8,
        uid=table_idx,
        sampling_params=core.SamplingParams(),
        cache_handle=None,
    )


def reference_attention(q, k, v):
    # Full causal attention, computed independently for each query head on CPU.
    q, k, v = q.float(), k.float(), v.float()
    num_q_heads, head_dim = q.shape[1:]
    heads_per_kv = num_q_heads // k.shape[1]
    future = torch.ones(len(q), len(k), dtype=torch.bool).triu(1)
    output = torch.empty_like(q)
    for head in range(num_q_heads):
        kv_head = head // heads_per_kv
        scores = q[:, head] @ k[:, kv_head].T / head_dim**0.5
        probs = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        output[:, head] = probs @ v[:, kv_head]
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(2, 2), (4, 2), (4, 1)])
@pytest.mark.parametrize(
    "phase,cached_lens,seq_lens,shared_prefix",
    [
        ("prefill", (0, 0), (5, 3), False),
        ("prefill", (2, 1), (5, 3), False),
        ("decode", (4, 2), (5, 3), False),
        ("prefill", (4, 4), (6, 5), True),
    ],
    ids=["prefill", "cached-prefill", "decode", "shared-prefix"],
)
def test_sdpa_matches_full_attention(
    device,
    dtype,
    page_size,
    num_q_heads,
    num_kv_heads,
    phase,
    cached_lens,
    seq_lens,
    shared_prefix,
):
    ctx, backend = make_backend(device, num_q_heads, num_kv_heads, page_size, dtype)
    generator = torch.Generator().manual_seed(42)
    pages = torch.randperm(16, generator=generator)
    ctx.page_table = torch.full((2, max(seq_lens)), -1, dtype=torch.int32, device=device)
    queries, keys, values, expected = [], [], [], []
    prefix_k, prefix_v = None, None
    page_offset = 0
    for i, (cached_len, seq_len) in enumerate(zip(cached_lens, seq_lens)):
        num_pages = (seq_len + page_size - 1) // page_size
        slots = (
            pages[page_offset : page_offset + num_pages, None] * page_size + torch.arange(page_size)
        ).flatten()[:seq_len]
        page_offset += num_pages
        ctx.page_table[i, :seq_len] = slots.to(device=device, dtype=torch.int32)
        q = torch.randn(seq_len, num_q_heads, 8, generator=generator).to(dtype)
        k = torch.randn(seq_len, num_kv_heads, 8, generator=generator).to(dtype)
        v = torch.randn(seq_len, num_kv_heads, 8, generator=generator).to(dtype)
        if shared_prefix:
            if i == 0:
                prefix_k, prefix_v = k[:cached_len].clone(), v[:cached_len].clone()
            else:
                k[:cached_len], v[:cached_len] = prefix_k, prefix_v
                ctx.page_table[i, :cached_len] = ctx.page_table[0, :cached_len]
        indices = ctx.page_table[i, :cached_len].long()
        ctx.kv_cache.k_cache(1).flatten(0, 1)[indices] = k[:cached_len].to(device)
        ctx.kv_cache.v_cache(1).flatten(0, 1)[indices] = v[:cached_len].to(device)
        queries.append(q[cached_len:].flatten(1))
        keys.append(k[cached_len:].flatten(1))
        values.append(v[cached_len:].flatten(1))
        expected.append(reference_attention(q, k, v)[cached_len:])

    batch = core.Batch(
        reqs=[
            make_req(i, cached, length)
            for i, (cached, length) in enumerate(zip(cached_lens, seq_lens))
        ],
        phase=phase,
    )
    batch.padded_reqs = batch.reqs
    batch.out_loc = torch.cat(
        [
            ctx.page_table[i, cached:length]
            for i, (cached, length) in enumerate(zip(cached_lens, seq_lens))
        ]
    )
    backend.prepare_metadata(batch)
    # Match AttentionLayer's non-contiguous Q/K/V views into a packed projection.
    packed = torch.cat([torch.cat(queries), torch.cat(keys), torch.cat(values)], dim=-1)
    q, k, v = packed.to(device).split([num_q_heads * 8, num_kv_heads * 8, num_kv_heads * 8], dim=-1)
    actual = backend.forward(q.view(-1, num_q_heads, 8), k, v, 1, batch)

    torch.testing.assert_close(actual.cpu().float(), torch.cat(expected), rtol=3e-3, atol=3e-3)
    last_indices = torch.tensor([length - cached for cached, length in zip(cached_lens, seq_lens)])
    torch.testing.assert_close(
        batch.attn_metadata.get_last_indices(batch.size).cpu(), last_indices.cumsum(0) - 1
    )
    assert ctx.kv_cache.k_cache(0).isnan().all()
    assert ctx.kv_cache.v_cache(0).isnan().all()


def test_chunked_prefill_then_decode_reuses_cache(device):
    ctx, backend = make_backend(device, 4, 2, page_size=4, dtype=torch.float32)
    ctx.page_table = torch.tensor([[8, 9, 10, 11, 4, 5, 6]], dtype=torch.int32, device=device)
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(7, 4, 8, generator=generator)
    k = torch.randn(7, 2, 8, generator=generator)
    v = torch.randn(7, 2, 8, generator=generator)
    expected = reference_attention(q, k, v)
    cached_len = 0
    for seq_len in (3, 5, 6, 7):
        batch = core.Batch(
            reqs=[make_req(0, cached_len, seq_len)],
            phase="prefill" if seq_len <= 5 else "decode",
        )
        batch.padded_reqs = batch.reqs
        batch.out_loc = ctx.page_table[0, cached_len:seq_len]
        backend.prepare_metadata(batch)
        actual = backend.forward(
            q[cached_len:seq_len].to(device),
            k[cached_len:seq_len].to(device),
            v[cached_len:seq_len].to(device),
            1,
            batch,
        )
        torch.testing.assert_close(actual.cpu(), expected[cached_len:seq_len], rtol=1e-4, atol=1e-5)
        cached_len = seq_len
