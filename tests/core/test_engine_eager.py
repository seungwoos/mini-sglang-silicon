from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as tp_info
import minisgl.layers.rotary as rotary
import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.llm import LLM
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture(params=["cpu", "mps"])
def eager_device(request):
    device = request.param
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return device


@pytest.fixture
def tiny_qwen3(eager_device, tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    monkeypatch.setattr(tp_info, "_TP_INFO", None)
    monkeypatch.setattr(rotary, "_ROPE_DEVICE", None)
    rotary.get_rope.cache_clear()

    torch.manual_seed(7)
    reference = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=16,
            tie_word_embeddings=True,
        )
    ).eval()
    reference.save_pretrained(tmp_path)
    yield reference
    rotary.get_rope.cache_clear()


@torch.inference_mode()
def test_eager_engine_prefill_and_decode(eager_device, tiny_qwen3, tmp_path, monkeypatch):
    device, reference = eager_device, tiny_qwen3
    engine = Engine(
        EngineConfig(
            model_path=str(tmp_path),
            tp_info=DistributedInfo(0, 1),
            dtype=torch.float32,
            device=device,
            num_page_override=8,
        )
    )
    try:
        actual_logits = []
        sample = engine.sampler.sample

        def capture_logits(logits, args):
            actual_logits.append(logits.cpu())
            return sample(logits, args)

        monkeypatch.setattr(engine.sampler, "sample", capture_logits)
        req = core.Req(
            input_ids=torch.tensor([1, 4], dtype=torch.int32),
            table_idx=0,
            cached_len=0,
            output_len=2,
            uid=0,
            sampling_params=core.SamplingParams(),
            cache_handle=NaiveCacheHandle(),
        )
        engine.page_table[0, :3] = torch.tensor([4, 7, 2], dtype=torch.int32, device=device)

        for phase in ("prefill", "decode"):
            expected = reference(req.input_ids.unsqueeze(0), use_cache=False).logits[:, -1]
            batch = core.Batch(reqs=[req], phase=phase)
            batch.padded_reqs = batch.reqs
            batch.input_ids = req.input_ids[req.cached_len :].to(device)
            batch.positions = torch.arange(req.cached_len, req.device_len, device=device)
            batch.out_loc = engine.page_table[0, req.cached_len : req.device_len]
            engine.attn_backend.prepare_metadata(batch)
            output = engine.forward_batch(batch, engine.sampler.prepare(batch))

            torch.testing.assert_close(actual_logits.pop(), expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(output.next_tokens_cpu, expected.argmax(-1).to(torch.int32))
            assert output.copy_done_event is None
            req.append_host(output.next_tokens_cpu)
    finally:
        engine.shutdown()


@torch.inference_mode()
def test_eager_scheduler_prefill_and_decode(eager_device, tiny_qwen3, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minisgl.scheduler.scheduler.load_tokenizer",
        lambda _: SimpleNamespace(eos_token_id=-1, decode=str),
    )
    llm = LLM(
        str(tmp_path),
        dtype=torch.float32,
        device=eager_device,
        num_page_override=16,
        cache_type="naive",
    )
    try:
        batches = []
        forward_batch = llm.engine.forward_batch

        def capture_batch(batch, args):
            output = forward_batch(batch, args)
            batches.append((batch.input_ids.cpu(), batch.positions.cpu(), batch.out_loc.cpu()))
            return output

        monkeypatch.setattr(llm.engine, "forward_batch", capture_batch)
        prompt = [1, 4, 7, 10, 12]
        tokens = []
        for _ in range(2):
            ids = torch.tensor([prompt + tokens])
            tokens.append(tiny_qwen3(ids, use_cache=False).logits[0, -1].argmax().item())

        [result] = llm.generate([prompt], core.SamplingParams(max_tokens=2))

        assert result["token_ids"] == tokens
        assert [(ids.tolist(), pos.tolist(), loc.tolist()) for ids, pos, loc in batches] == [
            (prompt, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
            ([tokens[0]], [5], [5]),
        ]
        llm.cache_manager.check_integrity()
    finally:
        llm.shutdown()
