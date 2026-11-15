from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Batch, SamplingParams
from minisgl.engine.sample import Sampler


def make_batch(*params):
    return Batch(reqs=[SimpleNamespace(sampling_params=p) for p in params], phase="decode")


def test_torch_sampler_applies_each_requests_filters(monkeypatch):
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    batch = make_batch(
        SamplingParams(),
        SamplingParams(temperature=0.5),
        SamplingParams(temperature=1.0, top_k=2),
        SamplingParams(temperature=1.0, top_p=0.6),
        SamplingParams(temperature=1.0, top_k=2, top_p=0.5),
    )
    weights = torch.tensor([0.15, 0.35, 0.30, 0.20])
    draws = []

    def choose_last_candidate(probs, num_samples):
        assert num_samples == 1
        draws.append(probs / probs.sum())
        return probs.nonzero()[-1]

    monkeypatch.setattr(torch, "multinomial", choose_last_candidate)
    tokens = sampler.sample(weights.log().repeat(batch.size, 1), sampler.prepare(batch))

    assert tokens.tolist() == [1, 3, 2, 2, 1]
    torch.testing.assert_close(draws[0], weights.square() / weights.square().sum())
    torch.testing.assert_close(draws[1], torch.tensor([0.35, 0.30]) / 0.65)
    torch.testing.assert_close(draws[2], torch.tensor([0.35, 0.30, 0.0, 0.0]) / 0.65)
    torch.testing.assert_close(draws[3], torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_sampled_token_ids_can_be_used_on_the_logits_device(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    sampler = Sampler(torch.device(device), vocab_size=4)
    batch = make_batch(SamplingParams(temperature=0.6, top_k=2, top_p=0.9))
    logits = torch.tensor([[0.0, 2.0, 0.0, 2.0]], device=device)
    tokens = sampler.sample(logits, sampler.prepare(batch))

    assert tokens.device == logits.device
    assert tokens.shape == (1,)
    assert logits[0, tokens].item() == 2.0
