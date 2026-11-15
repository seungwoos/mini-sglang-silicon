from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    is_cuda = device.type == "cuda"
    return torch.tensor(data, dtype=dtype, pin_memory=is_cuda).to(device, non_blocking=is_cuda)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


def _sample_with_torch(logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    # Use PyTorch's CPU sampler for eager CPU/MPS inference.
    rows = logits.to(device="cpu", dtype=torch.float32)
    assert args.temperatures is not None
    temperatures = args.temperatures.tolist()
    top_ks = args.top_k.tolist() if args.top_k is not None else [rows.size(-1)] * len(rows)
    top_ps = args.top_p.tolist() if args.top_p is not None else [1.0] * len(rows)
    tokens = []
    for row, temperature, k, p in zip(rows, temperatures, top_ks, top_ps, strict=True):
        if temperature == 0.0:
            tokens.append(row.argmax())
            continue

        indices = None
        if k < row.numel():
            row, indices = row.topk(k)
        elif p < 1.0:
            row, indices = row.sort(descending=True)
        probs = ((row - row.max()) / temperature).softmax(dim=0)
        if p < 1.0:
            # Keep the first token that reaches the cutoff as well.
            probs[1:].masked_fill_(probs.cumsum(dim=0)[:-1] >= p, 0)
        token = torch.multinomial(probs, num_samples=1).squeeze(0)
        tokens.append(token if indices is None else indices[token])

    return torch.stack(tokens).to(logits.device)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        is_cuda = self.device.type == "cuda"
        args_device = self.device if is_cuda else torch.device("cpu")
        MIN_P = MIN_T = 1e-6
        greedy_temperature = MIN_T if is_cuda else 0.0
        ts = [greedy_temperature if p.is_greedy else max(p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, args_device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, args_device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, args_device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.temperatures is None:  # greedy sampling
            return torch.argmax(logits, dim=-1)

        if self.device.type != "cuda":
            return _sample_with_torch(logits, args)

        return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
