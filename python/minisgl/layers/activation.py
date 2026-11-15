from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if x.is_cuda:
        from flashinfer import silu_and_mul

        return silu_and_mul(x, out=out)

    gate, up = x.chunk(2, dim=-1)
    result = F.silu(gate) * up

    return result if out is None else out.copy_(result)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    if x.is_cuda:
        from flashinfer import gelu_and_mul

        return gelu_and_mul(x, out=out)

    gate, up = x.chunk(2, dim=-1)
    result = F.gelu(gate) * up

    return result if out is None else out.copy_(result)


__all__ = ["silu_and_mul", "gelu_and_mul"]
