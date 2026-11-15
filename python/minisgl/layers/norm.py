from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from flashinfer import rmsnorm

            return rmsnorm(x, self.weight, self.eps)

        return F.rms_norm(x, normalized_shape=self.weight.shape, weight=self.weight, eps=self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        if x.is_cuda:
            from flashinfer import rmsnorm

            return rmsnorm(x, self.weight, self.eps, out=x)

        return x.copy_(self.forward(x))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:
            from flashinfer import fused_add_rmsnorm, rmsnorm

            if residual is None:
                return rmsnorm(x, self.weight, self.eps), x
            fused_add_rmsnorm(x, residual, self.weight, self.eps)
            return x, residual

        if residual is None:
            return F.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.eps), x

        # Match the fused kernel: normalize the FP32 sum before casting either output.
        summed = x.float() + residual.float()
        residual.copy_(summed)
        x.copy_(F.rms_norm(summed, self.weight.shape, weight=self.weight.float(), eps=self.eps))
        return x, residual
