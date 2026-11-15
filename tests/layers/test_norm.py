import pytest
import torch
from minisgl.layers.norm import RMSNormFused


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    return torch.device(request.param)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_fused_rmsnorm_values_and_mutation(device, dtype, has_residual, non_contiguous):
    generator = torch.Generator().manual_seed(42)
    x_base = torch.randn(3, 16, generator=generator).to(device=device, dtype=dtype)
    residual_base = torch.randn(3, 16, generator=generator).to(device=device, dtype=dtype)
    x = x_base[:, ::2] if non_contiguous else x_base[:, :8].contiguous()
    residual = residual_base[:, ::2] if non_contiguous else residual_base[:, :8].contiguous()
    original_x = x.clone()
    original_residual = residual.clone()
    untouched_x = x_base[:, 1::2].clone()
    untouched_residual = residual_base[:, 1::2].clone()
    layer = RMSNormFused(size=8, eps=1e-6)
    layer.weight = torch.randn(8, generator=generator).to(device=device, dtype=dtype)

    # Independent reference in double precision, before either buffer is modified.
    summed = original_x.cpu().double()
    if has_residual:
        summed = summed + original_residual.cpu().double()
    expected = summed / (summed.square().mean(dim=-1, keepdim=True) + layer.eps).sqrt()
    expected = (expected * layer.weight.cpu().double()).to(dtype)

    output, updated_residual = layer.forward(x, residual if has_residual else None)

    tolerance = {torch.float32: 1e-5, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(output.cpu(), expected, rtol=tolerance, atol=tolerance)
    if has_residual:
        assert output is x
        assert updated_residual is residual
        torch.testing.assert_close(residual.cpu(), summed.to(dtype), rtol=1e-5, atol=1e-6)
    else:
        assert output is not x
        assert updated_residual is x
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
        torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)
    if non_contiguous:
        torch.testing.assert_close(x_base[:, 1::2], untouched_x, rtol=0, atol=0)
        torch.testing.assert_close(residual_base[:, 1::2], untouched_residual, rtol=0, atol=0)


def test_fused_rmsnorm_normalizes_before_casting_residual(device):
    x = torch.full((2, 8), 60000.0, dtype=torch.float16, device=device)
    residual = x.clone()
    layer = RMSNormFused(size=8, eps=1e-6)
    layer.weight = torch.ones(8, dtype=x.dtype, device=device)

    output, updated_residual = layer.forward(x, residual)

    # The FP16 residual overflows, but normalization uses the finite FP32 sum.
    torch.testing.assert_close(output, torch.ones_like(x))
    assert updated_residual.isinf().all()


def test_fused_rmsnorm_initializes_on_meta():
    with torch.device("meta"):
        layer = RMSNormFused(size=8, eps=1e-6)
    assert layer.weight.is_meta
    assert set(layer.state_dict()) == {"weight"}
