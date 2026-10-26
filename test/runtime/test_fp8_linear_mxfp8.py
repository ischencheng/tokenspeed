"""Fp8LinearMethod MXFP8 (1,32) with canonical scales."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.dense.fp8 import (
    Fp8LinearMethod,
    mxfp8_linears_load_as_bf16,
)
from tokenspeed.runtime.layers.quantization.fp8 import Mxfp8Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _make_layer(n: int, k: int, device: str = "cuda") -> torch.nn.Module:
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = (torch.randn(n, k, device=device) * 0.02).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 130, (n, k // 32), device=device, dtype=torch.uint8)
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(scales, requires_grad=False)
    return layer


def _method() -> Fp8LinearMethod:
    return Fp8LinearMethod(
        Mxfp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            scale_fmt="ue8m0",
        )
    )


_KEEPS_FP8 = pytest.mark.skipif(
    torch.cuda.is_available() and mxfp8_linears_load_as_bf16(),
    reason="without a vendor MXFP8 GEMM the linear is dequantized to BF16",
)


@_KEEPS_FP8
def test_process_weights_preserves_canonical_scales() -> None:
    n, k = 256, 512
    layer = _make_layer(n, k)
    ref_weight = layer.weight.data.clone()
    ref_scales = layer.weight_scale_inv.data.clone()

    method = _method()
    method.process_weights_after_loading(layer)

    assert torch.equal(layer.weight_scale_inv, ref_scales)
    assert torch.equal(layer.weight.data, ref_weight)

    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x)

    dequant = ref_weight.float() * torch.exp2(
        ref_scales.float() - 127.0
    ).repeat_interleave(32, dim=1)
    ref = x.float() @ dequant.t()
    rel = (torch.norm(out.float() - ref) / torch.norm(ref)).item()
    assert rel < 5e-2, f"rel_l2={rel}"


@_KEEPS_FP8
def test_small_layers_keep_triton_fallback() -> None:
    # N < 128 is below the flashinfer problem-size floor; the layer must
    # keep row-major scales so the Triton kernel stays selectable.
    layer = _make_layer(64, 512)
    _method().process_weights_after_loading(layer)

    assert layer.weight_scale_inv.dim() == 2

    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    out = _method().apply(layer, x)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("vendor,arch", [("nvidia", (9, 0)), ("amd", (12, 1))])
def test_only_nvidia_without_a_vendor_gemm_loads_mxfp8_linears_as_bf16(vendor, arch):
    """gfx1201 has only portable kernels too, but DeepSeek-V4.1's projection
    method prepares MXFP8 scales there after the base processing; only NVIDIA
    switches to BF16."""
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor=vendor, arch_version=ArchVersion(*arch))
    )
    try:
        assert mxfp8_linears_load_as_bf16() is (vendor == "nvidia")
    finally:
        Platform.override(original)


def test_sm90_dequantizes_mxfp8_linears_to_exact_bf16() -> None:
    """SM90 has only the portable Triton block-FP8 GEMM for 1x32 scales, so the
    weights become exact BF16 at load and the layer runs the BF16 GEMM. The
    FP8-keeping path where a vendor GEMM exists is the tests above."""
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor="nvidia", arch_version=ArchVersion(9, 0))
    )
    try:
        assert mxfp8_linears_load_as_bf16()
        n, k = 256, 512
        layer = _make_layer(n, k)
        layer.orig_dtype = torch.bfloat16
        codes = layer.weight.data.clone()
        scales = layer.weight_scale_inv.data.clone()
        method = _method()
        method.process_weights_after_loading(layer)
    finally:
        Platform.override(original)

    assert method.dequantized
    assert layer.weight_scale_inv is None
    exact = codes.float() * torch.exp2(scales.float() - 127.0).repeat_interleave(
        32, dim=1
    )
    assert layer.weight.dtype == torch.bfloat16
    assert torch.equal(layer.weight.float(), exact)

    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x).float()
    ref = x.float() @ exact.t()
    two_ulps = 2.0**-7
    torch.testing.assert_close(
        out, ref, atol=two_ulps * ref.abs().max().item(), rtol=two_ulps
    )
    # A caller-owned FP32 destination receives the same values.
    out_fp32 = torch.empty(8, n, device="cuda", dtype=torch.float32)
    assert method.apply_into(layer, x, None, None, torch.float32, out_fp32) is out_fp32
    torch.testing.assert_close(out_fp32, out, atol=0, rtol=0)
    with pytest.raises(ValueError, match="dequantized to BF16"):
        method.apply(layer, x, block_scale=torch.ones(8, k // 32, device="cuda"))
