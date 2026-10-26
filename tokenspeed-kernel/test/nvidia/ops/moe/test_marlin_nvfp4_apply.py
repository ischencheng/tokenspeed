# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Numerical check: Marlin W4A16 NVFP4 MoE apply vs a dequantized reference.

Marlin dequantizes the packed E2M1 weights inside the GEMM and applies each
expert's global scale in its bf16 epilogue. The reference dequantizes the same
weights in FP32 with the bf16-rounded global scales, rounds where the kernel
writes bf16 (GEMM1 output, activation, each expert's GEMM2 output) and sums
the routes in FP32. Runs on SM90+.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.numerics.torch.activation import (
    torch_silu_and_mul,
    torch_swiglu_oai,
)
from tokenspeed_kernel.numerics.torch.quantization import dequantize_nvfp4_blocks
from tokenspeed_kernel.ops.activation.triton import _swiglu_oai_kernel
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.moe.marlin import marlin_nvfp4_moe_weights
from tokenspeed_kernel.platform import ArchVersion, Platform, current_platform
from tokenspeed_kernel.selection import NoKernelFoundError
from utils import assert_no_triton_compile


def _requires_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA required")
    if current_platform().arch_version < type(current_platform().arch_version)(9, 0):
        pytest.skip("SM90+ required")


@dataclass
class _SwigluArg:
    alpha: float | None
    limit: float | None


class _Weights(torch.nn.Module):
    """Expert params as the NVFP4 weight loader leaves them, plus EP geometry."""

    def __init__(
        self,
        raw: dict[str, torch.Tensor],
        *,
        num_local_experts: int,
        ep_size: int,
        ep_rank: int,
        swiglu_arg: _SwigluArg | None,
        swiglu_beta: float | None,
    ) -> None:
        super().__init__()
        for name, value in raw.items():
            self.register_parameter(
                name, torch.nn.Parameter(value.clone(), requires_grad=False)
            )
        self.num_local_experts = num_local_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.activation = "swiglu"
        self.swiglu_arg = swiglu_arg
        self.swiglu_beta = swiglu_beta


def _make_nvfp4_experts(
    num_experts: int, hidden: int, ispp: int, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Random NVFP4 experts: every E2M1 code, E4M3 scales over five binades."""

    def packed(n: int, k: int) -> torch.Tensor:
        return torch.randint(
            0,
            256,
            (num_experts, n, k // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )

    def scales(n: int, k: int) -> torch.Tensor:
        exponent = torch.randint(
            -3, 2, (num_experts, n, k // 16), device="cuda", generator=generator
        )
        mantissa = (
            1
            + torch.randint(0, 8, exponent.shape, device="cuda", generator=generator)
            / 8
        )
        return (mantissa * torch.exp2(exponent.float())).to(torch.float8_e4m3fn)

    def global_scales(*shape: int) -> torch.Tensor:
        return torch.rand(shape, device="cuda", generator=generator) * 0.04 + 0.01

    w13_global = global_scales(num_experts, 1).expand(num_experts, 2).contiguous()
    return {
        "w13_weight": packed(2 * ispp, hidden),
        "w13_weight_scale": scales(2 * ispp, hidden),
        "w13_weight_scale_2": w13_global,
        "w2_weight": packed(hidden, ispp),
        "w2_weight_scale": scales(hidden, ispp),
        "w2_weight_scale_2": global_scales(num_experts),
    }


def _reference(
    x: torch.Tensor,
    raw: dict[str, torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation,
) -> torch.Tensor:
    """The kernel's arithmetic with FP32 accumulation in a fixed order.

    Marlin multiplies FP32 accumulators of ``x @ (e2m1 * block_scale)`` (exact
    in bf16) by the expert's bf16 global scale after rounding them to bf16,
    and folds GEMM2's global scale into the bf16 route weight.
    """
    w13_global = raw["w13_weight_scale_2"][:, 0].to(torch.bfloat16)
    w2_global = raw["w2_weight_scale_2"].to(torch.bfloat16)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for expert in range(raw["w13_weight"].shape[0]):
        tokens, slots = torch.where(topk_ids == expert)
        if tokens.numel() == 0:
            continue
        w13 = dequantize_nvfp4_blocks(
            raw["w13_weight"][expert], raw["w13_weight_scale"][expert]
        )
        w2 = dequantize_nvfp4_blocks(
            raw["w2_weight"][expert], raw["w2_weight_scale"][expert]
        )
        fc1 = (x[tokens].float() @ w13.T).bfloat16() * w13_global[expert]
        fc2 = (activation(fc1).float() @ w2.T).bfloat16()
        route_weight = w2_global[expert] * topk_weights[tokens, slots].bfloat16()
        out.index_add_(0, tokens, (fc2 * route_weight[:, None]).float())
    # The apply sums the routes of a token in FP32 and rounds once to bf16.
    return out.bfloat16()


def _routes(num_tokens: int, top_k: int, num_experts: int, generator: torch.Generator):
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, device="cuda", generator=generator)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", generator=generator)
    return topk_ids, topk_weights / topk_weights.sum(-1, keepdim=True)


_ACTIVATIONS = {
    # MiniMax-M3.1: SwiGLU-OAI with alpha 1.702, up + 1, clamp 7.
    "swiglu_oai": (
        _SwigluArg(alpha=1.702, limit=7.0),
        1.0,
        lambda g: torch_swiglu_oai(g, alpha=1.702, limit=7.0),
    ),
    "swiglu_limit": (
        _SwigluArg(alpha=None, limit=7.0),
        None,
        lambda g: torch_silu_and_mul(g, 7.0),
    ),
    "swiglu": (None, None, lambda g: torch_silu_and_mul(g, None)),
}


def _plan(
    num_experts: int,
    top_k: int,
    hidden: int,
    ispp: int,
    activation: str,
    *,
    ep_size: int,
    solution: str | None,
) -> dict:
    swiglu_arg, _, _ = _ACTIVATIONS[activation]
    return moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=ispp,
        hidden=hidden,
        swiglu_form="generalized" if activation == "swiglu_oai" else "standard",
        activation_clamped=swiglu_arg is not None and swiglu_arg.limit is not None,
        expert_id_repeats=False,
        num_experts=num_experts,
        top_k=top_k,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution=solution,
    )


def _apply(raw, x, topk_ids, topk_weights, activation, *, ep_size=1):
    """Plan, preprocess and apply through the public MoE ops, one rank at a time."""
    num_experts = raw["w13_weight"].shape[0]
    num_local = num_experts // ep_size
    hidden = raw["w13_weight"].shape[2] * 2
    ispp = raw["w2_weight"].shape[2] * 2
    swiglu_arg, swiglu_beta, _ = _ACTIVATIONS[activation]
    plan = _plan(
        num_experts,
        topk_ids.shape[1],
        hidden,
        ispp,
        activation,
        ep_size=ep_size,
        solution="marlin",
    )
    out = torch.zeros_like(x, dtype=torch.float32)
    for ep_rank in range(ep_size):
        shard = {
            k: v[ep_rank * num_local : (ep_rank + 1) * num_local]
            for k, v in raw.items()
        }
        w = _Weights(
            shard,
            num_local_experts=num_local,
            ep_size=ep_size,
            ep_rank=ep_rank,
            swiglu_arg=swiglu_arg,
            swiglu_beta=swiglu_beta,
        )
        moe_process_weights(plan, w)
        out += moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        ).float()
    return out


@pytest.mark.parametrize("activation", sorted(_ACTIVATIONS))
@pytest.mark.parametrize("num_tokens", [1, 8, 64, 300])
def test_marlin_nvfp4_matches_reference(activation: str, num_tokens: int) -> None:
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(7)
    num_experts, top_k, hidden, ispp = 16, 4, 512, 192
    _, _, torch_activation = _ACTIVATIONS[activation]
    raw = _make_nvfp4_experts(num_experts, hidden, ispp, generator)
    x = (torch.randn(num_tokens, hidden, device="cuda", generator=generator) * 0.25).to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routes(num_tokens, top_k, num_experts, generator)

    expected = _reference(x, raw, topk_ids, topk_weights, torch_activation)
    actual = _apply(raw, x, topk_ids, topk_weights, activation)

    _check(actual, expected)


@pytest.mark.parametrize("ep_size", [2, 4])
def test_marlin_nvfp4_ep_ranks_sum_to_reference(ep_size: int) -> None:
    """Each rank computes only its experts; the rank sum is the full MoE."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(13)
    num_experts, top_k, hidden, ispp, num_tokens = 16, 4, 512, 192, 48
    _, _, torch_activation = _ACTIVATIONS["swiglu_oai"]
    raw = _make_nvfp4_experts(num_experts, hidden, ispp, generator)
    x = (torch.randn(num_tokens, hidden, device="cuda", generator=generator) * 0.25).to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routes(num_tokens, top_k, num_experts, generator)

    expected = _reference(x, raw, topk_ids, topk_weights, torch_activation)
    actual = _apply(raw, x, topk_ids, topk_weights, "swiglu_oai", ep_size=ep_size)

    _check(actual, expected)


def test_marlin_nvfp4_rejects_separate_gate_and_up_global_scales() -> None:
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(17)
    raw = _make_nvfp4_experts(4, 256, 128, generator)
    raw["w13_weight_scale_2"][1, 1] *= 2
    w = _Weights(
        raw,
        num_local_experts=4,
        ep_size=1,
        ep_rank=0,
        swiglu_arg=None,
        swiglu_beta=None,
    )
    with pytest.raises(ValueError, match="one global scale per expert"):
        marlin_nvfp4_moe_weights({"activation": "swiglu"}, w)


@pytest.mark.parametrize("swiglu_beta,limit", [(0.5, 7.0), (1.0, None)])
def test_marlin_nvfp4_rejects_swiglu_without_an_epilogue(swiglu_beta, limit) -> None:
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(19)
    raw = _make_nvfp4_experts(4, 256, 128, generator)
    w = _Weights(
        raw,
        num_local_experts=4,
        ep_size=1,
        ep_rank=0,
        swiglu_arg=_SwigluArg(alpha=1.702, limit=limit),
        swiglu_beta=swiglu_beta,
    )
    with pytest.raises(ValueError, match="SwiGLU-OAI"):
        marlin_nvfp4_moe_weights({"activation": "swiglu"}, w)


def test_marlin_nvfp4_token_counts_reuse_the_epilogue_binary() -> None:
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(23)
    num_experts, top_k, hidden, ispp = 16, 4, 512, 192
    swiglu_arg, swiglu_beta, _ = _ACTIVATIONS["swiglu_oai"]
    raw = _make_nvfp4_experts(num_experts, hidden, ispp, generator)
    w = _Weights(
        raw,
        num_local_experts=num_experts,
        ep_size=1,
        ep_rank=0,
        swiglu_arg=swiglu_arg,
        swiglu_beta=swiglu_beta,
    )
    plan = _plan(
        num_experts, top_k, hidden, ispp, "swiglu_oai", ep_size=1, solution="marlin"
    )
    moe_process_weights(plan, w)

    def run(num_tokens: int) -> None:
        x = torch.randn(num_tokens, hidden, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        topk_ids, topk_weights = _routes(num_tokens, top_k, num_experts, generator)
        moe_apply(plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids)

    run(3)
    run(16)
    with assert_no_triton_compile(_swiglu_oai_kernel):
        for num_tokens in (1, 5, 32, 77, 128, 513):
            run(num_tokens)
    torch.cuda.synchronize()


@pytest.mark.parametrize("activation", ["swiglu_oai", "swiglu_limit"])
@pytest.mark.parametrize("ispp,selected", [(64, True), (384, True), (32, False)])
def test_sm90_planner_selects_marlin_for_nvfp4_experts(activation, ispp, selected):
    """On SM90 Marlin is the NVFP4 expert kernel, for intermediate sizes per
    partition that are multiples of 64 (MiniMax-M3.1 TP8 is 384)."""
    original = current_platform()
    Platform.override(
        dataclasses.replace(original, vendor="nvidia", arch_version=ArchVersion(9, 0))
    )
    try:
        if selected:
            plan = _plan(128, 4, 6144, ispp, activation, ep_size=1, solution=None)
            assert plan["apply_kernel_name"] == "marlin_nvfp4_precomputed_moe_apply"
        else:
            with pytest.raises(NoKernelFoundError):
                _plan(128, 4, 6144, ispp, activation, ep_size=1, solution=None)
    finally:
        Platform.override(original)


def _check(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Within two bf16 ulps, relative to the element and to the largest output.

    The kernel and the reference accumulate in FP32 in different orders, so a
    bf16 rounding of GEMM1 can flip; GEMM2 carries that into outputs near zero
    as an error of the order of the largest output's ulp.
    """
    expected = expected.float()
    two_ulps = 2.0**-7
    torch.testing.assert_close(
        actual,
        expected,
        atol=two_ulps * expected.abs().max().item(),
        rtol=two_ulps,
    )
