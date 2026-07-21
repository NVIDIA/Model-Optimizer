# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent shape, conversion, fusion, and metadata tests for PDD projections."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modelopt.torch.fastgen import (
    PDDLayerSpec,
    PDDOutputProjection,
    convert_to_pdd_output_projection,
    get_module_by_path,
    replace_module_by_path,
)


class _NestedModel(nn.Module):
    def __init__(self, *, bias: bool = True):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.blocks = nn.Sequential(nn.Identity(), nn.Linear(2, 6, bias=bias))

    def forward(self, inputs):
        return self.transformer.blocks(inputs)


def _base_linear(*, bias: bool = True) -> nn.Linear:
    linear = nn.Linear(2, 6, bias=bias)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(6, 2) / 7)
        if linear.bias is not None:
            linear.bias.copy_(torch.arange(6, dtype=torch.float32) / 11)
    return linear


def _spec(layout: str) -> PDDLayerSpec:
    return PDDLayerSpec(
        projection_path="transformer.blocks.1",
        head_layout=layout,
        output_channels=2 if layout == "patch_major" else None,
    )


def _decode_heads(
    raw: torch.Tensor,
    *,
    layout: str,
    grid_size: int,
    base_out_features: int = 6,
    output_channels: int = 2,
) -> torch.Tensor:
    """Independent raw-layout decoder returning ``[batch, head, base_output]``."""
    if layout == "channel_major":
        return raw.reshape(raw.shape[0], grid_size, base_out_features)
    patch_factor = base_out_features // output_channels
    return (
        raw.reshape(raw.shape[0], patch_factor, grid_size, output_channels)
        .permute(0, 2, 1, 3)
        .reshape(raw.shape[0], grid_size, base_out_features)
    )


def _encode_head_parameters(
    head_values: torch.Tensor,
    *,
    layout: str,
    output_channels: int = 2,
) -> torch.Tensor:
    """Independent ``[head, base_output, ...]`` encoder for widened storage."""
    if layout == "channel_major":
        return head_values.reshape(-1, *head_values.shape[2:])
    grid_size, base_out_features = head_values.shape[:2]
    patch_factor = base_out_features // output_channels
    trailing_shape = head_values.shape[2:]
    return (
        head_values.reshape(grid_size, patch_factor, output_channels, *trailing_shape)
        .permute(1, 0, 2, *range(3, head_values.ndim + 1))
        .reshape(-1, *trailing_shape)
    )


@pytest.mark.parametrize("layout", ["channel_major", "patch_major"])
@pytest.mark.parametrize("bias", [False, True])
def test_from_linear_repeats_every_head_without_mutating_base(layout, bias):
    base = _base_linear(bias=bias)
    base.eval()
    base.weight.requires_grad_(False)
    if base.bias is not None:
        base.bias.requires_grad_(False)
    original = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    inputs = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    expected = base(inputs)

    projection = PDDOutputProjection.from_linear(base, 3, _spec(layout))
    actual_heads = _decode_heads(projection(inputs), layout=layout, grid_size=3)

    torch.testing.assert_close(actual_heads, expected[:, None].expand_as(actual_heads))
    assert projection.out_features == 18
    assert projection.base_out_features == 6
    assert projection.training is False
    assert projection.weight.requires_grad is False
    assert (projection.bias is None) is (base.bias is None)
    for name, tensor in base.state_dict().items():
        assert torch.equal(tensor, original[name])


def test_patch_major_requires_divisible_output_channels():
    spec = PDDLayerSpec("projection", "patch_major", output_channels=4)
    with pytest.raises(ValueError, match="must be divisible"):
        PDDOutputProjection.from_linear(_base_linear(), 3, spec)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection_path": "", "head_layout": "channel_major"},
        {"projection_path": "a..b", "head_layout": "channel_major"},
        {"projection_path": "a", "head_layout": "unknown"},
        {"projection_path": "a", "head_layout": "channel_major", "output_channels": 2},
        {"projection_path": "a", "head_layout": "patch_major"},
    ],
)
def test_layer_spec_rejects_unsupported_layout_metadata(kwargs):
    with pytest.raises(ValueError):
        PDDLayerSpec(**kwargs)


def test_nested_conversion_is_explicit_idempotent_and_conflict_safe():
    model = _NestedModel()
    spec = _spec("channel_major")
    original = get_module_by_path(model, spec.projection_path)

    projection = convert_to_pdd_output_projection(model, spec, grid_size=3)
    repeated = convert_to_pdd_output_projection(model, spec, grid_size=3)

    assert projection is repeated
    assert get_module_by_path(model, spec.projection_path) is projection
    assert original is not projection
    with pytest.raises(ValueError, match="incompatible"):
        convert_to_pdd_output_projection(model, spec, grid_size=4)
    with pytest.raises(ValueError, match="incompatible"):
        convert_to_pdd_output_projection(model, _spec("patch_major"), grid_size=3)
    assert get_module_by_path(model, spec.projection_path) is projection


def test_nested_module_helpers_require_existing_registered_modules():
    model = _NestedModel()
    replacement = nn.Linear(2, 6)
    previous = replace_module_by_path(model, "transformer.blocks.1", replacement)

    assert isinstance(previous, nn.Linear)
    assert get_module_by_path(model, "transformer.blocks.1") is replacement
    with pytest.raises(ValueError, match="does not resolve"):
        get_module_by_path(model, "transformer.missing")
    with pytest.raises(ValueError, match="non-empty dotted"):
        get_module_by_path(model, "")


@pytest.mark.parametrize("layout", ["channel_major", "patch_major"])
@pytest.mark.parametrize("bias", [False, True])
def test_fused_forward_matches_independent_weighted_head_sum(layout, bias):
    projection = PDDOutputProjection.from_linear(_base_linear(bias=bias), 3, _spec(layout))
    head_weights = torch.arange(36, dtype=torch.float32).reshape(3, 6, 2) / 13
    head_bias = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 17
    with torch.no_grad():
        projection.weight.copy_(_encode_head_parameters(head_weights, layout=layout))
        if projection.bias is not None:
            projection.bias.copy_(_encode_head_parameters(head_bias, layout=layout))

    inputs = torch.tensor([[0.25, -1.0], [2.0, 0.5]])
    original_inputs = inputs.clone()
    grid = torch.tensor([1.0, 0.8, 0.3, 0.0], dtype=torch.float64)
    original_grid = grid.clone()
    weight_id = id(projection.weight)
    weight_pointer = projection.weight.data_ptr()
    state_keys = tuple(projection.state_dict())
    explicit_heads = torch.stack(
        [
            F.linear(inputs, head_weights[index], head_bias[index] if bias else None)
            for index in range(3)
        ],
        dim=1,
    )
    coefficients = torch.tensor([0.5 / 0.8, 0.3 / 0.8])
    expected = torch.einsum("n,bno->bo", coefficients, explicit_heads[:, 1:3])

    with projection.fuse_block(1, 3, grid):
        actual = projection(inputs)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        _decode_heads(projection(inputs), layout=layout, grid_size=3), explicit_heads
    )
    assert id(projection.weight) == weight_id
    assert projection.weight.data_ptr() == weight_pointer
    assert tuple(projection.state_dict()) == state_keys
    assert torch.equal(inputs, original_inputs)
    assert torch.equal(grid, original_grid)


def test_fusion_contexts_nest_and_restore_in_order():
    projection = PDDOutputProjection.from_linear(_base_linear(), 3, _spec("channel_major"))
    inputs = torch.tensor([[1.0, -0.5]])
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])
    normal = projection(inputs)

    with projection.fuse_block(0, 2, grid):
        outer_before = projection(inputs)
        with projection.fuse_block(1, 3, grid):
            inner = projection(inputs)
        outer_after = projection(inputs)

    torch.testing.assert_close(outer_before, outer_after)
    assert outer_before.shape == inner.shape == (1, 6)
    assert projection(inputs).shape == normal.shape == (1, 18)
    torch.testing.assert_close(projection(inputs), normal)


def test_active_fusion_rejects_forward_from_another_thread():
    projection = PDDOutputProjection.from_linear(_base_linear(), 3, _spec("channel_major"))
    inputs = torch.tensor([[1.0, -0.5]])
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])

    with projection.fuse_block(0, 2, grid), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(projection, inputs)
        with pytest.raises(RuntimeError, match="non-owning thread"):
            future.result()


def test_simultaneous_fusion_entry_admits_exactly_one_thread():
    projection = PDDOutputProjection.from_linear(_base_linear(), 3, _spec("channel_major"))
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])
    start_barrier = threading.Barrier(3)
    release_owner = threading.Event()

    def _enter(start, end):
        start_barrier.wait()
        try:
            with projection.fuse_block(start, end, grid):
                release_owner.wait(timeout=5)
                return "admitted"
        except RuntimeError as error:
            return f"rejected: {error}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_enter, 0, 2), executor.submit(_enter, 1, 3)]
        start_barrier.wait()
        done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
        release_owner.set()
        results = [future.result(timeout=5) for future in futures]

    assert len(done) == 1
    assert results.count("admitted") == 1
    rejected = [result for result in results if result != "admitted"]
    assert len(rejected) == 1
    assert "another thread" in rejected[0]
    assert projection._fusion_stack == []
    assert projection._fusion_owner_thread is None


def test_fusion_context_exception_cleans_up_and_allows_reuse():
    projection = PDDOutputProjection.from_linear(_base_linear(), 3, _spec("channel_major"))
    inputs = torch.tensor([[1.0, -0.5]])
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])

    with pytest.raises(RuntimeError, match="body failed"), projection.fuse_block(0, 2, grid):
        raise RuntimeError("body failed")

    with projection.fuse_block(1, 3, grid):
        assert projection(inputs).shape == (1, 6)
    assert projection(inputs).shape == (1, 18)


def test_projection_deepcopy_recreates_inactive_fusion_lock():
    projection = PDDOutputProjection.from_linear(_base_linear(), 3, _spec("channel_major"))
    copied = copy.deepcopy(projection)
    inputs = torch.tensor([[1.0, -0.5]])
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])

    with copied.fuse_block(0, 2, grid):
        assert copied(inputs).shape == (1, 6)
    assert copied._fusion_lock is not projection._fusion_lock


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(-1, 2, "0 <= start"), (1, 1, "0 <= start"), (1, 4, "0 <= start")],
)
def test_fusion_context_rejects_invalid_blocks(start, end, message):
    projection = PDDOutputProjection.from_linear(_base_linear(), 3, _spec("channel_major"))
    grid = torch.tensor([1.0, 0.7, 0.2, 0.0])
    with pytest.raises(ValueError, match=message), projection.fuse_block(start, end, grid):
        pass


def _state_clone(state):
    return {name: tensor.clone() for name, tensor in state.items()}


def _assert_state_unchanged(state, original):
    assert state.keys() == original.keys()
    for name, tensor in state.items():
        assert torch.equal(tensor, original[name])


def test_base_and_widened_checkpoint_load_order_is_strict_and_nonmutating():
    spec = _spec("channel_major")
    base_model = _NestedModel()
    base_state = _state_clone(base_model.state_dict())
    base_original = _state_clone(base_state)

    student = _NestedModel()
    student.load_state_dict(base_state, strict=True)
    projection = convert_to_pdd_output_projection(student, spec, grid_size=3)
    _assert_state_unchanged(base_state, base_original)

    widened_state = _state_clone(student.state_dict())
    widened_original = _state_clone(widened_state)
    restored = _NestedModel()
    restored_projection = convert_to_pdd_output_projection(restored, spec, grid_size=3)
    restored.load_state_dict(widened_state, strict=True)
    _assert_state_unchanged(widened_state, widened_original)
    torch.testing.assert_close(restored_projection.weight, projection.weight)

    with pytest.raises(RuntimeError, match="size mismatch"):
        student.load_state_dict(base_state, strict=True)
    _assert_state_unchanged(base_state, base_original)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _NestedModel().load_state_dict(widened_state, strict=True)
    _assert_state_unchanged(widened_state, widened_original)
