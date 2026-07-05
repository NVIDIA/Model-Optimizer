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

"""Unit tests for ``modelopt.torch.quantization.utils.core_utils``.

Covers the CPU-testable surface: block/axis amax reductions, padding,
quantizer attribute-name helpers, quantized-module predicates, the various
patching/toggle context managers, quantizer state-dict round trips, MoE
expert amax syncing, and NVFP4 static-quantizer promotion. FSDP2 sharding
helpers that require an initialized process group are exercised only for
their non-distributed contracts (patch/restore, non-FSDP fallbacks).
"""

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import (
    NVFP4StaticQuantizer,
    SequentialQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.utils import core_utils as cu


def _fsdp_param_group_cls():
    """Single point of update for torch's private FSDP2 param-group class.

    The source module hardcodes the same path; if a torch upgrade moves it,
    only this helper needs adjusting.
    """
    return torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantized_simple_linear():
    """A SimpleLinear quantized with INT8 max calibration (deterministic)."""
    torch.manual_seed(0)
    model = SimpleLinear()
    calib_data = [model.get_input() for _ in range(2)]

    def forward_loop(m):
        for batch in calib_data:
            m(batch)

    return mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)


class _Expert(nn.Module):
    """Tiny single-linear expert used for MoE amax-sync tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def _quantized_expert(seed: int):
    torch.manual_seed(seed)
    expert = _Expert()
    x = torch.randn(2, 4)
    return mtq.quantize(expert, mtq.INT8_DEFAULT_CFG, lambda m: m(x))


# ---------------------------------------------------------------------------
# reduce_amax
# ---------------------------------------------------------------------------


class TestReduceAmax:
    def test_none_axis_reduces_all_to_scalar(self):
        x = torch.tensor([[1.0, -5.0], [2.0, 3.0]])
        out = cu.reduce_amax(x, axis=None)
        assert out.ndim == 0
        assert out.item() == 5.0  # |-5| dominates the positive max

    def test_int_axis_keepdims_shape_and_values(self):
        x = torch.tensor([[-7.0, 3.0], [2.0, -1.0]])
        out = cu.reduce_amax(x, axis=0)
        assert out.shape == (1, 2)
        assert torch.equal(out, torch.tensor([[7.0, 3.0]]))

    def test_keepdims_false_drops_reduced_dim(self):
        x = torch.randn(2, 3, 4)
        out = cu.reduce_amax(x, axis=1, keepdims=False)
        assert out.shape == (2, 4)
        assert torch.equal(out, x.abs().amax(dim=1))

    def test_tuple_axis(self):
        x = torch.randn(2, 3, 4)
        out = cu.reduce_amax(x, axis=(0, 2))
        assert out.shape == (1, 3, 1)
        assert torch.equal(out, x.abs().amax(dim=(0, 2), keepdim=True))

    def test_negative_axis_matches_positive(self):
        x = torch.randn(2, 3, 4)
        assert torch.equal(cu.reduce_amax(x, axis=-1), cu.reduce_amax(x, axis=2))

    def test_squeeze_scalar_squeezes_single_element_result(self):
        x = torch.tensor([[1.0, -4.0, 2.0]])  # shape (1, 3)
        out = cu.reduce_amax(x, axis=1)  # (1, 1) -> squeezed
        assert out.ndim == 0
        assert out.item() == 4.0

    def test_squeeze_scalar_false_retains_dims(self):
        x = torch.tensor([[1.0, -4.0, 2.0]])
        out = cu.reduce_amax(x, axis=1, squeeze_scalar=False)
        assert out.shape == (1, 1)

    def test_no_grad_even_for_grad_input(self):
        x = torch.randn(3, 3, requires_grad=True)
        out = cu.reduce_amax(x, axis=0)
        assert not out.requires_grad

    def test_inf_propagates(self):
        x = torch.tensor([float("-inf"), 1.0])
        assert cu.reduce_amax(x).item() == float("inf")  # abs(-inf) == inf

    def test_nan_propagates(self):
        x = torch.tensor([1.0, float("nan"), 2.0])
        assert torch.isnan(cu.reduce_amax(x))

    def test_zero_dim_input(self):
        out = cu.reduce_amax(torch.tensor(-3.0))
        assert out.ndim == 0
        assert out.item() == 3.0

    def test_fp8_upcast_keepdims_false(self):
        """FP8 upcast is covered dtype-parametrized in test_utils.py::test_reduce_amax_fp8;
        this only pins the keepdims=False interaction for an FP8 input."""
        ref = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
        out = cu.reduce_amax(ref.to(torch.float8_e4m3fn), axis=1, keepdims=False)
        assert out.dtype == torch.get_default_dtype()
        assert torch.equal(out, torch.tensor([2.0, 4.0]))


# ---------------------------------------------------------------------------
# reduce_sum
# ---------------------------------------------------------------------------


class TestReduceSum:
    def test_none_axis_total_sum(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = cu.reduce_sum(x)
        assert out.ndim == 0
        assert out.item() == 15.0

    def test_int_axis_keepdims(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = cu.reduce_sum(x, axis=0)
        assert out.shape == (1, 3)
        assert torch.equal(out, torch.tensor([[3.0, 5.0, 7.0]]))

    def test_keepdims_false(self):
        x = torch.randn(2, 3, 4)
        out = cu.reduce_sum(x, axis=2, keepdims=False)
        assert out.shape == (2, 3)
        assert torch.allclose(out, x.sum(dim=2))

    def test_tuple_axis(self):
        x = torch.randn(2, 3, 4)
        out = cu.reduce_sum(x, axis=(0, -1))
        assert out.shape == (1, 3, 1)
        assert torch.allclose(out, x.sum(dim=(0, 2), keepdim=True))

    def test_no_grad_even_for_grad_input(self):
        x = torch.randn(3, requires_grad=True)
        assert not cu.reduce_sum(x, axis=0).requires_grad


# ---------------------------------------------------------------------------
# convert_quantization_axis_to_reduce_axis
# ---------------------------------------------------------------------------


class TestConvertQuantizationAxisToReduceAxis:
    def test_none_means_reduce_all(self):
        assert cu.convert_quantization_axis_to_reduce_axis(torch.randn(2, 3), None) is None

    @pytest.mark.parametrize(
        ("shape", "axis", "expected"),
        # int-axis and None cases live in test_utils.py::test_convert_quantization_axis_to_reduce_axis;
        # these cover the list-axis edge cases missing there.
        [
            ((2, 3, 4), [0, 1, 2], []),  # all dims quantized -> nothing to reduce
            ((2, 3, 4), [], [0, 1, 2]),  # empty axis -> reduce everything
            ((2, 3, 4), [0, -3], [1, 2]),  # positive and negative alias of same dim
        ],
    )
    def test_reduce_axis(self, shape, axis, expected):
        assert cu.convert_quantization_axis_to_reduce_axis(torch.randn(shape), axis) == expected


# ---------------------------------------------------------------------------
# reduce_block_amax
# ---------------------------------------------------------------------------


class TestReduceBlockAmax:
    def test_1d_blocks_take_abs_max(self):
        x = torch.tensor([1.0, -8.0, 2.0, 3.0])
        out = cu.reduce_block_amax(x, {-1: 2})
        assert torch.equal(out, torch.tensor([8.0, 3.0]))

    def test_2d_blocks_last_and_second_last(self):
        x = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, -9.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]
        )
        out = cu.reduce_block_amax(x, {-1: 2, -2: 2})
        # 2x2 blocks: [[0,1],[4,-9]] -> 9, [[2,3],[6,7]] -> 7, ...
        assert torch.equal(out, torch.tensor([[9.0, 7.0], [13.0, 15.0]]))

    def test_second_last_dim_only(self):
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        out = cu.reduce_block_amax(x, {-2: 2})
        assert out.shape == (2, 3)
        assert torch.equal(out, torch.tensor([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]))

    def test_leading_dims_preserved(self):
        x = torch.randn(2, 4, 8)
        out = cu.reduce_block_amax(x, {-1: 4})
        assert out.shape == (2, 4, 2)
        expected = x.reshape(2, 4, 2, 4).abs().amax(dim=-1)
        assert torch.equal(out, expected)

    def test_block_size_equals_dim_keeps_outer_block_dim(self):
        # When block_size == dim size the outer block dim (size 1) is retained,
        # so a (2, 2) input with {-1: 2} yields shape (2, 1), not (2,).
        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        out = cu.reduce_block_amax(x, {-1: 2})
        assert torch.equal(out, torch.tensor([[2.0], [4.0]]))

    def test_non_divisible_dim_raises(self):
        with pytest.raises(AssertionError, match="not divisible"):
            cu.reduce_block_amax(torch.randn(4, 5), {-1: 3})

    def test_input_not_mutated(self):
        x = torch.randn(4, 4)
        x_copy = x.clone()
        cu.reduce_block_amax(x, {-1: 2, -2: 2})
        assert torch.equal(x, x_copy)


# ---------------------------------------------------------------------------
# reduce_block_padding
# ---------------------------------------------------------------------------


class TestReduceBlockPadding:
    def test_pads_last_dim_up_to_block_multiple(self):
        x = torch.ones(3, 5)
        out = cu.reduce_block_padding(x, {-1: 4})
        assert out.shape == (3, 8)
        assert torch.equal(out[:, :5], x)  # original content preserved
        assert torch.equal(out[:, 5:], torch.zeros(3, 3))  # right-padded with 0

    def test_custom_pad_value(self):
        x = torch.ones(2, 3)
        out = cu.reduce_block_padding(x, {-1: 4}, pad_value=7.0)
        assert torch.equal(out[:, 3:], torch.full((2, 1), 7.0))

    def test_divisible_dim_returns_input_unchanged(self):
        x = torch.ones(2, 8)
        out = cu.reduce_block_padding(x, {-1: 4})
        assert out is x  # no copy when no padding is needed

    def test_pads_second_last_dim(self):
        x = torch.ones(3, 5)
        out = cu.reduce_block_padding(x, {-2: 4})
        assert out.shape == (4, 5)
        assert torch.equal(out[3], torch.zeros(5))

    def test_pads_multiple_dims(self):
        x = torch.ones(3, 5)
        out = cu.reduce_block_padding(x, {-1: 4, -2: 2})
        assert out.shape == (4, 8)
        # Original content is preserved top-left; the padded band is zeros.
        assert torch.equal(out[:3, :5], x)
        assert torch.equal(out[3:, :], torch.zeros(1, 8))
        assert torch.equal(out[:, 5:], torch.zeros(4, 3))

    def test_positive_dim_key_equivalent_to_negative(self):
        x = torch.ones(3, 5)
        assert torch.equal(cu.reduce_block_padding(x, {1: 4}), cu.reduce_block_padding(x, {-1: 4}))


# ---------------------------------------------------------------------------
# quantizer_attr_names / QuantizerAttrNames
# ---------------------------------------------------------------------------


class TestQuantizerAttrNames:
    def test_default_weight_has_no_prefix(self):
        names = cu.quantizer_attr_names()
        assert names.weight_quantizer == "weight_quantizer"
        assert names.input_quantizer == "input_quantizer"
        assert names.output_quantizer == "output_quantizer"
        assert names.weight_scale == "weight_scale"
        assert names.weight_scale_2 == "weight_scale_2"
        assert names.input_scale == "input_scale"
        assert names.output_scale == "output_scale"

    def test_custom_weight_name_is_prefixed(self):
        names = cu.quantizer_attr_names("gate_up_proj")
        assert names.weight_quantizer == "gate_up_proj_weight_quantizer"
        assert names.weight_scale_2 == "gate_up_proj_weight_scale_2"
        assert names.input_quantizer == "gate_up_proj_input_quantizer"

    def test_explicit_weight_matches_default(self):
        assert cu.quantizer_attr_names("weight") == cu.quantizer_attr_names()


# ---------------------------------------------------------------------------
# representative_weight_quantizer / weight_attr_names
# ---------------------------------------------------------------------------


class TestRepresentativeWeightQuantizer:
    def test_singular_tensor_quantizer(self):
        m = nn.Module()
        m.weight_quantizer = TensorQuantizer()
        assert cu.representative_weight_quantizer(m) is m.weight_quantizer

    def test_singular_sequential_quantizer(self):
        m = nn.Module()
        m.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
        assert cu.representative_weight_quantizer(m) is m.weight_quantizer

    def test_plural_module_list_returns_first(self):
        m = nn.Module()
        m.weight_quantizers = nn.ModuleList([TensorQuantizer(), TensorQuantizer()])
        assert cu.representative_weight_quantizer(m) is m.weight_quantizers[0]

    def test_empty_plural_list_returns_none(self):
        m = nn.Module()
        m.weight_quantizers = nn.ModuleList()
        assert cu.representative_weight_quantizer(m) is None

    def test_plural_list_of_non_quantizers_returns_none(self):
        m = nn.Module()
        m.weight_quantizers = nn.ModuleList([nn.Identity()])
        assert cu.representative_weight_quantizer(m) is None

    def test_no_quantizer_returns_none(self):
        assert cu.representative_weight_quantizer(nn.Linear(2, 2)) is None

    def test_custom_weight_name(self):
        m = nn.Module()
        m.gate_up_proj_weight_quantizer = TensorQuantizer()
        assert (
            cu.representative_weight_quantizer(m, "gate_up_proj") is m.gate_up_proj_weight_quantizer
        )
        assert cu.representative_weight_quantizer(m, "weight") is None


class TestWeightAttrNames:
    def test_plain_linear_yields_nothing(self):
        assert list(cu.weight_attr_names(nn.Linear(4, 4))) == []

    def test_standard_weight_with_quantizer(self):
        m = nn.Linear(4, 4)
        m.weight_quantizer = TensorQuantizer()
        assert list(cu.weight_attr_names(m)) == ["weight"]

    def test_quantized_model_linear(self):
        model = _quantized_simple_linear()
        assert list(cu.weight_attr_names(model.net[0])) == ["weight"]

    def test_custom_param_name_with_quantizer(self):
        m = nn.Module()
        m.gate_up_proj = nn.Parameter(torch.randn(4, 4))
        m.gate_up_proj_weight_quantizer = TensorQuantizer()
        assert list(cu.weight_attr_names(m)) == ["gate_up_proj"]

    def test_param_without_quantizer_not_yielded(self):
        m = nn.Module()
        m.weight = nn.Parameter(torch.randn(4, 4))
        m.weight_quantizer = TensorQuantizer()
        m.other_proj = nn.Parameter(torch.randn(4, 4))  # no matching quantizer
        assert list(cu.weight_attr_names(m)) == ["weight"]

    def test_plural_weight_quantizers_layout(self):
        m = nn.Module()
        m.weight = nn.Parameter(torch.randn(4, 4))
        m.weight_quantizers = nn.ModuleList([TensorQuantizer()])
        assert list(cu.weight_attr_names(m)) == ["weight"]

    def test_non_quantizer_attr_is_ignored(self):
        m = nn.Module()
        m.weight = nn.Parameter(torch.randn(4, 4))
        m.weight_quantizer = nn.Identity()  # wrong type
        assert list(cu.weight_attr_names(m)) == []


# ---------------------------------------------------------------------------
# is_quantized* predicates
# ---------------------------------------------------------------------------


class _EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(self.emb(x))


class TestIsQuantizedPredicates:
    def test_plain_model_is_not_quantized(self):
        assert not cu.is_quantized(SimpleLinear())
        assert not cu.is_quantized_linear(nn.Linear(4, 4))

    def test_quantized_model_and_submodule(self):
        model = _quantized_simple_linear()
        assert cu.is_quantized(model)
        assert cu.is_quantized(model.net[0])
        assert cu.is_quantized_linear(model.net[0])

    def test_tensor_quantizer_itself_counts_as_quantized(self):
        assert cu.is_quantized(TensorQuantizer())

    def test_quantized_conv_is_not_linear(self):
        torch.manual_seed(0)
        conv_model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
        x = torch.randn(1, 3, 4, 4)
        mtq.quantize(conv_model, mtq.INT8_DEFAULT_CFG, lambda m: m(x))
        assert cu.is_quantized(conv_model[0])
        assert not cu.is_quantized_linear(conv_model[0])  # 4D weight is not a GEMM

    def test_quantized_embedding_is_excluded_from_linear(self):
        torch.manual_seed(0)
        model = _EmbeddingModel()
        idx = torch.randint(0, 16, (2, 3))
        mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(idx))
        # Embedding has a 2D weight but must never be treated as a linear.
        assert not cu.is_quantized_linear(model.emb)
        assert cu.is_quantized_linear(model.fc)

    def test_parallel_linear_flags(self):
        model = _quantized_simple_linear()
        qlin = model.net[0]
        assert not cu.is_quantized_column_parallel_linear(qlin)
        assert not cu.is_quantized_row_parallel_linear(qlin)
        assert not cu.is_quantized_parallel_linear(qlin)

        qlin._is_column_parallel = True
        assert cu.is_quantized_column_parallel_linear(qlin)
        assert not cu.is_quantized_row_parallel_linear(qlin)
        assert cu.is_quantized_parallel_linear(qlin)

        qlin._is_column_parallel = False
        qlin._is_row_parallel = True
        assert cu.is_quantized_row_parallel_linear(qlin)
        assert cu.is_quantized_parallel_linear(qlin)

    def test_parallel_flags_require_quantized_linear(self):
        lin = nn.Linear(4, 4)
        lin._is_column_parallel = True
        assert not cu.is_quantized_column_parallel_linear(lin)


# ---------------------------------------------------------------------------
# calibrate_with_adapters / disable_lora_quantizers_in_config
# ---------------------------------------------------------------------------


class _FakeLoraModel:
    def __init__(self):
        self.calls = []

    def disable_adapters(self):
        self.calls.append("disable")

    def enable_adapters(self):
        self.calls.append("enable")


class TestCalibrateWithAdapters:
    def test_lora_disabled_then_reenabled(self):
        model = _FakeLoraModel()
        with cu.calibrate_with_adapters(model, SimpleNamespace(lora=True)):
            assert model.calls == ["disable"]
        assert model.calls == ["disable", "enable"]

    def test_no_lora_flag_is_noop(self):
        model = _FakeLoraModel()
        with cu.calibrate_with_adapters(model, SimpleNamespace(lora=False)):
            pass
        assert model.calls == []

    def test_missing_lora_attr_is_noop(self):
        model = _FakeLoraModel()
        with cu.calibrate_with_adapters(model, SimpleNamespace()):
            pass
        assert model.calls == []

    def test_exception_leaves_adapters_disabled(self):
        # NOTE: documents current behavior; arguably a bug because the yield is
        # not wrapped in try/finally (unlike export_torch_mode et al.), so an
        # exception during calibration leaves LoRA adapters permanently
        # disabled. Same unprotected-yield pattern as replace_function.
        model = _FakeLoraModel()
        with (
            pytest.raises(RuntimeError),
            cu.calibrate_with_adapters(model, SimpleNamespace(lora=True)),
        ):
            raise RuntimeError("calibration failed")
        assert model.calls == ["disable"]  # no re-enable


class TestDisableLoraQuantizersInConfig:
    def test_appends_lora_and_layer_entries(self):
        config = {"quant_cfg": [{"quantizer_name": "*weight_quantizer", "num_bits": 8}]}
        out = cu.disable_lora_quantizers_in_config(config, ["fc1", "fc2"])
        assert out is config  # modified in place and returned
        entries = config["quant_cfg"]
        assert {"quantizer_name": "*lora*", "enable": False} in entries
        for layer in ("fc1", "fc2"):
            for qtype in ("input_quantizer", "weight_quantizer", "output_quantizer"):
                assert {"quantizer_name": f"*{layer}.{qtype}", "enable": False} in entries
        # 1 original + 1 lora wildcard + 3 per layer
        assert len(entries) == 1 + 1 + 3 * 2

    def test_no_layers_only_adds_lora_wildcard(self):
        config = {"quant_cfg": []}
        cu.disable_lora_quantizers_in_config(config, [])
        assert config["quant_cfg"] == [{"quantizer_name": "*lora*", "enable": False}]


# ---------------------------------------------------------------------------
# replace_function / multi_context / export_torch_mode
# ---------------------------------------------------------------------------


class TestReplaceFunction:
    def test_replace_and_restore(self):
        pkg = SimpleNamespace(func=lambda: "old")
        original = pkg.func

        with cu.replace_function(pkg, "func", lambda: "new"):
            assert pkg.func() == "new"
            assert pkg._func is original  # original cached under underscore name
        assert pkg.func is original
        assert not hasattr(pkg, "_func")  # cache attribute removed on exit

    def test_custom_cache_name(self):
        pkg = SimpleNamespace(func=lambda: "old")
        with cu.replace_function(pkg, "func", lambda: "new", og_func_cache_name="_orig_func"):
            assert pkg._orig_func() == "old"
            assert not hasattr(pkg, "_func")
        assert not hasattr(pkg, "_orig_func")

    def test_exception_leaks_replacement(self):
        # NOTE: documents current behavior; arguably a bug because replace_function
        # does not wrap the yield in try/finally, so an exception raised inside the
        # context permanently leaks the replacement function and the cached original.
        pkg = SimpleNamespace(func=lambda: "old")
        original = pkg.func
        new = lambda: "new"  # noqa: E731
        with pytest.raises(RuntimeError), cu.replace_function(pkg, "func", new):
            raise RuntimeError("boom")
        assert pkg.func is new  # replacement not rolled back
        assert pkg._func is original  # cache attribute not cleaned up


class TestMultiContext:
    def test_returns_entered_values_in_order(self):
        events = []

        @contextmanager
        def make(tag):
            events.append(f"enter {tag}")
            yield tag
            events.append(f"exit {tag}")

        with cu.multi_context(make("a"), make("b")) as values:
            assert values == ["a", "b"]
            assert events == ["enter a", "enter b"]
        # ExitStack unwinds in LIFO order.
        assert events == ["enter a", "enter b", "exit b", "exit a"]

    def test_empty_context_list(self):
        with cu.multi_context() as values:
            assert values == []


class TestExportTorchMode:
    def test_default_is_false(self):
        assert cu.is_torch_export_mode() is False

    def test_enabled_inside_context_and_restored(self):
        with cu.export_torch_mode():
            assert cu.is_torch_export_mode() is True
        assert cu.is_torch_export_mode() is False

    def test_nested_contexts_restore_correctly(self):
        with cu.export_torch_mode():
            with cu.export_torch_mode():
                assert cu.is_torch_export_mode() is True
            assert cu.is_torch_export_mode() is True  # inner exit restores outer True
        assert cu.is_torch_export_mode() is False

    def test_restored_on_exception(self):
        with pytest.raises(RuntimeError), cu.export_torch_mode():
            raise RuntimeError("boom")
        assert cu.is_torch_export_mode() is False


# ---------------------------------------------------------------------------
# is_pow2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, True),
        (2, True),
        (4, True),
        (64, True),
        (2**30, True),
        (0, False),
        (3, False),
        (6, False),
        (5, False),
        (-1, False),
        (-2, False),
        (-4, False),  # negative powers of two are rejected
    ],
)
def test_is_pow2(n, expected):
    assert cu.is_pow2(n) is expected


# ---------------------------------------------------------------------------
# _set_parameter / _get_module_name / _get_enclosing_fsdp_module /
# get_prefixed_param_names
# ---------------------------------------------------------------------------


class TestSetParameter:
    def test_dotted_name_sets_on_submodule(self):
        model = SimpleLinear()
        new_w = nn.Parameter(torch.ones(32, 16))
        cu._set_parameter(model, "net.0.weight", new_w)
        assert model.net[0].weight is new_w
        assert dict(model.named_parameters())["net.0.weight"] is new_w

    def test_plain_name_sets_on_module_itself(self):
        lin = nn.Linear(2, 2)
        new_w = nn.Parameter(torch.zeros(2, 2))
        cu._set_parameter(lin, "weight", new_w)
        assert lin.weight is new_w


class TestGetModuleName:
    def test_returns_dotted_name(self):
        model = SimpleLinear()
        assert cu._get_module_name(model.net[0], model) == "net.0"

    def test_root_module_name_is_empty_string(self):
        model = SimpleLinear()
        assert cu._get_module_name(model, model) == ""

    def test_foreign_module_returns_none(self):
        assert cu._get_module_name(nn.Linear(2, 2), SimpleLinear()) is None

    def test_precomputed_name_to_module_is_used(self):
        model = SimpleLinear()
        fake_map = {"custom.name": model.net[0]}
        assert cu._get_module_name(model.net[0], model, fake_map) == "custom.name"


class TestGetEnclosingFsdpModule:
    def test_non_fsdp_root_returns_none(self):
        model = SimpleLinear()
        assert cu._get_enclosing_fsdp_module(model.net[0], model) is None

    def test_root_module_itself_non_fsdp(self):
        model = SimpleLinear()
        assert cu._get_enclosing_fsdp_module(model, model) is None

    def test_foreign_module_raises(self):
        with pytest.raises(ValueError, match="not found in the root model"):
            cu._get_enclosing_fsdp_module(nn.Linear(2, 2), SimpleLinear())


class TestGetPrefixedParamNames:
    def test_returns_prefixed_module_name(self):
        model = SimpleLinear()
        assert cu.get_prefixed_param_names(model, model.net[0]) == "net.0"
        assert cu.get_prefixed_param_names(model, model.net[2]) == "net.2"

    def test_module_without_params_returns_none(self):
        model = SimpleLinear()
        assert cu.get_prefixed_param_names(model, nn.ReLU()) is None

    def test_foreign_module_returns_none(self):
        assert cu.get_prefixed_param_names(SimpleLinear(), nn.Linear(3, 3)) is None


# ---------------------------------------------------------------------------
# get_quantizer_state_dict / set_quantizer_state_dict
# ---------------------------------------------------------------------------


class TestQuantizerStateDict:
    def test_keys_are_quantizer_module_names(self):
        model = _quantized_simple_linear()
        state = cu.get_quantizer_state_dict(model)
        expected_names = {
            name for name, m in model.named_modules() if isinstance(m, TensorQuantizer)
        }
        assert set(state.keys()) == expected_names
        assert "net.0.input_quantizer" in state
        # Calibrated amax must be part of the captured state.
        assert "_amax" in state["net.0.input_quantizer"]

    def test_round_trip_restores_amax(self):
        model = _quantized_simple_linear()
        # Deep-copy: torch state_dicts share storage with live buffers.
        snapshot = copy.deepcopy(cu.get_quantizer_state_dict(model))
        q = model.net[0].input_quantizer
        original_amax = q.amax.clone()

        q.amax = original_amax + 5.0
        assert not torch.equal(q.amax, original_amax)

        cu.set_quantizer_state_dict(model, snapshot)
        assert torch.equal(q.amax, original_amax)

    def test_missing_keys_are_skipped(self):
        model = _quantized_simple_linear()
        q = model.net[0].input_quantizer
        perturbed = q.amax + 3.0
        q.amax = perturbed
        # A state dict without this quantizer's key leaves it untouched.
        cu.set_quantizer_state_dict(model, {})
        assert torch.equal(q.amax, perturbed)

    def test_no_quantizers_returns_empty(self):
        assert cu.get_quantizer_state_dict(SimpleLinear()) == {}


# ---------------------------------------------------------------------------
# enable_quant / disable_calib / enable_fake_quant
# ---------------------------------------------------------------------------


class TestQuantizerToggleContexts:
    def test_enable_quant_toggles_and_restores(self):
        q = TensorQuantizer()
        q._if_quant = False
        with cu.enable_quant(q):
            assert q._if_quant is True
        assert q._if_quant is False

    def test_enable_quant_restores_on_exception(self):
        q = TensorQuantizer()
        q._if_quant = False
        with pytest.raises(RuntimeError), cu.enable_quant(q):
            raise RuntimeError("boom")
        assert q._if_quant is False

    def test_disable_calib_toggles_and_restores(self):
        q = TensorQuantizer()
        q._if_calib = True
        with cu.disable_calib(q):
            assert q._if_calib is False
        assert q._if_calib is True

    def test_disable_calib_restores_on_exception(self):
        q = TensorQuantizer()
        q._if_calib = True
        with pytest.raises(RuntimeError), cu.disable_calib(q):
            raise RuntimeError("boom")
        assert q._if_calib is True

    def test_enable_fake_quant_restores_per_module_flags(self):
        model = _quantized_simple_linear()
        quant_linears = [m for m in model.modules() if hasattr(m, "weight_quantizer")]
        assert len(quant_linears) >= 2
        # Give the modules distinct initial flags to verify order-correct restore.
        quant_linears[0].weight_quantizer._fake_quant = False
        quant_linears[1].weight_quantizer._fake_quant = True

        with cu.enable_fake_quant(model):
            assert all(m.weight_quantizer._fake_quant for m in quant_linears)

        assert quant_linears[0].weight_quantizer._fake_quant is False
        assert quant_linears[1].weight_quantizer._fake_quant is True


# ---------------------------------------------------------------------------
# no_requires_grad
# ---------------------------------------------------------------------------


class TestNoRequiresGrad:
    def test_parameters_created_inside_have_no_grad(self):
        with cu.no_requires_grad():
            p = nn.Parameter(torch.zeros(2))
            assert p.requires_grad is False
            # Even an explicit requires_grad=True is overridden inside the context.
            p2 = nn.Parameter(torch.zeros(2), requires_grad=True)
            assert p2.requires_grad is False

    def test_integer_parameter_creation_allowed_inside(self):
        # The motivating use case: integer (compressed) weights cannot require grad.
        with pytest.raises(RuntimeError):
            nn.Parameter(torch.zeros(2, dtype=torch.int32))
        with cu.no_requires_grad():
            p = nn.Parameter(torch.zeros(2, dtype=torch.int32))
            assert p.requires_grad is False

    def test_default_behavior_restored_after_exit(self):
        with cu.no_requires_grad():
            pass
        assert nn.Parameter(torch.zeros(2)).requires_grad is True

    def test_restored_on_exception(self):
        with pytest.raises(RuntimeError), cu.no_requires_grad():
            raise RuntimeError("boom")
        assert nn.Parameter(torch.zeros(2)).requires_grad is True


# ---------------------------------------------------------------------------
# FSDP patch contexts (non-distributed contracts only)
# ---------------------------------------------------------------------------


class TestFsdpPatchContexts:
    def test_patch_fsdp_mp_dtypes_swaps_and_restores(self):
        pg_cls = _fsdp_param_group_cls()
        original = pg_cls._init_mp_dtypes
        with cu.patch_fsdp_mp_dtypes():
            assert pg_cls._init_mp_dtypes is not original
        assert pg_cls._init_mp_dtypes is original

    def test_patch_fsdp_mp_dtypes_restores_on_exception(self):
        pg_cls = _fsdp_param_group_cls()
        original = pg_cls._init_mp_dtypes
        with pytest.raises(RuntimeError), cu.patch_fsdp_mp_dtypes():
            raise RuntimeError("boom")
        assert pg_cls._init_mp_dtypes is original

    def test_persistent_materialization_noop_for_plain_module(self):
        pg_cls = _fsdp_param_group_cls()
        orig_unshard, orig_reshard = pg_cls.unshard, pg_cls.reshard
        with cu.persistent_materialization(nn.Linear(2, 2)):
            # Non-FSDP layers must not trigger the global monkeypatch.
            assert pg_cls.unshard is orig_unshard
            assert pg_cls.reshard is orig_reshard

    def test_enable_weight_access_plain_module_is_nullcontext(self):
        model = SimpleLinear()
        weight_before = model.net[0].weight
        with cu.enable_weight_access_and_writeback(model.net[0], model):
            assert model.net[0].weight is weight_before  # nothing swapped


# ---------------------------------------------------------------------------
# update_quant_cfg_with_kv_cache_quant
# ---------------------------------------------------------------------------


class TestUpdateQuantCfgWithKvCacheQuant:
    KV_CFG = [{"quantizer_name": "*[kv]_bmm_quantizer", "num_bits": (4, 3)}]

    def test_appends_kv_entries_and_preserves_algorithm(self):
        base = {
            "quant_cfg": [{"quantizer_name": "*weight_quantizer", "num_bits": 8}],
            "algorithm": "awq_lite",
        }
        out = cu.update_quant_cfg_with_kv_cache_quant(base, self.KV_CFG)
        assert out["quant_cfg"] == base["quant_cfg"] + self.KV_CFG
        assert out["algorithm"] == "awq_lite"

    def test_input_is_not_mutated(self):
        base = {"quant_cfg": [{"quantizer_name": "*", "num_bits": 8}], "algorithm": "max"}
        base_copy = copy.deepcopy(base)
        cu.update_quant_cfg_with_kv_cache_quant(base, self.KV_CFG)
        assert base == base_copy

    def test_none_quant_cfg_disables_everything_else(self):
        out = cu.update_quant_cfg_with_kv_cache_quant({"quant_cfg": None}, self.KV_CFG)
        assert out["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}
        assert out["quant_cfg"][1:] == self.KV_CFG

    def test_missing_algorithm_defaults_to_max(self):
        out = cu.update_quant_cfg_with_kv_cache_quant({"quant_cfg": []}, self.KV_CFG)
        assert out["algorithm"] == "max"

    def test_falsy_algorithm_defaults_to_max(self):
        out = cu.update_quant_cfg_with_kv_cache_quant(
            {"quant_cfg": [], "algorithm": None}, self.KV_CFG
        )
        assert out["algorithm"] == "max"


# ---------------------------------------------------------------------------
# sync_moe_expert_amax
# ---------------------------------------------------------------------------


class TestSyncMoeExpertAmax:
    def test_input_amax_synced_to_elementwise_max(self):
        e1, e2 = _quantized_expert(seed=0), _quantized_expert(seed=1)
        e1.fc.input_quantizer.amax = torch.tensor(1.0)
        e2.fc.input_quantizer.amax = torch.tensor(3.0)

        cu.sync_moe_expert_amax([e1, e2])

        assert e1.fc.input_quantizer.amax.item() == 3.0
        assert e2.fc.input_quantizer.amax.item() == 3.0

    def test_weight_amax_untouched_by_default(self):
        e1, e2 = _quantized_expert(seed=0), _quantized_expert(seed=1)
        w1 = e1.fc.weight_quantizer.amax.clone()
        w2 = e2.fc.weight_quantizer.amax.clone()

        cu.sync_moe_expert_amax([e1, e2], sync_weight_amax=False)

        assert torch.equal(e1.fc.weight_quantizer.amax, w1)
        assert torch.equal(e2.fc.weight_quantizer.amax, w2)

    def test_weight_amax_synced_when_requested(self):
        e1, e2 = _quantized_expert(seed=0), _quantized_expert(seed=1)
        expected = torch.maximum(e1.fc.weight_quantizer.amax, e2.fc.weight_quantizer.amax)

        cu.sync_moe_expert_amax([e1, e2], sync_weight_amax=True)

        assert torch.equal(e1.fc.weight_quantizer.amax, expected)
        assert torch.equal(e2.fc.weight_quantizer.amax, expected)

    def test_missing_weight_amax_recalibrated_from_weight(self):
        e1 = _quantized_expert(seed=0)
        del e1.fc.weight_quantizer._amax  # simulate an expert that saw no tokens
        assert e1.fc.weight_quantizer.amax is None

        cu.sync_moe_expert_amax([e1])

        # INT8_DEFAULT_CFG uses per-channel (axis=0) weight quantization.
        expected = e1.fc.weight.abs().amax(dim=1, keepdim=True)
        assert e1.fc.weight_quantizer.amax is not None
        assert torch.allclose(e1.fc.weight_quantizer.amax, expected)


# ---------------------------------------------------------------------------
# promote_nvfp4_static_quantizers
# ---------------------------------------------------------------------------


def _nvfp4_static_quantizer(with_amax=True):
    cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "static", "scale_bits": (4, 3)},
    )
    q = TensorQuantizer(quant_attribute_cfg=cfg)
    if with_amax:
        # Per-block amax for a (4, 32) weight with block size 16 -> (4, 2).
        q.amax = torch.tensor([[1.0, 2.0], [0.5, 6.0], [3.0, 0.25], [4.0, 1.0]])
    return q


class _QuantizerHolder(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.weight_quantizer = quantizer


class TestPromoteNvfp4StaticQuantizers:
    def test_promotes_and_sets_global_amax(self):
        holder = _QuantizerHolder(_nvfp4_static_quantizer())
        per_block = holder.weight_quantizer.amax.clone()

        converted = cu.promote_nvfp4_static_quantizers(holder)

        assert converted == 1
        q = holder.weight_quantizer
        assert isinstance(q, NVFP4StaticQuantizer)
        assert torch.equal(q.global_amax, per_block.abs().max())
        assert torch.equal(q.amax, per_block)  # per-block amax preserved

    def test_second_promotion_is_idempotent(self):
        holder = _QuantizerHolder(_nvfp4_static_quantizer())
        assert cu.promote_nvfp4_static_quantizers(holder) == 1
        assert cu.promote_nvfp4_static_quantizers(holder) == 0  # already promoted

    def test_missing_amax_skipped(self):
        holder = _QuantizerHolder(_nvfp4_static_quantizer(with_amax=False))
        assert cu.promote_nvfp4_static_quantizers(holder) == 0
        assert not isinstance(holder.weight_quantizer, NVFP4StaticQuantizer)

    def test_disabled_quantizer_skipped(self):
        holder = _QuantizerHolder(_nvfp4_static_quantizer())
        holder.weight_quantizer.disable()
        assert cu.promote_nvfp4_static_quantizers(holder) == 0

    def test_non_nvfp4_quantizer_skipped(self):
        q = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
        q.amax = torch.tensor([[1.0], [2.0]])
        holder = _QuantizerHolder(q)
        assert cu.promote_nvfp4_static_quantizers(holder) == 0
        assert not isinstance(holder.weight_quantizer, NVFP4StaticQuantizer)
