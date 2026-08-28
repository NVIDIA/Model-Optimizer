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

import json

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.transformers_models import get_tiny_llama, get_tiny_qwen3, get_tiny_qwen3vl

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_kv_cache_dtype, get_quant_config
from modelopt.torch.quantization import model_quant, tensor_quant
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.kv_cache_auto_quant import (
    _candidate_quantizers,
    _eligible_layers,
    _kv_scalar_weight,
    _solve_additive_recipe,
    _validate_kv_only_config,
    auto_quantize_kv_cache,
)
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.fixture
def nvfp4_fake_quant_stub(monkeypatch):
    """Keep CPU search tests independent of the CUDA-only NVFP4 fake-quant kernel."""

    monkeypatch.setattr(
        tensor_quant,
        "dynamic_block_quantize_op",
        lambda inputs, *_args, **_kwargs: torch.zeros_like(inputs),
    )


def _quantizer_cfg(bits, *, constant_amax=None):
    cfg = {"num_bits": bits}
    if bits == (2, 1):
        cfg["block_sizes"] = {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    if constant_amax is not None:
        cfg["constant_amax"] = constant_amax
    return cfg


def _kv_config(bits, effective_bits, *, algorithm="max", constant_amax=None):
    return QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": _quantizer_cfg(bits, constant_amax=constant_amax),
            }
        ],
        algorithm=algorithm,
        effective_bits=effective_bits,
    )


def _asymmetric_kv_config():
    return QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": _quantizer_cfg((2, 1), constant_amax=1.0),
            },
            {
                "quantizer_name": "*.k_bmm_quantizer",
                "cfg": _quantizer_cfg((4, 3), constant_amax=1.0),
            },
        ],
        algorithm=None,
        effective_bits=6.25,
    )


def test_kv_candidate_requires_exact_bits_and_both_sides():
    _validate_kv_only_config(_kv_config((4, 3), 8.0))

    with pytest.raises(ValueError, match="config-level effective_bits"):
        _validate_kv_only_config(
            QuantizeConfig(
                quant_cfg=[
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": (4, 3), "use_constant_amax": True},
                    }
                ]
            )
        )
    with pytest.raises(ValueError, match="completely configure both"):
        _validate_kv_only_config(
            QuantizeConfig(
                quant_cfg=[
                    {
                        "quantizer_name": "*k_bmm_quantizer",
                        "cfg": {"num_bits": (4, 3), "use_constant_amax": True},
                    }
                ],
                effective_bits=8.0,
            )
        )


@pytest.mark.parametrize("algorithm", ["svdquant", {"method": "smoothquant"}, {"method": "mse"}])
def test_kv_candidate_rejects_structural_or_unscoped_algorithms(algorithm):
    config = _kv_config((4, 3), 8.0).model_copy(update={"algorithm": algorithm})

    with pytest.raises(ValueError, match="only non-structural calibration algorithms"):
        _validate_kv_only_config(config)


def test_kv_additive_solver_spends_fp8_on_more_sensitive_layer():
    selections, status = _solve_additive_recipe(
        layer_names=["layer0", "layer1"],
        scalar_weights=[256, 256],
        candidate_names=["fp8", "nvfp4"],
        candidate_bits=[8.0, 4.5],
        scores=[[0.0, 10.0], [0.0, 1.0]],
        target_bits=6.25,
        verbose=False,
    )

    assert status == "Optimal"
    assert selections == [0, 1]


def test_kv_scalar_weight_counts_k_and_v_widths():
    module = nn.Module()
    module.k_proj = nn.Linear(32, 24, bias=False)
    module.v_proj = nn.Linear(32, 16, bias=False)

    assert _kv_scalar_weight(module, "attention") == 40


@pytest.mark.parametrize(
    ("config", "expected_format"),
    [
        (_kv_config((4, 3), 8.0), "FP8"),
        (_kv_config((2, 1), 4.5), "NVFP4"),
        (_kv_config((4, 3), 8.0, algorithm=None, constant_amax=1.0), "FP8"),
        (_kv_config((2, 1), 4.5, algorithm=None, constant_amax=1.0), "NVFP4"),
    ],
)
def test_kv_candidate_accepts_export_supported_persistent_formats(config, expected_format):
    _validate_kv_only_config(config)
    quantizers = _candidate_quantizers(config)
    module = nn.Module()
    for name, quantizer in quantizers.items():
        setattr(module, name, quantizer)

    assert get_kv_cache_dtype(module) == expected_format


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (_kv_config((4, 3), 8.0, algorithm=None), "no persistent export scale"),
        (
            QuantizeConfig(
                quant_cfg=[
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": (4, 3), "use_constant_amax": True},
                    }
                ],
                algorithm="max",
                effective_bits=8.0,
            ),
            "no persistent export scale",
        ),
        (_kv_config(8, 8.0, algorithm=None, constant_amax=1.0), "per-tensor FP8"),
        (_kv_config(4, 4.0, algorithm=None, constant_amax=1.0), "per-tensor FP8"),
        (_kv_config((4, 3), 6.0), "does not match its configured K/V storage cost"),
    ],
)
def test_kv_candidate_rejects_non_exportable_or_incorrect_cost(config, match):
    with pytest.raises(ValueError, match=match):
        _validate_kv_only_config(config)


def test_kv_candidate_rejects_top_level_dynamic_fp8():
    config = QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "type": "dynamic"},
            }
        ],
        algorithm="max",
        effective_bits=8.0,
    )

    with pytest.raises(ValueError, match="top-level dynamic"):
        _validate_kv_only_config(config)


class _ToyKVAttention(nn.Module):
    def __init__(self, width, gain):
        super().__init__()
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.k_bmm_quantizer = nn.Identity()
        self.v_bmm_quantizer = nn.Identity()
        self.gain = gain

    def forward(self, x):
        return x + self.gain * (self.k_bmm_quantizer(x) + self.v_bmm_quantizer(x))


class _ToyKVModel(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.attn0 = _ToyKVAttention(width, gain=0.25)
        self.attn1 = _ToyKVAttention(width, gain=2.0)
        self.lm_head = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.lm_head(self.attn1(self.attn0(x)))


def test_kv_autoquant_rejects_missing_scale_after_calibration(monkeypatch):
    model = _ToyKVModel()
    original_quantizers = {
        name: (module.k_bmm_quantizer, module.v_bmm_quantizer)
        for name, module in (("attn0", model.attn0), ("attn1", model.attn1))
    }
    monkeypatch.setattr(model_quant, "calibrate", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="no persistent export scale after calibration"):
        auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 8.0},
            [(_kv_config((4, 3), 8.0).model_dump(), "fp8")],
            [torch.randn(1, 2, 8)],
            lambda search_model, batch: search_model(batch),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert model.training
    for name, module in (("attn0", model.attn0), ("attn1", model.attn1)):
        assert (module.k_bmm_quantizer, module.v_bmm_quantizer) == original_quantizers[name]


def test_kv_eligible_layers_supports_hybrid_attention_mixers_only():
    """Hybrid decoders include attention mixers but exclude nonattention mixers."""
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    model.layers[0].mixer = nn.Linear(8, 8, bias=False)
    model.layers[1].mixer = _ToyKVAttention(8, gain=1.0)

    layers = _eligible_layers(model, disabled_layers=None)

    assert [(name, width) for name, _, width in layers] == [("layers.1.mixer", 16)]


def test_kv_eligible_layers_rejects_aliased_attention_boundary():
    """An attention object registered at multiple paths must not be selected by traversal order."""
    model = nn.Module()
    attention = _ToyKVAttention(8, gain=1.0)
    model.attention = attention
    model.attention_alias = attention

    with pytest.raises(ValueError, match="registered through aliases"):
        _eligible_layers(model, disabled_layers=None)


def test_kv_autoquant_scores_and_applies_one_format_per_layer(tmp_path, nvfp4_fake_quant_stub):
    torch.manual_seed(123)
    model = _ToyKVModel()
    data = [torch.randn(2, 3, 8), torch.randn(2, 3, 8)]
    candidates = [
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
                    }
                ],
                "algorithm": None,
                "effective_bits": 8.0,
            },
            "fp8",
        ),
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {
                            "num_bits": (2, 1),
                            "block_sizes": {
                                -1: 16,
                                "type": "dynamic",
                                "scale_bits": (4, 3),
                            },
                            "constant_amax": 1.0,
                        },
                    }
                ],
                "algorithm": None,
                "effective_bits": 4.5,
            },
            "nvfp4",
        ),
    ]

    model, state = auto_quantize_kv_cache(
        model,
        {"kv_effective_bits": 6.25},
        candidates,
        data,
        lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=str(tmp_path / "kv_search.pth"),
    )

    assert state["best"]["effective_bits"] == pytest.approx(6.25)
    assert state["best"]["is_satisfied"]
    assert model.training
    assert {layer["selected"] for layer in state["layers"].values()} == {
        "fp8",
        "nvfp4",
    }
    for layer_name, layer_state in state["layers"].items():
        layer = model.get_submodule(layer_name)
        assert layer.k_bmm_quantizer.num_bits == layer.v_bmm_quantizer.num_bits
        expected_bits = (4, 3) if layer_state["selected"] == "fp8" else (2, 1)
        assert layer.k_bmm_quantizer.num_bits == expected_bits

    restored_model = _ToyKVModel().eval()
    restored_model, restored_state = auto_quantize_kv_cache(
        restored_model,
        {"kv_effective_bits": 6.25},
        candidates,
        data,
        lambda *_: pytest.fail("A compatible checkpoint must skip calibration and scoring."),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=str(tmp_path / "kv_search.pth"),
    )

    assert restored_state == state
    assert not restored_model.training
    for layer_name, layer_state in restored_state["layers"].items():
        layer = restored_model.get_submodule(layer_name)
        expected_bits = (4, 3) if layer_state["selected"] == "fp8" else (2, 1)
        assert layer.k_bmm_quantizer.num_bits == expected_bits


def test_kv_autoquant_honors_ordered_qualified_override_and_cost(nvfp4_fake_quant_stub):
    model = _ToyKVModel()
    candidate = (_asymmetric_kv_config().model_dump(exclude_none=True), "fp8_k_nvfp4_v")

    model, state = auto_quantize_kv_cache(
        model,
        {"kv_effective_bits": 6.25},
        [candidate],
        [torch.randn(1, 2, 8)],
        lambda search_model, batch: search_model(batch),
        num_calib_steps=1,
        num_score_steps=1,
    )

    assert state["candidates"][0]["effective_bits"] == pytest.approx(6.25)
    assert state["best"]["effective_bits"] == pytest.approx(6.25)
    for layer_state in state["layers"].values():
        assert layer_state["selected"] == "fp8_k_nvfp4_v"
    for layer in (model.attn0, model.attn1):
        assert layer.k_bmm_quantizer.num_bits == (4, 3)
        assert layer.v_bmm_quantizer.num_bits == (2, 1)
    assert get_quant_config(model)["quantization"]["kv_cache_quant_algo"] == "FP8_K_NVFP4_V"


def test_kv_autoquant_rejects_asymmetric_candidate_for_unequal_kv_widths(
    nvfp4_fake_quant_stub,
):
    model = _ToyKVModel()
    model.attn0.k_proj = nn.Linear(8, 12, bias=False)
    model.attn0.v_proj = nn.Linear(8, 8, bias=False)

    with pytest.raises(ValueError, match=r"asymmetric K/V candidates.*unequal K/V widths"):
        auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 6.25},
            [(_asymmetric_kv_config().model_dump(exclude_none=True), "fp8_k_nvfp4_v")],
            [torch.randn(1, 2, 8)],
            lambda search_model, batch: search_model(batch),
            num_calib_steps=1,
            num_score_steps=1,
        )


def test_kv_autoquant_rejects_invalid_logits_and_restores_model_state():
    model = _ToyKVModel()
    original_quantizers = {
        name: (module.k_bmm_quantizer, module.v_bmm_quantizer)
        for name, module in (("attn0", model.attn0), ("attn1", model.attn1))
    }
    candidates = [
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {
                            "num_bits": (2, 1),
                            "block_sizes": {
                                -1: 16,
                                "type": "dynamic",
                                "scale_bits": (4, 3),
                            },
                            "constant_amax": 1.0,
                        },
                    }
                ],
                "algorithm": None,
                "effective_bits": 4.5,
            },
            "nvfp4",
        )
    ]

    with pytest.raises(ValueError, match="non-empty vocabulary dimension"):
        auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 4.5},
            candidates,
            [torch.randn(2, 3, 8)],
            lambda *_: torch.ones(8),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert model.training
    for name, module in (("attn0", model.attn0), ("attn1", model.attn1)):
        assert (module.k_bmm_quantizer, module.v_bmm_quantizer) == original_quantizers[name]


def test_public_kv_autoquant_converts_hf_attention_and_searches(tmp_path, nvfp4_fake_quant_stub):
    torch.manual_seed(123)
    model = get_tiny_llama(num_hidden_layers=2)
    data = [{"input_ids": torch.randint(0, model.config.vocab_size, (1, 8))} for _ in range(2)]
    candidates = [
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": (4, 3)},
                    },
                ],
                "algorithm": "max",
                "effective_bits": 8.0,
            },
            "fp8",
        ),
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {
                            "num_bits": (2, 1),
                            "block_sizes": {
                                -1: 16,
                                "type": "dynamic",
                                "scale_bits": (4, 3),
                            },
                            "constant_amax": 1.0,
                        },
                    }
                ],
                "algorithm": None,
                "effective_bits": 4.5,
            },
            "nvfp4",
        ),
    ]

    model, state = mtq.auto_quantize_kv_cache(
        model,
        {"kv_effective_bits": 6.25},
        candidates,
        data,
        lambda search_model, batch: search_model(**batch).logits,
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=str(tmp_path / "hf_kv_search.pth"),
    )

    assert len(state["layers"]) == model.config.num_hidden_layers
    assert state["best"]["effective_bits"] == pytest.approx(6.25)
    assert all(
        layer.self_attn.k_bmm_quantizer.is_enabled and layer.self_attn.v_bmm_quantizer.is_enabled
        for layer in model.model.layers
    )
    non_kv_quantizers = [
        quantizer
        for name, quantizer in model.named_modules()
        if isinstance(quantizer, TensorQuantizer)
        and not name.endswith(("k_bmm_quantizer", "v_bmm_quantizer"))
    ]
    assert non_kv_quantizers
    assert all(not quantizer.is_enabled for quantizer in non_kv_quantizers)
    assert all(
        layer.self_attn.q_proj.weight_quantizer.num_bits == 8 for layer in model.model.layers
    )
    exported_quantization = get_quant_config(model)["quantization"]
    assert exported_quantization["quantized_layers"] == {}
    assert exported_quantization["kv_cache_quantized_layers"]
    assert set(exported_quantization["kv_cache_quantized_layers"]) <= {
        f"model.layers.{idx}.self_attn" for idx in range(model.config.num_hidden_layers)
    }

    restored_model = get_tiny_llama(num_hidden_layers=2)
    restored_model, restored_state = mtq.auto_quantize_kv_cache(
        restored_model,
        {"kv_effective_bits": 6.25},
        candidates,
        data,
        lambda *_: pytest.fail("A compatible checkpoint must skip calibration and scoring."),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=str(tmp_path / "hf_kv_search.pth"),
    )

    assert restored_state == state
    assert any(
        hasattr(layer.self_attn.k_bmm_quantizer, "_amax")
        for layer in restored_model.model.layers
        if layer.self_attn.k_bmm_quantizer.num_bits == (4, 3)
    )


@pytest.mark.parametrize(
    ("model_factory", "expected_layer", "disabled_layers"),
    [
        (get_tiny_qwen3, "model.layers.0.self_attn", None),
        (get_tiny_qwen3vl, "model.language_model.layers.0.self_attn", "*visual*"),
    ],
)
def test_public_kv_autoquant_selects_qwen_causal_attention_only(
    model_factory, expected_layer, disabled_layers
):
    """Plain and conditional Qwen models expose only causal attention to the KV search."""
    model = model_factory(num_hidden_layers=1)
    text_config = getattr(model.config, "text_config", model.config)
    data = [{"input_ids": torch.randint(0, text_config.vocab_size, (1, 8))}]
    candidate = (
        _kv_config((4, 3), 8.0, algorithm=None, constant_amax=1.0).model_dump(),
        "fp8",
    )

    model, state = mtq.auto_quantize_kv_cache(
        model,
        {"kv_effective_bits": 8.0},
        [candidate],
        data,
        lambda search_model, batch: search_model(**batch).logits,
        num_calib_steps=1,
        num_score_steps=1,
        disabled_layers=disabled_layers,
    )

    assert set(state["layers"]) == {expected_layer}
    assert json.loads(json.dumps(state))["layers"][expected_layer]["selected"] == "fp8"
    exported = get_quant_config(model)["quantization"]
    if "kv_cache_quantized_layers" in exported:
        assert set(exported["kv_cache_quantized_layers"]) == {expected_layer}
    else:
        assert exported["kv_cache_quant_algo"] == "FP8"


def test_public_kv_autoquant_validation_and_runtime_failures_are_atomic():
    model = get_tiny_llama(num_hidden_layers=1)
    original_types = {name: type(module) for name, module in model.named_modules()}
    invalid_candidate = _kv_config((4, 3), 8.0).model_dump()
    invalid_candidate["algorithm"] = "svdquant"

    with pytest.raises(ValueError, match="only non-structural calibration algorithms"):
        mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 8.0},
            [invalid_candidate],
            [],
            lambda *_: pytest.fail("Validation must run before model conversion."),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert not hasattr(model, "_modelopt_state")
    assert {name: type(module) for name, module in model.named_modules()} == original_types

    valid_candidate = _kv_config((4, 3), 8.0, algorithm=None, constant_amax=1.0).model_dump()
    data = [{"input_ids": torch.randint(0, model.config.vocab_size, (1, 8))}]
    with pytest.raises(ValueError, match="non-empty vocabulary dimension"):
        mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 8.0},
            [valid_candidate],
            data,
            lambda *_: torch.ones(8),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert not hasattr(model, "_modelopt_state")
    assert {name: type(module) for name, module in model.named_modules()} == original_types


def test_public_kv_autoquant_rejects_unmatched_or_unexportable_candidates_before_conversion():
    model = get_tiny_llama(num_hidden_layers=1)
    original_types = {name: type(module) for name, module in model.named_modules()}
    unmatched = _kv_config((4, 3), 8.0).model_dump()
    unmatched["quant_cfg"].append({"quantizer_name": "q_proj.*_quantizer", "cfg": {"num_bits": 2}})
    invalid_candidates = [
        (unmatched, "does not match a supported qualified K/V quantizer"),
        (_kv_config((4, 3), 8.0, algorithm=None).model_dump(), "no persistent export scale"),
        (
            _kv_config(8, 8.0, algorithm=None, constant_amax=1.0).model_dump(),
            "per-tensor FP8",
        ),
    ]

    for candidate, match in invalid_candidates:
        with pytest.raises(ValueError, match=match):
            mtq.auto_quantize_kv_cache(
                model,
                {"kv_effective_bits": 8.0},
                [candidate],
                [],
                lambda *_: pytest.fail("Validation must run before model conversion."),
                num_calib_steps=1,
                num_score_steps=1,
            )
        assert not hasattr(model, "_modelopt_state")
        assert {name: type(module) for name, module in model.named_modules()} == original_types


def test_public_kv_autoquant_rejects_distributed_execution_before_mutation(monkeypatch):
    model = get_tiny_llama(num_hidden_layers=1)
    original_types = {name: type(module) for name, module in model.named_modules()}
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="single-process only"):
        mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 8.0},
            [],
            [],
            lambda *_: pytest.fail("Distributed validation must fail before search."),
        )

    assert not hasattr(model, "_modelopt_state")
    assert {name: type(module) for name, module in model.named_modules()} == original_types


def test_public_kv_autoquant_preserves_fixed_layers_and_weight_quantizers(
    monkeypatch, nvfp4_fake_quant_stub
):
    torch.manual_seed(123)
    model = get_tiny_llama(num_hidden_layers=2)
    data = [{"input_ids": torch.randint(0, model.config.vocab_size, (1, 8))}]
    fixed_kv_config = {
        "quant_cfg": [
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            }
        ],
        "algorithm": None,
    }
    model = mtq.quantize(model, fixed_kv_config)
    fixed_weight_quantizer = model.model.layers[0].self_attn.q_proj.weight_quantizer
    fixed_weight_quantizer.enable()
    fixed_weight_quantizer.amax = torch.tensor(1.0)
    fixed_weight_quantizer.disable_quant()
    fixed_weight_quantizer.disable_calib()
    observed_fixed_states = []
    fixed_hook = fixed_weight_quantizer.register_forward_hook(
        lambda module, _inputs, _output: observed_fixed_states.append(
            (module.is_enabled, module._if_quant, module._if_calib)
        )
    )
    fixed_qdq_quantizer = model.model.layers[1].self_attn.q_proj.weight_quantizer
    fixed_qdq_quantizer.enable()
    fixed_qdq_quantizer.amax = torch.tensor(1.0)
    observed_qdq_states = []
    qdq_hook = fixed_qdq_quantizer.register_forward_hook(
        lambda module, _inputs, _output: observed_qdq_states.append(
            (module.is_enabled, module._if_quant, module._if_calib)
        )
    )
    calibration_states = []
    real_calibrate = model_quant.calibrate

    def calibrate_with_state_check(*args, **kwargs):
        calibration_states.append(
            (fixed_weight_quantizer._if_quant, fixed_weight_quantizer._if_calib)
        )
        result = real_calibrate(*args, **kwargs)
        calibration_states.append(
            (fixed_weight_quantizer._if_quant, fixed_weight_quantizer._if_calib)
        )
        return result

    monkeypatch.setattr(model_quant, "calibrate", calibrate_with_state_check)

    try:
        model, state = mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 4.5},
            [
                (
                    {
                        "quant_cfg": [
                            {
                                "quantizer_name": "*[kv]_bmm_quantizer",
                                "cfg": {
                                    "num_bits": (2, 1),
                                    "block_sizes": {
                                        -1: 16,
                                        "type": "dynamic",
                                        "scale_bits": (4, 3),
                                    },
                                },
                            },
                        ],
                        "algorithm": "max",
                        "effective_bits": 4.5,
                    },
                    "nvfp4",
                )
            ],
            data,
            lambda search_model, batch: search_model(**batch).logits,
            num_calib_steps=1,
            num_score_steps=1,
            disabled_layers="model.layers.1.self_attn",
        )
    finally:
        fixed_hook.remove()
        qdq_hook.remove()

    assert set(state["layers"]) == {"model.layers.0.self_attn"}
    assert observed_fixed_states
    assert all(state == (True, False, False) for state in observed_fixed_states)
    assert calibration_states == [(False, False), (False, False)]
    assert model.model.layers[0].self_attn.k_bmm_quantizer.num_bits == (2, 1)
    assert model.model.layers[0].self_attn.q_proj.weight_quantizer.is_enabled
    assert not fixed_weight_quantizer._if_quant
    assert not fixed_weight_quantizer._if_calib
    assert fixed_weight_quantizer.amax.item() == pytest.approx(1.0)
    assert observed_qdq_states
    assert all(quantizer_state == (True, True, False) for quantizer_state in observed_qdq_states)
    assert fixed_qdq_quantizer._if_quant
    assert not fixed_qdq_quantizer._if_calib
    assert fixed_qdq_quantizer.amax.item() == pytest.approx(1.0)
    fixed_attention = model.model.layers[1].self_attn
    assert fixed_attention.k_bmm_quantizer.is_enabled
    assert fixed_attention.v_bmm_quantizer.is_enabled
    assert fixed_attention.k_bmm_quantizer.num_bits == (4, 3)
    assert fixed_attention.v_bmm_quantizer.num_bits == (4, 3)
