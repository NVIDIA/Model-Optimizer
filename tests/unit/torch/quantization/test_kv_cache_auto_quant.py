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

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.kv_cache_auto_quant import (
    _candidate_quantizers,
    _kv_scalar_weight,
    _solve_additive_recipe,
    _validate_kv_only_config,
    auto_quantize_kv_cache,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def _kv_config(bits, effective_bits):
    return QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": bits, "use_constant_amax": True},
            }
        ],
        effective_bits=effective_bits,
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


@pytest.mark.parametrize("algorithm", ["svdquant", {"method": "smoothquant"}])
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


def test_kv_candidate_format_can_use_dynamic_amax():
    module = nn.Module()
    module.k_bmm_quantizer = nn.Identity()
    module.v_bmm_quantizer = nn.Identity()
    config = QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": (4, 3)},
            }
        ],
        algorithm=None,
        effective_bits=8.0,
    )

    quantizers = _candidate_quantizers(module, config)

    assert all(quantizer.is_enabled for quantizer in quantizers.values())
    assert all(not quantizer._use_constant_amax for quantizer in quantizers.values())


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


def test_kv_autoquant_scores_and_applies_one_format_per_layer(tmp_path):
    torch.manual_seed(123)
    model = _ToyKVModel()
    data = [torch.randn(2, 3, 8), torch.randn(2, 3, 8)]
    candidates = [
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": 8, "constant_amax": 1.0},
                    }
                ],
                "algorithm": None,
                "effective_bits": 8.0,
            },
            "int8",
        ),
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": 4, "constant_amax": 1.0},
                    }
                ],
                "algorithm": None,
                "effective_bits": 4.0,
            },
            "int4",
        ),
    ]

    model, state = auto_quantize_kv_cache(
        model,
        {"kv_effective_bits": 6.0},
        candidates,
        data,
        lambda model, batch: model(batch),
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=str(tmp_path / "kv_search.pth"),
    )

    assert state["best"]["effective_bits"] == pytest.approx(6.0)
    assert state["best"]["is_satisfied"]
    assert model.training
    assert {layer["selected"] for layer in state["layers"].values()} == {
        "int8",
        "int4",
    }
    for layer_name, layer_state in state["layers"].items():
        layer = model.get_submodule(layer_name)
        assert layer.k_bmm_quantizer.num_bits == layer.v_bmm_quantizer.num_bits
        expected_bits = 8 if layer_state["selected"] == "int8" else 4
        assert layer.k_bmm_quantizer.num_bits == expected_bits

    restored_model = _ToyKVModel().eval()
    restored_model, restored_state = auto_quantize_kv_cache(
        restored_model,
        {"kv_effective_bits": 6.0},
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
        expected_bits = 8 if layer_state["selected"] == "int8" else 4
        assert layer.k_bmm_quantizer.num_bits == expected_bits


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
                        "cfg": {"num_bits": 4, "constant_amax": 1.0},
                    }
                ],
                "algorithm": None,
                "effective_bits": 4.0,
            },
            "int4",
        )
    ]

    with pytest.raises(ValueError, match="non-empty vocabulary dimension"):
        auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 4.0},
            candidates,
            [torch.randn(2, 3, 8)],
            lambda *_: torch.ones(8),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert model.training
    for name, module in (("attn0", model.attn0), ("attn1", model.attn1)):
        assert (module.k_bmm_quantizer, module.v_bmm_quantizer) == original_quantizers[name]


def test_public_kv_autoquant_converts_hf_attention_and_searches(tmp_path):
    torch.manual_seed(123)
    model = get_tiny_llama(num_hidden_layers=2)
    data = [{"input_ids": torch.randint(0, model.config.vocab_size, (1, 8))} for _ in range(2)]
    candidates = [
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": 8},
                    }
                ],
                "algorithm": "max",
                "effective_bits": 8.0,
            },
            "int8",
        ),
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": 4, "constant_amax": 1.0},
                    }
                ],
                "algorithm": None,
                "effective_bits": 4.0,
            },
            "int4",
        ),
    ]

    model, state = mtq.auto_quantize_kv_cache(
        model,
        {"kv_effective_bits": 6.0},
        candidates,
        data,
        lambda search_model, batch: search_model(**batch).logits,
        num_calib_steps=2,
        num_score_steps=2,
        checkpoint=str(tmp_path / "hf_kv_search.pth"),
    )

    assert len(state["layers"]) == model.config.num_hidden_layers
    assert state["best"]["effective_bits"] == pytest.approx(6.0)
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
    exported_quantization = get_quant_config(model)["quantization"]
    assert exported_quantization["quantized_layers"] == {}
    assert exported_quantization["kv_cache_quantized_layers"]
    assert set(exported_quantization["kv_cache_quantized_layers"]) <= {
        f"model.layers.{idx}.self_attn" for idx in range(model.config.num_hidden_layers)
    }

    restored_model = get_tiny_llama(num_hidden_layers=2)
    restored_model, restored_state = mtq.auto_quantize_kv_cache(
        restored_model,
        {"kv_effective_bits": 6.0},
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
        if layer.self_attn.k_bmm_quantizer.num_bits == 8
    )


def test_public_kv_autoquant_validation_and_runtime_failures_are_atomic():
    model = get_tiny_llama(num_hidden_layers=1)
    original_types = {name: type(module) for name, module in model.named_modules()}
    invalid_candidate = _kv_config((4, 3), 4.5).model_dump()
    invalid_candidate["algorithm"] = "svdquant"

    with pytest.raises(ValueError, match="only non-structural calibration algorithms"):
        mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 4.5},
            [invalid_candidate],
            [],
            lambda *_: pytest.fail("Validation must run before model conversion."),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert not hasattr(model, "_modelopt_state")
    assert {name: type(module) for name, module in model.named_modules()} == original_types

    valid_candidate = _kv_config((4, 3), 4.5).model_dump()
    valid_candidate["algorithm"] = None
    data = [{"input_ids": torch.randint(0, model.config.vocab_size, (1, 8))}]
    with pytest.raises(ValueError, match="non-empty vocabulary dimension"):
        mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 4.5},
            [valid_candidate],
            data,
            lambda *_: torch.ones(8),
            num_calib_steps=1,
            num_score_steps=1,
        )

    assert not hasattr(model, "_modelopt_state")
    assert {name: type(module) for name, module in model.named_modules()} == original_types


def test_public_kv_autoquant_preserves_fixed_layers_and_weight_quantizers():
    torch.manual_seed(123)
    model = get_tiny_llama(num_hidden_layers=2)
    data = [{"input_ids": torch.randint(0, model.config.vocab_size, (1, 8))}]
    fixed_kv_config = {
        "quant_cfg": [
            {
                "quantizer_name": "*[kv]_bmm_quantizer",
                "cfg": {"num_bits": 8, "constant_amax": 1.0},
            }
        ],
        "algorithm": None,
    }
    model = mtq.quantize(model, fixed_kv_config)
    fixed_weight_quantizer = model.model.layers[0].self_attn.q_proj.weight_quantizer
    fixed_weight_quantizer.enable()
    assert not hasattr(fixed_weight_quantizer, "_amax")
    observed_fixed_states = []
    hook = fixed_weight_quantizer.register_forward_hook(
        lambda module, _inputs, _output: observed_fixed_states.append(
            (module.is_enabled, module._if_quant, module._if_calib)
        )
    )

    try:
        model, state = mtq.auto_quantize_kv_cache(
            model,
            {"kv_effective_bits": 4.0},
            [
                (
                    {
                        "quant_cfg": [
                            {
                                "quantizer_name": "*[kv]_bmm_quantizer",
                                "cfg": {"num_bits": 4},
                            }
                        ],
                        "algorithm": "max",
                        "effective_bits": 4.0,
                    },
                    "int4",
                )
            ],
            data,
            lambda search_model, batch: search_model(**batch).logits,
            num_calib_steps=1,
            num_score_steps=1,
            disabled_layers="model.layers.1.self_attn",
        )
    finally:
        hook.remove()

    assert set(state["layers"]) == {"model.layers.0.self_attn"}
    assert observed_fixed_states
    assert all(state == (True, True, False) for state in observed_fixed_states)
    assert model.model.layers[0].self_attn.k_bmm_quantizer.num_bits == 4
    assert model.model.layers[0].self_attn.q_proj.weight_quantizer.is_enabled
    assert not hasattr(model.model.layers[0].self_attn.q_proj.weight_quantizer, "_amax")
    fixed_attention = model.model.layers[1].self_attn
    assert fixed_attention.k_bmm_quantizer.is_enabled
    assert fixed_attention.v_bmm_quantizer.is_enabled
    assert fixed_attention.k_bmm_quantizer.num_bits == 8
    assert fixed_attention.v_bmm_quantizer.num_bits == 8
