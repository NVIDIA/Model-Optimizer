# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
import gc
import os
import warnings
import weakref
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.misc import set_seed
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    get_tiny_gpt_oss,
    get_tiny_llama,
    get_tiny_nemotron_h,
    get_tiny_qwen3_moe,
    tf_modelopt_state_and_output_tester,
)

import modelopt.torch.quantization as mtq
from modelopt.recipe.loader import load_recipe
from modelopt.torch.quantization.algorithms import _AutoQuantizeBaseSearcher
from modelopt.torch.quantization.config import LayerwiseConfig
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.nn import (
    QuantLinear,
    QuantModuleRegistry,
    SequentialQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.plugins.huggingface import (
    _QuantHFParallelLinear,
    _TransposedExpertsCalibMixin,
    convert_hf_parallel_linears_on_the_fly,
    get_homogeneous_hf_decoder_layers,
    is_homogeneous_hf_model,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan

pytest.importorskip("transformers")

import transformers
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.integrations.finegrained_fp8 import FP8Linear


class HFModel(nn.Module):
    def __init__(self, in_features=3, out_features=5):
        super().__init__()
        self.net = nn.Sequential(
            # initialization is (out_features, in_features) instead of (in_features, out_features)
            transformers.pytorch_utils.Conv1D(out_features, in_features),
            nn.ReLU(),
            transformers.pytorch_utils.Conv1D(out_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class PytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            QuantLinear(3, 5),
            nn.ReLU(),
            QuantLinear(5, 5),
        )

    def forward(self, x):
        return self.net(x)


def test_hf_parallel_weight_access_restores_dtensor_after_exception(monkeypatch):
    class FakeDTensor:
        placements = ("shard",)

        @staticmethod
        def to_local():
            return torch.ones(2, 2)

    class ParallelLinear:
        weight = FakeDTensor()
        shard = FakeDTensor.placements

    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)
    linear = ParallelLinear()
    original_weight = linear.weight

    with (
        pytest.raises(RuntimeError, match="test error"),
        _QuantHFParallelLinear.enable_weight_access_and_writeback(linear),
    ):
        assert isinstance(linear.weight, nn.Parameter)
        raise RuntimeError("test error")

    assert linear.weight is original_weight


@pytest.mark.parametrize("expert_parallel", [False, True], ids=["no-plan", "expert-parallel"])
def test_a_mesh_without_tp_sharded_linears_warns_unless_the_model_is_expert_parallel(
    expert_parallel,
):
    """A transformers mesh with no TP-sharded Linear under it is worth a warning, except on a model
    loaded expert-parallel: its mesh is the experts', and no Linear is meant to be sharded."""
    model = nn.Sequential(nn.Linear(4, 4))
    model._device_mesh = object()
    if expert_parallel:
        set_distribution_plan(model, DistributionPlan(2, ep_size=2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert_hf_parallel_linears_on_the_fly(model)

    warned = any("no TP-sharded linear layer" in str(w.message) for w in caught)
    assert warned is not expert_parallel


def test_convert_conv1d():
    set_seed()
    assert transformers.pytorch_utils.Conv1D in QuantModuleRegistry

    model_ref = HFModel()
    model_test = HFModel()
    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)
    for name, module in model_test.named_modules():
        if isinstance(module, transformers.pytorch_utils.Conv1D):
            assert hasattr(module, "input_quantizer")
            assert hasattr(module, "weight_quantizer")
            assert hasattr(module, "output_quantizer")

    mtq.set_quantizer_attributes_partial(model_test, "*", {"enable": False})

    x = torch.randn(2, 3)
    out_1 = model_ref(x)
    out_2 = model_test(x)

    assert torch.allclose(out_1, out_2)

    mtq.set_quantizer_attributes_partial(model_test, "*input_quantizer", {"enable": True})
    mtq.set_quantizer_attributes_partial(model_test, "*weight_quantizer", {"enable": True})
    model_ref = PytorchModel()
    model_ref.load_state_dict(model_test.state_dict())

    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1, out_2)


_Q8_0_GPTQ_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": "q8_0",
                "block_sizes": {-1: 32},
                "backend": "ggml",
                "effective_bits": GGML_FORMAT_REGISTRY["q8_0"].effective_bits,
            },
        },
    ],
    "algorithm": {"method": "gptq", "block_size": 256},
}


@pytest.mark.parametrize("rewrites", [False, True], ids=["int8", "q8_0_gptq"])
def test_auto_quantize_frees_the_conv1d_weights_conversion_replaces(monkeypatch, rewrites):
    # Conversion transposes each Conv1D weight into a new parameter. Only a search with a
    # weight-rewriting candidate keeps the old ones, to hand back on a refusal, and only until its
    # checks before calibration pass.
    model = HFModel(in_features=256, out_features=256)
    replaced = [
        weakref.ref(m.weight)
        for m in model.modules()
        if isinstance(m, transformers.pytorch_utils.Conv1D)
    ]
    alive = []

    def count_alive():
        gc.collect()
        alive.append(sum(ref() is not None for ref in replaced))

    before_search = _AutoQuantizeBaseSearcher.before_search

    def counting_before_search(self):
        count_alive()
        return before_search(self)

    def forward_step(model, batch):
        count_alive()
        return model(batch)

    monkeypatch.setattr(_AutoQuantizeBaseSearcher, "before_search", counting_before_search)
    mtq.auto_quantize(
        model,
        constraints={"effective_bits": 10.0},
        quantization_formats=[_Q8_0_GPTQ_CFG if rewrites else mtq.INT8_DEFAULT_CFG],
        data_loader=[torch.randn(8, 256) for _ in range(2)],
        forward_step=forward_step,
        loss_func=lambda output, data: output.sum(),
        num_calib_steps=2,
        num_score_steps=2,
    )
    after_conversion, *in_forwards = alive
    assert after_conversion == (len(replaced) if rewrites else 0)
    assert in_forwards and not any(in_forwards)


@pytest.mark.parametrize(
    "layerwise",
    [{"export_dir": "export"}, {"export_dir": ""}, {"checkpoint_dir": "checkpoint"}],
    ids=["export_dir", "empty_export_dir", "checkpoint_dir_beside_gptq"],
)
@pytest.mark.parametrize("as_config", [False, True], ids=["dict", "LayerwiseConfig"])
def test_auto_quantize_refuses_layerwise_dirs_before_converting(
    monkeypatch, tmp_path, layerwise, as_config
):
    # Exporting each calibrated layer would replace its weights with packed ones before scoring,
    # and an empty path exports into the working directory. Resuming the max candidate's layers
    # would load their weights into new Parameters, which the restore after GPTQ does not reach.
    monkeypatch.chdir(tmp_path)
    source = get_tiny_llama()
    model = copy.deepcopy(source)
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    layerwise = {"enable": True, **layerwise}
    config["algorithm"] = {
        "method": "max",
        "layerwise": LayerwiseConfig(**layerwise) if as_config else layerwise,
    }
    formats = [config, _Q8_0_GPTQ_CFG] if "checkpoint_dir" in layerwise else [config]
    input_ids = model.dummy_inputs["input_ids"]

    def search():
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 11.0},
            quantization_formats=formats,
            data_loader=[{"input_ids": input_ids} for _ in range(2)],
            forward_step=lambda model, batch: model(**batch),
            loss_func=lambda output, batch: output.logits.float().square().sum(),
            num_calib_steps=2,
            num_score_steps=2,
            # Otherwise layerwise export stops at the lm_head quantizer before exporting a layer.
            disabled_layers=["*lm_head*"],
        )

    with pytest.raises(NotImplementedError):
        search()

    assert [type(m) for m in model.modules()] == [type(m) for m in source.modules()]
    for weight, original in zip(model.parameters(), source.parameters()):
        assert torch.equal(weight, original)
    assert not any(tmp_path.iterdir())
    config["algorithm"] = "max"
    search()


def test_fp8_linear_per_tensor_dequant(monkeypatch):
    module = FP8Linear(2, 2, block_size=(128, 128))
    module.weight_scale_inv = nn.Parameter(torch.tensor(2.0))
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[-2.0, 1.0], [0.5, 4.0]], dtype=torch.float8_e4m3fn))

    mtq.replace_quant_module(module)
    monkeypatch.setattr("modelopt.torch.quantization.plugins.huggingface.weight_dequant", None)

    assert module.block_size is None
    torch.testing.assert_close(
        module._dequantize_weight(torch.float32), module.weight.float() * 2.0
    )


@pytest.mark.skipif(
    not hasattr(transformers, "NemotronHConfig"),
    reason="NemotronH is not supported by this Transformers version",
)
@pytest.mark.parametrize(
    "recipe_path",
    [
        "general/ptq/nvfp4_experts_only-kv_fp8",
        "general/ptq/nvfp4_experts_only-kv_fp8_cast",
        "general/ptq/nvfp4_experts_only-kv_fp8_layerwise",
        "general/ptq/nvfp4_experts_only_mse-kv_fp8_cast",
    ],
)
def test_nemotron_h_experts_only_recipes_target_routed_experts(recipe_path):
    model = get_tiny_nemotron_h(
        num_hidden_layers=1,
        hybrid_override_pattern="E",
        n_routed_experts=2,
    )
    mtq.replace_quant_module(model)

    recipe = load_recipe(recipe_path)
    mtq.set_quantizer_by_cfg(model, recipe.quantize.model_dump()["quant_cfg"])

    routed_expert_quantizers = {
        name: module
        for name, module in model.named_modules()
        if ".mixer.experts." in name and isinstance(module, TensorQuantizer)
    }
    shared_expert_quantizers = {
        name: module
        for name, module in model.named_modules()
        if ".mixer.shared_experts." in name and isinstance(module, TensorQuantizer)
    }

    assert routed_expert_quantizers
    assert all(module.is_enabled for module in routed_expert_quantizers.values())
    assert shared_expert_quantizers
    assert not any(module.is_enabled for module in shared_expert_quantizers.values())


@pytest.mark.parametrize("method", ["gradient", "kl_div"])
@pytest.mark.parametrize("model_provider", [get_tiny_llama, get_tiny_qwen3_moe])
def test_autoquantize_huggingface(model_provider, method):
    model = model_provider()
    input_ids = model.dummy_inputs["input_ids"]

    def forward_step(model, batch):
        return model(**batch) if method == "gradient" else model(**batch).logits

    warnings.filterwarnings(
        "error", message="AutoQuantize: Error enabling gradient checkpointing for huggingface model"
    )

    # Gradient checkpointing warning should only appear for gradient-based method
    context = (
        pytest.warns(
            UserWarning,
            match="AutoQuantize: Huggingface model detected - Enabling gradient checkpointing. ",
        )
        if method == "gradient"
        else nullcontext()
    )

    with context:
        best_model, search_history = mtq.auto_quantize(
            model,
            constraints={"effective_bits": 11.0},
            quantization_formats=[mtq.INT8_DEFAULT_CFG],
            data_loader=[{"input_ids": input_ids, "labels": input_ids} for _ in range(2)],
            forward_step=forward_step,
            loss_func=lambda output, data: output.loss,
            num_calib_steps=2,
            num_score_steps=2,
            verbose=True,
            method=method,
        )


@pytest.mark.parametrize(
    ("model_cls", "quant_config"),
    [
        (LlamaForCausalLM, mtq.INT4_AWQ_CFG),
        (AutoModelForCausalLM, mtq.INT4_AWQ_CFG),
    ],
)
def test_quantized_transformers_save_restore(tmp_path, model_cls, quant_config):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path, dtype=torch.float32)
    # update config to fit test cases
    if quant_config == mtq.INT4_AWQ_CFG:
        quant_config = copy.deepcopy(quant_config)
        for entry in quant_config["quant_cfg"]:
            if entry["quantizer_name"] == "*weight_quantizer":
                entry.setdefault("cfg", {})["block_sizes"] = {-1: 16}
                break
    else:
        raise ValueError(f"Unsupported quant_config: {quant_config}")

    model_ref = model_cls.from_pretrained(tiny_llama_dir)
    mtq.quantize(model_ref, quant_config, lambda model: model(**model.dummy_inputs))
    mtq.compress(model_ref)
    model_ref.save_pretrained(tiny_llama_dir / "modelopt_model")
    assert os.path.exists(tiny_llama_dir / "modelopt_model/modelopt_state.pth")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")
    tf_modelopt_state_and_output_tester(model_ref, model_test)


def test_is_homogeneous_hf_model_llama():
    model = get_tiny_llama()
    assert is_homogeneous_hf_model(model)


def test_is_homogeneous_hf_vlm_language_model():
    model = get_tiny_llama()
    language_model = nn.Module()
    language_model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    model.model.language_model = language_model

    assert is_homogeneous_hf_model(model)
    assert get_homogeneous_hf_decoder_layers(model) is language_model.layers


def test_is_homogeneous_hf_model_gpt_oss():
    model = get_tiny_gpt_oss(num_hidden_layers=1)
    assert is_homogeneous_hf_model(model)


def test_gpt_oss_experts_iter_weights_for_calibration_transposed():
    """``_QuantGptOssExperts`` quantizes its expert weights *transposed* in the forward
    (``_transposed_quantize`` puts the contraction ``in_dim`` last). Weight-only
    calibration must yield the same transposed view; otherwise the unconditional
    ``weight_only_quantize`` locks a non-transposed block-quant ``_original_shape`` and the
    calibration forward then raises "Input shape has changed" for static-block NVFP4.
    """
    # Use intermediate_size != hidden_size so both expert weights are non-square and the
    # transpose is observable in the shape.
    model = get_tiny_gpt_oss(num_hidden_layers=1, hidden_size=32, intermediate_size=48)
    mtq.replace_quant_module(model)
    experts = model.model.layers[0].mlp.experts
    assert hasattr(experts, "gate_up_proj_weight_quantizer")

    yielded = {q: w for w, q in experts.iter_weights_for_calibration()}
    # Stored weights are (num_experts, in_dim, out_dim); calibration must see (…, out_dim, in_dim).
    assert (
        yielded[experts.gate_up_proj_weight_quantizer].shape
        == experts.gate_up_proj.transpose(-1, -2).shape
    )
    assert (
        yielded[experts.down_proj_weight_quantizer].shape
        == experts.down_proj.transpose(-1, -2).shape
    )


def test_transposed_experts_calib_mixin_yields_transposed_views():
    """Unit-level guard for the shared ``_TransposedExpertsCalibMixin`` (no GPU / no model
    conversion needed): it must yield the transposed ``(num_experts, out, in)`` weight view
    paired with the matching weight quantizer, so weight-only calibration agrees with the
    experts' transposed forward (regression for the static-block "Input shape has changed").
    """

    class _FakeExperts(_TransposedExpertsCalibMixin):
        def __init__(self):
            # Non-square so the transpose is observable; (num_experts, in_dim, out_dim).
            self.gate_up_proj = torch.randn(8, 64, 192)
            self.down_proj = torch.randn(8, 96, 64)
            self.gate_up_proj_weight_quantizer = nn.Identity()
            self.down_proj_weight_quantizer = nn.Identity()

    experts = _FakeExperts()
    pairs = list(experts.iter_weights_for_calibration())

    assert len(pairs) == 2
    (gate_up_w, gate_up_q), (down_w, down_q) = pairs
    assert torch.equal(gate_up_w, experts.gate_up_proj.transpose(-1, -2))
    assert gate_up_q is experts.gate_up_proj_weight_quantizer
    assert torch.equal(down_w, experts.down_proj.transpose(-1, -2))
    assert down_q is experts.down_proj_weight_quantizer


@pytest.mark.parametrize("model_type", ["gpt_oss", "llama4"])
@pytest.mark.parametrize("keep_attrs", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offload", [None, "cpu", "disk"])
def test_fold_transposed_expert_weights_preserves_forward(
    model_type, keep_attrs, dtype, offload, tmp_path
):
    set_seed()
    if model_type == "gpt_oss":
        model = get_tiny_gpt_oss(num_hidden_layers=1, hidden_size=32, intermediate_size=48)
    else:
        model = AutoModelForCausalLM.from_config(
            transformers.Llama4TextConfig(
                num_hidden_layers=1,
                num_local_experts=4,
                hidden_size=32,
                intermediate_size=48,
                intermediate_size_mlp=48,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                vocab_size=32,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                no_rope_layers=[0],
            )
        )
    model = model.to(dtype=dtype).eval()
    input_ids = torch.randint(0, 32, (2, 7))
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*.experts.*weight_quantizer",
                "cfg": {"num_bits": 8, "axis": (0, 1)},
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, lambda m: m(input_ids))
    experts = next(
        module for module in model.modules() if isinstance(module, _TransposedExpertsCalibMixin)
    )
    pairs = list(experts.iter_weights_for_calibration())
    for weight, quantizer in pairs:
        quantizer.pre_quant_scale = torch.linspace(0.75, 1.25, weight.shape[-1], dtype=dtype)
    pointers = [weight.data_ptr() for weight, _ in pairs]

    with torch.no_grad():
        expected = model(input_ids).logits
        if offload is not None:
            # Accelerate is optional; only offload cases require it.
            accelerate = pytest.importorskip("accelerate")
            if offload == "cpu":
                accelerate.cpu_offload(experts, execution_device=torch.device("cpu"))
            else:
                accelerate.disk_offload(
                    experts, tmp_path / "offload", execution_device=torch.device("cpu")
                )
            assert experts.gate_up_proj.is_meta
            torch.testing.assert_close(model(input_ids).logits, expected)
        mtq.fold_weight(model, keep_attrs=keep_attrs)
        torch.testing.assert_close(model(input_ids).logits, expected)
        for (weight, quantizer), pointer in zip(pairs, pointers):
            if offload is None:
                assert weight.data_ptr() == pointer
            assert not quantizer.is_enabled
            assert quantizer.pre_quant_scale is None
            assert hasattr(quantizer, "_amax") == keep_attrs
            assert hasattr(quantizer, "_pre_quant_scale") == keep_attrs
        mtq.fold_weight(model, keep_attrs=keep_attrs)
        torch.testing.assert_close(model(input_ids).logits, expected)
        if offload is not None:
            assert experts.gate_up_proj.is_meta
            assert experts.down_proj.is_meta


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fold_transposed_expert_weights_with_sequential_quantizer(dtype):
    set_seed()
    model = get_tiny_gpt_oss(num_hidden_layers=1, hidden_size=32, intermediate_size=48)
    model = model.to(dtype=dtype).eval()
    input_ids = torch.randint(0, 32, (2, 7))
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*.experts.*weight_quantizer",
                "cfg": [{"num_bits": 8, "axis": (0, 1)}, {"num_bits": 4, "axis": (0, 1)}],
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, lambda m: m(input_ids))
    experts = next(
        module for module in model.modules() if isinstance(module, _TransposedExpertsCalibMixin)
    )
    quantizers = [quantizer for _, quantizer in experts.iter_weights_for_calibration()]
    assert all(isinstance(quantizer, SequentialQuantizer) for quantizer in quantizers)

    with torch.no_grad():
        expected = model(input_ids).logits
        mtq.fold_weight(model)
        torch.testing.assert_close(model(input_ids).logits, expected)
    assert not any(member.is_enabled for quantizer in quantizers for member in quantizer)


def test_hf_decoder_discoverer_registration_path():
    model = get_tiny_llama()
    assert any(
        is_supported is is_homogeneous_hf_model and discoverer is get_homogeneous_hf_decoder_layers
        for is_supported, discoverer in LayerActivationCollector._decoder_layer_support
    )
    assert LayerActivationCollector.get_decoder_layers(model) is get_homogeneous_hf_decoder_layers(
        model
    )
