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
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Lock
from types import MethodType, ModuleType, SimpleNamespace

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
from packaging.version import Version

import modelopt.torch.quantization as mtq
from modelopt.recipe.loader import load_recipe
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.quantization.nn import QuantLinear, QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.huggingface import (
    CompressedLinearCompat,
    _adapt_compressed_tensors_packed_linears,
    _QuantCompressedLinear,
    _reconcile_compressed_tensors_config,
    _TransposedExpertsCalibMixin,
    get_homogeneous_hf_decoder_layers,
    is_homogeneous_hf_model,
    patch_compressed_linear_loading,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

pytest.importorskip("transformers")

import transformers
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.integrations.finegrained_fp8 import FP8Linear
from transformers.models.dbrx.configuration_dbrx import DbrxConfig, DbrxFFNConfig
from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU, DbrxExperts, DbrxFFN


class HFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # initialization is (out_features, in_features) instead of (in_features, out_features)
            transformers.pytorch_utils.Conv1D(5, 3),
            nn.ReLU(),
            transformers.pytorch_utils.Conv1D(5, 5),
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
    Version(transformers.__version__) < Version("5.0"),
    reason="test_dbrx is not supported for transformers<5.0",
)
def test_dbrx():
    assert DbrxExperts in QuantModuleRegistry
    assert DbrxExpertGLU in QuantModuleRegistry

    config = DbrxConfig(
        ffn_config=DbrxFFNConfig(ffn_hidden_size=8, moe_num_experts=2, hidden_size=32),
        hidden_size=32,
    )

    model_ref = DbrxFFN(config)
    model_test = DbrxFFN(config)
    with torch.no_grad():
        model_ref.experts.mlp.w1.copy_(torch.randn(16, 32))
        model_ref.experts.mlp.v1.copy_(torch.randn(16, 32))
        model_ref.experts.mlp.w2.copy_(torch.randn(16, 32))

    model_test.load_state_dict(model_ref.state_dict())

    mtq.replace_quant_module(model_test)

    expertglu_ref = model_ref.experts.mlp
    expertglu_test = model_test.experts.mlp

    assert hasattr(expertglu_test, "w1_linear") and not hasattr(expertglu_test, "w1")
    assert hasattr(expertglu_test, "v1_linear") and not hasattr(expertglu_test, "v1")
    assert hasattr(expertglu_test, "w2_linear") and not hasattr(expertglu_test, "w2")

    # Weights are stored transposed (W = w1[i].T) to match F.linear semantics with
    # transformers 5.0's raw matmul: x @ w1[i] = F.linear(x, w1[i].T)
    assert torch.allclose(
        torch.concat([m.weight.T for m in expertglu_test.w1_linear], dim=0),
        expertglu_ref.w1,
    )

    mtq.set_quantizer_attributes_partial(model_test, "*", {"enable": False})

    # In transformers 5.0, the FFN input dimension is ffn_hidden_size (not hidden_size)
    x = torch.randn(1, 4, 8)
    out_1 = model_ref(x)
    out_2 = model_test(x)
    assert torch.allclose(out_1[0], out_2[0])


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
    if model_provider == get_tiny_qwen3_moe and Version(torch.__version__) < Version("2.9"):
        pytest.skip("torch 2.8 grouped_mm is CUDA-only")

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


def _install_fake_compressed_tensors_loading_modules(monkeypatch):
    compressed_tensors = ModuleType("compressed_tensors")
    compressed_tensors.__path__ = []
    quantization = ModuleType("compressed_tensors.quantization")
    linear = ModuleType("compressed_tensors.linear")
    linear.__path__ = []
    compressed_linear_module = ModuleType("compressed_tensors.linear.compressed_linear")
    utils = ModuleType("compressed_tensors.utils")
    utils.__path__ = []
    match = ModuleType("compressed_tensors.utils.match")

    class CompressedLinear(nn.Module):
        pass

    apply_calls = []
    apply_calls_lock = Lock()

    def apply_quantization_config(model, config, *args, **kwargs):
        with apply_calls_lock:
            apply_calls.append((model, config, args, kwargs))
        return config

    def is_match(name, _module, patterns):
        return name in patterns

    quantization.apply_quantization_config = apply_quantization_config
    compressed_linear_module.CompressedLinear = CompressedLinear
    match.is_match = is_match
    compressed_tensors.quantization = quantization
    compressed_tensors.linear = linear
    compressed_tensors.utils = utils
    linear.compressed_linear = compressed_linear_module
    utils.match = match

    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed_tensors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.quantization", quantization)
    monkeypatch.setitem(sys.modules, "compressed_tensors.linear", linear)
    monkeypatch.setitem(
        sys.modules, "compressed_tensors.linear.compressed_linear", compressed_linear_module
    )
    monkeypatch.setitem(sys.modules, "compressed_tensors.utils", utils)
    monkeypatch.setitem(sys.modules, "compressed_tensors.utils.match", match)
    return quantization, CompressedLinear, apply_calls


def _get_loading_patch_test_model():
    model = nn.Module()
    model.outer_dense = nn.Linear(32, 32, bias=False)
    model.inner_dense = nn.Linear(32, 32, bias=False)
    return model


def test_compressed_linear_loading_patch_nested_contexts_and_exceptions(monkeypatch):
    quantization, compressed_linear, apply_calls = _install_fake_compressed_tensors_loading_modules(
        monkeypatch
    )
    original_apply = quantization.apply_quantization_config
    assert "__getattr__" not in compressed_linear.__dict__
    assert "_modelopt_init_patched" not in compressed_linear.__dict__

    model = _get_loading_patch_test_model()
    outer_keys = {"outer_dense.weight", "inner_dense.weight_packed"}
    inner_keys = {"outer_dense.weight_packed", "inner_dense.weight"}
    outer_config = SimpleNamespace(format="pack-quantized", ignore=[])
    inner_config = SimpleNamespace(format="pack-quantized", ignore=[])

    with patch_compressed_linear_loading(outer_keys):
        dispatcher = quantization.apply_quantization_config
        with (
            pytest.raises(RuntimeError, match="inner failure"),
            patch_compressed_linear_loading(inner_keys),
        ):
            assert quantization.apply_quantization_config is dispatcher
            quantization.apply_quantization_config(model, inner_config)
            raise RuntimeError("inner failure")

        assert quantization.apply_quantization_config is dispatcher
        quantization.apply_quantization_config(model, outer_config)

    assert inner_config.ignore == ["inner_dense"]
    assert outer_config.ignore == ["outer_dense"]
    assert len(apply_calls) == 2
    assert quantization.apply_quantization_config is original_apply
    assert "__getattr__" not in compressed_linear.__dict__
    assert "_modelopt_init_patched" not in compressed_linear.__dict__

    with (
        pytest.raises(ValueError, match="outer failure"),
        patch_compressed_linear_loading(outer_keys),
    ):
        raise ValueError("outer failure")
    assert quantization.apply_quantization_config is original_apply
    assert "__getattr__" not in compressed_linear.__dict__
    assert "_modelopt_init_patched" not in compressed_linear.__dict__


def test_compressed_linear_loading_patch_isolates_concurrent_contexts(monkeypatch, caplog):
    quantization, compressed_linear, apply_calls = _install_fake_compressed_tensors_loading_modules(
        monkeypatch
    )
    original_apply = quantization.apply_quantization_config
    model = _get_loading_patch_test_model()
    contexts_ready = Barrier(3)
    dispatches_complete = Barrier(3)

    def load_with_schema(checkpoint_weight_keys):
        config = SimpleNamespace(format="pack-quantized", ignore=[])
        with patch_compressed_linear_loading(checkpoint_weight_keys):
            contexts_ready.wait(timeout=10)
            quantization.apply_quantization_config(model, config)
            dispatches_complete.wait(timeout=10)
        return config.ignore

    with ThreadPoolExecutor(max_workers=2) as executor:
        outer_future = executor.submit(
            load_with_schema, {"outer_dense.weight", "inner_dense.weight_packed"}
        )
        inner_future = executor.submit(
            load_with_schema, {"outer_dense.weight_packed", "inner_dense.weight"}
        )
        contexts_ready.wait(timeout=10)

        # This thread has no ContextVar value and cannot safely select between the
        # two active schemas. It must skip reconciliation instead of choosing one.
        ambiguous_config = SimpleNamespace(format="pack-quantized", ignore=[])
        with caplog.at_level("WARNING"):
            quantization.apply_quantization_config(model, ambiguous_config)
        dispatches_complete.wait(timeout=10)

        assert outer_future.result() == ["outer_dense"]
        assert inner_future.result() == ["inner_dense"]

    assert ambiguous_config.ignore == []
    assert "multiple compressed-tensors loads are active" in caplog.text
    assert len(apply_calls) == 3
    assert quantization.apply_quantization_config is original_apply
    assert "__getattr__" not in compressed_linear.__dict__
    assert "_modelopt_init_patched" not in compressed_linear.__dict__


def test_compressed_linear_loading_patch_does_not_clobber_later_patch(monkeypatch, caplog):
    quantization, compressed_linear, _apply_calls = (
        _install_fake_compressed_tensors_loading_modules(monkeypatch)
    )

    def third_party_apply(model, config):
        return config

    def third_party_getattr(self, name):
        raise AttributeError(name)

    with caplog.at_level("WARNING"), patch_compressed_linear_loading({"outer_dense.weight"}):
        quantization.apply_quantization_config = third_party_apply
        compressed_linear.__getattr__ = third_party_getattr

    assert quantization.apply_quantization_config is third_party_apply
    assert compressed_linear.__dict__["__getattr__"] is third_party_getattr
    assert "another patch replaced ModelOpt's dispatcher" in caplog.text
    assert "another patch replaced ModelOpt's implementation" in caplog.text
    assert "_modelopt_init_patched" not in compressed_linear.__dict__


def test_compressed_linear_calibrates_transient_packed_weight(monkeypatch):
    """Weight-only calibration must cover packed experts that receive no tokens."""

    class _QuantizationStatus:
        COMPRESSED = object()
        FROZEN = object()

    compressed_tensors = ModuleType("compressed_tensors")
    compressed_tensors.__path__ = []
    quantization = ModuleType("compressed_tensors.quantization")
    quantization.QuantizationStatus = _QuantizationStatus
    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed_tensors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.quantization", quantization)

    expected_weight = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 8

    class _Compressor:
        def __init__(self):
            self.calls = 0

        def decompress_weight(self, compressed_data, quantization_args):
            self.calls += 1
            assert compressed_data["weight_shape"] == [4, 16]
            assert quantization_args == "int4-args"
            return expected_weight

    class _PackedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight_packed", torch.zeros(4, 2, dtype=torch.int32))
            self.register_buffer("weight_scale", torch.ones(4, 1))
            self.register_buffer("weight_shape", torch.tensor([4, 16], dtype=torch.int32))
            self.bias = None
            self.compressor = _Compressor()
            self.quantization_scheme = SimpleNamespace(weights="int4-args")
            self.quantization_status = _QuantizationStatus.COMPRESSED

        def forward(self, input):
            raise AssertionError("calibration must not route through this expert")

    class _QuantPackedLinear(_QuantCompressedLinear, _PackedLinear):
        pass

    module = _QuantPackedLinear.convert(_PackedLinear())
    module.input_quantizer.disable()
    module.weight_quantizer.num_bits = (2, 1)
    module.weight_quantizer.block_sizes = {
        -1: 16,
        "type": "dynamic",
        "scale_bits": (4, 3),
    }

    max_calibrate(module, distributed_sync=False)

    assert module.compressor.calls > 0
    assert not hasattr(module, "weight")
    assert hasattr(module.weight_quantizer, "_amax")
    torch.testing.assert_close(module.weight_quantizer._amax, expected_weight.abs().max())

    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    assert torch.isfinite(
        NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(module.weight_quantizer)
    )


def test_compressed_tensors_v1_real_packed_linear_calibrates_without_forward():
    compressed_tensors = pytest.importorskip("compressed_tensors")
    if Version(compressed_tensors.__version__) >= Version("0.15"):
        pytest.skip("compressed-tensors no longer supports the legacy CompressedLinear API")

    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.linear.compressed_linear import CompressedLinear
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    expected_weight = torch.arange(64, dtype=torch.float32).reshape(4, 16).remainder(15) - 7
    scale = torch.ones(4, 1)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            strategy="channel",
            symmetric=True,
        ),
        format="pack-quantized",
    )
    compressor = BaseCompressor.load_from_registry("pack-quantized")
    compressed_data = compressor.compress_weight(
        weight=expected_weight,
        scale=scale,
        quantization_args=scheme.weights,
    )

    packed = CompressedLinear.from_linear(
        nn.Linear(16, 4, bias=False),
        quantization_scheme=scheme,
        quantization_format="pack-quantized",
    )
    with torch.no_grad():
        packed.weight_packed.copy_(compressed_data["weight_packed"])
        packed.weight_shape.copy_(compressed_data["weight_shape"])
        packed.weight_scale.copy_(scale)
    assert not hasattr(packed, "weight")

    model = nn.Sequential(packed)
    with pytest.warns(UserWarning, match="monkey patched forward"):
        mtq.replace_quant_module(model)
    packed = model[0]
    packed.input_quantizer.disable()
    packed.weight_quantizer.num_bits = (2, 1)
    packed.weight_quantizer.block_sizes = {
        -1: 16,
        "type": "dynamic",
        "scale_bits": (4, 3),
    }

    max_calibrate(packed, distributed_sync=False)

    assert not hasattr(packed, "weight")
    assert hasattr(packed.weight_quantizer, "_amax")
    torch.testing.assert_close(packed.weight_quantizer._amax, expected_weight.abs().max())

    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    assert torch.isfinite(
        NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(packed.weight_quantizer)
    )


def test_reconcile_compressed_tensors_config_uses_checkpoint_schema(monkeypatch):
    """A stale config must not try to group-pack a checkpoint-dense 4304-column weight."""
    compressed_tensors = ModuleType("compressed_tensors")
    compressed_tensors.__path__ = []
    utils = ModuleType("compressed_tensors.utils")
    utils.__path__ = []
    match = ModuleType("compressed_tensors.utils.match")
    match.is_match = lambda name, _module, patterns: (
        name in patterns or (name == "mm_projector" and "re:mm_projector.*" in patterns)
    )
    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed_tensors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.utils", utils)
    monkeypatch.setitem(sys.modules, "compressed_tensors.utils.match", match)

    model = nn.Module()
    model.vision_tower = nn.Module()
    model.vision_tower.fc1 = nn.Linear(4304, 32, bias=False)
    model.mm_projector = nn.Linear(32, 32, bias=False)
    model.expert = nn.Linear(32, 64, bias=False)
    config = SimpleNamespace(format="pack-quantized", ignore=["re:mm_projector.*"])
    checkpoint_weight_keys = {
        "vision_tower.fc1.weight",
        "mm_projector.weight",
        "expert.weight_packed",
        "expert.weight_scale",
    }

    added = _reconcile_compressed_tensors_config(model, config, checkpoint_weight_keys)

    assert added == ["vision_tower.fc1"]
    assert config.ignore == ["re:mm_projector.*", "vision_tower.fc1"]
    assert _reconcile_compressed_tensors_config(model, config, checkpoint_weight_keys) == []


def test_adapt_compressed_tensors_packed_linears_removes_full_decompress_hook(monkeypatch):
    class _QuantizationStatus:
        COMPRESSED = object()

    expected_weight = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 16
    calls = {}

    class _PackedCompressor:
        @staticmethod
        def decompress(compressed_data, scheme):
            calls["compressed_data"] = compressed_data
            calls["scheme"] = scheme
            return {"weight": expected_weight}

    class _BaseCompressor:
        @staticmethod
        def get_value_from_registry(compression_format):
            calls["format"] = compression_format
            return _PackedCompressor

    compressed_tensors = ModuleType("compressed_tensors")
    compressed_tensors.__path__ = []
    compressors = ModuleType("compressed_tensors.compressors")
    compressors.BaseCompressor = _BaseCompressor
    quantization = ModuleType("compressed_tensors.quantization")
    quantization.QuantizationStatus = _QuantizationStatus
    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed_tensors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.compressors", compressors)
    monkeypatch.setitem(sys.modules, "compressed_tensors.quantization", quantization)

    model = nn.Sequential(nn.Linear(16, 4, bias=False))
    packed = model[0]
    packed.register_buffer("weight_packed", torch.zeros(4, 2, dtype=torch.int32))
    packed.register_buffer("weight_scale", torch.ones(4, 1))
    packed.register_buffer("weight_shape", torch.tensor([4, 16], dtype=torch.int32))
    del packed._parameters["weight"]
    packed.quantization_scheme = SimpleNamespace(format="pack-quantized")
    packed.quantization_status = _QuantizationStatus.COMPRESSED

    def compressed_tensors_forward(self, input):
        raise AssertionError("the compressed-tensors instance forward must be replaced")

    packed.forward = MethodType(compressed_tensors_forward, packed)

    def ct_decompress_hook(module, args):
        raise AssertionError("the full-model decompression hook must be removed")

    model.ct_decompress_hook = model.register_forward_pre_hook(ct_decompress_hook)

    assert _adapt_compressed_tensors_packed_linears(model) == 1
    assert isinstance(packed, CompressedLinearCompat)
    assert packed.forward.__func__ is type(packed).forward
    assert not hasattr(model, "ct_decompress_hook")
    assert not model._forward_pre_hooks

    inputs = torch.randn(2, 16)
    torch.testing.assert_close(model(inputs), nn.functional.linear(inputs, expected_weight))
    assert calls["format"] == "pack-quantized"
    assert calls["scheme"] is packed.quantization_scheme
    assert not hasattr(packed, "weight")

    with warnings.catch_warnings(record=True) as caught:
        mtq.replace_quant_module(model)
    assert type(packed).__name__ == "QuantCompressedLinearCompat"
    assert not any("monkey patched forward" in str(item.message) for item in caught)


def test_compressed_tensors_v2_adapter_executes_real_packed_linear():
    compressed_tensors = pytest.importorskip("compressed_tensors")
    if Version(compressed_tensors.__version__) < Version("0.15"):
        pytest.skip("compressed-tensors uses the legacy CompressedLinear API")

    from compressed_tensors.compressors import PackedQuantizationCompressor
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStatus,
    )
    from compressed_tensors.quantization.lifecycle.forward import set_forward_quantized

    packed = nn.Linear(16, 4, bias=False)
    expected_weight = torch.arange(64, dtype=torch.float32).reshape(4, 16).remainder(15) - 7
    with torch.no_grad():
        packed.weight.copy_(expected_weight)
    packed.register_parameter("weight_scale", nn.Parameter(torch.ones(4, 1), requires_grad=False))
    packed.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            strategy="channel",
            symmetric=True,
        ),
        format="pack-quantized",
    )
    packed.quantization_status = QuantizationStatus.FROZEN
    set_forward_quantized(packed)
    PackedQuantizationCompressor.compress_module(packed)

    model = nn.Sequential(packed)

    def ct_decompress_hook(module, args):
        raise AssertionError("the full-model decompression hook must be removed")

    model.ct_decompress_hook = model.register_forward_pre_hook(ct_decompress_hook)

    assert _adapt_compressed_tensors_packed_linears(model) == 1
    assert isinstance(packed, CompressedLinearCompat)
    assert packed.quantization_status == QuantizationStatus.COMPRESSED
    assert not hasattr(packed, "weight")

    inputs = torch.randn(2, 16)
    torch.testing.assert_close(model(inputs), nn.functional.linear(inputs, expected_weight))
    assert not hasattr(packed, "weight")


def test_compressed_tensors_v2_adapter_preserves_accelerate_forward_wrapper():
    model = nn.Sequential(nn.Linear(16, 4, bias=False))
    packed = model[0]
    packed.register_buffer("weight_packed", torch.zeros(4, 2, dtype=torch.int32))

    def compressed_tensors_forward(self, input):
        raise AssertionError("the compressed-tensors forward must be replaced")

    def accelerate_forward(self, input):
        return self._old_forward(input)

    packed._hf_hook = object()
    packed._old_forward = MethodType(compressed_tensors_forward, packed)
    packed.forward = MethodType(accelerate_forward, packed)
    accelerate_wrapper = packed.forward

    assert _adapt_compressed_tensors_packed_linears(model) == 1
    assert packed.forward is accelerate_wrapper
    assert packed._old_forward.__func__ is CompressedLinearCompat.forward


def test_hf_decoder_discoverer_registration_path():
    model = get_tiny_llama()
    assert any(
        is_supported is is_homogeneous_hf_model and discoverer is get_homogeneous_hf_decoder_layers
        for is_supported, discoverer in LayerActivationCollector._decoder_layer_support
    )
    assert LayerActivationCollector.get_decoder_layers(model) is get_homogeneous_hf_decoder_layers(
        model
    )
