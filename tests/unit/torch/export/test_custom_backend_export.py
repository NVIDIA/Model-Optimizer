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

"""Export of a model quantized by a custom quantization backend that an external package registers.

A toy backend fake-quantizes the MLP weights and registers an export handler that packs its linears
and declares the format it writes. The exporters that ``hf_ptq.py`` and ``llm_qat/export.py`` call
then export the model with no change to them, the built-in packing refuses the modules whose format
a handler declares, and the other exporters ignore the declared format.
"""

import json
from fnmatch import fnmatch
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")
from _test_utils.torch.transformers_models import get_tiny_gpt_oss, get_tiny_llama, get_tiny_mixtral
from safetensors.torch import load_file

import modelopt.torch.quantization as mtq
from modelopt.torch.export import (
    capture_quantized_weight_export_state,
    export_hf_checkpoint,
    export_hf_vllm_fq_checkpoint,
)
from modelopt.torch.export.quant_format import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_WO,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_INT8_WO,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_Q8_0,
    QUANTIZATION_W4A16_NVFP4,
)
from modelopt.torch.export.quant_utils import get_quant_config, get_quantization_format
from modelopt.torch.export.registry import ExportFormat, ExportModuleRegistry
from modelopt.torch.export.unified_export_hf import (
    _export_quantized_weight,
    _export_transformers_checkpoint,
)
from modelopt.torch.export.unified_export_megatron import GPTModelExporter
from modelopt.torch.quantization.nn import register_quant_backend, unregister_quant_backend
from modelopt.torch.quantization.qtensor import NVFP4QTensor

BACKEND = "toy_backend"
MLP_LINEARS = [
    f"model.layers.{i}.mlp.{proj}"
    for i in range(2)
    for proj in ("gate_proj", "up_proj", "down_proj")
]
TOY_CFG = {"num_bits": BACKEND, "backend": BACKEND, "axis": None}
NVFP4_BLOCK_SIZE = 16
# The formats that the toy backend's handler declares, with their block size and the quant_algo
# and group_size that its layers then take.
DECLARED_FORMATS = {
    QUANTIZATION_W4A16_NVFP4: (NVFP4_BLOCK_SIZE, "W4A16_NVFP4", NVFP4_BLOCK_SIZE),
    QUANTIZATION_INT4_AWQ: (128, "W4A16_AWQ", 128),
    QUANTIZATION_INT8_WO: (0, "W8A16", None),
    QUANTIZATION_FP8: (0, "FP8", None),
    QUANTIZATION_FP8_PB_WO: (128, "FP8_PB_WO", 128),
}

# The exporters as examples/hf_ptq/hf_ptq.py (with and without --vllm_fakequant_export) and
# examples/llm_qat/export.py call them.
EXPORTERS = {
    "export_hf_checkpoint": lambda model, export_dir: export_hf_checkpoint(
        model, export_dir=export_dir
    ),
    "export_hf_vllm_fq_checkpoint": lambda model, export_dir: export_hf_vllm_fq_checkpoint(
        model, export_dir=export_dir, inplace_mem_efficient=True
    ),
    "_export_transformers_checkpoint": lambda model, export_dir: _export_transformers_checkpoint(
        model
    ),
}
PACKING_EXPORTERS = ["export_hf_checkpoint", "_export_transformers_checkpoint"]


def _quantizer_cfg(config: dict, quantizer_name: str) -> dict:
    """Return the attributes of a quantizer of a built-in quantization config."""
    return next(
        entry["cfg"] for entry in config["quant_cfg"] if entry["quantizer_name"] == quantizer_name
    )


def _toy_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """The toy backend's fake quantization: round to a multiple of 1/64."""
    return torch.round(inputs * 64) / 64


def _quantize(
    model: nn.Module,
    quantizer_name: str = "*mlp*weight_quantizer",
    extra_cfg: tuple = (),
    algorithm: str | None = None,
    cfg: dict | None = None,
) -> nn.Module:
    """Quantize the weights that ``quantizer_name`` matches, with the toy backend by default."""
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": quantizer_name, "cfg": cfg or TOY_CFG},
        *extra_cfg,
    ]

    def forward_loop(model):
        model(torch.randint(model.config.vocab_size, (2, 8)))

    return mtq.quantize(
        model, {"quant_cfg": quant_cfg, "algorithm": algorithm}, forward_loop if algorithm else None
    )


def _has_toy_weights(module: nn.Module) -> bool:
    quantizer = getattr(module, "weight_quantizer", None)
    return quantizer is not None and quantizer.is_enabled and quantizer.backend == BACKEND


def _has_fused_experts(module: nn.Module) -> bool:
    return hasattr(module, "gate_up_proj_weight_quantizers")


def _excluded(name: str, quantization: dict) -> bool:
    return any(fnmatch(name, pattern) for pattern in quantization["exclude_modules"])


def _declare(predicate, quantization_format: str, block_size: int, prepend: bool = True):
    """Register a handler for the modules ``predicate`` matches that declares their format."""

    @ExportModuleRegistry.register(
        predicate=predicate,
        prepend=prepend,
        export_format=ExportFormat(quantization_format=quantization_format, block_size=block_size),
    )
    def _export_declared(name, module, ctx):
        raise AssertionError(f"{name} was not to be packed")

    return _export_declared


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Drop the handlers that a test registers."""
    monkeypatch.setattr(ExportModuleRegistry, "_entries", list(ExportModuleRegistry._entries))
    monkeypatch.setattr(ExportModuleRegistry, "_declared", dict(ExportModuleRegistry._declared))


@pytest.fixture
def toy_backend():
    register_quant_backend(BACKEND, _toy_fake_quant)
    yield
    unregister_quant_backend(BACKEND)


def _register_toy_packer(quantization_format: str, block_size: int) -> dict:
    """Pack the toy backend's linears as NVFP4 tensors; returns the tensors by module name."""
    packed = {}

    @ExportModuleRegistry.register(
        predicate=_has_toy_weights,
        prepend=True,
        export_format=ExportFormat(quantization_format=quantization_format, block_size=block_size),
    )
    def _export_toy_linear(name, module, ctx):
        out_features, in_features = module.weight.shape
        tensors = {
            "weight": torch.randint(256, (out_features, in_features // 2), dtype=torch.uint8),
            "weight_scale": torch.rand(out_features, in_features // NVFP4_BLOCK_SIZE).to(
                torch.float8_e4m3fn
            ),
            "weight_scale_2": torch.tensor(0.5),
        }
        module.weight = nn.Parameter(tensors["weight"], requires_grad=False)
        module.register_buffer("weight_scale", tensors["weight_scale"])
        module.register_buffer("weight_scale_2", tensors["weight_scale_2"])
        packed[name] = tensors

    return packed


@pytest.mark.parametrize("exporter", PACKING_EXPORTERS)
@pytest.mark.parametrize("quantization_format", sorted(DECLARED_FORMATS))
def test_exports_in_the_declared_format_with_its_handler(
    toy_backend, quantization_format, exporter, tmp_path
):
    block_size, quant_algo, group_size = DECLARED_FORMATS[quantization_format]
    packed = _register_toy_packer(quantization_format, block_size)
    result = EXPORTERS[exporter](_quantize(get_tiny_llama()), tmp_path)
    if exporter == "export_hf_checkpoint":
        state_dict = load_file(tmp_path / "model.safetensors")
        hf_quant_config = json.loads((tmp_path / "hf_quant_config.json").read_text())
    else:
        state_dict, hf_quant_config = result

    assert sorted(packed) == sorted(MLP_LINEARS)
    for name, tensors in packed.items():
        for key, tensor in tensors.items():
            exported = state_dict[f"{name}.{key}"]
            assert exported.dtype == tensor.dtype and torch.equal(exported.float(), tensor.float())
    quantization = hf_quant_config["quantization"]
    assert quantization["quant_algo"] == quant_algo
    assert quantization.get("group_size") == group_size
    assert not [name for name in MLP_LINEARS if _excluded(name, quantization)]
    # The rest of the model is exported unquantized, as usual.
    assert _excluded("model.layers.0.self_attn.q_proj", quantization)
    assert state_dict["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16


def _nvfp4_fake_quant(inputs: torch.Tensor, quantizer) -> torch.Tensor:
    """Weight-only NVFP4 fake quantization, in blocks of NVFP4_BLOCK_SIZE weights."""
    qtensor, scale, scale_2 = NVFP4QTensor.quantize(inputs, NVFP4_BLOCK_SIZE)
    return qtensor.dequantize(scale=scale, double_scale=scale_2, block_sizes={-1: NVFP4_BLOCK_SIZE})


def test_the_exported_weights_decode_to_the_backend_weights_by_the_config(tmp_path):
    register_quant_backend(BACKEND, _nvfp4_fake_quant)
    try:
        model = _quantize(get_tiny_llama())
        expected = {
            name: (module.weight.detach().clone(), module.weight_quantizer(module.weight).detach())
            for name, module in model.named_modules()
            if _has_toy_weights(module)
        }

        @ExportModuleRegistry.register(
            predicate=_has_toy_weights,
            prepend=True,
            export_format=ExportFormat(
                quantization_format=QUANTIZATION_W4A16_NVFP4, block_size=NVFP4_BLOCK_SIZE
            ),
        )
        def _export_nvfp4_linear(name, module, ctx):
            qtensor, scale, scale_2 = NVFP4QTensor.quantize(
                module.weight.detach(), NVFP4_BLOCK_SIZE
            )
            module.weight = nn.Parameter(qtensor._quantized_data, requires_grad=False)
            module.register_buffer("weight_scale", scale)
            module.register_buffer("weight_scale_2", scale_2)

        export_hf_checkpoint(model, export_dir=tmp_path)
    finally:
        unregister_quant_backend(BACKEND)

    state_dict = load_file(tmp_path / "model.safetensors")
    quantization = json.loads((tmp_path / "hf_quant_config.json").read_text())["quantization"]
    assert quantization["quant_algo"] == "W4A16_NVFP4"
    assert sorted(expected) == sorted(MLP_LINEARS)
    for name, (weight, fake_quantized) in expected.items():
        assert not torch.equal(fake_quantized, weight)
        decoded = NVFP4QTensor(weight.shape, weight.dtype, state_dict[f"{name}.weight"]).dequantize(
            scale=state_dict[f"{name}.weight_scale"],
            double_scale=state_dict[f"{name}.weight_scale_2"],
            block_sizes={-1: quantization["group_size"]},
        )
        assert torch.equal(decoded, fake_quantized)


@pytest.mark.parametrize(
    ("quantization_format", "quant_algo"),
    [(QUANTIZATION_NVFP4, "NVFP4"), (QUANTIZATION_W4A16_NVFP4, "W4A16_NVFP4")],
)
def test_the_declared_format_holds_whatever_the_input_quantizer(
    toy_backend, quantization_format, quant_algo
):
    nvfp4_input = _quantizer_cfg(mtq.NVFP4_DEFAULT_CFG, "*input_quantizer")
    model = _quantize(
        get_tiny_llama(),
        extra_cfg=({"quantizer_name": "*mlp*input_quantizer", "cfg": nvfp4_input},),
        algorithm="max",
    )
    _declare(_has_toy_weights, quantization_format, NVFP4_BLOCK_SIZE)

    with ExportModuleRegistry.reading_declared_formats():
        quantization = get_quant_config(model)["quantization"]
    assert quantization["quant_algo"] == quant_algo
    assert not [name for name in MLP_LINEARS if _excluded(name, quantization)]


@pytest.mark.parametrize("quantization_format", [QUANTIZATION_W4A16_NVFP4, QUANTIZATION_FP8_PB_WO])
def test_fused_experts_take_the_declared_format(toy_backend, quantization_format):
    model = _quantize(get_tiny_mixtral(), "*experts*weight_quantizer*")
    experts = [name for name, module in model.named_modules() if name.endswith(".mlp.experts")]
    assert experts
    assert all(_has_fused_experts(model.get_submodule(name)) for name in experts)
    block_size, quant_algo, group_size = DECLARED_FORMATS[quantization_format]
    _declare(_has_fused_experts, quantization_format, block_size)

    with ExportModuleRegistry.reading_declared_formats():
        quantization = get_quant_config(model)["quantization"]
    assert quantization["quant_algo"] == quant_algo
    assert quantization["group_size"] == group_size
    assert not [name for name in experts if _excluded(name, quantization)]


def test_a_disabled_weight_takes_no_declared_format(toy_backend):
    model = _quantize(get_tiny_llama())
    for layer in model.model.layers:
        layer.mlp.down_proj.weight_quantizer.disable()
    # Matches the down projections too, whose weight quantizer is disabled.
    _declare(lambda module: hasattr(module, "weight_quantizer"), QUANTIZATION_W4A16_NVFP4, 16)

    with ExportModuleRegistry.reading_declared_formats():
        quantization = get_quant_config(model)["quantization"]
    assert quantization["quant_algo"] == "W4A16_NVFP4"
    assert {name for name in MLP_LINEARS if _excluded(name, quantization)} == {
        f"model.layers.{i}.mlp.down_proj" for i in range(2)
    }


def test_the_built_in_packing_refuses_each_weight_of_a_declared_module(toy_backend):
    # GPT-OSS experts hold two weights, each with its own quantizer; down_proj's stays disabled.
    model = _quantize(
        get_tiny_gpt_oss(num_hidden_layers=1), "*experts.gate_up_proj_weight_quantizer"
    )
    experts = model.model.layers[0].mlp.experts
    assert not experts.down_proj_weight_quantizer.is_enabled
    _declare(lambda module: module is experts, QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE)

    with ExportModuleRegistry.reading_declared_formats():
        assert get_quantization_format(experts) == QUANTIZATION_W4A16_NVFP4
        # The packing that the built-in handler of these experts does for each weight, the
        # disabled one included, which would otherwise take its scales in the declared format.
        for weight_name in ("gate_up_proj", "down_proj"):
            with pytest.raises(NotImplementedError, match="built-in export cannot pack"):
                _export_quantized_weight(experts, torch.bfloat16, weight_name)


def test_only_the_handler_that_exports_a_module_declares_its_format(toy_backend):
    model = _quantize(get_tiny_llama())
    # Registered after the built-in handler of quantized linears, which exports them instead.
    _declare(_has_toy_weights, QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE, prepend=False)

    with (
        ExportModuleRegistry.reading_declared_formats(),
        pytest.raises(NotImplementedError, match=rf"num_bits: {BACKEND}$"),
    ):
        get_quant_config(model)


def test_re_registering_a_handler_replaces_its_declared_format(toy_backend):
    model = _quantize(get_tiny_llama())
    handler = _declare(_has_toy_weights, QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE)
    # As when its module is reloaded.
    ExportModuleRegistry.register(predicate=_has_toy_weights, prepend=True)(handler)

    with (
        ExportModuleRegistry.reading_declared_formats(),
        pytest.raises(NotImplementedError, match=rf"num_bits: {BACKEND}$"),
    ):
        get_quant_config(model)


def test_only_the_hf_exports_read_the_declared_format(toy_backend):
    model = _quantize(get_tiny_llama())
    down_proj = model.model.layers[0].mlp.down_proj
    export_format = ExportFormat(
        quantization_format=QUANTIZATION_W4A16_NVFP4, block_size=NVFP4_BLOCK_SIZE
    )
    _declare(_has_toy_weights, QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE)

    assert ExportModuleRegistry.declared_format(down_proj) is None
    with ExportModuleRegistry.reading_declared_formats():
        assert ExportModuleRegistry.declared_format(down_proj) == export_format
    # The export reads it and calls the handler, which raises; then it is unread again.
    with pytest.raises(AssertionError, match="was not to be packed"):
        _export_transformers_checkpoint(model)
    assert ExportModuleRegistry.declared_format(down_proj) is None


def test_the_declared_format_must_be_an_export_format():
    entries = list(ExportModuleRegistry._entries)
    with pytest.raises(TypeError, match="export_format must be an ExportFormat"):
        ExportModuleRegistry.register(
            predicate=_has_toy_weights, export_format=QUANTIZATION_W4A16_NVFP4
        )
    assert ExportModuleRegistry._entries == entries


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"quantization_format": QUANTIZATION_NONE}, "quantization_format must name a format"),
        ({"quantization_format": ""}, "quantization_format must name a format"),
        (
            {"quantization_format": QUANTIZATION_W4A16_NVFP4, "block_size": -1},
            "block_size must be an int >= 0",
        ),
        (
            {"quantization_format": QUANTIZATION_W4A16_NVFP4, "block_size": True},
            "block_size must be an int >= 0",
        ),
        (
            {"quantization_format": QUANTIZATION_W4A16_NVFP4, "block_size": 16.0},
            "block_size must be an int >= 0",
        ),
        (
            {"quantization_format": "w4a16_nvpf4", "block_size": NVFP4_BLOCK_SIZE},
            "not a format that export writes a config for",
        ),
        ({"quantization_format": QUANTIZATION_W4A16_NVFP4}, "block_size must be > 0"),
        ({"quantization_format": QUANTIZATION_FP8, "block_size": 128}, "block_size must be 0"),
        ({"quantization_format": QUANTIZATION_Q8_0, "block_size": 16}, "requires block size 32"),
    ],
)
def test_an_export_format_refuses_invalid_fields(fields, error):
    with pytest.raises(ValueError, match=error):
        ExportFormat(**fields)


def test_an_export_format_takes_its_fields_by_keyword():
    with pytest.raises(TypeError):
        ExportFormat(QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE)


def _megatron_packing(model: nn.Module, tmp_path) -> None:
    """Pack a layer as the Megatron export does, with the parts of the exporter that it calls."""
    exporter = SimpleNamespace(
        _get_quantization_format=get_quantization_format,
        _record_excluded_module=lambda prefix: None,
        _get_weight_bias=lambda module, dtype, name_to_value, keep_weight_device: {
            "weight": module.weight.detach()
        },
    )
    GPTModelExporter._get_quantized_state(
        exporter, model.model.layers[1].mlp.down_proj, torch.bfloat16
    )


# Exporters that do not read a declared format.
OTHER_EXPORTERS = {
    "export_hf_vllm_fq_checkpoint": EXPORTERS["export_hf_vllm_fq_checkpoint"],
    "megatron": _megatron_packing,
    "functional": lambda model, tmp_path: capture_quantized_weight_export_state(
        model.model.layers[1].mlp.down_proj
    ),
}


@pytest.mark.parametrize("exporter", sorted(OTHER_EXPORTERS))
def test_the_other_exporters_ignore_a_declared_format(toy_backend, exporter, tmp_path):
    model = _quantize(get_tiny_llama())
    packed = _register_toy_packer(QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE)

    # As without a declared format, but naming the exporters that read it.
    with pytest.raises(
        NotImplementedError,
        match=rf"num_bits: {BACKEND}\. Its ExportModuleRegistry handler declares the format",
    ):
        OTHER_EXPORTERS[exporter](model, tmp_path)
    assert not packed


def _offload(*modules: nn.Module) -> None:
    """Offload ``modules`` to the CPU with accelerate, so that export_hf_checkpoint streams."""
    from accelerate import cpu_offload

    for module in modules:
        cpu_offload(module, execution_device="cpu")


def _load_shards(export_dir) -> dict[str, torch.Tensor]:
    return {k: v for f in export_dir.glob("*.safetensors") for k, v in load_file(f).items()}


@pytest.mark.parametrize("offload_lm_head", [True, False])
def test_offloaded_export_packs_each_declared_module_once(toy_backend, offload_lm_head, tmp_path):
    pytest.importorskip("accelerate")
    lm_head_cfg = {"quantizer_name": "*lm_head*weight_quantizer", "cfg": TOY_CFG}
    model = _quantize(get_tiny_llama(), extra_cfg=(lm_head_cfg,))
    packed = _register_toy_packer(QUANTIZATION_W4A16_NVFP4, NVFP4_BLOCK_SIZE)
    # A resident lm_head, which no offload window reaches, is packed too.
    _offload(model.model.layers[0], *([model.lm_head] if offload_lm_head else []))
    export_hf_checkpoint(model, export_dir=tmp_path)

    state_dict = _load_shards(tmp_path)
    # A handler called twice for a module would leave tensors other than the exported ones here.
    assert sorted(packed) == sorted([*MLP_LINEARS, "lm_head"])
    for name, tensors in packed.items():
        for key, tensor in tensors.items():
            exported = state_dict[f"{name}.{key}"]
            assert exported.dtype == tensor.dtype and torch.equal(exported.float(), tensor.float())


def test_offloaded_export_packs_a_resident_lm_head(tmp_path):
    pytest.importorskip("accelerate")
    fp8 = _quantizer_cfg(mtq.FP8_DEFAULT_CFG, "*weight_quantizer")
    model = _quantize(get_tiny_llama(), "*lm_head*weight_quantizer", cfg=fp8, algorithm="max")
    _offload(model.model.layers[0])
    export_hf_checkpoint(model, export_dir=tmp_path)

    state_dict = _load_shards(tmp_path)
    assert state_dict["lm_head.weight"].dtype == torch.float8_e4m3fn
    assert "lm_head.weight_scale" in state_dict


def test_the_fused_experts_packing_refuses_a_declared_expert(tmp_path):
    fp8 = _quantizer_cfg(mtq.FP8_DEFAULT_CFG, "*weight_quantizer")
    model = _quantize(get_tiny_mixtral(), "*experts*weight_quantizer*", cfg=fp8)
    # Matches the wrapper that the built-in packing of fused experts passes for each expert.
    _declare(
        lambda module: type(module) is nn.Module and hasattr(module, "weight_quantizer"),
        QUANTIZATION_FP8,
        0,
    )

    with pytest.raises(NotImplementedError, match="built-in export cannot pack"):
        export_hf_checkpoint(model, export_dir=tmp_path)


@pytest.mark.parametrize("exporter", sorted(EXPORTERS))
def test_backend_without_declared_format_is_unsupported(toy_backend, exporter, tmp_path):
    model = _quantize(get_tiny_llama())
    with pytest.raises(
        NotImplementedError, match=rf"^Unsupported quantizer with num_bits: {BACKEND}$"
    ):
        EXPORTERS[exporter](model, tmp_path)
