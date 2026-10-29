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

"""Integration tests for layerwise_calibrate and LayerActivationCollector."""

import copy
import sys

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import TOY_MOE_EXPERTS, TOY_MOE_HIDDEN, ToyMoE
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

import modelopt.torch.quantization as mtq
from modelopt.recipe import (
    ModelOptAutoQuantizeRecipe,
    freeze_auto_quantize_assignment,
    plan_auto_quantize,
)
from modelopt.torch.quantization._layerwise_checkpoint import _assignment_checkpoint
from modelopt.torch.quantization._selected_calibration import _SelectedLayerCalibrator
from modelopt.torch.quantization._selected_model import _selected_model
from modelopt.torch.quantization._selected_search import (
    _load_selected_search,
    _selected_search_binding,
)
from modelopt.torch.quantization._source_estimation import _estimate_source_activations
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import pinned_packed_weight, preserve_source_weights
from modelopt.torch.quantization.model_calib import layerwise_calibrate
from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.utils import layerwise_calib
from modelopt.torch.quantization.utils.layerwise_calib import (
    LayerActivationCollector,
    get_max_layerwise_batch_size,
)


class _DecoderBlock(nn.Module):
    """Minimal transformer decoder block."""

    def __init__(self, dim=16):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(dim * 4, dim, bias=False),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(x)
        return x


class _SimpleTransformerModel(nn.Module):
    """model.layers (ModuleList) — the simplest pattern recognised by get_decoder_layers."""

    def __init__(self, n_layers=3, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])
        self.embed = nn.Embedding(32, dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


class _TupleReturningBlock(nn.Module):
    """Decoder layer that returns a tuple, mimicking HuggingFace decoder layers."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return (self.linear(x), None)


class _TupleUnpackingModel(nn.Module):
    """Parent model that unpacks layer outputs as tuples."""

    def __init__(self, n_layers=4, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_TupleReturningBlock(dim) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x


def _make_model_and_data(n_layers=3, dim=16, n_batches=2, batch_size=4):
    torch.manual_seed(42)
    model = _SimpleTransformerModel(n_layers=n_layers, dim=dim)
    tokens = [torch.randint(0, 32, (batch_size, 8)) for _ in range(n_batches)]
    return model, tokens


def _run_forward(model, data):
    for batch in data:
        model(batch)


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_fsdp_capture_releases_completed_layers(dist_workers, cpu_offload):
    dist_workers.run(_fsdp_capture_releases_completed_layers, cpu_offload)


def _fsdp_capture_releases_completed_layers(rank, size, cpu_offload):
    for qdq_from_prev in (False, True):
        model, data = _make_model_and_data(n_layers=8, dim=256)
        model.cuda()
        data = [batch.cuda() for batch in data]
        with torch.no_grad():
            expected = model(data[0]).clone()
        options = {"offload_policy": CPUOffloadPolicy()} if cpu_offload else {}
        for layer in model.layers:
            fully_shard(layer, **options)
        fully_shard(model, **options)
        layers = list(model.layers)
        visited = []

        def calibrate(layer, forward_loop):
            assert all(isinstance(p, DTensor) for m in visited for p in m.parameters())
            forward_loop(layer)
            visited.append(layer)

        with pytest.MonkeyPatch.context() as patch:
            _register_test_discoverer(patch)
            layerwise_calibrate(
                model,
                forward_loop=lambda m: _run_forward(m, data),
                calib_func=calibrate,
                calib_mutates_weights=False,
                get_qdq_activations_from_prev_layer=qdq_from_prev,
            )
        assert visited == layers
        assert all(isinstance(p, DTensor) for m in layers for p in m.parameters())
        with torch.no_grad():
            torch.testing.assert_close(model(data[0]), expected, rtol=0, atol=0)


def _register_test_discoverer(monkeypatch):
    """Register a simple discoverer that finds model.layers on any model."""
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("qdq_from_prev", "estimate", "recover"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        pytest.param(
            False,
            False,
            "selected",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX checkpoint leases"),
        ),
        pytest.param(
            True,
            False,
            "selected",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX checkpoint leases"),
        ),
        pytest.param(
            False,
            False,
            True,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX checkpoint leases"),
        ),
        pytest.param(
            True,
            False,
            True,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX checkpoint leases"),
        ),
    ],
)
def test_selected_layerwise_gptq_mixed_formats(
    monkeypatch, tmp_path, qdq_from_prev, estimate, recover
):
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(42)
    source = _SimpleTransformerModel(n_layers=0)
    source.embed = nn.Identity()
    source.layers = nn.ModuleList([ToyMoE() for _ in range(3)])
    source = source.to(device="cuda", dtype=torch.bfloat16).eval()
    data = [torch.randn(32, TOY_MOE_HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    config = {"quant_cfg": [{"quantizer_name": "*", "enable": False}], "algorithm": None}
    groups = []
    for i, fmt in enumerate(("iq2_xxs", "iq2_xs", "iq2_s")):
        prefix = f"layers.{i}.mlp"
        config["quant_cfg"].append(
            {
                "quantizer_name": f"{prefix}.experts.*weight_quantizer",
                "cfg": {
                    "num_bits": fmt,
                    "effective_bits": GGML_FORMAT_REGISTRY[fmt].effective_bits,
                    "block_sizes": {-1: 256},
                    "backend": "ggml",
                },
            }
        )
        preset = mtq.NVFP4_DEFAULT_CFG if i % 2 == 0 else mtq.FP8_DEFAULT_CFG
        for entry in preset["quant_cfg"]:
            if entry.get("quantizer_name") in {"*weight_quantizer", "*input_quantizer"}:
                entry = copy.deepcopy(entry)
                entry["quantizer_name"] = prefix + ".shared_expert." + entry["quantizer_name"]
                config["quant_cfg"].append(entry)
        traversal = {
            "enable": recover != "selected" or i > 0,
            "get_qdq_activations_from_prev_layer": qdq_from_prev or estimate,
        }
        groups.extend(
            [
                (
                    [prefix + ".experts"],
                    {
                        "method": "gptq",
                        "block_size": 256,
                        "perc_damp": 0.05 * (i + 1),
                        "importance_weighted": True,
                        "layerwise": traversal,
                    },
                ),
                ([prefix + ".shared_expert"], {"method": "max", "layerwise": traversal}),
            ]
        )
    original_groups = copy.deepcopy(groups)
    if recover == "selected":
        spaces = []
        for names, algorithm in groups:
            (name,) = names
            entries = [
                {
                    **copy.deepcopy(entry),
                    "quantizer_name": entry["quantizer_name"].removeprefix(name + "."),
                }
                for entry in config["quant_cfg"]
                if entry["quantizer_name"].startswith(name + ".")
            ]
            spaces.append(
                {
                    "module_name_patterns": [name],
                    "candidate_formats": [
                        {
                            "quant_cfg": [{"quantizer_name": "*", "enable": False}, *entries],
                            "algorithm": algorithm,
                        }
                    ],
                    "allow_no_quant": False,
                }
            )
        no_quant = {"quant_cfg": [{"quantizer_name": "*", "enable": False}]}
        recipe = ModelOptAutoQuantizeRecipe(
            auto_quantize={
                "constraints": {"effective_bits": 16},
                "candidate_formats": [no_quant],
                "module_search_spaces": spaces,
            }
        )
        options = {
            "constraints": {"effective_bits": 16},
            "quantization_formats": [no_quant],
            "module_search_spaces": [
                {
                    **{k: v for k, v in space.items() if k != "candidate_formats"},
                    "quantization_formats": space["candidate_formats"],
                }
                for space in spaces
            ],
            "data_loader": data,
            "forward_step": lambda model, batch: model(batch),
            "loss_func": lambda output, batch: output.float().square().mean(),
            "num_calib_steps": 2,
            "num_score_steps": 1,
        }
        limits = {
            "directory": tmp_path / "formats",
            "collect_scores": True,
            "source_activation_bytes": 2**20,
            "max_artifact_bytes": 64 * 2**20,
        }
        for candidate in range(len(spaces)):
            mtq.auto_quantize(
                copy.deepcopy(source), **options, calibration=dict(limits, candidate=candidate)
            )
        checkpoint = tmp_path / "search"
        _, search_state = mtq.auto_quantize(
            copy.deepcopy(source), **options, calibration=limits, checkpoint=str(checkpoint)
        )
        identity = search_state["calibration_provenance"]["contract"]["identity"]
        plan = plan_auto_quantize(recipe)
        assignment = freeze_auto_quantize_assignment(search_state, plan, identity)
        binding = _selected_search_binding(checkpoint, assignment, 64 * 2**20)
        search_state = _load_selected_search(
            checkpoint, binding, assignment, plan, identity, (2, 1), 64 * 2**20
        )

    def fresh():
        return mtq.quantize(copy.deepcopy(source), copy.deepcopy(config))

    reference = fresh()
    inputs = data
    for i, layer in enumerate(reference.layers):
        captured = {"experts": [], "shared_expert": []}
        handles = [
            getattr(layer.mlp, name).register_forward_pre_hook(
                lambda m, args, name=name: captured[name].append(
                    tuple(x.detach().clone() for x in args)
                )
            )
            for name in captured
        ]
        with mtq.set_quantizer_by_cfg_context(layer, [{"quantizer_name": "*", "enable": False}]):
            source_outputs = [layer(batch).detach() for batch in inputs]
        for handle in handles:
            handle.remove()
        for j, name in enumerate(captured):
            algorithm = copy.deepcopy(groups[2 * i + j][1])
            algorithm["layerwise"]["enable"] = False
            with preserve_source_weights():
                mtq.calibrate(
                    getattr(layer.mlp, name),
                    algorithm,
                    lambda module, batches=captured[name]: [module(*args) for args in batches],
                )
        inputs = [layer(batch).detach() for batch in inputs] if qdq_from_prev else source_outputs

    for order_index, order in enumerate((groups, list(reversed(groups)))):
        actual = copy.deepcopy(source) if recover == "selected" else fresh()
        if estimate:
            result = _estimate_source_activations(
                actual, order, lambda model: _run_forward(model, data), max_activation_bytes=2**20
            )
            assert result["peak_activation_bytes"] <= 2**20
            assert all(
                pinned_packed_weight(q, q.num_bits).device.type == "cpu"
                for q in actual.modules()
                if isinstance(q, TensorQuantizer) and q.is_enabled and q.backend == "ggml"
            )
        else:
            callback = (
                _selected_model(actual, assignment, plan, search_state, identity)
                if recover == "selected"
                else _SelectedLayerCalibrator(actual, order)
            )
            if recover is True:
                options = dict(callback.layerwise_config, save_every=2, calib_mutates_weights=False)
                directory = tmp_path / str(order_index)
                binding = {
                    "assignment_digest": "1" * 64,
                    "provenance": {
                        "source": "fixture-source",
                        "inputs": "fixture-data",
                        "code": "fixture-code",
                    },
                    "num_layers": 3,
                    "max_bytes": 16 * 2**20,
                    "save_every": 2,
                }
                last_layer = actual.layers[2]

                def interrupt(layer, forward_loop):
                    if layer is last_layer:
                        raise RuntimeError("test interruption")
                    callback(layer, forward_loop)

                with (
                    _assignment_checkpoint(directory, **binding) as checkpoint,
                    pytest.raises(RuntimeError, match="test interruption"),
                ):
                    layerwise_calibrate(
                        actual,
                        lambda model: _run_forward(model, data),
                        interrupt,
                        _checkpoint_state=checkpoint,
                        **options,
                    )
                actual = fresh()
                callback = _SelectedLayerCalibrator(actual, order)
                with _assignment_checkpoint(directory, **binding) as checkpoint:
                    assert checkpoint.start_layer == 2
                    layerwise_calibrate(
                        actual,
                        lambda model: _run_forward(model, data),
                        callback,
                        _checkpoint_state=checkpoint,
                        **options,
                    )

                def unexpected(*args):
                    pytest.fail("Completed recovery repeated calibration or activation capture")

                actual = fresh()
                with _assignment_checkpoint(directory, **binding) as checkpoint:
                    assert checkpoint.start_layer == 3
                    layerwise_calibrate(
                        actual, unexpected, unexpected, _checkpoint_state=checkpoint, **options
                    )
                assert not list(directory.glob("layer_*/next_inputs.pt"))
            else:
                layerwise_calibrate(
                    actual,
                    lambda model: _run_forward(model, data),
                    callback,
                    **callback.layerwise_config,
                )
        assert actual.state_dict().keys() == reference.state_dict().keys()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(
                actual.state_dict()[name].to(value.device), value, rtol=0, atol=0
            )
        torch.testing.assert_close(actual(data[0]), reference(data[0]), rtol=0, atol=0)
        for name, parameter in actual.named_parameters():
            torch.testing.assert_close(
                parameter, dict(source.named_parameters())[name], rtol=0, atol=0
            )
        packed_counts = {}
        for quantizer in actual.modules():
            if (
                isinstance(quantizer, TensorQuantizer)
                and quantizer.is_enabled
                and quantizer.backend == "ggml"
            ):
                payload = pinned_packed_weight(quantizer, quantizer.num_bits)
                assert payload is not None and payload.numel() > 0
                packed_counts[quantizer.num_bits] = packed_counts.get(quantizer.num_bits, 0) + 1
        assert packed_counts == dict.fromkeys(("iq2_xxs", "iq2_xs", "iq2_s"), 2 * TOY_MOE_EXPERTS)
        for module in actual.modules():
            if isinstance(module, QuantModule):
                for weight, quantizer in module.iter_weights_for_calibration():
                    if quantizer.is_enabled and quantizer.backend == "ggml":
                        packed = GGML_FORMAT_REGISTRY[quantizer.num_bits].pack(weight, quantizer)
                        expected = pinned_packed_weight(quantizer, quantizer.num_bits)
                        torch.testing.assert_close(
                            packed.flatten().to(expected.device), expected.flatten(), rtol=0, atol=0
                        )
    assert groups == original_groups


def test_layerwise_calib_func_called_per_layer(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=4)
    call_count = [0]

    def counting_calib(layer, forward_loop, **kwargs):
        call_count[0] += 1

    layerwise_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=counting_calib,
    )

    assert call_count[0] == 4


def test_layerwise_calib_func_receives_correct_layer(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=3)
    called_layers = []

    def track_layers(layer, forward_loop, **kwargs):
        called_layers.append(layer)

    layerwise_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=track_layers,
    )

    for i, layer in enumerate(model.layers):
        assert called_layers[i] is layer


def test_layerwise_calib_kwargs_forwarded(monkeypatch):
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=2)
    received_kwargs = []

    def capture_kwargs(layer, forward_loop, **kwargs):
        received_kwargs.append(kwargs)

    layerwise_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=capture_kwargs,
        alpha=0.5,
        method="max",
    )

    assert len(received_kwargs) == 2
    for kw in received_kwargs:
        assert kw["alpha"] == 0.5
        assert kw["method"] == "max"


def test_layerwise_calib_layer_forward_loop_runs_all_batches(monkeypatch):
    """The per-layer forward loop passed to calib_func should replay all batches."""
    _register_test_discoverer(monkeypatch)
    n_batches = 5
    model, data = _make_model_and_data(n_layers=2, n_batches=n_batches)
    batch_counts = []

    def count_batches(layer, forward_loop, **kwargs):
        counter = {"n": 0}
        orig_forward = layer.forward

        def counting_forward(*args, **kw):
            counter["n"] += 1
            return orig_forward(*args, **kw)

        layer.forward = counting_forward
        forward_loop(layer)
        layer.forward = orig_forward
        batch_counts.append(counter["n"])

    layerwise_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=count_batches,
    )

    for count in batch_counts:
        assert count == n_batches


def test_layerwise_calib_does_not_alter_weights(monkeypatch):
    """layerwise_calibrate itself should not modify model weights."""
    _register_test_discoverer(monkeypatch)
    model, data = _make_model_and_data(n_layers=3)
    weights_before = {n: p.clone() for n, p in model.named_parameters()}

    layerwise_calibrate(
        model,
        forward_loop=lambda m: _run_forward(m, data),
        calib_func=lambda layer, forward_loop, **kw: None,
    )

    for n, p in model.named_parameters():
        assert torch.equal(p, weights_before[n]), f"Weight {n} was modified"


def test_layerwise_calib_activations_update_across_layers(monkeypatch):
    """Subsequent layers should see activations transformed by prior layers."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _SimpleTransformerModel(n_layers=2, dim=16)
    tokens = [torch.randint(0, 32, (2, 4))]

    layer_inputs_record = {}

    def record_inputs(layer, forward_loop, **kwargs):
        activations = []
        orig_forward = layer.forward

        def capture_forward(*args, **kw):
            activations.append(args[0].clone())
            return orig_forward(*args, **kw)

        layer.forward = capture_forward
        forward_loop(layer)
        layer.forward = orig_forward

        layer_idx = list(model.layers).index(layer)
        layer_inputs_record[layer_idx] = activations

    layerwise_calibrate(
        model,
        forward_loop=lambda m: [m(t) for t in tokens],
        calib_func=record_inputs,
    )

    assert not torch.allclose(layer_inputs_record[0][0], layer_inputs_record[1][0]), (
        "Layer 1 should receive different activations than layer 0"
    )


def test_mode_transitions_across_calibration_steps(monkeypatch):
    """Verify layer modes after each layerwise calibration step.

    After get_input_activations(layers[i]) returns, the current layer is reset
    to 'original'.  Layers further back are left in 'run' (just calibrated) or
    'skip' (fully done), reflecting the state the next forward loop will see.
    """
    _register_test_discoverer(monkeypatch)
    model = _TupleUnpackingModel(n_layers=5, dim=16)
    data = [torch.randn(2, 16)]

    def forward_loop(m):
        for d in data:
            m(d)

    collector = LayerActivationCollector(model)
    collector._patch_all_layers()
    try:

        def modes():
            return [model.layers[i]._layerwise_calib.mode for i in range(5)]

        collector.get_input_activations(model.layers[0], forward_loop)
        assert modes() == ["original", "original", "original", "original", "original"]

        collector.get_input_activations(model.layers[1], forward_loop)
        assert modes() == ["run", "original", "original", "original", "original"]

        collector.get_input_activations(model.layers[2], forward_loop)
        assert modes() == ["skip", "run", "original", "original", "original"]

        collector.get_input_activations(model.layers[3], forward_loop)
        assert modes() == ["skip", "skip", "run", "original", "original"]

        collector.get_input_activations(model.layers[4], forward_loop)
        assert modes() == ["skip", "skip", "skip", "run", "original"]
    finally:
        collector._unpatch_all_layers()


def test_run_layer_reflects_weight_updates(monkeypatch):
    """After calib_func modifies weights, the next layer should see updated activations."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    dim = 8

    class _ScaleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * self.weight

    class _TwoScaleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_ScaleLayer(), _ScaleLayer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = _TwoScaleModel()
    x = torch.randn(2, dim)

    activations_before_weight_update = model.layers[0](x).clone()

    def forward_loop(m):
        m(x)

    def weight_doubling_calib(layer, layer_forward_loop, **kwargs):
        with torch.no_grad():
            layer.weight.mul_(2.0)
        layer_forward_loop(layer)

    layerwise_calibrate(
        model,
        forward_loop=forward_loop,
        calib_func=weight_doubling_calib,
    )

    # Layer 0's weight was doubled by calib_func.  When collecting inputs
    # for layer 1, the run-mode replay of layer 0 should use the updated
    # weight, so layer 1 should have received 2x the original activations.
    expected = activations_before_weight_update * 2.0
    # Verify by running model.layers[0] with its updated weights
    actual = model.layers[0](x)
    assert torch.allclose(actual, expected)


def test_skip_placeholder_uses_meta_device():
    recorded_device = torch.device("cpu")
    meta = ("tensor", torch.Size([2, 3]), torch.float32, recorded_device)

    out = LayerActivationCollector._zeros_from_meta(meta)
    assert out.device == torch.device("meta")
    assert out.shape == torch.Size([2, 3])
    assert out.dtype == torch.float32


class _WideDecoderBlock(_DecoderBlock):
    """A heavier kind of decoder layer: its FFN is 8x wider than _DecoderBlock's."""

    def __init__(self, dim=16):
        super().__init__(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 32, bias=False),
            nn.ReLU(),
            nn.Linear(dim * 32, dim, bias=False),
        )


@pytest.mark.parametrize("same_class", [False, True], ids=["new_class", "same_class"])
def test_layerwise_batch_probe_fits_the_heaviest_kind_of_layer(monkeypatch, same_class):
    """The batch probe runs every kind of decoder layer, not just the first one, including layers
    of one class that differ in width, like Nemotron-NAS's."""
    _register_test_discoverer(monkeypatch)
    budget = 256 * 2**20
    monkeypatch.setattr(layerwise_calib, "get_available_gpu_memory", lambda gpu: budget)
    model = _SimpleTransformerModel(n_layers=3, dim=256).cuda()

    def probe():
        return get_max_layerwise_batch_size(model, seq_len=512, max_batch_size=4096, num_samples=1)

    light = probe()
    if same_class:
        model.layers[2].ffn = _WideDecoderBlock(256).ffn.cuda()
    else:
        model.layers[2] = _WideDecoderBlock(256).cuda()
    heavy = probe()
    assert 1 < heavy < light < 4096
    assert not any(hasattr(layer, "_layerwise_calib") for layer in model.layers)

    # The heavy layer's forward at the probed batch takes at most half of the available memory.
    x = torch.randn(heavy, 512, 256, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        model.layers[2](x)
    assert torch.cuda.max_memory_allocated() - base <= budget / 2


class _OutOfMemoryOnTwoSamples(nn.Module):
    """Runs *module*, but runs out of GPU memory on a batch of more than one sample."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, **kwargs):
        if x.shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return self.module(x, **kwargs)


@pytest.mark.parametrize("where", ["layer_forward", "input_capture"])
def test_layerwise_batch_probe_falls_back_to_one_sample(monkeypatch, where):
    """A decoder layer that fits one sample but not two, or whose inputs for two samples cannot
    be captured, gets a batch of one instead of stopping the run."""
    _register_test_discoverer(monkeypatch)
    model = _SimpleTransformerModel(n_layers=2, dim=64).cuda()
    if where == "layer_forward":
        model.layers[1] = _OutOfMemoryOnTwoSamples(model.layers[1])
    else:
        model.embed = _OutOfMemoryOnTwoSamples(model.embed)

    assert get_max_layerwise_batch_size(model, seq_len=8, max_batch_size=64, num_samples=1) == 1
    assert not any(hasattr(layer, "_layerwise_calib") for layer in model.layers)


class _MultiGPUModel(_SimpleTransformerModel):
    """Decoder layers spread over GPUs, as a ``device_map`` places them."""

    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x.to(layer.norm.weight.device))
        return x


def test_layerwise_batch_probe_fits_every_gpu_a_kind_of_layer_runs_on(need_2_gpus, monkeypatch):
    """Of two identical layers, only the first is probed, but the second's GPU, with less memory,
    sets the batch."""
    _register_test_discoverer(monkeypatch)
    model = _MultiGPUModel(n_layers=2, dim=256).cuda()
    model.layers[1].to("cuda:1")

    def probe(budgets):
        monkeypatch.setattr(layerwise_calib, "get_available_gpu_memory", lambda gpu: budgets[gpu])
        return get_max_layerwise_batch_size(model, seq_len=512, max_batch_size=4096, num_samples=1)

    small, large = 256 * 2**20, 1024 * 2**20
    assert probe({0: large, 1: small}) == probe({0: small, 1: small}) < probe({0: large, 1: large})
