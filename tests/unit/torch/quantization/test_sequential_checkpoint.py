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

"""Unit tests for layerwise calibration checkpoint save/resume."""

import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.model_calib import layerwise_calibrate
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.network import get_module_device


class _DecoderBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x, **kwargs):
        return self.linear(x)


class _SimpleTransformerModel(nn.Module):
    def __init__(self, n_layers=3, dim=16):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderBlock(dim) for _ in range(n_layers)])
        self.embed = nn.Embedding(32, dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


def _register_test_discoverer(monkeypatch):
    monkeypatch.setattr(
        LayerActivationCollector,
        "_decoder_layer_support",
        [(lambda m: hasattr(m, "layers"), lambda m: m.layers)],
    )


def _dummy_calib_func(layer, forward_loop, **kwargs):
    """Scale all weights by 0.5 to produce a visible, deterministic change."""
    forward_loop(layer)
    with torch.no_grad():
        for p in layer.parameters():
            p.mul_(0.5)


def _make_model_and_forward(n_layers=3, dim=16, seed=42):
    torch.manual_seed(seed)
    model = _SimpleTransformerModel(n_layers=n_layers, dim=dim)
    tokens = [torch.randint(0, 32, (2, 8)) for _ in range(2)]

    def forward_loop(m):
        for t in tokens:
            m(t)

    return model, forward_loop


def test_full_run_creates_checkpoints(monkeypatch, tmp_path):
    """layerwise_calibrate with checkpoint_dir creates correct layer dirs and manifest."""
    _register_test_discoverer(monkeypatch)
    model, forward_loop = _make_model_and_forward(n_layers=3)
    ckpt_dir = str(tmp_path / "ckpt")

    layerwise_calibrate(model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)

    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["last_completed_layer"] == 2
    assert manifest["num_layers"] == 3

    for i in range(3):
        layer_dir = os.path.join(ckpt_dir, f"layer_{i:04d}")
        assert os.path.isdir(layer_dir)
        assert os.path.isfile(os.path.join(layer_dir, "weights.pt"))
        assert os.path.isfile(os.path.join(layer_dir, "quantizer_state.pt"))
        assert os.path.isfile(os.path.join(layer_dir, "output_meta.pt"))
    # All layers except the last should have next_inputs
    assert os.path.isfile(os.path.join(ckpt_dir, "layer_0000", "next_inputs.pt"))
    assert os.path.isfile(os.path.join(ckpt_dir, "layer_0001", "next_inputs.pt"))
    assert not os.path.isfile(os.path.join(ckpt_dir, "layer_0002", "next_inputs.pt"))


def test_resume_matches_full_run(monkeypatch, tmp_path):
    """Resume from a truncated checkpoint produces the same final weights as a full run."""
    _register_test_discoverer(monkeypatch)
    ckpt_dir = str(tmp_path / "ckpt")

    # Full reference run
    ref_model, forward_loop = _make_model_and_forward(n_layers=3)
    layerwise_calibrate(ref_model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)
    ref_weights = {n: p.clone() for n, p in ref_model.named_parameters()}

    # Simulate crash after layer 0: truncate manifest
    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"last_completed_layer": 0, "num_layers": 3}, f)

    # Resume from a fresh model
    resumed_model, forward_loop = _make_model_and_forward(n_layers=3)
    layerwise_calibrate(resumed_model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)

    for name, ref_param in ref_weights.items():
        resumed_param = dict(resumed_model.named_parameters())[name]
        assert torch.allclose(ref_param, resumed_param, atol=1e-6), (
            f"Parameter {name} diverged after resume"
        )


class _Payload:
    """Pickle that creates ``marker`` when loaded (stands in for attacker code)."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


@pytest.mark.parametrize(
    "planted",
    [
        "output_meta.pt",
        "next_inputs.pt",
        "quantizer_state.pt",
        "weights.pt",
        "quantizer_buffers.pt",
    ],
)
@pytest.mark.parametrize("force_no_weights_only", [False, True])
def test_resume_does_not_unpickle_planted_files(
    monkeypatch, tmp_path, planted, force_no_weights_only
):
    """A forged checkpoint file must be refused, not executed, on resume.

    Also pins that ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`` cannot downgrade the explicit load.
    """
    _register_test_discoverer(monkeypatch)
    ckpt_dir = str(tmp_path / "ckpt")
    model, forward_loop = _make_model_and_forward(n_layers=3)
    layerwise_calibrate(model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)
    with open(os.path.join(ckpt_dir, "manifest.json"), "w") as f:
        json.dump({"last_completed_layer": 1, "num_layers": 3}, f)

    marker = str(tmp_path / "pwned")
    layer_dir = os.path.join(
        ckpt_dir, "layer_0000" if planted == "output_meta.pt" else "layer_0001"
    )
    if planted == "quantizer_buffers.pt":  # only read when weights.pt is absent
        os.remove(os.path.join(layer_dir, "weights.pt"))
    torch.save(_Payload(marker), os.path.join(layer_dir, planted))
    if force_no_weights_only:
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    resumed_model, forward_loop = _make_model_and_forward(n_layers=3)
    with pytest.raises(RuntimeError, match="Cannot resume") as exc:
        layerwise_calibrate(resumed_model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)
    assert os.path.join(layer_dir, planted) in str(exc.value)
    assert not os.path.exists(marker)


def test_resume_truncated_file_gives_fresh_dir_guidance(monkeypatch, tmp_path):
    _register_test_discoverer(monkeypatch)
    ckpt_dir = str(tmp_path / "ckpt")
    model, forward_loop = _make_model_and_forward(n_layers=3)
    layerwise_calibrate(model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)
    with open(os.path.join(ckpt_dir, "manifest.json"), "w") as f:
        json.dump({"last_completed_layer": 1, "num_layers": 3}, f)
    path = os.path.join(ckpt_dir, "layer_0001", "next_inputs.pt")
    with open(path, "r+b") as f:
        f.truncate(10)

    resumed_model, forward_loop = _make_model_and_forward(n_layers=3)
    with pytest.raises(RuntimeError, match="fresh checkpoint directory"):
        layerwise_calibrate(resumed_model, forward_loop, _dummy_calib_func, checkpoint_dir=ckpt_dir)


class _Opaque:
    """A layer input that ``weights_only`` loading cannot restore."""


class _OpaqueKwargModel(_SimpleTransformerModel):
    def forward(self, x, **kwargs):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, extra=_Opaque())
        return x


def test_save_warns_when_checkpoint_not_resumable(monkeypatch, tmp_path):
    """An unloadable layer input is reported at calibration time, and calibration still finishes."""
    _register_test_discoverer(monkeypatch)
    torch.manual_seed(0)
    model = _OpaqueKwargModel(n_layers=3)
    tokens = torch.randint(0, 32, (2, 8))

    with pytest.warns(UserWarning, match="will not be resumable"):
        layerwise_calibrate(
            model, lambda m: m(tokens), _dummy_calib_func, checkpoint_dir=str(tmp_path / "ckpt")
        )


def test_resume_hf_model_with_quantizers(tmp_path):
    """Resume of a real HF model (KV-cache and tuple inputs, quantizer state) still loads safely."""
    transformers = pytest.importorskip("transformers")
    ckpt_dir = str(tmp_path / "ckpt")
    ids = [torch.randint(0, 128, (2, 16)) for _ in range(2)]

    def build():
        torch.manual_seed(0)
        cfg = transformers.LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
        return transformers.LlamaForCausalLM(cfg).eval()

    def forward_loop(m):
        for t in ids:
            m(t)

    def quant_cfg():
        cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
        cfg["algorithm"] = {
            "method": "max",
            "layerwise": {"enable": True, "checkpoint_dir": ckpt_dir},
        }
        return cfg

    ref = mtq.quantize(build(), quant_cfg(), forward_loop)
    # Exercise the cache allowlist: next_inputs.pt must really hold a DynamicCache.
    saved = torch.load(  # trusted: written by this test
        os.path.join(ckpt_dir, "layer_0001", "next_inputs.pt"), weights_only=False
    )
    assert any(type(v).__name__ == "DynamicCache" for _, kw in saved for v in kw.values())
    ref_state = {k: v.clone() for k, v in ref.state_dict().items()}

    manifest_path = os.path.join(ckpt_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["last_completed_layer"] = 1
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    resumed = mtq.quantize(build(), quant_cfg(), forward_loop)
    for k, v in resumed.state_dict().items():
        assert torch.allclose(ref_state[k].float(), v.float()), k


def test_no_checkpoint_unchanged(monkeypatch):
    """Without checkpoint_dir, calibration still works and modifies parameters."""
    _register_test_discoverer(monkeypatch)
    model, forward_loop = _make_model_and_forward(n_layers=3)
    original_weights = {n: p.clone() for n, p in model.named_parameters()}

    layerwise_calibrate(model, forward_loop, _dummy_calib_func)

    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(original_weights[name], param):
            changed = True
            break
    assert changed, "Expected calibration to modify at least one parameter"


# ---------------------------------------------------------------------------
# get_module_device tests
# ---------------------------------------------------------------------------


def test_get_module_device_no_hook():
    """Falls back to parameter device when no _hf_hook is present."""
    layer = nn.Linear(4, 4)
    assert get_module_device(layer) == torch.device("cpu")


def test_get_module_device_with_direct_hook():
    """Returns execution_device from a direct AlignDevicesHook-style hook."""
    layer = nn.Linear(4, 4)
    layer._hf_hook = SimpleNamespace(execution_device=torch.device("cuda:0"))
    assert get_module_device(layer) == torch.device("cuda:0")


def test_get_module_device_with_sequential_hook():
    """Returns execution_device from an AlignDevicesHook wrapped in SequentialHook."""
    layer = nn.Linear(4, 4)
    inner_hook = SimpleNamespace(execution_device=torch.device("cuda:1"))
    layer._hf_hook = SimpleNamespace(hooks=[inner_hook])
    assert get_module_device(layer) == torch.device("cuda:1")


def test_get_module_device_hook_without_execution_device():
    """Falls back to parameters when hook has no execution_device."""
    layer = nn.Linear(4, 4)
    layer._hf_hook = SimpleNamespace()
    assert get_module_device(layer) == torch.device("cpu")


def test_get_module_device_parameterless_module():
    """Returns cpu for a module with no parameters and no hook."""
    module = nn.Module()
    assert get_module_device(module) == torch.device("cpu")
