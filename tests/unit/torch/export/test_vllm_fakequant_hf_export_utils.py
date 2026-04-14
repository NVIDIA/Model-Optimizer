# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import types

import pytest
import torch
import torch.nn as nn
from modelopt.torch.export.plugins import vllm_fakequant_hf as vllm_fq


class _MinimalModel(nn.Module):
    def __init__(self, meta_weight: bool = False):
        super().__init__()
        device = "meta" if meta_weight else "cpu"
        self.weight = nn.Parameter(torch.ones(4, device=device))
        self.save_calls = []

    def save_pretrained(self, export_dir, **kwargs):
        self.save_calls.append((export_dir, kwargs))


class _DummyHook:
    def __init__(self, weights_map):
        self.weights_map = weights_map


def _patch_minimal_modelopt_state(monkeypatch):
    monkeypatch.setattr(vllm_fq, "get_quantizer_state_dict", lambda _model: {})
    monkeypatch.setattr(vllm_fq, "quantizer_state", lambda _model: {})
    monkeypatch.setattr(vllm_fq.mto, "modelopt_state", lambda _model: {"modelopt_state_dict": []})
    monkeypatch.setattr(vllm_fq.torch, "save", lambda _obj, _path: None)


def test_materialize_uses_longest_module_prefix(monkeypatch):
    class _NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Module()
            self.a.b = nn.Module()
            self.a._hf_hook = _DummyHook({"b.weight": torch.tensor([1.0])})
            self.a.b._hf_hook = _DummyHook({"weight": torch.tensor([2.0])})

    model = _NestedModel()
    state_dict = {"a.b.weight": torch.empty(1, device="meta")}

    fake_accel = types.ModuleType("modelopt.torch.quantization.plugins.accelerate")
    fake_accel._get_cpu_offload_hook = lambda hook: hook
    monkeypatch.setitem(sys.modules, "modelopt.torch.quantization.plugins.accelerate", fake_accel)

    vllm_fq._materialize_offloaded_weights(model, state_dict, ["a.b.weight"])
    assert torch.allclose(state_dict["a.b.weight"], torch.tensor([2.0]))


def test_export_raises_if_non_quant_meta_tensors_remain(monkeypatch, tmp_path):
    _patch_minimal_modelopt_state(monkeypatch)
    model = _MinimalModel(meta_weight=True)

    monkeypatch.setattr(vllm_fq, "_materialize_offloaded_weights", lambda *_args, **_kwargs: None)

    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="Failed to materialize offloaded tensors") as exc,
    ):
        vllm_fq.export_hf_vllm_fq_checkpoint(model, export_dir=tmp_path / "export_meta_fail")
    assert "_save_clean_checkpoint" in str(exc.value)


def test_export_uses_model_save_pretrained_when_not_offloaded(monkeypatch, tmp_path):
    _patch_minimal_modelopt_state(monkeypatch)
    model = _MinimalModel(meta_weight=False)
    called = {"clean": 0}

    def _save_clean_checkpoint(*_args, **_kwargs):
        called["clean"] += 1

    monkeypatch.setattr(vllm_fq, "_save_clean_checkpoint", _save_clean_checkpoint)
    vllm_fq.export_hf_vllm_fq_checkpoint(model, export_dir=tmp_path / "export_non_offloaded")

    assert called["clean"] == 0
    assert len(model.save_calls) == 1
    assert model.save_calls[0][1]["save_modelopt_state"] is False
    assert "state_dict" in model.save_calls[0][1]


def test_export_uses_clean_checkpoint_when_offloaded(monkeypatch, tmp_path):
    _patch_minimal_modelopt_state(monkeypatch)
    model = _MinimalModel(meta_weight=True)
    called = {"clean": 0}

    def _materialize(_model, state_dict, _meta_keys):
        state_dict["weight"] = torch.ones(4)

    def _save_clean_checkpoint(*_args, **_kwargs):
        called["clean"] += 1

    def _unexpected_save_pretrained(*_args, **_kwargs):
        raise AssertionError("model.save_pretrained should not be called for offloaded export")

    monkeypatch.setattr(vllm_fq, "_materialize_offloaded_weights", _materialize)
    monkeypatch.setattr(vllm_fq, "_save_clean_checkpoint", _save_clean_checkpoint)
    model.save_pretrained = _unexpected_save_pretrained

    vllm_fq.export_hf_vllm_fq_checkpoint(model, export_dir=tmp_path / "export_offloaded")
    assert called["clean"] == 1


def test_export_raises_when_cuda_device_cannot_be_found(monkeypatch, tmp_path):
    _patch_minimal_modelopt_state(monkeypatch)

    class _DummyTensorQuantizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fake_quant = True
            self.is_enabled = True
            self.rotate_is_enabled = False
            self._rotate = False

        def disable(self):
            self.is_enabled = False

        def enable(self):
            self.is_enabled = True

        def forward(self, x):
            return x

    class _DummyQuantModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))
            self.weight_quantizer = _DummyTensorQuantizer()

    class _DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = _DummyQuantModule()

        def save_pretrained(self, _export_dir, **_kwargs):
            return None

    monkeypatch.setattr(vllm_fq, "QuantModule", _DummyQuantModule)
    monkeypatch.setattr(vllm_fq, "TensorQuantizer", _DummyTensorQuantizer)

    with torch.inference_mode(), pytest.raises(
        RuntimeError, match="Cannot find CUDA device for quantizer kernel"
    ):
        vllm_fq.export_hf_vllm_fq_checkpoint(
            _DummyModel(), export_dir=tmp_path / "export_cuda_missing"
        )
