# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    get_tiny_qwen3,
    tf_output_tester,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.opt.plugins.transformers import _restore_qtensor_wrappers
from modelopt.torch.quantization.qtensor import QTensorWrapper


@pytest.mark.parametrize(
    ("model_cls", "teacher_model_type"),
    [
        (AutoModelForCausalLM, "llama"),
        (AutoModelForCausalLM, "qwen3"),
    ],
)
# Skipped on Windows - Flaky; root cause unknown; not critical
def test_nested_model_save_restore(skip_on_windows, tmp_path, model_cls, teacher_model_type):
    tiny_llama_dir = create_tiny_llama_dir(tmp_path)

    model_ref = model_cls.from_pretrained(tiny_llama_dir)

    if teacher_model_type == "qwen3":
        teacher_model = get_tiny_qwen3()
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(tiny_llama_dir)

    kd_config = {
        "teacher_model": teacher_model,
        "criterion": mtd.LogitsDistillationLoss(),
    }
    model = mtd.convert(model_ref, mode=[("kd_loss", kd_config)])
    model.save_pretrained(tiny_llama_dir / "modelopt_model")

    model_test = model_cls.from_pretrained(tiny_llama_dir / "modelopt_model")

    tf_output_tester(model, model_test)
    # KD state is not saved and it should be empty
    assert not mto.ModeloptStateManager(model_test).has_state


class _LoraLike(nn.Module):
    """Stand-in for peft's `lora.Linear`, which nests the original module under `base_layer`."""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer


def _compressed_model_and_state_dir(tmp_path, state_keyed_with_base_layer=False):
    model = nn.Sequential()
    model.fc = nn.Linear(64, 32)
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(torch.randn(2, 64)))
    mtq.compress(model)
    assert isinstance(model.fc.weight, QTensorWrapper)

    state = mto.modelopt_state(model)
    if state_keyed_with_base_layer:
        # Compressing after the adapters are attached saves the keys with the peft suffix.
        for _, mode_config in state["modelopt_state_dict"]:
            q_tensor_state = mode_config.get("metadata", {}).get("q_tensor_state", {})
            for key in list(q_tensor_state):
                q_tensor_state[f"{key}.base_layer"] = q_tensor_state.pop(key)
    torch.save(state, tmp_path / "modelopt_state.pth")

    # transformers>=5 assigns a plain Parameter holding the packed data, dropping the wrapper.
    packed = model.fc.weight.data.clone()
    del model.fc._parameters["weight"]
    model.fc._parameters["weight"] = nn.Parameter(packed, requires_grad=False)
    assert not isinstance(model.fc.weight, QTensorWrapper)
    return model


@pytest.mark.parametrize("wrap_in_lora", [False, True])
@pytest.mark.parametrize("state_keyed_with_base_layer", [False, True])
def test_restore_qtensor_wrappers(tmp_path, wrap_in_lora, state_keyed_with_base_layer):
    """Either side may carry the `.base_layer` suffix, so the lookup must work in both directions."""
    model = _compressed_model_and_state_dir(tmp_path, state_keyed_with_base_layer)
    if wrap_in_lora:
        model.fc = _LoraLike(model.fc)

    _restore_qtensor_wrappers(model, str(tmp_path))

    linear = model.fc.base_layer if wrap_in_lora else model.fc
    assert isinstance(linear.weight, QTensorWrapper)
    assert linear.weight.metadata["shape"] == torch.Size([32, 64])


def test_restore_qtensor_wrappers_warns_when_nothing_matches(tmp_path):
    """A total miss must be loud -- it otherwise surfaces as an opaque shape error at dequant."""
    model = _compressed_model_and_state_dir(tmp_path)
    model.fc = _LoraLike(_LoraLike(model.fc))  # a nesting the lookup does not know about

    with pytest.warns(UserWarning, match="re-wrapped none"):
        _restore_qtensor_wrappers(model, str(tmp_path))

    assert not isinstance(model.fc.base_layer.base_layer.weight, QTensorWrapper)
