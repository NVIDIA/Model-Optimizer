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

"""ModelOpt plugin for enabling automatic save/restore of ModelOpt state for `peft` library."""

import os
from collections.abc import Callable

import torch
from peft import PeftModel

from modelopt.torch.utils import get_unwrapped_name, print_rank_0

from ..conversion import ModeloptStateManager, modelopt_state, restore_from_modelopt_state
from .huggingface import register_for_patching

__all__ = []

_MODELOPT_STATE_SAVE_NAME = "peft_modelopt_state.pth"


def _get_modelopt_state_path(model_name_or_path: str) -> str:
    return os.path.join(model_name_or_path, _MODELOPT_STATE_SAVE_NAME)


def _get_quantizer_state_save_path(dir):
    return os.path.join(dir, "quantizer_state_dict.pth")


def _new_save_pretrained_peft(self, save_directory, *args, **kwargs):
    outputs = self._modelopt_cache["save_pretrained"](self, save_directory, *args, **kwargs)
    if not ModeloptStateManager.is_converted(self):
        return outputs
    path = _get_modelopt_state_path(save_directory)
    torch.save(modelopt_state(self), path)
    print_rank_0(f"Saved ModelOpt state to {path}")

    if not ModeloptStateManager.has_state_for_mode_type("quantization", model=self):
        return outputs

    # Lets save the quantizer state_dict separately
    # PEFT save_pretrained only saves the state_dict corresponding to the adapters
    # However our quantizers are part of the LoraLinear layers and not the adapters
    # Also there might non-LoraLinear layers which have quantizers in the model which also wont be saved by PEFT
    # So we need to save the quantizer state_dict separately

    # TODO: Move this to modelopt.torch.quantization.plugins.peft
    from modelopt.torch.quantization.utils import get_quantizer_state_dict

    quantizer_state_dict = get_quantizer_state_dict(self)
    if len(quantizer_state_dict) > 0:
        torch.save(quantizer_state_dict, _get_quantizer_state_save_path(save_directory))
    return outputs


def _new_load_adapter(self, model_id, adapter_name, *args, **kwargs):
    modelopt_state_path = _get_modelopt_state_path(model_id)

    if os.path.isfile(modelopt_state_path):
        assert adapter_name in self.peft_config, (
            f"ModelOpt modified model should have adapter_name={adapter_name} in peft_config"
        )
        restore_from_modelopt_state(
            self, torch.load(modelopt_state_path, map_location="cpu", weights_only=False)
        )

    outputs = self._modelopt_cache["load_adapter"](self, model_id, adapter_name, *args, **kwargs)

    if not ModeloptStateManager.has_state_for_mode_type("quantization", model=self):
        return outputs

    # TODO: Move this to modelopt.torch.quantization.plugins.peft
    if os.path.isfile(_get_quantizer_state_save_path(model_id)):
        from modelopt.torch.quantization.nn import TensorQuantizer

        quantizer_state_dict = torch.load(
            _get_quantizer_state_save_path(model_id), map_location="cpu", weights_only=False
        )
        for name, module in self.named_modules():
            if isinstance(module, TensorQuantizer):
                module.load_state_dict(quantizer_state_dict[get_unwrapped_name(name, self)])

    return outputs


patch_methods: list[tuple[str, Callable]] = [
    ("save_pretrained", _new_save_pretrained_peft),
    ("load_adapter", _new_load_adapter),
]

register_for_patching("peft", PeftModel, patch_methods)
