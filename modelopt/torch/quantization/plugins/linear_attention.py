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

"""Shared module policy and checkpoint support for linear-attention QAT."""

import fnmatch

from modelopt.torch.opt.conversion import ApplyModeError
from modelopt.torch.utils import get_unwrapped_name

from ..config import QuantizerAttributeConfig
from ..linear_attention.config import LinearAttentionConfig
from ..linear_attention.utils import state_quantizer_config
from ..nn import QuantModule, TensorQuantizer

__all__ = []


def _linear_attention_modules(model):
    return {
        get_unwrapped_name(name, model): module
        for name, module in model.named_modules()
        if isinstance(module, _LinearAttentionQuantMixin)
    }


def _apply_linear_attention_policy(model, config):
    """Assign complete policies in rule order, before compatibility validation."""
    modules = _linear_attention_modules(model)
    # getattr also handles pickled configs that predate the policy field.
    for entry in getattr(config, "linear_attention", []):
        matches = [name for name in modules if fnmatch.fnmatch(name, entry.module_name)]
        if not matches:
            raise ValueError(
                f"linear_attention rule {entry.module_name!r} matches no supported modules"
            )
        for name in matches:
            modules[name].linear_attention_config = entry.cfg.model_copy(deep=True)


def _validate_linear_attention(model):
    """Validate configured modules after both policy and quantizers are assigned."""
    for module in _linear_attention_modules(model).values():
        module.validate_linear_attention()


def _linear_attention_state(model):
    return {
        name: module.linear_attention_config.model_dump()
        for name, module in _linear_attention_modules(model).items()
    }


def _restore_linear_attention_policy(model, saved_policies):
    """Restore saved policies; modelopt_post_restore validates the completed state."""
    if saved_policies is None:
        return
    modules = _linear_attention_modules(model)
    if saved_policies.keys() != modules.keys():
        raise ApplyModeError("Saved linear_attention policies do not match the restored modules")
    for name, policy in saved_policies.items():
        modules[name].linear_attention_config = LinearAttentionConfig(**policy)


def _restore_legacy_linear_attention_quantizers(model, quantizer_state):
    """Keep newly introduced, disabled handles loadable from older checkpoints."""
    for name, module in _linear_attention_modules(model).items():
        for handle in module.linear_attention_quantizer_names:
            key = f"{name}.{handle}" if name else handle
            quantizer = getattr(module, handle)
            if key not in quantizer_state and not quantizer.is_enabled:
                quantizer_state[key] = quantizer.get_modelopt_state()


class _LinearAttentionQuantMixin(QuantModule):
    linear_attention_quantizer_names: tuple[str, ...] = ()

    def _setup(self):
        for name in self.linear_attention_quantizer_names:
            self._register_temp_attribute(
                name, TensorQuantizer(QuantizerAttributeConfig(enable=False))
            )
        self._register_temp_attribute("linear_attention_config", LinearAttentionConfig())
        self._register_temp_attribute("_linear_attention_prefill_lengths", None)

    @property
    def _linear_attn_state(self):
        return getattr(self, self.linear_attention_quantizer_names[0])

    @property
    def linear_attention_is_enabled(self):
        """Whether state quantization or the serving arithmetic policy is enabled."""
        return (
            any(getattr(self, name).is_enabled for name in self.linear_attention_quantizer_names)
            or self.linear_attention_config.backend == "serving"
        )

    def validate_linear_attention(self):
        """Validate quantizer contracts shared by GDN and KDA."""
        if self._linear_attn_state.is_enabled:
            state_format, group_size = state_quantizer_config(
                self._linear_attn_state,
                name=self.linear_attention_quantizer_names[0],
            )
            if self.linear_attention_config.backend != "serving":
                raise ValueError(
                    "GDN/KDA state QAT requires backend='serving' and prefill lengths "
                    "through linear_attention_training_phase"
                )
            if self.linear_attention_config.state_codec == "int8_hadamard32":
                if state_format != "int8":
                    raise ValueError("int8_hadamard32 requires INT8 state quantization")
                if group_size:
                    raise ValueError("TensorQuantizer block_sizes requires state_codec='tile'")

    def modelopt_post_restore(self, prefix=""):
        """Validate the restored numerical policy and quantizers."""
        super().modelopt_post_restore(prefix)
        self.validate_linear_attention()
