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

"""Checkpoint loading and calibration support for the Nemotron-H MTP tail."""

from __future__ import annotations

import copy
import inspect
import json
import re
import weakref
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import SafetensorError
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.masking_utils import create_causal_mask

from modelopt.torch.quantization.model_calib import (
    _LocalHessianInputHook,
    _needs_activation_forward_for_max_calib,
)
from modelopt.torch.quantization.utils.calib_utils import GPTQHelper
from modelopt.torch.utils import warn_rank_0
from modelopt.torch.utils.plugins.hf_checkpoint_utils import indexed_weight_map

__all__ = [
    "has_mtp_weights",
    "prepare_for_calibration",
    "prepare_for_loading",
]

_MTP_FORWARD_MODELS = weakref.WeakSet()


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if not path.is_dir():
        # Inspect an index without downloading its shards; an unsharded checkpoint needs its file.
        path = Path(
            snapshot_download(
                repo_id=str(checkpoint_path),
                allow_patterns=["config.json", "model.safetensors.index.json", "model.safetensors"],
            )
        )
    return path


def has_mtp_weights(checkpoint_path: str) -> bool:
    """Inspect local or Hub checkpoint tensors without constructing or executing model code."""
    return _get_mtp_layout(_resolve_checkpoint_path(checkpoint_path)) is not None


def _normalize_mtp_block_types(block_types, mixer_types):
    """Normalize checkpoint attention names to the installed Transformers registry."""
    aliases = {"attention": "full_attention", "full_attention": "attention"}
    normalized = []
    for block_type in block_types:
        if block_type in mixer_types:
            normalized.append(block_type)
        elif aliases.get(block_type) in mixer_types:
            normalized.append(aliases[block_type])
        else:
            raise ValueError(
                f"Unsupported NemotronH MTP block type {block_type!r}; "
                f"available mixer types: {sorted(mixer_types)}"
            )
    return normalized


def _get_mtp_layout(checkpoint_path: str | Path) -> tuple[str, int] | None:
    """Return the flattened MTP prefix and block count, rejecting unsupported tensor layouts."""
    try:
        weight_map = indexed_weight_map(checkpoint_path)
    except (OSError, ValueError, SafetensorError):
        return None
    matches = [
        match
        for name in weight_map
        if (match := re.fullmatch(r"((?:.*\.)?)mtp\.layers\.(\d+)\.(.+)", name))
    ]
    if not matches:
        return None
    prefixes = {match[1] for match in matches}
    prefix = matches[0][1]
    layers = {int(match[2]) for match in matches}
    if (
        len(prefixes) != 1
        or prefix not in ("", "language_model.")
        or sorted(layers) != list(range(len(layers)))
        or f"{prefix}mtp.layers.0.eh_proj.weight" not in weight_map
        or f"{prefix}mtp.layers.{max(layers)}.final_layernorm.weight" not in weight_map
    ):
        raise ValueError(
            "Unsupported Nemotron-H MTP tensor layout; refusing unquantized passthrough"
        )
    return prefix, len(layers)


class _NemotronHMTP(torch.nn.Module):
    """MTP fusion around native attention and MoE blocks with checkpoint-compatible names."""

    def __init__(self, config):
        super().__init__()
        # Native Nemotron-H is optional: older Transformers can still load unrelated models.
        from transformers.models.nemotron_h.modeling_nemotron_h import (
            MIXER_TYPES,
            NemotronHBlock,
            NemotronHRMSNorm,
        )

        block_config = copy.deepcopy(config)
        # Some remote-code revisions drop this field while materializing ``llm_config``.
        # The detected tensor layout is the canonical attention + MoE MTP tail.
        block_types = list(getattr(config, "mtp_layers_block_type", None) or ("attention", "moe"))
        block_config.layers_block_type = _normalize_mtp_block_types(block_types, MIXER_TYPES)
        self.layers = torch.nn.ModuleList(
            [NemotronHBlock(block_config, layer_idx) for layer_idx in range(len(block_types))]
        )
        first_layer, last_layer = self.layers[0], self.layers[-1]
        first_layer.enorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        first_layer.hnorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        first_layer.eh_proj = torch.nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        last_layer.final_layernorm = NemotronHRMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

    def forward(self, hidden_states, decoder_input, attention_mask=None, position_ids=None):
        first_layer = self.layers[0]
        decoder_input = torch.cat(
            (decoder_input[:, 1:, :], torch.zeros_like(decoder_input[:, :1, :])), dim=1
        )
        mtp_hidden = first_layer.eh_proj(
            torch.cat((first_layer.enorm(decoder_input), first_layer.hnorm(hidden_states)), dim=-1)
        )
        attention_mask = create_causal_mask(
            config=first_layer.config,
            inputs_embeds=mtp_hidden,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        for layer in self.layers:
            mtp_hidden = layer(
                mtp_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        return self.layers[-1].final_layernorm(mtp_hidden)


@contextmanager
def prepare_for_loading(checkpoint_path: str, trust_remote_code: bool, *, model_class=None):
    """Scope MTP construction to the selected class, or resolve the default AutoModel class."""
    local_path = _resolve_checkpoint_path(checkpoint_path)
    layout = _get_mtp_layout(local_path)
    config_path = local_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if layout is None:
        configs = (config, config.get("llm_config") or {}, config.get("text_config") or {})
        if any((cfg.get("num_nextn_predict_layers") or 0) > 0 for cfg in configs):
            warn_rank_0(
                "Nemotron-H config declares MTP but no MTP tensors were found; "
                "MTP will not be constructed or quantized.",
                stacklevel=2,
            )
        yield
        return

    prefix, num_blocks = layout
    if model_class is None:
        class_ref = config.get("auto_map", {}).get("AutoModelForCausalLM")
        if class_ref is None and prefix:
            class_ref = "modeling_nemotron_h_omni.NemotronH_Omni_Reasoning_V3"
        # AutoModel can use native Nemotron-H without consenting to the optional remote class.
        if not trust_remote_code and not prefix and config.get("model_type") == "nemotron_h":
            class_ref = None
        if class_ref is not None:
            if not trust_remote_code:
                raise ValueError(
                    "Loading Nemotron-H MTP remote code requires trust_remote_code=True"
                )
            # Preserve Hub IDs so the dynamic class has the same identity as in the loader.
            model_class = get_class_from_dynamic_module(class_ref, checkpoint_path)
        else:
            # The native class is optional for unrelated models and remote-code checkpoints.
            from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

            model_class = NemotronHForCausalLM
    original_init = model_class.__init__
    constructed = False

    @wraps(original_init)
    def init_with_mtp(self, *args, **kwargs):
        nonlocal constructed
        original_init(self, *args, **kwargs)
        language_model = self.get_submodule(prefix.rstrip(".")) if prefix else self
        if not hasattr(language_model, "mtp"):
            language_model.mtp = _NemotronHMTP(language_model.config)
        if len(language_model.mtp.layers) != num_blocks:
            raise ValueError("Nemotron-H MTP block count does not match the checkpoint tensors")
        constructed = True

    model_class.__init__ = init_with_mtp
    try:
        yield
        if not constructed:
            warn_rank_0(
                "Nemotron-H MTP tensors were found but the prepared model class was not "
                "constructed; MTP weights may remain unquantized.",
                stacklevel=2,
            )
    finally:
        model_class.__init__ = original_init


def prepare_for_calibration(full_model) -> bool:
    """Install an MTP forward for activation statistics and activation-dependent weight search."""
    language_model = getattr(full_model, "language_model", full_model)

    mtp = getattr(language_model, "mtp", None)
    if mtp is None:
        return False
    if language_model in _MTP_FORWARD_MODELS:
        return True

    original_forward = language_model.forward
    forward_signature = inspect.signature(original_forward)

    @wraps(original_forward)
    def forward_with_mtp(*args, **kwargs):
        # AWQ, GPTQ, and local-Hessian need inputs even with activation quantizers disabled.
        # Their wrappers/hooks exist only during calibration, unlike retained debug statistics.
        if not _needs_activation_forward_for_max_calib(mtp) and not any(
            hasattr(module, "_forward_no_awq")
            or hasattr(module, GPTQHelper.CACHE_NAME)
            or any(
                isinstance(hook, _LocalHessianInputHook)
                for hook in module._forward_pre_hooks.values()
            )
            for module in mtp.modules()
        ):
            return original_forward(*args, **kwargs)
        captured = []
        handle = language_model.model.norm_f.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0])
        )
        try:
            outputs = original_forward(*args, **kwargs)
        finally:
            handle.remove()
        if captured:
            arguments = forward_signature.bind(*args, **kwargs).arguments
            decoder_input = arguments.get("inputs_embeds")
            if decoder_input is None and arguments.get("input_ids") is not None:
                decoder_input = language_model.model.embeddings(arguments["input_ids"])
            if decoder_input is not None:
                mtp(
                    captured[-1],
                    decoder_input,
                    attention_mask=arguments.get("attention_mask"),
                    position_ids=arguments.get("position_ids"),
                )
        return outputs

    language_model.forward = forward_with_mtp
    _MTP_FORWARD_MODELS.add(language_model)
    print("Installed NemotronH MTP calibration forward", flush=True)
    return True
