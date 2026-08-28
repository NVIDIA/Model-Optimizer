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
"""Megatron-Bridge plugins for using with Model-Optimizer."""

from typing import Any

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.post_training.checkpointing import (
    _get_modelopt_checkpoint_path,
    has_modelopt_state,
    load_modelopt_state,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoTokenizer

from modelopt.torch.nas.plugins.megatron import get_te_hybrid_stack_spec
from modelopt.torch.utils import print_rank_0

__all__ = [
    "get_language_model",
    "is_vlm_config",
    "load_mbridge_model_from_hf",
    "load_modelopt_megatron_checkpoint",
]


def get_language_model(model: MegatronModule) -> tuple[MegatronModule, bool]:
    """Return ``(language_model, is_vlm)``; VLM wrappers nest it under ``.language_model``.

    Must agree with :func:`is_vlm_config`, which answers the same question before the
    Megatron model exists.
    """
    language_model = getattr(model, "language_model", None)
    return (model, False) if language_model is None else (language_model, True)


def is_vlm_config(hf_model_name_or_path: str, trust_remote_code: bool = False) -> bool:
    """Whether a HuggingFace checkpoint describes a VLM, from its config alone."""
    config = AutoConfig.from_pretrained(hf_model_name_or_path, trust_remote_code=trust_remote_code)
    return hasattr(config, "vision_config")


def load_mbridge_model_from_hf(
    *,
    hf_model_name_or_path: str,
    trust_remote_code: bool = False,
    provider_overrides: dict[str, Any] | None = None,
    init_model_parallel: bool = True,
    moe_grouped_gemm: bool = True,
    load_weights: bool = True,
) -> tuple[
    AutoBridge,
    GPTModelProvider | HybridModelProvider,
    list[MegatronModule],
    MegatronModule,
    AutoTokenizer,
]:
    """Load a Megatron-Bridge model from HF.

    Args:
        hf_model_name_or_path: The name or path of the HF model.
        trust_remote_code: Whether to trust remote code.
        provider_overrides: Overrides for the provider.
        init_model_parallel: Whether to initialize model parallel.
        moe_grouped_gemm: Whether to use grouped GEMM for MoE.
            Pruning does not support grouped GEMM yet.
        load_weights: Whether to load the HF weights into the model. Set to ``False`` when the
            weights will be loaded from a Megatron checkpoint instead (e.g. for export), in which
            case only the model structure (with the correct layer spec) is built.

    Returns:
        A tuple of (bridge, provider, model, unwrapped_model, tokenizer).
    """
    print_rank_0(f"Loading Megatron-Bridge model from HF: {hf_model_name_or_path}")
    trust_remote_code = is_safe_repo(
        trust_remote_code=trust_remote_code,
        hf_path=hf_model_name_or_path,
    )
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name_or_path, trust_remote_code=trust_remote_code
    )

    provider = bridge.to_megatron_provider(load_weights=load_weights)
    if provider_overrides:
        for key, value in provider_overrides.items():
            assert hasattr(provider, key), f"{type(provider)} does not have attribute {key}"
            setattr(provider, key, value)

    # Set moe_grouped_gemm on the provider (the bridge's native, possibly custom/hybrid spec reads
    # it at build time) rather than replacing the whole layer spec -- overwriting it would drop
    # custom layers (e.g. Qwen3.5's GatedDeltaNet + gated-attention or Gemma3's custom spec).
    if isinstance(provider, HybridModelProvider):
        provider.hybrid_stack_spec = get_te_hybrid_stack_spec(moe_grouped_gemm=moe_grouped_gemm)
        provider.moe_grouped_gemm = moe_grouped_gemm
    elif (provider.num_moe_experts or 0) > 0:
        provider.moe_grouped_gemm = moe_grouped_gemm
    provider.finalize()
    if init_model_parallel:
        provider.initialize_model_parallel(seed=0)

    model = provider.provide_distributed_model(wrap_with_ddp=False)
    assert len(model) == 1
    unwrapped_model = unwrap_model(model[0])
    # The optimization target is the inner GPTModel/HybridModel, but callers get the full
    # wrapper back so they can save the whole VLM.
    language_model, _ = get_language_model(unwrapped_model)
    assert isinstance(language_model, GPTModel | HybridModel), (
        f"Expected a GPTModel/HybridModel (optionally wrapped as `.language_model`), "
        f"got {type(unwrapped_model)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name_or_path, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for calibration

    return bridge, provider, model, unwrapped_model, tokenizer


def _checkpoint_keys(checkpoint_path: str) -> list[str]:
    """Tensor names in a Megatron distributed checkpoint."""
    return list(FileSystemReader(checkpoint_path).read_metadata().state_dict_metadata)


def _has_vision_model_weights(checkpoint_path: str) -> bool:
    """Whether a Megatron distributed checkpoint holds a VLM's vision tower (``vision_model.*``)."""
    return any(key.startswith("vision_model.") for key in _checkpoint_keys(checkpoint_path))


def load_modelopt_megatron_checkpoint(
    model: list[MegatronModule], megatron_path: str, restore_modelopt_state: bool = True
) -> None:
    """Load Megatron checkpoint weights (with modelopt_state).

    Args:
        model: The (pre-built) Megatron model to load the checkpoint into.
        megatron_path: Path to the quantized Megatron checkpoint (produced by ``quantize.py``)
        restore_modelopt_state: Whether to restore the ModelOpt state (e.g. quantizers) before loading
            weights. Set ``False`` to load weights only -- e.g. to reload a full-precision distilled
            student without reconstructing the ``kd_loss`` mode (which would require a teacher model).
    """
    # _load_model_weights_from_checkpoint does not resolve the latest iter_* directory, so resolve it explicitly
    checkpoint_path = _get_modelopt_checkpoint_path(megatron_path)

    # ``distill.py`` distills a VLM's language model only, so its checkpoint holds no ``vision_model.*``
    # weights and must be loaded into ``.language_model`` rather than the full VLM wrapper.
    unwrapped_model = unwrap_model(model)
    if any(get_language_model(m)[1] for m in unwrapped_model) and not _has_vision_model_weights(
        checkpoint_path
    ):
        print_rank_0("Language-model-only checkpoint: loading into the VLM's `.language_model`.")
        model = [get_language_model(m)[0] for m in unwrapped_model]

    # Restore the ModelOpt state before loading weights.
    # has_modelopt_state / load_modelopt_state resolves the latest iter_* directory
    if restore_modelopt_state:
        if has_modelopt_state(megatron_path):
            load_modelopt_state(model, megatron_path)
        elif any("_quantizer." in key for key in _checkpoint_keys(checkpoint_path)):
            # The quantizers cannot be rebuilt without the state, and the loader ignores the
            # leftover amax tensors, so the model would silently load unquantized.
            raise RuntimeError(
                f"{megatron_path} holds quantizer tensors but no restorable ModelOpt state. "
                "The state was dropped when the checkpoint was written -- re-run the step that "
                "produced it."
            )
    _load_model_weights_from_checkpoint(checkpoint_path, model)
