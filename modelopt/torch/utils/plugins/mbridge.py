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
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.post_training.checkpointing import (
    _get_modelopt_checkpoint_path,
    has_modelopt_state,
    load_modelopt_state,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.mamba import MambaModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model
from transformers import AutoTokenizer

from modelopt.torch.nas.plugins.megatron import get_te_mamba_stack_spec
from modelopt.torch.utils import print_rank_0

__all__ = ["load_mbridge_model_from_hf", "load_modelopt_megatron_checkpoint"]


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
    GPTModelProvider | MambaModelProvider,
    list[MegatronModule],
    GPTModel | MambaModel,
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

    # Only MoE models need their layer spec overridden to disable moe_grouped_gemm (not supported
    # by pruning yet). Dense models keep the bridge's native spec, which is required for models
    # with custom layers (e.g. Gemma3's gemma3_layer_spec) to be built correctly.
    if isinstance(provider, MambaModelProvider):
        provider.mamba_stack_spec = get_te_mamba_stack_spec(moe_grouped_gemm=moe_grouped_gemm)
    elif (provider.num_moe_experts or 0) > 0:
        provider.transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=provider.num_moe_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=provider.qk_layernorm,
        )
    provider.finalize()
    if init_model_parallel:
        provider.initialize_model_parallel(seed=0)

    model = provider.provide_distributed_model(wrap_with_ddp=False)
    assert len(model) == 1
    unwrapped_model = unwrap_model(model[0])
    assert isinstance(unwrapped_model, (GPTModel, MambaModel))

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name_or_path, trust_remote_code=trust_remote_code
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for calibration

    return bridge, provider, model, unwrapped_model, tokenizer


def load_modelopt_megatron_checkpoint(model: list[MegatronModule], megatron_path: str) -> None:
    """Load Megatron checkpoint weights (with modelopt_state).

    Args:
        model: The (pre-built) Megatron model to load the checkpoint into.
        megatron_path: Path to the quantized Megatron checkpoint (produced by ``quantize.py``)
    """
    # Restore the ModelOpt state before loading weights.
    # has_modelopt_state / load_modelopt_state resolves the latest iter_* directory
    if has_modelopt_state(megatron_path):
        load_modelopt_state(model, megatron_path)
    # _load_model_weights_from_checkpoint does not resolve the latest iter_* directory, so resolve it explicitly
    _load_model_weights_from_checkpoint(_get_modelopt_checkpoint_path(megatron_path), model)
