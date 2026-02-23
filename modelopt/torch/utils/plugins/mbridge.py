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

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from datasets import DatasetDict
from megatron.bridge import AutoBridge
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.loaders import setup_data_iterators
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.models.gpt_provider import GPTModelProvider, modelopt_transformer_layer_spec
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.models.mamba.mamba_provider import (
    MambaModelProvider,
    modelopt_mamba_stack_spec,
)
from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
    runtime_config_update,
)
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.core.parallel_state import get_data_parallel_group
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model
from transformers import AutoTokenizer

from modelopt.torch.utils import get_dataset_samples, print_rank_0, warn_rank_0

__all__ = [
    "get_hf_mbridge_calibration_loop",
    "load_mbridge_model_from_hf",
    "resolve_prunable_backbone",
]

_SUPPORTED_BACKBONE_TYPES = (GPTModel, MambaModel)


def _shape_from_hidden_states(t: torch.Tensor) -> tuple[int, int, torch.device] | None:
    """Infer (batch_size, seq_len, device) from hidden-state like tensors."""
    if t.ndim >= 3:
        # Megatron hidden states are [seq, batch, hidden].
        return t.shape[1], t.shape[0], t.device
    if t.ndim == 2:
        dim0, dim1 = t.shape[0], t.shape[1]
        seq_len, batch_size = (dim0, dim1) if dim0 >= dim1 else (dim1, dim0)
        return batch_size, seq_len, t.device
    return None


def _shape_from_decoder_input(module: nn.Module | None) -> tuple[int, int, torch.device] | None:
    """Infer (batch_size, seq_len, device) from decoder.input_tensor."""
    if module is None:
        return None
    decoder = getattr(module, "decoder", None)
    input_tensor = getattr(decoder, "input_tensor", None)
    if isinstance(input_tensor, torch.Tensor):
        return _shape_from_hidden_states(input_tensor)
    return None


def _shape_from_forward_kwargs(kwargs: dict[str, Any]) -> tuple[int, int, torch.device] | None:
    """Infer (batch_size, seq_len, device) from forward kwargs."""
    for key in ("input_ids", "labels"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return value.shape[0], value.shape[1], value.device

    decoder_input = kwargs.get("decoder_input")
    if isinstance(decoder_input, torch.Tensor):
        inferred = _shape_from_hidden_states(decoder_input)
        if inferred is not None:
            return inferred

    attention_mask = kwargs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        if attention_mask.ndim == 2:
            return attention_mask.shape[0], attention_mask.shape[1], attention_mask.device
        if attention_mask.ndim >= 4:
            return attention_mask.shape[0], attention_mask.shape[-1], attention_mask.device
    return None


def _infer_batch_shape(
    kwargs: dict[str, Any], module: nn.Module | None = None
) -> tuple[int, int, torch.device] | None:
    """Infer (batch_size, seq_len, device), preferring PP decoder input."""
    inferred = _shape_from_decoder_input(module)
    if inferred is not None:
        return inferred
    return _shape_from_forward_kwargs(kwargs)


def _ensure_mrope_position_ids(
    module: nn.Module | None, ensured_mrope_ids: set[int] | None = None
) -> None:
    """Ensure mRoPE forward receives position_ids.

    Megatron-Bridge's default GPT forward_step only provides position_ids on the first PP stage.
    For Qwen3-VL mRoPE models, later stages still require position_ids to build rotary embeddings.
    """
    if module is None:
        return
    if ensured_mrope_ids is not None and id(module) in ensured_mrope_ids:
        return
    cfg = getattr(module, "config", None)
    if getattr(cfg, "position_embedding_type", None) != "mrope":
        return

    original_forward = module.forward

    def _forward_with_position_ids(*args, **kwargs):
        if kwargs.get("position_ids") is None:
            inferred = _infer_batch_shape(kwargs, module)
            if inferred is not None:
                batch_size, seq_len, device = inferred
                kwargs["position_ids"] = (
                    torch.arange(seq_len, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
        return original_forward(*args, **kwargs)

    module.forward = _forward_with_position_ids  # type: ignore[assignment]
    if ensured_mrope_ids is not None:
        ensured_mrope_ids.add(id(module))


def resolve_prunable_backbone(
    unwrapped_model: nn.Module,
) -> tuple[GPTModel | MambaModel, bool, str | None]:
    """Resolve the mcore text backbone to prune from an unwrapped model.

    Returns:
        (backbone, is_vlm_wrapper, wrapper_name)
    """
    if isinstance(unwrapped_model, _SUPPORTED_BACKBONE_TYPES):
        return unwrapped_model, False, None

    language_model = getattr(unwrapped_model, "language_model", None)
    if isinstance(language_model, _SUPPORTED_BACKBONE_TYPES):
        return language_model, True, type(unwrapped_model).__name__

    raise TypeError(
        f"Expected {_SUPPORTED_BACKBONE_TYPES}, or a VLM wrapper whose .language_model "
        f"is one of them. Got {type(unwrapped_model)}"
    )


def load_mbridge_model_from_hf(
    *,
    hf_model_name_or_path: str,
    trust_remote_code: bool = False,
    provider_overrides: dict[str, Any] | None = None,
    init_model_parallel: bool = True,
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

    provider = bridge.to_megatron_provider()
    if provider_overrides:
        for key, value in provider_overrides.items():
            assert hasattr(provider, key), f"{type(provider)} does not have attribute {key}"
            setattr(provider, key, value)

    print_rank_0("Setting ModelOpt spec for model provider")
    if isinstance(provider, MambaModelProvider):
        provider.mamba_stack_spec = modelopt_mamba_stack_spec
    else:
        provider.transformer_layer_spec = modelopt_transformer_layer_spec

    provider.finalize()
    if init_model_parallel:
        provider.initialize_model_parallel(seed=0)

    model = provider.provide_distributed_model(wrap_with_ddp=False)
    assert len(model) == 1
    unwrapped_model = unwrap_model(model[0])
    resolve_prunable_backbone(unwrapped_model)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name_or_path, trust_remote_code=trust_remote_code
    )

    return bridge, provider, model, unwrapped_model, tokenizer


def _get_dataset_cfg(
    dataset_name: str,
    num_samples: int,
    seq_length: int,
    apply_chat_template: bool = True,
    tokenizer: AutoTokenizer | None = None,
) -> HFDatasetConfig:
    """Get a dataset config for the dataset."""
    dataset = get_dataset_samples(
        dataset_name, num_samples, apply_chat_template=apply_chat_template, tokenizer=tokenizer
    )
    dataset_cfg = HFDatasetConfig(
        dataset_name=f"{dataset_name}_{num_samples}",
        dataset_dict=DatasetDict({"train": dataset}),
        process_example_fn=lambda example, tokenizer: {"input": example, "output": ""},
        seq_length=seq_length,
        dataloader_type="batch",
        num_workers=1,
        do_validation=False,
        do_test=False,
        val_proportion=None,
        split_val_from_train=False,
        rewrite=True,
    )

    return dataset_cfg


def get_hf_mbridge_calibration_loop(
    *,
    model: list[MegatronModule],
    provider: GPTModelProvider | MambaModelProvider,
    tokenizer: AutoTokenizer,
    hf_model_name_or_path: str,
    trust_remote_code: bool = False,
    dataset_name: str = "nemotron-post-training-dataset-v2",
    num_samples: int = 512,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
) -> Callable[[nn.Module], None]:
    """Get a modelopt calibration loop for a Megatron-Bridge model.

    Args:
        model: The model to calibrate.
        provider: The provider to use for the model.
        tokenizer: The tokenizer to use for the model.
        hf_model_name_or_path: The name or path of the HF model.
        trust_remote_code: Whether to trust remote code.
        dataset_name: The name of the dataset to use for evaluation.
        num_samples: The number of samples to use for evaluation.
        micro_batch_size: The micro batch size to use for evaluation.
        global_batch_size: The global batch size to use for evaluation.

    Returns:
        A function that can be used to calibrate the model with a modelopt.torch API.
    """
    if global_batch_size < micro_batch_size:
        warn_rank_0(
            f"{global_batch_size=} is smaller than {micro_batch_size=}. Setting gbs to {micro_batch_size}."
        )
        global_batch_size = micro_batch_size
    num_iters = num_samples // global_batch_size

    # NOTE: Issue with NemotronH tokenizer's len() hence using use_fast=True as a WAR
    use_fast_tokenizer = isinstance(provider, NemotronHModelProvider)

    cfg = ConfigContainer(
        model=provider,
        train=TrainingConfig(
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            train_iters=num_iters,
            eval_iters=num_iters,
            skip_train=True,
        ),
        # TODO: Replace validation args in train with validation config in nemo:26.04
        # validation=ValidationConfig(eval_iters=num_iters, eval_interval=1, skip_train=True),
        dataset=_get_dataset_cfg(
            dataset_name,
            num_samples,
            provider.seq_length,
            apply_chat_template=True,
            tokenizer=tokenizer,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name_or_path,
            hf_tokenizer_kwargs={
                "trust_remote_code": trust_remote_code,
                "use_fast": use_fast_tokenizer,
            },
        ),
        # Unused
        optimizer=OptimizerConfig(optimizer="adam", lr=1e-4, use_distributed_optimizer=False),
        scheduler=SchedulerConfig(lr_decay_style="constant"),
        logger=LoggerConfig(),
        checkpoint=CheckpointConfig(),
    )
    runtime_config_update(cfg)

    state = GlobalState()
    state.cfg = cfg

    dataset_provider = get_dataset_provider(cfg.dataset)

    def _train_valid_test_datasets_provider(
        train_val_test_num_samples: tuple, dataset_cfg: HFDatasetConfig
    ):
        return dataset_provider(train_val_test_num_samples, dataset_cfg, tokenizer=state.tokenizer)

    train_data_iterator, _, _ = setup_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        model_length=len(model),
        train_valid_test_datasets_provider=_train_valid_test_datasets_provider,
        dp_group=get_data_parallel_group(),
    )

    ensured_mrope_ids: set[int] = set()

    def forward_loop(m):
        eval_model = m if isinstance(m, list) else [m]
        # mcore_minitron converts modules to DynamicModule before invoking this loop.
        # Ensure mRoPE modules on the converted eval model always receive position_ids.
        for local_model in eval_model:
            local_unwrapped = unwrap_model(local_model)
            _ensure_mrope_position_ids(local_unwrapped, ensured_mrope_ids)
            _ensure_mrope_position_ids(
                getattr(local_unwrapped, "language_model", None), ensured_mrope_ids
            )
        evaluate_and_print_results(
            state,
            prefix="iteration 1",
            forward_step_func=forward_step,
            data_iterator=train_data_iterator,
            model=eval_model,
            config=cfg,
            verbose=True,
            write_to_tensorboard=False,
        )

    return forward_loop
