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
# mypy: ignore-errors
"""
Provides a function to validate a model. Runs a model forward pass on a dataset and calculates
the loss, and optionally registers hooks to capture the inputs and the outputs
of pytorch modules that are used for activation scoring for pruning.

TODO: Consider moving this a separate module dedicated for scoring

Uses native HuggingFace models with deci_x_patcher for heterogeneous layer configurations.
"""

import textwrap
from pathlib import Path
from typing import Type

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress.activation_scoring.activation_hooks.utils import (
    register_activation_hooks,
)
from modelopt.torch._compress.anymodel.model_descriptor import (
    ModelDescriptor,
    ModelDescriptorFactory,
)
from modelopt.torch._compress.anymodel.puzzformer.no_op import Same
from modelopt.torch._compress.tools.logger import aprint, mprint
from modelopt.torch._compress.tools.sharded_checkpoint_utils import (
    load_and_shard_model,
    set_submodule,
)
from modelopt.torch._compress.utils.data.dataloaders import create_validation_dataloader
from modelopt.torch._compress.utils.parsing import (
    simple_parse_args_string,  # noqa: F401 (kept for backwards compat)
)
from modelopt.torch._compress.utils.validate_runtime_pipeline import (
    HiddenStatesAndLMHead,
    calculate_losses_pipeline,
)

"""
Two goals:
1) Calculate lm loss and token accuracy for a model.
May raise lots of NCCL warnings when it finishes, don't be alarmed.
Can be used to validate a HuggingFace model.
Automatically uses pipeline parallelism via device_map="auto".

2) Register hooks to capture the inputs and the outputs of pytorch modules.
For example, to collect activations scores for various layers (ffn, layer_norm, etc.)
that are used for pruning (ffn_hidden_size, embedding_pruning, etc).
See activations_log_dir and activation_hooks_kwargs arguments.
"""


@torch.no_grad()
def validate_model(
    args: DictConfig,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    target_hidden_states_per_batch: list[torch.Tensor] | None = None,
    return_hidden_states: bool = False,
    calculate_full_score_ablations: bool = False,
    val_dataloader: DataLoader | None = None,
) -> tuple[dict[str, dict], HiddenStatesAndLMHead | None] | tuple[None, None]:
    """Validate a language model on a dataset by calculating loss and optionally capturing activations.

    Args:
        args: Configuration object containing the following attributes:

            Model Configuration:
            - model_name_or_path (str): Path to model checkpoint or HuggingFace model name.
                                        Required unless model is passed directly.
            - model_dtype (str or torch.dtype): Model data type (e.g., "torch.bfloat16", torch.float16).
            - autocast_dtype (str or torch.dtype): Autocast data type for mixed precision.

            Dataset Configuration:
            - dataset_path (str): Path to the validation dataset.
            - tokenizer_name (str, optional): Tokenizer name/path. Uses model_name_or_path if not specified.
            - data_column (str): Column name in dataset containing text data.
            - block_size (int): Maximum sequence length for tokenization.
            - eval_samples (int, optional): Number of samples to evaluate. Uses all if None.
            - val_dataset_name (str): Name of validation dataset split.
            - source_datasets_to_discard (list[str], optional): List of source datasets to exclude.
            - load_dataset_fn (callable, optional): Custom function to load the dataset.

            Data Processing:
            - micro_batch_size (int): Batch size for evaluation.
            - seed (int): Random seed for reproducibility.
            - shuffle_seed (int, optional): Seed for shuffling data. Uses seed if None.
            - varlen (bool): Enable variable-length sequences.
            - bos_rate (float): Rate of adding BOS token.
            - fim_rate (float): Fill-in-the-middle rate for code completion tasks.
            - fim_spm_rate (float): SPM-based fill-in-the-middle rate.

            Activation Hooks:
            - activations_log_dir (str, optional): Directory to log activation scores. If provided,
                                                   hooks will be registered to capture activations.
            - activation_hooks_kwargs (str or dict, optional): Arguments for activation hooks.
                                                               If string, comma-separated format: "arg1=val1,arg2=val2".

            Execution Options:
            - calc_losses_on_cpu (bool): Calculate losses on CPU to avoid OOM. Very slow, not recommended.
            - write_results (bool): Write validation results to file.

        model: Pre-loaded model. If None, will be loaded from args.model_name_or_path.
        tokenizer: Pre-loaded tokenizer. If None, will be loaded based on args.
        target_hidden_states_per_batch: Target hidden states for pipeline parallel evaluation.
        return_hidden_states: Whether to return hidden states from the model.
        calculate_full_score_ablations: Calculate comprehensive teacher similarity scores.
                                         False calculates only a small suite for efficiency.
        val_dataloader: Pre-created validation dataloader. If None, will be created from args.

    Returns:
        A tuple containing:
        - losses: Dictionary mapping loss names to loss statistics (avg, per_sample).
        - hidden_states_per_batch: Hidden states and LM head outputs if return_hidden_states is True, else None.
        Returns (None, None) if not on master rank.
    """
    descriptor = ModelDescriptorFactory.get(args.descriptor)

    if val_dataloader is None:
        val_dataloader = prepare_dataloader(args, tokenizer) if dist.is_master() else None
    validation_full_iters = (
        args.eval_samples // args.micro_batch_size
    )  # model pipeline, single data rank

    model = prepare_model(args, descriptor=descriptor, model=model)

    just_model_forward = False
    checkpoint_manager = None
    activation_hooks = None

    if args.activations_log_dir is not None:
        activation_hooks_kwargs = args.activation_hooks_kwargs or {}
        activation_hooks_kwargs["validation_full_iters"] = validation_full_iters
        hook_class = args.hook_class

        # Create activation hooks using pruning mixin
        activation_hooks = register_activation_hooks(
            model=model,
            activation_hooks_kwargs=activation_hooks_kwargs,
            hook_class=hook_class,
            pruning_mixin=args.pruning_mixin,
        )

        # Create checkpoint manager with hooks
        from modelopt.torch._compress.utils.checkpoint_manager import ScoringCheckpointManager

        mprint(
            f"Creating checkpoint manager with {len(activation_hooks)} hooks for dir: {args.activations_log_dir}"
        )
        checkpoint_manager = ScoringCheckpointManager(
            checkpoint_dir=args.activations_log_dir,
            activation_hooks=activation_hooks,
            checkpoint_interval=50,  # Save every 50 batches
        )

        # Load existing checkpoint if available
        mprint("Attempting to load existing checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            mprint(f"Checkpoint loaded successfully: {checkpoint_data}")
        else:
            mprint("No checkpoint found, starting fresh")
        just_model_forward = True
        set_submodule(model, descriptor.output_embedding_name(), Same())

    losses, hidden_states_per_batch = calculate_losses_pipeline(
        stitched_model=model,
        dataloader=val_dataloader,
        target_hidden_states_per_batch=target_hidden_states_per_batch,
        return_hidden_states=return_hidden_states,
        calculate_full_score_ablations=calculate_full_score_ablations,
        calc_on_cpu=args.calc_losses_on_cpu,
        just_model_forward=just_model_forward,
        checkpoint_manager=checkpoint_manager,
        autocast_dtype=getattr(
            torch, getattr(args, "autocast_dtype", "torch.bfloat16").strip("torch.")
        ),
        descriptor=descriptor,
    )

    if losses is not None:
        avg_losses = {loss_name: loss_log["avg"] for loss_name, loss_log in losses.items()}

        results_str = f"""
            validate_model:
            {args.model_name_or_path=}
            Average losses = {avg_losses}
            Actual num samples = {len(next(iter(losses.values()))["per_sample"])}
            {args=}
        """
        results_str = textwrap.dedent(results_str)
        aprint(results_str)
        if args.write_results:
            Path(f"{args.model_name_or_path}/validate_model_results.txt").write_text(results_str)

    if activation_hooks is not None:
        hook_class.dump_activations_logs(activation_hooks, args.activations_log_dir, args)

    return losses, hidden_states_per_batch


def prepare_model(
    args: DictConfig,
    descriptor: Type[ModelDescriptor],
    model: PreTrainedModel | None = None,
) -> nn.Module:
    if model is None:
        assert args.model_name_or_path is not None
        model = load_and_shard_model(descriptor=descriptor, checkpoint_path=args.model_name_or_path)

    model.eval()
    return model


def prepare_dataloader(
    args: DictConfig, tokenizer: PreTrainedTokenizerBase | None = None
) -> DataLoader:
    if tokenizer is None:
        tokenizer_name = getattr(args, "tokenizer_name", None)
        assert (tokenizer_name is not None) or (args.model_name_or_path is not None)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or args.model_name_or_path, trust_remote_code=True
        )

    val_dataloader = create_validation_dataloader(
        accelerator=None,
        seed=args.seed,
        tokenizer=tokenizer,
        block_size=args.block_size,
        dataset=args.dataset_path,
        content_field=args.data_column,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        micro_batch_size=args.micro_batch_size,
        eval_samples=args.eval_samples,
        dataset_name=args.val_dataset_name,
        source_datasets_to_discard=args.source_datasets_to_discard,
        bos_rate=args.bos_rate,
        varlen=args.varlen,
        shuffle_seed=args.shuffle_seed,
        load_dataset_fn=args.load_dataset_fn,
    )

    return val_dataloader
