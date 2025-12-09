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

"""
Provides a function to validate a model. Runs a model forward pass on a dataset and calculates
the loss, and optionally registers hooks to capture the inputs and the outputs
of pytorch modules that are used for activation scoring for pruning.

TODO: Consider moving this a separate module dedicated for scoring.
"""

import argparse
import textwrap
from pathlib import Path

import torch.distributed
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress.activation_scoring.activation_hooks.utils import (
    register_activation_hooks,
)
from modelopt.torch._compress.tools.checkpoint_utils_hf import load_checkpoint
from modelopt.torch._compress.tools.logger import aprint, mprint
from modelopt.torch._compress.tools.sharded_checkpoint_utils import load_and_shard_model
from modelopt.torch._compress.utils.data.dataloaders import create_validation_dataloader
from modelopt.torch._compress.utils.parsing import simple_parse_args_string
from modelopt.torch._compress.utils.validate_runtime_pipeline import (
    HiddenStatesAndLMHead,
    calculate_losses_pipeline,
)
from modelopt.torch._compress.utils.validation import calculate_losses

"""
Two goals:
1) Calculate lm loss and token accuracy for a model.
May raise lots of NCCL warnings when it finishes, don't be alarmed.
Can be used to validate a HuggingFace model.
Automatically uses pipeline parallelism via device_map="auto".

2) Register hooks to capture the inputs and the outputs of pytorch modules.
For example, to collect activations scores for various layers (ffn, layer_norm, etc.)
that are used for pruning (ffn_hidden_size, embedding_pruning, etc).
See --activations_log_dir and --activation_hooks_kwargs args arguments.

"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Required unless a model is passed to the function",
    )
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--output_dir_name", type=str, default="validation")
    parser.add_argument(
        "--calculate_full_score_ablations",
        action="store_true",
        help="Calculates a diverse suite of teacher similarity scores. "
        "By default only a small suite is calculated, which is good for most use-cases.",
    )

    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--data_column", type=str, default="content")
    # TODO: Add help text for FIM rate, also for others less obvious args
    parser.add_argument("--fim_rate", type=float, default=0)
    parser.add_argument("--fim_spm_rate", type=float, default=0)
    parser.add_argument("--eval_samples", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--val_dataset_name", type=str, default="__auto__")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_datasets_to_discard", nargs="+", type=str)
    parser.add_argument("--bos_rate", type=float, default=1.0)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--pipeline_parallel", action="store_true")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--activations_log_dir", type=str, default=None)
    parser.add_argument(
        "--activation_hooks_kwargs",
        type=str,
        default=None,
        help="Comma separated string arguments, e.g. `arg1=val1,arg2=val2`",
    )
    parser.add_argument(
        "--calc_losses_on_cpu",
        action="store_true",
        help="Very slow, not recommended. Can help avoid OOM.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    args, unknown_args = parser.parse_known_args()
    return args


@torch.no_grad()
def validate_model(
    args: argparse.Namespace | DictConfig,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    target_hidden_states_per_batch: list[torch.Tensor] | None = None,
    return_hidden_states: bool = False,
    pipeline_parallel: bool = False,
    calculate_full_score_ablations: bool = False,
    val_dataloader: DataLoader | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, dict], HiddenStatesAndLMHead | None] | tuple[None, None]:
    if val_dataloader is None:
        val_dataloader = prepare_dataloader(args, tokenizer) if dist.is_master() else None
    validation_full_iters = (
        args.eval_samples // args.micro_batch_size
    )  # model pipeline, single data rank

    model = prepare_model(args, model, pipeline_parallel, dtype=dtype)

    just_model_forward = False
    checkpoint_manager = None
    activation_hooks = None

    if args.activations_log_dir is not None:
        activation_hooks_kwargs = (
            simple_parse_args_string(args.activation_hooks_kwargs)
            if isinstance(args.activation_hooks_kwargs, str)
            else args.activation_hooks_kwargs
        )
        activation_hooks_kwargs["validation_full_iters"] = validation_full_iters

        # Create activation hooks first
        activation_hooks, hook_class = register_activation_hooks(
            model=model, activation_hooks_kwargs=activation_hooks_kwargs
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
        model.lm_head = nn.Identity()

    if not pipeline_parallel:
        losses, hidden_states_per_batch = calculate_losses(
            model=model,
            dataloader=val_dataloader,
            checkpoint_manager=checkpoint_manager,
        )
    else:
        losses, hidden_states_per_batch = calculate_losses_pipeline(
            stitched_model=model,
            dataloader=val_dataloader,
            target_hidden_states_per_batch=target_hidden_states_per_batch,
            return_hidden_states=return_hidden_states,
            calculate_full_score_ablations=calculate_full_score_ablations,
            calc_on_cpu=args.calc_losses_on_cpu,
            just_model_forward=just_model_forward,
            checkpoint_manager=checkpoint_manager,
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
    args: argparse.Namespace,
    model: PreTrainedModel | None = None,
    pipeline_parallel: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if model is None:
        assert args.model_name_or_path is not None
        if pipeline_parallel:
            model = load_and_shard_model(
                args.model_name_or_path,
                model_config_overrides={"block_size": args.block_size},
                dtype=dtype,
            )
        else:
            try:
                model = load_checkpoint(
                    args.model_name_or_path,
                    model_config_overrides={"block_size": args.block_size},
                    ignore_unexpected_config_keys=True,
                )
                model.to("cuda")
            except FileNotFoundError:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True,
                )

    model.eval()
    return model


def prepare_dataloader(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase | None = None,
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


def main():
    args = parse_args()
    if args.pipeline_parallel:
        dist.setup(timeout=args.nccl_timeout_minutes)
    validate_model(args=args, pipeline_parallel=args.pipeline_parallel, dtype=torch.bfloat16)
    dist.cleanup()


if __name__ == "__main__":
    main()
