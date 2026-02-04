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
"""Example script for pruning a GPT / Mamba model using Minitron algorithm on a Megatron-Bridge model (load from HF).

Example usage to prune Qwen3-8B to 6B on 2-GPUs (Pipeline Parallelism = 2)
while skipping pruning of num_attention_heads using following defaults:
    1024 samples from nemotron-post-training-dataset-v2 for calibration,
    at-most 20% depth (num_layers) and 40% width is pruned per prunable hparam (hidden_size, ffn_hidden_size, ...),
    top-10 candidates are evaluated for MMLU score (5% sampled data) to select the best model.

    torchrun --nproc_per_node 2 prune_minitron.py \
        --hf_model_name_or_path Qwen/Qwen3-8B \
        --prune_target_params 6e9 \
        --hparams_to_skip num_attention_heads \
        --output_hf_path /tmp/Qwen3-8B-Pruned-6B

To see the full usage for advanced configurations, run:
    python prune_minitron.py --help
"""

import argparse
import json
import os

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider
from transformers import AutoConfig, AutoModelForCausalLM

import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import get_supported_datasets, num2hrb, print_rank_0, warn_rank_0
from modelopt.torch.utils.plugins.mbridge import (
    get_hf_mbridge_calibration_loop,
    load_mbridge_model_from_hf,
)
from modelopt.torch.utils.plugins.megatron_mmlu import megatron_mmlu


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_model_name_or_path", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--output_megatron_path",
        type=str,
        help="Path to save the pruned model in Megatron checkpoint format",
    )
    target_group.add_argument(
        "--output_hf_path", type=str, help="Path to save the pruned model in HF checkpoint format"
    )

    # Uneven Pipeline Parallelism parameters
    parser.add_argument("--num_layers_in_first_pipeline_stage", type=int, default=None)
    parser.add_argument("--num_layers_in_last_pipeline_stage", type=int, default=None)

    # Calibration dataset parameters
    parser.add_argument(
        "--calib_dataset_name",
        type=str,
        default="nemotron-post-training-dataset-v2",
        choices=get_supported_datasets(),
        help="Dataset name for calibration",
    )
    parser.add_argument(
        "--calib_num_samples", type=int, default=1024, help="Number of samples for calibration"
    )
    # TODO: Add support for pre-training dataset (pre-tokenized)
    # TODO: only allow mbs>1 for pretraining dataset
    parser.add_argument(
        "--calib_mbs", type=int, default=1, choices=[1], help="Calibration micro-batch size"
    )
    parser.add_argument("--calib_gbs", type=int, default=1, help="Calibration global batch size")
    parser.add_argument("--seq_length", type=int, default=4096)

    # Pruning parameters
    parser.add_argument(
        "--prune_intermediate_ckpt",
        type=str,
        default=None,
        help=(
            "Path to save/restore intermediate pruning scores for resuming / faster re-run. "
            "If not provided, it will default to `<output_path>/modelopt_pruning_scores.pth`"
        ),
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--prune_export_config",
        type=str,
        help=(
            'Target pruned config as JSON e.g., \'{"hidden_size": 512, "ffn_hidden_size": 2048}\'. '
            f"Supported hyperparameters: {mtp.mcore_minitron.SUPPORTED_HPARAMS}. "
            "Cannot be used with --prune_target_params."
        ),
    )
    target_group.add_argument(
        "--prune_target_params",
        type=float,
        help=(
            "Target parameter count for pruning e.g., 6e9 for pruning to 6B params (total params, not active params). "
            "Uses Neural Architecture Search (NAS) to find the best pruned model that maximizes the --prune_score_func."
            "Cannot be used with --prune_export_config."
        ),
    )

    parser.add_argument(
        "--prune_score_func",
        type=str,
        choices=["mmlu_5pct"],
        default="mmlu_5pct",
        help=(
            "Score function to use for NAS-based pruning (--prune_target_params). Currently supported: "
            "mmlu_5pct (MMLU on 5% sampled data per subject for faster eval). "
        ),
    )
    parser.add_argument(
        "--max_width_pruning",
        type=float,
        default=0.4,
        help=(
            f"Maximum width pruning percentage ({mtp.mcore_minitron.SUPPORTED_HPARAMS - {'num_layers'}}) "
            "for NAS-based pruning (--prune_target_params)"
        ),
    )
    parser.add_argument(
        "--max_depth_pruning",
        type=float,
        default=0.2,
        help="Maximum depth pruning percentage ('num_layers') for NAS-based pruning (--prune_target_params)",
    )
    parser.add_argument(
        "--hparams_to_skip",
        nargs="*",
        type=str,
        default=[],
        choices=mtp.mcore_minitron.SUPPORTED_HPARAMS,
        help=(
            "Space-separated list of hparams to skip for NAS-based pruning (--prune_target_params) "
            "e.g. dont prune 'num_attention_heads'"
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help=(
            "Number of top candidates to consider for NAS-based pruning (--prune_target_params). "
            "Higher values will take longer to prune but may find a better model."
        ),
    )

    args = parser.parse_args()

    # Post-process arguments
    if args.prune_intermediate_ckpt is None:
        if args.output_megatron_path:
            args.prune_intermediate_ckpt = (
                f"{args.output_megatron_path}/modelopt_pruning_scores.pth"
            )
        elif args.output_hf_path:
            args.prune_intermediate_ckpt = f"{args.output_hf_path}/modelopt_pruning_scores.pth"
        print_rank_0(
            "No checkpoint provided to cache intermediate pruning scores. "
            f"Setting to: {args.prune_intermediate_ckpt}"
        )

    if args.prune_export_config:
        try:
            prune_export_config = json.loads(args.prune_export_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON for --prune_export_config: {args.prune_export_config}"
            ) from exc
        if not isinstance(prune_export_config, dict):
            raise ValueError("--prune_export_config must parse to a dictionary.")
        args.prune_export_config = prune_export_config

    print_rank_0("\n==================== Arguments ====================")
    for k, v in args.__dict__.items():
        print_rank_0(f"{k:<35} {v}")
    print_rank_0("===================================================\n")

    return args


def main(args: argparse.Namespace):
    pp_size = dist.size()
    print_rank_0(f"Setting pipeline_model_parallel_size to {pp_size}")

    if args.output_megatron_path and os.path.exists(
        f"{args.output_megatron_path}/latest_checkpointed_iteration.txt"
    ):
        warn_rank_0(f"\nPruned model already exists at {args.output_megatron_path}. Exiting...")
        return
    elif args.output_hf_path and os.path.exists(f"{args.output_hf_path}/config.json"):
        warn_rank_0(f"\nPruned model already exists at {args.output_hf_path}. Exiting...")
        return

    bridge, provider, model, unwrapped_model, tokenizer = load_mbridge_model_from_hf(
        hf_model_name_or_path=args.hf_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        provider_overrides={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": pp_size,
            "num_layers_in_first_pipeline_stage": args.num_layers_in_first_pipeline_stage,
            "num_layers_in_last_pipeline_stage": args.num_layers_in_last_pipeline_stage,
            "pipeline_dtype": torch.bfloat16,
            "seq_length": args.seq_length,
        },
        init_model_parallel=True,
    )
    print_rank_0(f"\nPruning {unwrapped_model=}")
    print_rank_0(
        f"Original model params: {num2hrb(mtp.mcore_minitron.get_mcore_param_count(unwrapped_model))}"
    )

    forward_loop = get_hf_mbridge_calibration_loop(
        model=model,
        provider=provider,
        tokenizer=tokenizer,
        hf_model_name_or_path=args.hf_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        dataset_name=args.calib_dataset_name,
        num_samples=args.calib_num_samples,
        micro_batch_size=args.calib_mbs,
        global_batch_size=args.calib_gbs,
    )

    pruning_config = {
        "forward_loop": forward_loop,
        "checkpoint": args.prune_intermediate_ckpt,
    }
    if args.prune_target_params is not None:
        # Restrict search space to a smaller set of candidates
        # NOTE: You can reduce the divisors and increase config['top_k'] to potentially find a better model.
        ss_config = mtp.mcore_minitron.get_mcore_minitron_config(
            hidden_size_divisor=256,
            ffn_hidden_size_divisor=512,
            mamba_head_dim_divisor=8,
            num_moe_experts_divisor=8,
            num_layers_divisor=2,
        )

        pruning_constraints = {"params": args.prune_target_params}
        print_rank_0(
            f"Using NAS-based automatic pruning with score function: {args.prune_score_func}"
            "You can change this to be any other metric you want to maximize (e.g. negative validation loss)."
        )

        def score_func_mmlu(m):
            return megatron_mmlu(m, tokenizer, percentage=0.05)

        pruning_config["score_func"] = score_func_mmlu
        pruning_config["max_width_pruning"] = args.max_width_pruning
        pruning_config["max_depth_pruning"] = args.max_depth_pruning
        pruning_config["hparams_to_skip"] = args.hparams_to_skip
        pruning_config["top_k"] = args.top_k
    elif args.prune_export_config is not None:
        # Less restrictive search space for manual pruning
        ss_config = mtp.mcore_minitron.get_mcore_minitron_config(
            hidden_size_divisor=64,
            ffn_hidden_size_divisor=64,
            mamba_head_dim_divisor=8,
            num_moe_experts_divisor=8,
            num_layers_divisor=1,
        )

        pruning_constraints = {"export_config": args.prune_export_config}
    print_rank_0(f"Pruning constraints: {pruning_constraints}")

    unwrapped_model, pruning_scores = mtp.prune(  # in-place pruning
        unwrapped_model,
        mode=[("mcore_minitron", ss_config)],  # type: ignore[arg-type]
        constraints=pruning_constraints,
        dummy_input=None,
        config=pruning_config,
    )
    # Remove unnecessary modelopt_state since ckpt is homogeneous
    if mto.ModeloptStateManager.has_state_for_mode_type("prune", model=unwrapped_model):
        mto.ModeloptStateManager.remove_state(unwrapped_model)
    if isinstance(provider, MambaModelProvider):
        provider.hybrid_override_pattern = unwrapped_model.hybrid_override_pattern
    print_rank_0(f"\nPruned {unwrapped_model=}")
    print_rank_0(
        f"Pruned model params: {num2hrb(mtp.mcore_minitron.get_mcore_param_count(unwrapped_model))}"
    )

    if args.output_megatron_path is not None:
        print_rank_0(
            f"Saved pruned model to {args.output_megatron_path} in Megatron checkpoint format"
        )

        # NOTE: Issue with NemotronH tokenizer's len() hence using use_fast=True as a WAR
        use_fast_tokenizer = isinstance(provider, NemotronHModelProvider)
        bridge.save_megatron_model(
            model,
            args.output_megatron_path,
            hf_tokenizer_path=args.hf_model_name_or_path,
            hf_tokenizer_kwargs={
                "trust_remote_code": args.trust_remote_code,
                "use_fast": use_fast_tokenizer,
            },
        )
        print_rank_0(
            f"Saved pruned model to {args.output_megatron_path} in Megatron checkpoint format"
        )
    else:
        print_rank_0(f"Saving pruned model to {args.output_hf_path} in HF checkpoint format")

        # [WAR] Hacky way to save pruned HF model until Megatron-Bridge natively supports it
        bridge.hf_pretrained.save_artifacts(args.output_hf_path)
        hf_cfg = AutoConfig.from_pretrained(
            args.output_hf_path, trust_remote_code=args.trust_remote_code
        )
        mcore_cfg = unwrapped_model.config

        hf_cfg.hidden_size = mcore_cfg.hidden_size
        hf_cfg.intermediate_size = mcore_cfg.ffn_hidden_size
        hf_cfg.num_attention_heads = mcore_cfg.num_attention_heads
        hf_cfg.head_dim = mcore_cfg.kv_channels
        hf_cfg.num_key_value_heads = mcore_cfg.num_query_groups
        if hasattr(hf_cfg, "mamba_num_heads"):
            hf_cfg.mamba_num_heads = mcore_cfg.mamba_num_heads
        if hasattr(hf_cfg, "mamba_head_dim"):
            hf_cfg.mamba_head_dim = mcore_cfg.mamba_head_dim
        if hasattr(hf_cfg, "moe_intermediate_size"):
            hf_cfg.moe_intermediate_size = mcore_cfg.moe_ffn_hidden_size
        if hasattr(hf_cfg, "moe_shared_expert_intermediate_size"):
            hf_cfg.moe_shared_expert_intermediate_size = (
                mcore_cfg.moe_shared_expert_intermediate_size
            )
        if hasattr(hf_cfg, "num_experts"):
            hf_cfg.num_experts = mcore_cfg.num_moe_experts
        if hasattr(hf_cfg, "n_routed_experts"):
            hf_cfg.n_routed_experts = mcore_cfg.num_moe_experts
        if hasattr(hf_cfg, "n_shared_experts"):
            hf_cfg.n_shared_experts = (
                mcore_cfg.moe_shared_expert_intermediate_size // mcore_cfg.moe_ffn_hidden_size
            )
        if hasattr(hf_cfg, "layer_types"):
            kept_layer_nums = pruning_scores["sorted_layers"][: mcore_cfg.num_layers]  # 1-indexed
            hf_cfg.layer_types = [
                lt for i, lt in enumerate(hf_cfg.layer_types) if i + 1 in kept_layer_nums
            ]
        if hasattr(hf_cfg, "hybrid_override_pattern"):
            hf_cfg.hybrid_override_pattern = unwrapped_model.hybrid_override_pattern
        hf_cfg.num_hidden_layers = mcore_cfg.num_layers

        # Save dummy pruned HF model to get the correct bridge for saving pruned weights
        AutoModelForCausalLM.from_config(
            hf_cfg, trust_remote_code=args.trust_remote_code
        ).save_pretrained(args.output_hf_path, trust_remote_code=args.trust_remote_code)
        pruned_bridge = AutoBridge.from_hf_pretrained(args.output_hf_path)
        pruned_bridge.save_hf_weights(model, args.output_hf_path)
        print_rank_0(f"Saved pruned model to {args.output_hf_path} in HF checkpoint format")

    print_rank_0("Done!")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        main(args)
    finally:
        dist.cleanup()
