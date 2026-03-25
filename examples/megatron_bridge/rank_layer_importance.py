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

"""Estimate per-layer importance by comparing final hidden states with a full-model reference."""

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.utils import WrappedTensor
from torch import Tensor
from transformer_engine.pytorch.module.layernorm import LayerNorm
from transformer_engine.pytorch.module.rmsnorm import RMSNorm

import modelopt.torch.prune as mtp
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import get_supported_datasets, num2hrb, print_rank_0
from modelopt.torch.utils.plugins.mbridge import (
    get_hf_mbridge_calibration_loop,
    load_mbridge_model_from_hf,
)

kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True).cuda()
mse_loss = torch.nn.MSELoss(reduction="mean").cuda()


def noop_mlp_forward_patch(
    hidden_states,
):
    """Return zero activations for the MLP sublayer (no contribution to hidden states)."""
    return torch.zeros_like(hidden_states), None


def noop_attn_forward_patch(
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
):
    """Return zero activations for the attention sublayer (no contribution to hidden states)."""
    return torch.zeros_like(hidden_states), None


def noop_mamba_forward_patch(
    hidden_states,
    attention_mask,
    inference_context=None,
    inference_params=None,
    rotary_pos_emb=None,
):
    """Return zero state from a Mamba-style block (no contribution to hidden states)."""
    return torch.zeros_like(hidden_states)


def noop_transformer_forward_patch(
    hidden_states,
    attention_mask,
    inference_context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
):
    """Pass through hidden states unchanged for a full transformer layer (identity layer)."""
    return hidden_states.clone(), inference_context


def noop_gpt_block_forward_patch(
    hidden_states: Tensor | WrappedTensor,
    attention_mask: Tensor | None,
    context: Tensor | None = None,
    context_mask: Tensor | None = None,
    rotary_pos_emb: Tensor | None = None,
    rotary_pos_cos: Tensor | None = None,
    rotary_pos_sin: Tensor | None = None,
    attention_bias: Tensor | None = None,
    inference_context: BaseInferenceContext | None = None,
    packed_seq_params: PackedSeqParams | None = None,
    padding_mask: PackedSeqParams | None = None,
    sequence_len_offset: Tensor | None = None,
    *,
    inference_params: BaseInferenceContext | None = None,
):
    """Replace a GPT decoder block with identity (clone input); used when simulating a dropped layer."""
    return hidden_states.clone(), inference_context


def normalized_mse_loss_per_sample(
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Per-batch-item normalized MSE between ``hidden_states`` and ``target_hidden_states``."""
    return torch.stack(
        [
            normalized_mse_loss(hidden_states[i_sample], target_hidden_states[i_sample])
            for i_sample in range(hidden_states.shape[0])
        ]
    )


def normalized_mse_loss(
    input: torch.Tensor, target: torch.Tensor, reduction: str = "mean", epsilon: float = 1e-6
) -> torch.Tensor:
    """MSE between ``input`` and ``target`` scaled by MSE of ``target`` to a near-zero baseline."""
    loss = F.mse_loss(input, target, reduction=reduction) / F.mse_loss(
        target, torch.zeros_like(target) + epsilon, reduction=reduction
    )
    return loss


class LastHiddenImportanceHook(torch.nn.Module):
    """Forward hook on final norm: stores reference hiddens, then MSE/logits distance vs. reference."""

    def __init__(self, module, name, nlast_tokens=0):
        """Initialize hook with module and name; store reference hiddens, logits, and hook."""
        super().__init__()

        self.forward_hook = module.register_forward_hook(self.hook_fn, with_kwargs=False)
        self.pre_forward_hook = None
        self.name = name
        self.activations_stats = defaultdict(list)
        self.hidden_distance = []
        self.logits_distance = []
        self.reference_hidden = []
        self.reference_load = True
        self.lm_head = None

    def set_lm_head(self, lm_head):
        """Attach the output LM head so logits distances can be computed alongside hidden MSE."""
        self.lm_head = lm_head

    def hook_fn(self, module, input, output):
        """Collect reference final hiddens or append normalized MSE (and optional logits MSE) vs. reference."""
        # seq x batch x dim
        hidden_out = output.detach().permute(1, 0, 2)  # batch x seq x dim

        # if loading the reference form teacher
        if self.reference_load:
            self.reference_hidden.append(hidden_out)
            return

        # if computing the distance to the reference
        sample_id = len(self.hidden_distance)
        # MSE
        self.hidden_distance.append(
            normalized_mse_loss_per_sample(hidden_out, self.reference_hidden[sample_id]).mean()
        )
        # if computing the distance to the teacher's logits
        if self.lm_head:
            teacher_logits = self.lm_head(self.reference_hidden[sample_id].permute(1, 0, 2))[
                0
            ].detach()
            logits = self.lm_head(hidden_out.permute(1, 0, 2))[0].detach()
            self.logits_distance.append(
                normalized_mse_loss_per_sample(logits, teacher_logits).mean()
            )

    def load_reference(self):
        """Clear buffers and switch to mode that records reference outputs on the next forward passes."""
        self.reference_hidden = []
        self.reference_load = True
        print_rank_0("> Loading reference outputs")

    def load_rankings(self):
        """First call ends reference capture; later calls gather per-step distances into ``activations_stats``."""
        if self.reference_load:  # the first call only swithches the accumultors
            self.reference_load = False
            return
        print_rank_0("> Computing distances to stored references")

        if len(self.hidden_distance) > 0:
            hidden_state_stats = self.gather_across_dp(torch.stack(self.hidden_distance))
            logits_stats = self.gather_across_dp(torch.stack(self.logits_distance))
        else:
            hidden_state_stats = torch.empty((0,)).cuda()
            logits_stats = torch.empty((0,)).cuda()

        self.activations_stats["mse"].append(hidden_state_stats)
        self.activations_stats["logits"].append(logits_stats)
        self.hidden_distance = []
        self.logits_distance = []

    def gather_across_dp(self, tensor):
        """Concatenate ``tensor`` from all data-parallel ranks along dim 0."""
        # Get the data parallel group
        dp_group = get_data_parallel_group()
        dp_world_size = get_data_parallel_world_size()

        # Create a list to hold tensors from all DP ranks
        tensor_list = [torch.empty_like(tensor) for _ in range(dp_world_size)]

        # Gather tensors from all DP ranks
        torch.distributed.all_gather(tensor_list, tensor, group=dp_group)
        return torch.cat(tensor_list, dim=0)

    def reset_stats(self):
        """Drop accumulated layer-wise stats before a new importance pass."""
        self.activations_stats = defaultdict(list)

    def close(self):
        """Remove the registered forward hook."""
        self.forward_hook.remove()


def setup_gates(unwrapped_model):
    """Register ``LastHiddenImportanceHook`` on final LayerNorm/RMSNorm modules as ``logits_gate_list``."""

    def setup_out_gate(unwrapped_model):
        """Attach hooks to every final norm module named with ``'final'``."""
        logits_importance = torch.nn.ModuleList()
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, (LayerNorm, RMSNorm)) and "final" in name:
                logits_importance.append(LastHiddenImportanceHook(module, name))
        unwrapped_model.logits_gate_list = logits_importance

    setup_out_gate(unwrapped_model)


def get_args() -> argparse.Namespace:
    """Parse CLI arguments for HF path, pipeline layout, calibration data, and optional dropped layers."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_model_name_or_path", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")

    # Uneven Pipeline Parallelism parameters
    parser.add_argument("--num_layers", type=int, required=True)
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
    parser.add_argument("--calib_gbs", type=int, default=1, help="Calibration global batch size")
    parser.add_argument("--seq_length", type=int, default=4096)

    parser.add_argument(
        "--drop_layers",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Layers to drop from the model (compute importance ranking as if these layers were already dropped)"
            "Useful for iterative pruning"
        ),
    )

    parser.add_argument(
        "--save_scores_path", type=str, default="scores.pt", help="Path to save scores"
    )
    args = parser.parse_args()

    print_rank_0("\n==================== Arguments ====================")
    for k, v in args.__dict__.items():
        print_rank_0(f"{k:<35} {v}")
    print_rank_0("===================================================\n")

    return args


def collect_scores(
    unwrapped_model,
    use_metric: str = "mse",
    aggregation: str = "mean",
    drop_blocks: list[int] = [],
    drop_group: int = 1,
):
    """Build dict layer_idx -> metric tensors from hook stats; print layer order by mean MSE."""
    stats = unwrapped_model.logits_gate_list[0].activations_stats
    metrics = list(stats.keys())
    num_layers = len(stats[metrics[0]])

    scores = {}
    for i in range(num_layers):
        scores[i] = {}
        for metric in metrics:
            scores[i][metric] = stats[metric][i].cpu()
    # print(f"{scores=}")
    print("Layers ordered by <MSE> importance:")
    res = sorted(
        [(k, v["mse"].mean()) for k, v in scores.items() if v["mse"].numel() > 0],
        key=lambda x: (x[1]),
    )
    print(f"{[r[0] for r in res]}")

    return scores


def estimate_layer_importance(args: argparse.Namespace):
    """Load model via Megatron-Bridge, record baseline final hiddens, then score each layer as identity-pass."""
    pp_size = dist.size()
    if args.num_layers_in_first_pipeline_stage is None:
        args.num_layers_in_first_pipeline_stage = args.num_layers // pp_size
    if args.num_layers_in_last_pipeline_stage is None:
        args.num_layers_in_last_pipeline_stage = args.num_layers - (
            args.num_layers_in_first_pipeline_stage * (pp_size - 1)
        )
    assert (
        args.num_layers_in_first_pipeline_stage + args.num_layers_in_last_pipeline_stage
        == args.num_layers
    ), "Number of layers in first and last pipeline stages must sum to the total number of layers"

    print_rank_0(f"Setting pipeline_model_parallel_size to {pp_size}")
    print_rank_0(
        f"Setting num_layers_in_first_pipeline_stage to {args.num_layers_in_first_pipeline_stage}"
    )
    print_rank_0(
        f"Setting num_layers_in_last_pipeline_stage to {args.num_layers_in_last_pipeline_stage}"
    )

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

    pp_rank = get_pipeline_model_parallel_rank()
    offset = (
        pp_rank * args.num_layers_in_first_pipeline_stage
        if args.num_layers_in_first_pipeline_stage
        else 0
    )

    print_rank_0(f"\nPruning {unwrapped_model=}")
    print_rank_0(
        f"Original model params: {num2hrb(mtp.mcore_minitron.get_mcore_param_count(unwrapped_model))}"
    )
    setup_gates(unwrapped_model)
    # set lm head in the last hidden hook
    if is_pipeline_last_stage():
        unwrapped_model.logits_gate_list[0].set_lm_head(unwrapped_model.output_layer)

    # Prepare model
    def patch_model(layer_id, block="transformer"):
        """No-op if ``layer_id < 0``; otherwise replace local decoder block forward with identity.

        Returns original forward.
        """
        if layer_id == -1:
            return None
        patch_register = unwrapped_model.decoder.layers[layer_id].forward
        unwrapped_model.decoder.layers[layer_id].forward = noop_gpt_block_forward_patch
        print_rank_0(f"Patched gpt block {layer_id} to noop_gpt_block_forward")

        return patch_register

    def unpatch_model(layer_id, patch_register, block="transformer"):
        """Restore original forward for local layer ``layer_id`` when ``layer_id >= 0``."""
        if layer_id == -1:
            return None
        print_rank_0(f"Unpatching gpt block {layer_id} ")
        unwrapped_model.decoder.layers[layer_id].forward = patch_register

    def layer_id_in_this_rank(layer_id):
        """Map global layer index to local decoder index on this pipeline stage, or ``-1`` if not local."""
        if (
            layer_id >= offset and layer_id < offset + args.num_layers_in_first_pipeline_stage
            if args.num_layers_in_first_pipeline_stage
            else 0
        ):
            return layer_id - offset
        else:
            return -1

    def load_reference():
        """On last pipeline stage, reset hook to capture reference final hiddens on next forwards."""
        if is_pipeline_last_stage():
            unwrapped_model.logits_gate_list[0].load_reference()

    def load_rankings():
        """On last pipeline stage, finalize hook pass (reference switch or append gathered distances)."""
        if is_pipeline_last_stage():
            unwrapped_model.logits_gate_list[0].load_rankings()

    def reset_stats():
        """On last pipeline stage, clear accumulated importance metrics in the hook."""
        if is_pipeline_last_stage():
            unwrapped_model.logits_gate_list[0].reset_stats()

    def reset_train_data_iterator():
        """New HF calibration forward loop over the configured dataset (fresh iterator)."""
        forward_loop = get_hf_mbridge_calibration_loop(
            model=model,
            provider=provider,
            tokenizer=tokenizer,
            hf_model_name_or_path=args.hf_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            dataset_name=args.calib_dataset_name,
            num_samples=args.calib_num_samples,
            micro_batch_size=1,
            global_batch_size=args.calib_gbs,
        )
        return forward_loop

    forward_loop = reset_train_data_iterator()
    load_reference()
    forward_loop(unwrapped_model)

    load_rankings()
    for layer_id in args.drop_layers:
        _ = patch_model(layer_id_in_this_rank(layer_id))

    reset_stats()

    # for each block compute logits and difference to the reference
    for layer_id in range(args.num_layers):
        # ignore blocks that are already dropped
        if layer_id in args.drop_layers:
            load_rankings()
            continue

        patch_register = patch_model(layer_id_in_this_rank(layer_id))
        forward_loop = reset_train_data_iterator()
        forward_loop(unwrapped_model)

        unpatch_model(layer_id_in_this_rank(layer_id), patch_register)  # , block=block)
        load_rankings()

    if is_pipeline_last_stage() and get_data_parallel_rank() == 0:
        scores = collect_scores(unwrapped_model)
        assert scores is not None
        torch.save(scores, f"scores_{get_pipeline_model_parallel_rank()}.pt")


if __name__ == "__main__":
    dist.setup()
    args = get_args()
    try:
        estimate_layer_importance(args)
    finally:
        dist.cleanup()
