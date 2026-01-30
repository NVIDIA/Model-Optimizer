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
import copy
import re
from collections import defaultdict
from warnings import warn

import torch
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron()

from megatron.core import dist_checkpointing
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    initialize_model_parallel,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    restore_sharded_modelopt_state,
    save_sharded_modelopt_state,
)
from modelopt.torch.utils import to_empty_if_meta_device

try:
    from megatron.core.ssm.mamba_layer import MambaLayer

    HAS_MAMBA = True
except ImportError as e:
    warn(f"Mamba not installed: {e}")
    HAS_MAMBA = False


@torch.no_grad()
def run_mcore_inference(
    model: GPTModel | MambaModel,
    prompt_tokens: torch.Tensor,
    active_hidden_size: int | None = None,
) -> torch.Tensor:
    """Run inference on a wrapped Megatron GPT or Mamba model.

    Args:
        model: Megatron GPT or Mamba model.
        prompt_tokens: Input tokens for inference.
        active_hidden_size: Hidden size to use for inference. If not provided, infer the hidden_size
            NOTE: `model.config.hidden_size` may not be the same as the active hidden size
                for the model since for a NAS search space-converted model, the hidden size
                may be different until the model is exported.
            NOTE: If depth pruned model and some PP have 0 layers, this would not work.
    """
    batch_size = prompt_tokens.shape[0]
    if active_hidden_size is None:
        if HAS_MAMBA and isinstance(model.decoder.layers[0], MambaLayer):
            active_hidden_size = model.decoder.layers[0].mixer.d_model
        elif isinstance(model.decoder.layers[0].self_attention, SelfAttention):
            if hasattr(model.decoder.layers[0].self_attention.linear_qkv, "in_features"):
                active_hidden_size = model.decoder.layers[0].self_attention.linear_qkv.in_features
            else:
                active_hidden_size = model.decoder.layers[0].self_attention.linear_qkv.input_size
        elif isinstance(model.decoder.layers[0].mlp, MLP):
            active_hidden_size = model.decoder.layers[0].mlp.linear_fc1.input_size
        else:
            raise ValueError(f"Cannot infer hidden size from {type(model.decoder.layers[0])=}")

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=active_hidden_size,
        inference_batch_times_seqlen_threshold=batch_size * model.max_sequence_length,
        fp32_residual_connection=False,
        params_dtype=torch.bfloat16 if model.config.bf16 else torch.float32,
        padded_vocab_size=model.vocab_size,
    )
    # Get full sequence output instead of only last token logits
    inference_context = StaticInferenceContext.from_config(inference_wrapper_config)
    inference_context.materialize_only_last_token_logits = False

    wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config, inference_context)
    wrapped_model.prep_model_for_inference()

    inference_input = wrapped_model.prep_inference_input(prompt_tokens)
    inference_input = wrapped_model.get_batch_for_context_window(
        inference_input, 0, model.max_sequence_length
    )

    # Note: This is returned in all TP ranks or last PP stage in PP models
    logits = wrapped_model.run_one_forward_step(inference_input)
    logits = broadcast_from_last_pipeline_stage(
        [batch_size, model.max_sequence_length, model.vocab_size],
        dtype=torch.bfloat16 if model.config.bf16 else torch.float32,
        tensor=logits,
    )
    return logits  # shape: (batch_size, max_sequence_length, vocab_size)


def run_mcore_inference_with_dummy_input(
    model: GPTModel | MambaModel, batch_size: int = 2, hidden_size: int | None = None
) -> torch.Tensor:
    """Run inference on a Megatron GPT or Mamba model with random dummy input."""
    prompt_tokens = torch.randint(
        0, model.vocab_size, (batch_size, model.max_sequence_length)
    ).cuda()
    return run_mcore_inference(model, prompt_tokens, hidden_size)


def get_batch(model, batch_size=2):
    seq_length = model.max_sequence_length
    vocab_size = model.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()
    labels = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()
    position_ids = (
        torch.arange(seq_length, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1).cuda()
    )
    attention_mask = torch.tril(
        torch.ones((batch_size, 1, seq_length, seq_length), dtype=torch.bool)
    ).cuda()
    loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float32).cuda()

    return input_ids, labels, position_ids, attention_mask, loss_mask


def get_forward(model, batch_size=2):
    """Return a forward function with cached batch inputs."""
    input_ids, labels, position_ids, attention_mask, loss_mask = get_batch(model, batch_size)

    def forward(model):
        # MambaModel doesn't accept loss_mask argument
        if isinstance(model, MambaModel):
            return model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        return model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

    return forward


def initialize_for_megatron(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    seed=1234,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
):
    """Initialize Megatron model parallelism.

    NOTE: If used in a non-spawned process, make sure to call `megatron.core.parallel_state.destroy_model_parallel()`.
    """
    initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
    )
    model_parallel_cuda_manual_seed(seed)


def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


def sharded_state_dict_test_helper(
    tmp_path, model_ref, model_test, forward_fn, meta_device=False, version=None
):
    logits_ref = forward_fn(model_ref)
    state_dict = copy.deepcopy(model_ref.state_dict())

    # Save Megatron-Core checkpoint and modelopt_state with `torch-dist` format.
    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)

    # Restore model_test from `torch-dist`.
    restore_sharded_modelopt_state([model_test], tmp_path)
    if meta_device:
        to_empty_if_meta_device(model_test, device="cuda")
    model_test = load_distributed_checkpoint(tmp_path, model_test)

    state_dict_test = model_test.state_dict()
    assert state_dict.keys() == state_dict_test.keys(), (
        f"{set(state_dict.keys()) - set(state_dict_test.keys())}"
    )

    def convert_maybe_fp8(v):
        if v.dtype == torch.float8_e4m3fn:
            return v.to(torch.float16)
        return v

    for k, v in state_dict.items():
        # sharded_state_dict will omit output_layer since we are lacking support on vocab padding
        # extra_state can be a byte Tensor where the value can change due to different serialized
        # order (serialized from a dict). As a result, we must skip checking extra_state.
        if (
            "_extra_state" in k
            or "output_layer" in k
            or k.endswith("._amax_for_smoothing")
            or not isinstance(v, torch.Tensor)
        ):
            continue
        assert v.dtype == state_dict_test[k].dtype, f"{k} v:{v}, s[k]: {state_dict_test[k]}"
        assert torch.allclose(convert_maybe_fp8(v), convert_maybe_fp8(state_dict_test[k])), (
            f"{k} v:{v}, s[k]: {state_dict_test[k]}"
        )

    logits_test = forward_fn(model_test)

    logits_diff = (logits_test - logits_ref) / logits_ref
    assert torch.allclose(logits_ref, logits_test), (
        f"diff: {logits_diff.max()} ref: {logits_ref}, test: {logits_test}"
    )

    # Test backward pass on model_test
    model_test.train()
    loss = forward_fn(model_test).sum()
    loss.backward()

    # Assert that trainable parameters have gradients computed
    for name, param in model_test.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient computed"


def copy_weights_from_grouped_to_non_grouped(te_grouped_moe_model, sequential_moe_model):
    """Copy weights from TEGrouped MoE model to sequential MoE model."""
    te_grouped_state = te_grouped_moe_model.state_dict()
    sequential_state = sequential_moe_model.state_dict()

    # Map grouped weights to sequential weights
    weight_mapping = {}
    sequential_key_template = "decoder.layers.{}.mlp.experts.local_experts.{}.linear_fc{}"
    for key, value in te_grouped_state.items():
        if "experts.linear_fc" in key and any(param in key for param in ("weight", "bias")):
            # Extract expert index from grouped weight name
            # Format: decoder.layers.X.mlp.experts.linear_fcY.weightZ
            parts = key.split(".")
            layer_idx = parts[2]  # X
            fc_idx = parts[5]  # Y (linear_fc1 or linear_fc2)
            param_idx = parts[6]  # weight0 / bias0 / etc.
            match = re.search(r"\d+", param_idx)
            expert_idx = match.group(0) if match else "0"  # Z for expert index
            # Map to sequential format: decoder.layers.X.mlp.experts.local_experts.Y.linear_fcZ
            sequential_key = sequential_key_template.format(layer_idx, expert_idx, fc_idx[-1])
            param_name = "weight" if "weight" in param_idx else "bias"
            weight_mapping[f"{sequential_key}.{param_name}"] = value
        elif isinstance(value, torch.Tensor):
            weight_mapping[key] = value

    # Copy weights to sequential model
    for sequential_key in sequential_state:
        if sequential_key in weight_mapping:
            sequential_state[sequential_key] = weight_mapping[sequential_key].clone()

    sequential_moe_model.load_state_dict(sequential_state)


def compare_amax_sync_across_expert_parallel(model, compare_across_experts=True):
    """
    Test if amax values are synchronized across expert parallel groups.

    Returns True if synchronized, False otherwise.
    """

    ep_group = get_expert_model_parallel_group(check_initialized=False)
    etp_group = get_expert_tensor_parallel_group(check_initialized=False)

    # Check if we have either expert model parallel or expert tensor parallel
    has_expert_parallel = (ep_group is not None and ep_group.size() > 1) or (
        etp_group is not None and etp_group.size() > 1
    )

    assert has_expert_parallel, "No expert parallelism detected"
    # Collect amax values from expert quantizers only
    expert_amax_values = {}
    for name, module in model.named_modules():
        if isinstance(module, mtq.nn.TensorQuantizer) and hasattr(module, "_amax"):
            # Check for both TEGrouped and sequential MoE patterns
            if "local_experts" in name or ("experts" in name and "linear_fc" in name):
                # Convert to scalar only if tensor has a single element
                expert_amax_values[name] = module.amax.detach().clone().cpu()

    # Early return if no expert quantizers found
    assert expert_amax_values, "No expert quantizers found"

    # Gather amax values from all ranks
    world_size = torch.distributed.get_world_size()
    all_amax_values = [None] * world_size
    torch.distributed.all_gather_object(all_amax_values, expert_amax_values)

    # Group quantizers by type (ignoring specific expert indices) and check sync
    expert_quantizers = defaultdict(dict)
    for rank_idx, rank_amax in enumerate(all_amax_values):
        for name, amax_val in rank_amax.items():
            # Create quantizer type key by normalizing the name
            quantizer_type = (
                re.sub(r"local_experts\.\d+", "local_experts.*", name)
                if "local_experts" in name
                else name
            )

            if (
                quantizer_type in expert_quantizers
                and rank_idx in expert_quantizers[quantizer_type]
            ):
                if compare_across_experts:
                    # compare expert value across expert for sequential MoE
                    prev_val = expert_quantizers[quantizer_type][rank_idx]
                    # Handle both scalar and tensor comparisons
                    if isinstance(amax_val, torch.Tensor) and isinstance(prev_val, torch.Tensor):
                        are_equal = torch.allclose(prev_val, amax_val, rtol=1e-6, atol=1e-6)
                    else:
                        are_equal = prev_val == amax_val
                    assert are_equal, (
                        f"{rank_idx}, {quantizer_type}, expert_quantizers[quantizer_type][rank_idx]: "
                        f"{expert_quantizers[quantizer_type][rank_idx]}, amax_val: {amax_val}"
                    )
            expert_quantizers[quantizer_type][rank_idx] = amax_val

    rank_info = {
        "global_rank": torch.distributed.get_rank(),
        "etp_rank": get_expert_tensor_parallel_rank(),
    }

    all_rank_info = [None] * world_size
    torch.distributed.all_gather_object(all_rank_info, rank_info)

    # Group ranks by ETP rank for fc1 (ColumnParallel: same output channels should match)
    etp_groups = defaultdict(list)
    for info in all_rank_info:
        etp_groups[info["etp_rank"] if info["etp_rank"] else 0].append(info["global_rank"])

    for quantizer_type, rank_values in expert_quantizers.items():
        # Determine which ranks should have same amax
        # Find which rank should have same amax
        #
        # fc1: ColumnParallel: X @ [A_1, A_2] (weights split along Cout)
        # so amax should be the same across same ETP rank
        # if EP is 2, ETP is 2, we have 4 ranks, EP1, ETP1: 0, EP1, ETP2: 1, EP2, ETP1: 2, EP2, ETP2: 3
        # so we need to compare amax across same ETP rank [0, 2] [1, 3] for per-channel quantization
        #
        # fc2: RowParallel:    [X_1, X_2] @  [A_1
        #                                     A_2] (weights split along Cin)
        # amax should be the same across all ranks
        rank_groups = (
            list(etp_groups.values())
            if "linear_fc1" in quantizer_type and (next(iter(rank_values.values()))).ndim > 0
            else [list(range(world_size))]
        )

        # Check each group independently
        for group in rank_groups:
            group_values = [rank_values[r] for r in group if r in rank_values]
            if len(group_values) > 1:
                # All values in this group should be identical
                first_val = group_values[0]
                for val in group_values[1:]:
                    if isinstance(first_val, torch.Tensor):
                        if not torch.allclose(first_val, val, rtol=1e-6, atol=1e-6):
                            group_rank_values = {
                                r: rank_values[r] for r in group if r in rank_values
                            }
                            return False, f"{quantizer_type} (group {group})", group_rank_values
                    elif abs(first_val - val) > 1e-6:
                        group_rank_values = {r: rank_values[r] for r in group if r in rank_values}
                        return False, f"{quantizer_type} (group {group})", group_rank_values

    return True, None, None
