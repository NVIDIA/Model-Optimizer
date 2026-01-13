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

"""Model evaluation utilities for models split across multiple GPUs in pipeline-parallel mode.

Coordinates forward passes and loss computation through model shards distributed across GPUs
using sewing_kit's StitchedModule framework. Relies on validation.py for core loss computation.

Used by validate_model.py during activation scoring for sharded models.
"""
# mypy: ignore-errors

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import (
    DeciLMForCausalLM,
    LMHead,
)
from modelopt.torch.puzzletron.sewing_kit import (
    ExternalTarget,
    InputArgs,
    ModuleTarget,
    Needle,
    RemoteTarget,
    StitchedModule,
)
from modelopt.torch.puzzletron.sewing_kit.core import InputReducer
from modelopt.torch.puzzletron.sewing_kit.utils import (
    distributed_recv_obj,
    distributed_send_obj,
    fake_tensor,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import init_module_with_state_dict
from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import DummyBlock
from modelopt.torch.puzzletron.utils.validation import _organize_outputs, calculate_batch_outputs


class HiddenStatesAndLMHead(list):
    def __init__(self, hidden_states: list[torch.Tensor], lm_head_weights: torch.Tensor):
        super().__init__(hidden_states)
        self.lm_head_weights = lm_head_weights


@torch.no_grad()
def calculate_losses_pipeline(
    stitched_model: StitchedModule | DeciLMForCausalLM,
    dataloader: DataLoader | None,
    target_hidden_states_per_batch: HiddenStatesAndLMHead | None = None,
    return_hidden_states: bool = False,
    calculate_full_score_ablations: bool = False,
    calc_on_cpu: bool = False,
    just_model_forward: bool = False,
    checkpoint_manager=None,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, dict], HiddenStatesAndLMHead | None] | tuple[None, None]:
    """Do model forward on each batch and calculate LM loss.
    Optionally also calculate kl_div loss and other metrics from given target_hidden_states_per_batch.
    Optionally return hidden states per batch.
    Does not support data-parallel.
    just_model_forward: skip loss calculation, just forward the model. Useful for activation hooks.


    Returns:
        losses: dict = {
            "lm_loss": {
                "avg": float,
                "per_sample": list[float]
            }
            more metrics if provided with target_hidden_states_per_batch
        }
        target_hidden_states_per_batch: list[torch.Tensor], returned if return_hidden_states=True

    """
    if isinstance(stitched_model, DeciLMForCausalLM):
        stitched_model = perform_pipeline_stitches(stitched_model)

    params = list(stitched_model.parameters())
    model_device = params[0].device if params else "cpu"

    # Pre-populate outputs with dummy values for skipped batches
    start_batch = checkpoint_manager.current_batch if checkpoint_manager else 0
    if dist.is_last_process():
        outputs = [{"lm_loss": [0.0]}] * start_batch
    else:
        outputs = None

    if dist.is_master():
        all_input_ids, all_targets = zip(
            *[(batch["input_ids"], batch["targets"]) for batch in dataloader]
        )
        if dist.size() > 1:
            distributed_send_obj(all_targets, dst=dist.size() - 1)

    if dist.is_last_process():
        if dist.size() > 1:
            all_targets = distributed_recv_obj(src=0)

        lm_head: LMHead = next(
            module
            for module_name, module in stitched_model.named_modules()
            if "lm_head" in module_name
        )

        if target_hidden_states_per_batch is not None:
            lm_head_weights = target_hidden_states_per_batch.lm_head_weights
            with torch.device(model_device):
                target_lm_head = init_module_with_state_dict(
                    {"weight": lm_head_weights}, LMHead, *lm_head_weights.shape[::-1], bias=False
                )

    if dist.is_master():
        num_batches = len(all_input_ids)
        seq_len = all_input_ids[0].shape[1]
        if dist.size() > 1:
            torch.distributed.broadcast_object_list([num_batches, seq_len])

        # Create progress bar with sliced range starting from checkpoint position
        desc = (
            f"[rank {dist.rank()}] calculate_losses_pipeline("
            f"{(target_hidden_states_per_batch is None)=}, {return_hidden_states=}, {num_batches=})"
        )
        progress_bar = tqdm(range(start_batch, num_batches), desc=desc)
    else:
        obj_list = [None, None]
        if dist.size() > 1:
            torch.distributed.broadcast_object_list(obj_list)
        num_batches, seq_len = obj_list
        progress_bar = range(start_batch, num_batches)

    stitched_model.eval()

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        for i_batch in progress_bar:
            if dist.is_master():
                input_ids = all_input_ids[i_batch].to(model_device)
            else:
                input_ids = fake_tensor(1, seq_len, dtype=torch.long)

            output = stitched_model({}, {}, input_ids)

            if dist.is_last_process():
                logits = output.captured_outputs.get("model_output")
                logits = getattr(logits, "logits", logits)
                hidden_states = output.captured_outputs.get("hidden_states")
                targets = all_targets[i_batch].to(model_device)

                target_hidden_states = None
                target_logits = None
                if target_hidden_states_per_batch is not None:
                    target_hidden_states = target_hidden_states_per_batch[i_batch]
                    target_hidden_states = target_hidden_states.to(hidden_states.device)
                    target_logits = target_lm_head(target_hidden_states)

                if just_model_forward:
                    batch_outputs = {"lm_loss": [-1.0] * len(targets)}
                else:
                    batch_outputs = calculate_batch_outputs(
                        hidden_states,
                        target_hidden_states,
                        logits,
                        target_logits,
                        targets,
                        return_hidden_states,
                        calculate_full_score_ablations,
                        calc_on_cpu,
                    )

                outputs.append(batch_outputs)

            # Update checkpoint progress periodically
            if checkpoint_manager:
                checkpoint_manager.update_progress(i_batch + 1, num_batches)

    losses, hidden_states_per_batch = (
        _organize_outputs(outputs) if outputs is not None else (None, None)
    )

    if hidden_states_per_batch is not None:
        hidden_states_per_batch = HiddenStatesAndLMHead(
            hidden_states_per_batch, lm_head.weight.cpu()
        )

    dist.barrier()
    return losses, hidden_states_per_batch


def perform_pipeline_stitches(model: DeciLMForCausalLM) -> StitchedModule:
    target = ModuleTarget("module", model)
    stitcher = Needle()

    is_real_block = np.flatnonzero(
        [not isinstance(block, DummyBlock) for block in model.model.layers]
    )
    first_block, last_block = is_real_block.min(), is_real_block.max()

    if dist.rank() != 0:
        # receive activations from previous rank
        stitcher.stitch(
            RemoteTarget(peer_rank=dist.rank() - 1).value(
                name="activations", adapter=lambda x: InputArgs(x)
            ),
            target.input(
                name=f"model.layers.{first_block}",
                reducer=InputReducer(
                    lambda acc, override, orig, *args: override + orig.drop_args(0)
                ),
            ),
        )

    if not dist.is_last_process():
        # send activations to next rank
        stitcher.stitch(
            target.output(f"model.layers.{last_block}"),
            RemoteTarget(peer_rank=dist.rank() + 1).value(name="activations"),
        )
    else:
        # register model output
        stitcher.stitch(
            target.output(name="lm_head"),
            ExternalTarget().output("model_output"),
        )
        stitcher.stitch(
            target.output(name="model.norm"),
            ExternalTarget().output("hidden_states"),
        )

    stitched_module = stitcher.knot(ignore_extra_overrides=True)
    return stitched_module
