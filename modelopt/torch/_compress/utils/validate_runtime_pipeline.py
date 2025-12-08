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
Model evaluation utilities for models split across multiple GPUs in pipeline-parallel mode.

Coordinates forward passes and loss computation through model shards distributed across GPUs
using sewing_kit's StitchedModule framework. Relies on validation.py for core loss computation.

Used by validate_model.py during activation scoring for sharded models.
"""
# mypy: ignore-errors

from statistics import mean

import numpy as np
import torch
import torch.distributed
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import modelopt.torch.utils.distributed as dist
from modelopt.torch._compress.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch._compress.decilm.deci_lm_hf_code.modeling_decilm import (
    DeciLMForCausalLM,
    LMHead,
)
from modelopt.torch._compress.sewing_kit import (
    ExternalTarget,
    InputArgs,
    ModuleTarget,
    Needle,
    RemoteTarget,
    StitchedModule,
)
from modelopt.torch._compress.sewing_kit.core import InputReducer
from modelopt.torch._compress.sewing_kit.utils import (
    distributed_recv_obj,
    distributed_send_obj,
    fake_tensor,
)
from modelopt.torch._compress.tools.checkpoint_utils import init_module_with_state_dict
from modelopt.torch._compress.tools.logger import mprint
from modelopt.torch._compress.tools.sharded_checkpoint_utils import DummyBlock
from modelopt.torch._compress.utils.validation import _organize_outputs, calculate_batch_outputs


@torch.no_grad()
def validate_pipeline_inner(
    stitched_model: StitchedModule, val_dataloader: DataLoader | None
) -> float:
    if dist.is_master():
        assert val_dataloader.batch_size is not None
    model_device = next(stitched_model.parameters()).device

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,  # TODO: make this configurable
    ):
        stitched_model.eval()

        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        losses: list[float] = []

        if dist.is_master():
            input_ids: torch.Tensor
            targets: torch.Tensor

            for i_batch, batch in enumerate(tqdm(val_dataloader)):
                input_ids, targets = (
                    batch["input_ids"].to(model_device),
                    batch["targets"].to(model_device),
                )

                if i_batch == 0:
                    num_batches = len(val_dataloader)
                    seq_len = input_ids.shape[1]
                    if torch.distributed.is_initialized():
                        torch.distributed.broadcast_object_list([(num_batches, seq_len)])

                all_targets.append(targets.cpu())

                output = stitched_model({}, {}, input_ids)
                logits = output.captured_outputs.get("model_output")
                logits = getattr(logits, "logits", logits)

                if logits is not None:
                    all_logits.append(logits.cpu())

                del output, logits

            if len(all_targets) > 0:
                distributed_send_obj(all_targets, dst=dist.size() - 1)

        else:
            obj_list: list[tuple] = [None]
            torch.distributed.broadcast_object_list(obj_list)
            num_batches, seq_len = obj_list[0]

            fake_input_ids = fake_tensor(
                1,
                seq_len,
                dtype=torch.bfloat16,  # TODO: make this configurable
            )

            for i in range(num_batches):
                output = stitched_model({}, {}, fake_input_ids)
                logits = output.captured_outputs.get("model_output")
                logits = getattr(logits, "logits", logits)
                if logits is not None:
                    all_logits.append(logits.cpu())
                del output, logits

            if len(all_targets) == 0 and dist.is_last_process():
                all_targets = distributed_recv_obj(src=0)

        torch.distributed.barrier()

        if len(all_logits) > 0:
            for logits, targets in zip(all_logits, all_targets):
                logits = logits.to("cuda")
                targets = targets.to("cuda")
                logit_losses = torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), targets, ignore_index=-1, reduction="none"
                )

                mean_losses = logit_losses.cpu().mean(dim=-1)
                losses.extend(mean_losses.tolist())

            val_loss = mean(losses)

            if not dist.is_master():
                distributed_send_obj(val_loss, dst=0)
        elif dist.is_master():
            val_loss = distributed_recv_obj()
        else:
            val_loss = float("nan")

        stitched_model.train()

    loss_list = [val_loss]
    torch.distributed.broadcast_object_list(loss_list)
    val_loss = loss_list[0]

    return val_loss


@torch.no_grad()
def validate_pipeline(
    stitched_model: StitchedModule,
    model_config: DeciLMConfig,
    val_dataloader: DataLoader,
    iter_num: int | None = None,
    max_iters: int | None = None,
    model_name: str | None = None,
    enable_print: bool = True,
    enable_wandb_log: bool = False,
    # pad_to_batchsize: bool = True,
) -> float:
    if enable_print:
        mprint("Validating ...")

    val_loss = validate_pipeline_inner(stitched_model=stitched_model, val_dataloader=val_dataloader)

    if dist.is_master():
        key = "val/loss" if model_name is None else f"val/{model_name}_loss"
        if enable_print:
            prefix = ""
            if iter_num is not None:
                prefix += f"iter {iter_num}"
                if max_iters is not None:
                    prefix += f"/{max_iters}"
                prefix += " - "
            mprint(f"{prefix}{key}: {val_loss:.4f}")
        if enable_wandb_log:
            wandb.log({key: val_loss}, step=iter_num)

    dist.barrier()

    return val_loss


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
) -> tuple[dict[str, dict], HiddenStatesAndLMHead | None] | tuple[None, None]:
    """
    Do model forward on each batch and calculate LM loss.
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

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,  # TODO: make this configurable
    ):
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
