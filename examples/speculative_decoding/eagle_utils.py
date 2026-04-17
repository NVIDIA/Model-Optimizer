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

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType
import numpy as np
import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from scripts.ar_validate import validate_ar
from transformers import Trainer, TrainerCallback

import modelopt
from modelopt.torch.speculative.eagle.utils import (
    EagleOfflineDataCollator,
    OfflineSupervisedDataset,
)
from modelopt.torch.speculative.utils import get_ttt_msk_func
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import is_master
from modelopt.torch.utils.plugins.transformers_dataset import (
    LanguageDataCollator,
    ShardedDataset,
    VisionLanguageDataCollator,
)

try:
    import wandb

    wandb.log  # Verify wandb is functional (not a stub module).
except (ImportError, AttributeError):
    wandb = None

# Re-export for backward compatibility
__all__ = ["EagleOfflineDataCollator", "OfflineSupervisedDataset"]


def make_speculative_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    train_len=None,
    answer_only_loss=False,
    shift_labels=True,
) -> dict:
    """Create data module for speculative decoding training.

    Args:
        shift_labels: If True, labels are shifted by 1 for autoregressive training (EAGLE3).
            If False, labels are unshifted for diffusion-style training (DFlash).
    """
    # Load chat template from file if provided
    chat_template = None
    if getattr(data_args, "chat_template", None):
        template_path = data_args.chat_template
        with open(template_path) as f:
            chat_template = f.read()
        print_rank_0(f"Loaded chat template from {template_path}")

    if data_args.offline_data_path is None:
        train_dataset = ShardedDataset("json", data_files=data_args.data_path)

        if not data_args.vlm_processor:
            data_collator = LanguageDataCollator(
                tokenizer=tokenizer,
                train_len=train_len,
                return_labels=True,
                answer_only_loss=answer_only_loss,
                shift_labels=shift_labels,
                chat_template=chat_template,
            )
        else:
            data_collator = VisionLanguageDataCollator(
                processor=data_args.vlm_processor,
                train_len=train_len,
                local_image_path=data_args.vlm_img_dir,
                return_labels=True,
            )

    else:
        print_rank_0("Loading pre-processed data for offline training...")
        assert not data_args.vlm_processor, "Offline data is not supported for VLM."

        offline_data_path = Path(data_args.offline_data_path)
        dumped_files = [str(p) for p in offline_data_path.rglob("*.pt")]
        if not dumped_files:
            raise ValueError(f"No .pt files found in {data_args.offline_data_path}")

        # sample_size=-1 means use all samples; positive integer selects that many
        if data_args.sample_size == 0 or data_args.sample_size < -1:
            raise ValueError("sample_size must be -1 (use all samples) or a positive integer")
        if data_args.sample_size > 0:
            dumped_files = dumped_files[: data_args.sample_size]
        train_dataset = OfflineSupervisedDataset(dumped_files)
        data_collator = EagleOfflineDataCollator(train_len=train_len)

    return {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }


class EagleTrainerWithAccLog(Trainer):
    """Wrapper around Trainer that logs training accuracy."""

    def __init__(
        self,
        *args,
        lora_lr_multiplier: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lora_lr_multiplier = lora_lr_multiplier

    def create_optimizer(self):
        """Override to give LoRA parameters a higher learning rate."""
        super().create_optimizer()
        if self.lora_lr_multiplier != 1.0:
            lora_ids = {
                id(p) for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad
            }
            if lora_ids:
                new_groups = []
                for group in self.optimizer.param_groups:
                    lora = [p for p in group["params"] if id(p) in lora_ids]
                    others = [p for p in group["params"] if id(p) not in lora_ids]
                    if lora and others:
                        new_groups.append({**group, "params": others})
                        new_groups.append(
                            {**group, "params": lora, "lr": group["lr"] * self.lora_lr_multiplier}
                        )
                    elif lora:
                        new_groups.append({**group, "lr": group["lr"] * self.lora_lr_multiplier})
                    else:
                        new_groups.append(group)
                self.optimizer.param_groups = new_groups
        return self.optimizer

    def compute_loss(self, *args, **kwargs):
        """Override compute_loss to save train accs and per-component losses in trainer state."""
        if not hasattr(self.state, "training_accs"):
            self.state.training_accs = []
        if not hasattr(self.state, "component_losses"):
            self.state.component_losses = {"eagle": [], "preservation": []}
        kwargs.pop("num_items_in_batch", None)
        loss, outputs = super().compute_loss(return_outputs=True, *args, **kwargs)
        if hasattr(outputs, "train_acc") and any(outputs.train_acc):
            self.state.training_accs.append(outputs.train_acc)
        # Track per-component losses
        for key, attr in [
            ("eagle", "eagle_loss"),
            ("preservation", "preservation_loss"),
        ]:
            val = getattr(outputs, attr, None)
            if val is not None:
                self.state.component_losses[key].append(val.item())
        return loss


class LoRAWarmupCallback(TrainerCallback):
    """Manages LoRA warmup: freezes LoRA during warmup, unfreezes after."""

    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self._activated = False

    def on_step_begin(self, args, state, control, **kwargs):
        """Check if warmup is over and activate LoRA co-training."""
        if self._activated:
            return control
        if state.global_step >= self.warmup_steps:
            model = kwargs["model"]
            # Unwrap DDP/FSDP if needed
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "_lora_cotraining_active"):
                raw_model._lora_cotraining_active = True
                # Unfreeze LoRA parameters
                lora_params = []
                for name, param in raw_model._base_model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True
                        lora_params.append(param)

                # Add LoRA params to optimizer — they were excluded at creation time
                # because requires_grad was False during warmup.
                optimizer = kwargs.get("optimizer")
                if optimizer is not None and lora_params:
                    existing_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
                    new_params = [p for p in lora_params if id(p) not in existing_ids]
                    if new_params:
                        optimizer.add_param_group(
                            {
                                "params": new_params,
                                "lr": optimizer.param_groups[0]["lr"],
                                "weight_decay": 0.0,
                            }
                        )
                        print_rank_0(f"  Added {len(new_params)} LoRA params to optimizer")

                print_rank_0(
                    f"Step {state.global_step}: LoRA warmup complete, enabling co-training."
                )
            self._activated = True
        return control


class EagleTrainingPlot(TrainerCallback):
    """Callback that plot training acc and AR during training."""

    def __init__(self, ar_validate_steps: int = 1000, estimate_ar: bool = False):
        self.ar_validate_steps = ar_validate_steps
        if wandb is not None and wandb.run is not None and is_master():
            wandb.init()
        self.estimate_ar = estimate_ar

    def on_log(self, args, state, control, **kwargs):
        """Log training acc and estimate AR during log step."""
        if not hasattr(state, "training_accs") or len(state.training_accs) == 0:
            return control
        average_acc = np.mean(state.training_accs, axis=0)
        # Always print accuracy to console
        try:
            acc_str = ", ".join(f"{a:.4f}" for a in np.array(average_acc).flatten())
            print_rank_0(f"Step {state.global_step} Training Acc: [{acc_str}]")
        except Exception:
            print_rank_0(f"Step {state.global_step} Training Acc: {average_acc}")
        # Log accuracy to HF Trainer's logs dict (picked up by TensorBoard)
        logs = kwargs.get("logs") or {}
        for i, draft_acc in enumerate(average_acc):
            for j, step_acc in enumerate(draft_acc):
                logs[f"train_acc/parallel_{i}_step_{j}"] = float(step_acc)
        if self.estimate_ar:
            # Calculate mean training AR since last log
            # NOTE: This is only an estimate of the real AR.
            est_ar = 1
            acc_cumprod = 1
            for step_acc in average_acc[0]:
                acc_cumprod *= step_acc
                est_ar += acc_cumprod
            # Parallel draft tokens only used after all eagle tokens
            for draft_acc in average_acc[1:]:
                acc_cumprod *= draft_acc[-1]
                est_ar += acc_cumprod
            print_rank_0(f"Step {state.global_step} Estimated Training AR: {est_ar:.4f}")
            logs["estimated_training_ar"] = est_ar

        # log to wandb
        if wandb is not None and wandb.run is not None and is_master():
            if logs:
                wandb.log({k: v for k, v in logs.items() if v is not None}, step=state.global_step)

            # Log per-component losses
            if hasattr(state, "component_losses"):
                for key, vals in state.component_losses.items():
                    if vals:
                        wandb.log({f"{key}_loss": np.mean(vals)}, step=state.global_step)

        # reset training_accs and component_losses
        state.training_accs = []
        if hasattr(state, "component_losses"):
            state.component_losses = {"eagle": [], "preservation": []}
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Run AR validation periodically, if available."""
        if self.ar_validate_steps <= 0:
            return control
        if state.global_step % self.ar_validate_steps == 0 and state.global_step > 0:
            print_rank_0("Running AR validation...")
            torch.cuda.empty_cache()
            try:
                ars = validate_ar(
                    model=kwargs["model"],
                    tokenizer=kwargs["processing_class"],
                    ds=load_dataset("HuggingFaceH4/mt_bench_prompts")["train"],
                    device=kwargs["model"].device,
                )
                print_rank_0(f"Step {state.global_step} AR: {sum(ars) / len(ars):.4f}")
                if wandb is not None and wandb.run is not None and is_master():
                    wandb.log({"validate_ar": sum(ars) / len(ars)}, step=state.global_step)
            except Exception:
                print_rank_0("AR validation not available.")
        return control


def get_patched_templated_ring_attn(orig_templated_attn: Callable):
    """
    Return patched version of
    torch.distributed.tensor.experimental._context_parallel._attention._templated_ring_attention
    to support TTT.
    """

    def _get_sharded_ttt_msk(i, rank, size, q_len, ttt_step, dtype):
        """Get chunk-interleaved TTT mask for current rank.
        e.g.:
        2 ranks, ttt_step=1;
        full_ttt_mask = [[0, 0, 0, 0,  x, 0, 0, 0],
                         [x, 0, 0, 0,  0, x, 0, 0],
                         [x, x, 0, 0,  0, 0, x, 0],
                         [x, x, x, 0,  0, 0, 0, x],

        rank 0, step0: [[0, 0,  x, 0],
                        [x, 0,  0, x]]

        rank 1, step0: [[0, 0,  x, 0],
                        [x, 0,  0, x]]

        rank 0, step1: [[0, 0,  0, 0],
                        [0, 0,  0, 0]]

        rank 1, step1: [[x, x,  0, 0],
                        [x, x,  0, 0]]

        """
        device = torch.cuda.current_device()
        q_indices = torch.arange(q_len * rank, q_len * (rank + 1), device=device)
        kv_indices = (
            torch.arange(q_len * size * (ttt_step + 1), device=device)
            .view(ttt_step + 1, size, q_len)[:, (rank - i) % size, :]
            .reshape(-1)
        )
        msk_func = get_ttt_msk_func(q_len * size, ttt_step)
        attn_mask = msk_func(
            None,
            None,
            q_indices.view(1, 1, -1, 1),
            kv_indices.view(1, 1, 1, -1),
        )
        attn_bias = torch.where(
            attn_mask,
            torch.zeros((), dtype=dtype, device=attn_mask.device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=attn_mask.device),
        )

        return attn_bias

    def patched_templated_attn(*args, **kwargs):
        """Patched version of _templated_ring_attention."""
        # Get original attention op
        # Sensitive to impl of _templated_ring_attention
        original_op = args[2]

        # This patch is only enabled for eagle model by context manager, not base model.
        patch_enbabled = modelopt.torch.speculative.plugins.transformers.ENABLE_CP_TTT_PATCH

        if patch_enbabled and original_op != torch.ops.aten._scaled_dot_product_cudnn_attention:
            raise ValueError(f"CP TTT only supports cudnn attention now. Got: {original_op}")

        # Unset is_causal to use custom attn mask
        if patch_enbabled:
            kwargs["is_causal"] = False

        def patched_op(*args, **kwargs):
            # Inspect the parent frame to get current shard info
            # This is sensitive to torch _templated_ring_attention impl
            try:
                frame: FrameType = inspect.currentframe()
                f_back: FrameType = frame.f_back
                rank = f_back.f_locals["rank"]
                size = f_back.f_locals["size"]
                query = f_back.f_locals["query"]
                key = f_back.f_locals["key"]
                i = f_back.f_locals["i"]
                ttt_step = (key.shape[2] // query.shape[2]) - 1
            except Exception as e:
                raise RuntimeError(
                    f"Failed to capture loop variables in patched _templated_ring_attention: {e}"
                ) from e
            # Set attn mask to permuted TTT mask
            if "attn_bias" in kwargs:
                kwargs["attn_bias"] = _get_sharded_ttt_msk(
                    i, rank, size, query.shape[2], ttt_step, query.dtype
                )
            # Perform shard attention
            return original_op(*args, **kwargs)

        return orig_templated_attn(args[0], args[1], patched_op, *args[3:], **kwargs)

    return patched_templated_attn


def patch_ring_attention_for_ttt():
    """Patch torch ring attention to support context parallelism for TTT."""
    # Torch Ring Attention only supports no mask or causal mask. We apply the following patches to enable TTT mask.

    if Version(torch.__version__) < Version("2.10.0"):
        raise RuntimeError(
            f"Context parallel TTT only supported for PyTorch >= 2.10.0. "
            f"Got {torch.__version__}. "
            f"Please use torch 2.10.0 or cp_size=1."
        )

    from torch.distributed.tensor.experimental._context_parallel import _attention

    # 1. Disable load balance, which is designed for causal mask.
    # This affect how buffers are sharded. So need to be done permanently before accelerate/hf trainer init.
    _attention._cp_options.enable_load_balance = False

    # 2. Patch templated ring attention for TTT mask.
    original_templated_ring_attention = _attention._templated_ring_attention
    original_templated_ring_attention_backward = _attention._templated_ring_attention_backward
    _attention._templated_ring_attention = get_patched_templated_ring_attn(
        original_templated_ring_attention
    )
    _attention._templated_ring_attention_backward = get_patched_templated_ring_attn(
        original_templated_ring_attention_backward
    )

    # 3. Patch merger to skip the blank shard to avoid difference in output.
    original_sdpa_merger_step = _attention._SDPAMerger.step

    def patched_sdpa_merger_step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool):
        if lse.sum() <= 0:
            return
        return original_sdpa_merger_step(self, out, lse, partial)

    _attention._SDPAMerger.step = patched_sdpa_merger_step
