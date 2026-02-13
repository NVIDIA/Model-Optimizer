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
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset
from packaging.version import Version
from scripts.ar_validate import validate_ar
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

import modelopt
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
except ImportError:
    wandb = None

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class OfflineSupervisedDataset(Dataset):
    """Offline dataset for supervised fine-tuning.

    This dataset loads data on-the-fly from pre-processed .pt data files.

    Args:
        dumped_files (list): A list of file paths to the dumped .pt files.
    """

    def __init__(
        self,
        dumped_files,
    ):
        super().__init__()
        self.dumped_files = dumped_files

    def __len__(self):
        return len(self.dumped_files)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        offline_data = torch.load(self.dumped_files[i])

        labels = torch.full_like(offline_data["input_ids"], IGNORE_TOKEN_ID)
        labels[..., :-1] = offline_data["input_ids"][..., 1:]

        ret = {
            "input_ids": offline_data["input_ids"],
            "base_model_hidden_states": offline_data["hidden_states"],
            "aux_hidden_states": offline_data["aux_hidden_states"],
            "attention_mask": torch.ones_like(offline_data["input_ids"]),
            "loss_mask": torch.ones_like(offline_data["input_ids"]),
            "labels": labels,
        }
        return ret


class EagleOfflineDataCollator:
    """Data collator that truncate or pads data for offline training."""

    def __init__(self, train_len):
        self.train_len = train_len

    def _pad_or_truncate(self, x: torch.Tensor, length: int, dim: int = 0):
        """Pad or truncate a tensor to length along a given dimension."""
        dim = dim % x.ndim  # support negative dimension

        # allocate output tensor
        out_shape = list(x.shape)
        out_shape[dim] = length
        out = x.new_zeros(out_shape)

        # consturct copy slice
        slc = [slice(None)] * x.ndim
        slc[dim] = slice(0, min(length, x.size(dim)))

        # populate output tensor
        out[tuple(slc)] = x[tuple(slc)]
        return out

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        base_batch = {
            k: torch.stack([self._pad_or_truncate(item[k], self.train_len) for item in features])
            for k in ["input_ids", "attention_mask", "loss_mask", "labels"]
        }

        base_model_outputs = {
            k: torch.stack([self._pad_or_truncate(item[k], self.train_len) for item in features])
            for k in ["base_model_hidden_states", "aux_hidden_states"]
        }

        batch = {
            **base_batch,
            "base_model_outputs": base_model_outputs,
        }
        return batch


def make_eagle_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    train_len=None,
) -> dict:
    if data_args.offline_data_path is None:
        train_dataset = ShardedDataset("json", data_files=data_args.data_path)

        if not data_args.vlm_processor:
            data_collator = LanguageDataCollator(
                tokenizer=tokenizer,
                train_len=train_len,
                return_labels=True,
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
        dumped_files = [str(p) for p in offline_data_path.glob("*.pt")]
        if not dumped_files:
            raise ValueError(f"No .pt files found in {data_args.offline_data_path}")

        train_dataset = OfflineSupervisedDataset(dumped_files)
        data_collator = EagleOfflineDataCollator(train_len=train_len)

    return {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }


def load_vlm_or_llm_with_kwargs(model_name_or_path: str, **kwargs):
    """Load a VLM or LLM with kwargs. Returns the model and model config."""
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if "vl" in model_config.model_type.lower():
        model_cls = transformers.AutoModelForVision2Seq
    else:
        model_cls = transformers.AutoModelForCausalLM

    return model_config, model_cls.from_pretrained(
        model_name_or_path, trust_remote_code=True, **kwargs
    )


class EagleTrainerWithAccLog(Trainer):
    """Wrapper around Trainer that logs training accuracy."""

    def compute_loss(self, *args, **kwargs):
        """Override compute_loss to save train accs in trainer state."""
        if not hasattr(self.state, "training_accs"):
            self.state.training_accs = []
        kwargs.pop("num_items_in_batch", None)
        loss, outputs = super().compute_loss(return_outputs=True, *args, **kwargs)
        if hasattr(outputs, "train_acc"):
            self.state.training_accs.append(outputs.train_acc)
        return loss


class EagleTrainingPlot(TrainerCallback):
    """Callback that plot training acc and AR during training."""

    def __init__(self, ar_validate_steps: int = 1000, estimate_ar: bool = False):
        self.ar_validate_steps = ar_validate_steps
        if wandb and is_master():
            wandb.init()
        self.estimate_ar = estimate_ar

    def on_log(self, args, state, control, **kwargs):
        """Log training acc and estimate AR during log step."""
        if not hasattr(state, "training_accs") or len(state.training_accs) == 0:
            return control
        average_acc = np.mean(state.training_accs, axis=0)
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

        # log to wandb
        if wandb and is_master():
            for i, draft_acc in enumerate(average_acc):
                for j, step_acc in enumerate(draft_acc):
                    wandb.log(
                        {f"parallel_{i}_step_{j}_train_acc": step_acc}, step=state.global_step
                    )
            if self.estimate_ar:
                wandb.log({"estimated_training_ar": est_ar}, step=state.global_step)

        # reset training_accs
        state.training_accs = []
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Run AR validation periodically, if available."""
        if self.ar_validate_steps <= 0:
            return control
        if state.global_step % self.ar_validate_steps == 0 and state.global_step > 0:
            print_rank_0("Running AR validation...")
            try:
                ars = validate_ar(
                    model=kwargs["model"],
                    tokenizer=kwargs["processing_class"],
                    ds=load_dataset("HuggingFaceH4/mt_bench_prompts")["train"],
                    device=kwargs["model"].device,
                )
                print_rank_0(f"Step {state.global_step} AR: {sum(ars) / len(ars):.4f}")
                if wandb and is_master():
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
