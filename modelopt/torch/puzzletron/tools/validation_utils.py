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

"""Utility functions for validating models and extracting hidden states and similarity metrics.

TODO: Consider moving this a separate module dedicated for scoring.
"""

# mypy: ignore-errors

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import PreTrainedTokenizerBase

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools import validate_model
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.robust_json import json_dump
from modelopt.torch.puzzletron.utils.validation import LowMemorySparseTensor

if TYPE_CHECKING:
    from modelopt.torch.puzzletron.sewing_kit import StitchedModule


def validate_model_and_extract_hidden_states(
    args: DictConfig,
    model: "nn.Module | StitchedModule",
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    model_name: str,
    extra_payload: dict[str, Any] | None = None,
    pipeline_parallel: bool = False,
    val_dataloader=None,
) -> list[torch.Tensor | LowMemorySparseTensor]:
    mprint(f"""

################################################################
validate_model_and_extract_token_probs({model_name=})
################################################################

""")
    losses, hidden_states_per_batch = validate_model.validate_model(
        args,
        model,
        tokenizer,
        return_hidden_states=True,
        pipeline_parallel=pipeline_parallel,
        val_dataloader=val_dataloader,
    )
    if dist.is_last_process():
        output_dir = output_dir if (output_dir is not None) else args.bypass_dir
        extra_payload = extra_payload if (extra_payload is not None) else dict()
        write_results(output_dir, model_name, args, {**losses, **extra_payload})
    return hidden_states_per_batch


def validate_model_with_teacher_similarity_metrics(
    args: DictConfig,
    model: "nn.Module | StitchedModule",
    tokenizer: PreTrainedTokenizerBase,
    target_hidden_states_per_batch: list[torch.Tensor],
    output_dir: str | Path,
    model_name: str,
    extra_payload: dict[str, Any] | None = None,
    pipeline_parallel: bool = False,
    calculate_full_score_ablations: bool = False,
    val_dataloader=None,
) -> None:
    is_calc_kl_div = target_hidden_states_per_batch is not None
    mprint(f"""

################################################################
validate_model_with_kl_div({model_name=}, {is_calc_kl_div=})
################################################################

""")
    losses, _ = validate_model.validate_model(
        args,
        model,
        tokenizer,
        target_hidden_states_per_batch=target_hidden_states_per_batch,
        pipeline_parallel=pipeline_parallel,
        calculate_full_score_ablations=calculate_full_score_ablations,
        val_dataloader=val_dataloader,
    )
    if dist.is_last_process():
        extra_payload = extra_payload if (extra_payload is not None) else dict()
        write_results(output_dir, model_name, args, {**losses, **extra_payload})


def write_results(
    output_dir: str | Path, result_name: str, args: DictConfig, payload: dict[str, Any]
) -> None:
    output_path = Path(output_dir) / f"{result_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        **payload,
        "args": OmegaConf.to_container(args, resolve=True)
        if isinstance(args, DictConfig)
        else args.__dict__,
    }
    json_dump(results, output_path)
