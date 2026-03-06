# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.vllm import update_kv_cfg_for_mla


def _create_new_data_cls(data_cls, **kwargs):
    """vLLM's low-level API changes frequently. This function creates a class with parameters
    compatible with the different vLLM versions."""
    valid_params = {field.name for field in dataclasses.fields(data_cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return data_cls(**filtered_kwargs)


def calibrate_fun(calib_dataloader: DataLoader, self: Any) -> Callable[[Any], None]:
    def calibrate_loop(model: Any) -> None:
        for batch_idx, batch in tqdm(enumerate(calib_dataloader)):
            input_ids_batch = batch["input_ids"]

            # Convert to list of flat token id lists (one per sequence in batch)
            if torch.is_tensor(input_ids_batch):
                input_ids_batch = input_ids_batch.cpu()
                # Handle both [batch_size, seq_len] and [seq_len]
                if input_ids_batch.dim() == 1:
                    input_ids_batch = input_ids_batch.unsqueeze(0)
                input_ids_list_batch = [seq.tolist() for seq in input_ids_batch]
            else:
                input_ids_list_batch = [
                    list(seq) if not isinstance(seq, list) else seq for seq in input_ids_batch
                ]
                if input_ids_list_batch and isinstance(input_ids_list_batch[0], int):
                    input_ids_list_batch = [input_ids_list_batch]

            num_groups = len(self.model_runner.kv_cache_config.kv_cache_groups)
            empty_block_ids = tuple([] for _ in range(num_groups))

            scheduled_new_reqs = []
            num_scheduled_tokens = {}
            total_tokens = 0
            for seq_idx, input_ids_list in enumerate(input_ids_list_batch):
                req_id = f"req-{batch_idx}-{seq_idx}"
                new_req = _create_new_data_cls(
                    NewRequestData,
                    req_id=req_id,
                    prompt_token_ids=input_ids_list,
                    mm_kwargs=[],
                    mm_hashes=[],
                    mm_positions=[],
                    mm_features=[],
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    block_ids=empty_block_ids,
                    num_computed_tokens=0,
                    lora_request=None,
                )
                scheduled_new_reqs.append(new_req)
                num_scheduled_tokens[req_id] = len(input_ids_list)
                total_tokens += len(input_ids_list)

            scheduler_output = _create_new_data_cls(
                SchedulerOutput,
                scheduled_new_reqs=scheduled_new_reqs,
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_tokens,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0] * num_groups,
                finished_req_ids=set(),
                free_encoder_mm_hashes=[],
                kv_connector_metadata=None,
                structured_output_request_ids={},
                grammar_bitmask=None,
            )
            output = self.execute_model(scheduler_output)
            if hasattr(self, "sample_tokens"):
                if output is None:  # TODO: make this default when vllm <= 0.11 is outdated
                    self.sample_tokens(None)

    return calibrate_loop


def get_quant_config(quant_config: dict[str, Any], model: Any) -> dict[str, Any]:
    quant_cfg = getattr(mtq, quant_config["quant_cfg"]) if quant_config["quant_cfg"] else {}
    quant_kv_cfg = (
        getattr(mtq, quant_config["kv_quant_cfg"]) if quant_config["kv_quant_cfg"] else {}
    )

    # Check if model has MLA and update KV config accordingly
    if quant_kv_cfg:
        quant_kv_cfg["quant_cfg"] = update_kv_cfg_for_mla(model, quant_kv_cfg["quant_cfg"])

    if quant_kv_cfg:
        quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
            quant_cfg, quant_kv_cfg["quant_cfg"]
        )

    return quant_cfg
