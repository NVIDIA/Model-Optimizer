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
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm_reload_utils import convert_dict_to_vllm, process_state_dict_for_tp

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


def load_state_dict_from_path(
    fakequant_runner: Any, quantizer_file_path: str, model: Any
) -> dict[str, Any]:
    fakequant_runner.model_runner._dummy_run(1)
    print(f"Loading quantizer values from {quantizer_file_path}")
    # Load on CPU to avoid failures when the checkpoint was saved from a different
    # GPU mapping
    saved_quant_dict = torch.load(quantizer_file_path, weights_only=True, map_location="cpu")
    # convert quant keys to vLLM format
    if hasattr(fakequant_runner.model_runner.model, "hf_to_vllm_mapper"):
        saved_quant_dict = fakequant_runner.model_runner.model.hf_to_vllm_mapper.apply_dict(
            saved_quant_dict
        )
        saved_quant_dict = {
            key.replace("quantizer_", "quantizer._"): value
            for key, value in saved_quant_dict.items()
            if "quantizer_" in key
        }
    saved_quant_dict = convert_dict_to_vllm(saved_quant_dict)

    current_state_dict = model.state_dict()
    # Count quant keys in checkpoint and model
    checkpoint_quant_keys = [key for key in saved_quant_dict if "quantizer" in key]
    model_quant_keys = [key for key in current_state_dict if "quantizer" in key]
    for key in checkpoint_quant_keys:
        if key not in model_quant_keys:
            print(f"Key {key} not found in model state dict, but exists in checkpoint")
    for key in model_quant_keys:
        if key not in checkpoint_quant_keys:
            raise ValueError(f"Key {key} not found in checkpoint state dict, but exists in model")

    checkpoint_quant_count = len(checkpoint_quant_keys)
    model_quant_count = len(model_quant_keys)

    # Ensure counts match
    if checkpoint_quant_count != model_quant_count:
        warnings.warn(
            f"Mismatch in quantizer state key counts: checkpoint has {checkpoint_quant_count} "
            f"quant keys but model has {model_quant_count} quantizer state keys. "
            f"This can happen if the model is using PP."
        )

    # Update quant values
    saved_quant_dict = process_state_dict_for_tp(saved_quant_dict, current_state_dict)
    for key, value in saved_quant_dict.items():
        if key in current_state_dict:
            current_state_dict[key] = value.to(current_state_dict[key].device)
    return current_state_dict


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
