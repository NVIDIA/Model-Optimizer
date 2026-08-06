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

import modelopt.torch.quantization as mtq
from modelopt.recipe import ModelOptPTQRecipe, load_recipe


def _create_new_data_cls(data_cls, **kwargs):
    """vLLM's low-level API changes frequently. This function creates a class with parameters
    compatible with the different vLLM versions."""
    valid_params = {field.name for field in dataclasses.fields(data_cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return data_cls(**filtered_kwargs)


def _patch_mamba_mixer2(model: Any) -> None:
    """Clamp NaN in MambaMixer2 and causal_conv1d_fn for hybrid Mamba+Attention models.

    Enable via CLAMP_MAMBA_NAN=1 when calibration produces NaN on models with
    MambaMixer2 layers (e.g. NemotronH at low TP).
    """
    mamba_cls = next(
        (type(m) for _, m in model.named_modules() if "MambaMixer2" in type(m).__name__),
        None,
    )
    if mamba_cls is None:
        return

    class PatchedMambaMixer2(mamba_cls):  # type: ignore[misc,valid-type]
        def forward(self, *args, **kwargs):
            return torch.nan_to_num(super().forward(*args, **kwargs), nan=0.0)

    for _, parent in model.named_modules():
        for _, child in list(parent.named_children()):
            if type(child) is mamba_cls:
                child.__class__ = PatchedMambaMixer2

    try:
        import vllm.model_executor.layers.mamba.mamba_mixer2 as _mm2

        _orig_conv = _mm2.causal_conv1d_fn
        _mm2.causal_conv1d_fn = lambda x, w, b=None, *a, **kw: torch.nan_to_num(
            _orig_conv(x, w, b, *a, **kw), nan=0.0
        )
    except Exception as e:
        warnings.warn(f"causal_conv1d_fn patch failed: {e}")


def calibrate_fun(
    calib_dataloader: DataLoader, self: Any, clamp_mamba_nan: bool = False
) -> Callable[[Any], None]:
    if clamp_mamba_nan:
        _patch_mamba_mixer2(self.model_runner.model)

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
            calib_block_ids = tuple([batch_idx] for _ in range(num_groups))

            scheduled_new_reqs = []
            num_scheduled_tokens = {}
            total_tokens = 0
            for seq_idx, input_ids_list in enumerate(input_ids_list_batch):
                req_id = f"req-{batch_idx}-{seq_idx}"
                new_req = _create_new_data_cls(
                    NewRequestData,
                    req_id=req_id,
                    prompt_token_ids=input_ids_list,
                    prefill_token_ids=input_ids_list,
                    mm_kwargs=[],
                    mm_hashes=[],
                    mm_positions=[],
                    mm_features=[],
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    block_ids=calib_block_ids,
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
                finished_req_ids=set(num_scheduled_tokens),
                free_encoder_mm_hashes=[],
                kv_connector_metadata=None,
                structured_output_request_ids={},
                grammar_bitmask=None,
            )
            try:
                output = self.execute_model(scheduler_output)
                if hasattr(self, "sample_tokens"):
                    if output is None:  # TODO: make this default when vllm <= 0.11 is outdated
                        self.sample_tokens(None)
            finally:
                # finish_requests runs before add_requests inside execute_model, so
                # req IDs aren't registered yet at that point — call it directly after.
                # Wrap in try/except so a cleanup error never masks the original exception.
                try:
                    if hasattr(self.model_runner, "finish_requests"):
                        cleanup_output = dataclasses.replace(
                            scheduler_output,
                            scheduled_new_reqs=[],
                            num_scheduled_tokens={},
                            total_num_scheduled_tokens=0,
                            finished_req_ids=set(num_scheduled_tokens.keys()),
                        )
                        self.model_runner.finish_requests(cleanup_output)
                    else:
                        warnings.warn(
                            "model_runner.finish_requests not found; request state may leak during calibration."
                        )
                except Exception as cleanup_err:
                    warnings.warn(f"Failed to clean up request state: {cleanup_err}")

    return calibrate_loop


def update_kv_cfg_for_mla(model: torch.nn.Module, kv_quant_cfg: list) -> list:
    """Update KV cache quantization config for MLA models.

    MLA uses `kv_c_bmm_quantizer` (compressed KV) instead of separate
    `k_bmm_quantizer` and `v_bmm_quantizer`. This function copies the
    config from `*[kv]_bmm_quantizer` to also cover `*kv_c_bmm_quantizer`.
    """
    try:
        from vllm.attention.layer import MLAAttention
    except ImportError:
        return kv_quant_cfg

    if not any(isinstance(m, MLAAttention) for m in model.modules()):
        return kv_quant_cfg

    kv_entry = next(
        (
            e
            for e in kv_quant_cfg
            if isinstance(e, dict) and e.get("quantizer_name") == "*[kv]_bmm_quantizer"
        ),
        None,
    )
    if kv_entry is not None:
        kv_config = kv_entry.get("cfg", {})
        kv_quant_cfg.append(
            {"quantizer_name": "*kv_c_bmm_quantizer", "cfg": kv_config, "enable": True}
        )
        kv_quant_cfg.append(
            {"quantizer_name": "*k_pe_bmm_quantizer", "cfg": kv_config, "enable": True}
        )
        print("MLA detected: added *kv_c_bmm_quantizer and k_pe_bmm_quantizer config")

    return kv_quant_cfg


def get_quant_config(quant_config: dict[str, Any], model: Any) -> dict[str, Any]:
    import copy

    if quant_config["recipe_path"]:
        recipe = load_recipe(quant_config["recipe_path"])
        assert isinstance(recipe, ModelOptPTQRecipe), (
            f"Expected PTQ recipe, but got {type(recipe).__name__} from {quant_config['recipe_path']}"
        )
        quant_cfg = recipe.quantize
    else:
        quant_cfg = (
            copy.deepcopy(getattr(mtq, quant_config["quant_cfg"]))
            if quant_config["quant_cfg"]
            else {}
        )
        quant_kv_cfg = (
            copy.deepcopy(getattr(mtq, quant_config["kv_quant_cfg"]))
            if quant_config["kv_quant_cfg"]
            else {}
        )

        # Check if model has MLA and update KV config accordingly
        if quant_kv_cfg:
            quant_kv_cfg["quant_cfg"] = update_kv_cfg_for_mla(model, quant_kv_cfg["quant_cfg"])

        if quant_kv_cfg:
            quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
                quant_cfg, quant_kv_cfg["quant_cfg"]
            )

    return quant_cfg
