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

import dataclasses
import os
import warnings
from contextlib import contextmanager
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from vllm_reload_utils import (
    convert_dict_to_vllm,
    convert_modelopt_state_to_vllm,
    process_state_dict_for_tp,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


@contextmanager
def disable_compilation(model):
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):
        do_not_compile = model.language_model.model.do_not_compile
        model.language_model.model.do_not_compile = True
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    try:
        yield
    finally:
        if hasattr(model, "model"):
            model.model.do_not_compile = do_not_compile
        elif hasattr(model, "language_model"):
            model.language_model.model.do_not_compile = do_not_compile


quant_config: dict[str, Any] = {
    "dataset": os.environ.get("QUANT_DATASET", "cnn_dailymail"),
    "calib_size": int(os.environ.get("QUANT_CALIB_SIZE", 512)),
    "quant_cfg": os.environ.get("QUANT_CFG", None),
    "kv_quant_cfg": os.environ.get("KV_QUANT_CFG", None),
    "quant_file_path": os.environ.get("QUANT_FILE_PATH", None),
    "modelopt_state_path": os.environ.get("MODELOPT_STATE_PATH", None),
    "calib_batch_size": int(os.environ.get("CALIB_BATCH_SIZE", 1)),
}


def update_kv_cfg_for_mla(model: torch.nn.Module, kv_quant_cfg: dict[str, Any]) -> dict[str, Any]:
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

    if kv_config := kv_quant_cfg.get("*[kv]_bmm_quantizer"):
        kv_quant_cfg["*kv_c_bmm_quantizer"] = kv_config
        kv_quant_cfg["*k_pe_bmm_quantizer"] = kv_config
        print("MLA detected: added *kv_c_bmm_quantizer and k_pe_bmm_quantizer config")

    return kv_quant_cfg


def _create_new_data_cls(data_cls, **kwargs):
    """vLLM's low-level API changes frequently. This function creates a class with parameters
    compatible with the different vLLM versions."""
    valid_params = {field.name for field in dataclasses.fields(data_cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return data_cls(**filtered_kwargs)


def _fakequant_run_prolog_worker(self) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        self.model_runner.model_config.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = self.model_runner.model
    if quant_config["modelopt_state_path"]:
        print(f"Loading modelopt state from {quant_config['modelopt_state_path']}")
        # Load on CPU to avoid failures when the checkpoint was saved from a different
        # GPU mapping
        modelopt_state = torch.load(
            quant_config["modelopt_state_path"], weights_only=False, map_location="cpu"
        )
        modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
        modelopt_state = convert_modelopt_state_to_vllm(modelopt_state)
        restore_from_modelopt_state(model, modelopt_state)

        if modelopt_weights is not None:
            modelopt_weights = convert_dict_to_vllm(modelopt_weights)
            mtq.utils.set_quantizer_state_dict(model, modelopt_weights)

    else:
        if quant_config["quant_file_path"]:
            print("Will load quant, so only do a single sample calibration")
            quant_config["calib_size"] = 1
        calib_dataloader = get_dataset_dataloader(
            dataset_name=quant_config["dataset"],
            tokenizer=tokenizer,
            batch_size=quant_config["calib_batch_size"],
            num_samples=quant_config["calib_size"],
            device=self.device,
        )

        def calibrate_loop(model: Any = None) -> None:
            for batch_idx, batch in tqdm(enumerate(calib_dataloader)):
                input_ids = batch["input_ids"][0]

                # Convert tensor to list of integers for vLLM compatibility
                if torch.is_tensor(input_ids):
                    input_ids_list = input_ids.cpu().tolist()
                else:
                    input_ids_list = list(input_ids)

                num_groups = len(self.model_runner.kv_cache_config.kv_cache_groups)
                empty_block_ids = tuple([] for _ in range(num_groups))

                req_id = f"req-{batch_idx}"
                # Pass all possible parameters - the helper will filter based on vLLM version
                new_req = _create_new_data_cls(
                    NewRequestData,
                    req_id=req_id,
                    prompt_token_ids=input_ids_list,
                    # Old API parameters
                    mm_kwargs=[],  # TODO: remove this when vllm <= 0.11 is outdated
                    mm_hashes=[],  # TODO: remove this when vllm <= 0.11 is outdated
                    mm_positions=[],  # TODO: remove this when vllm <= 0.11 is outdated
                    # New API parameter
                    mm_features=[],
                    sampling_params=SamplingParams(max_tokens=1),
                    pooling_params=None,
                    block_ids=empty_block_ids,
                    num_computed_tokens=0,
                    lora_request=None,
                )

                scheduler_output = _create_new_data_cls(
                    SchedulerOutput,
                    scheduled_new_reqs=[new_req],
                    scheduled_cached_reqs=CachedRequestData.make_empty(),
                    num_scheduled_tokens={req_id: len(input_ids_list)},
                    total_num_scheduled_tokens=len(input_ids_list),
                    scheduled_spec_decode_tokens={},
                    scheduled_encoder_inputs={},
                    num_common_prefix_blocks=[0] * num_groups,
                    finished_req_ids=set(),
                    free_encoder_mm_hashes=[],
                    kv_connector_metadata=None,
                    # Old API parameters
                    structured_output_request_ids={},  # TODO: remove this when vllm <= 0.11 is outdated
                    grammar_bitmask=None,  # TODO: remove this when vllm <= 0.11 is outdated
                )
                output = self.execute_model(scheduler_output)
                if hasattr(self, "sample_tokens"):
                    if output is None:  # TODO: make this default when vllm <= 0.11 is outdated
                        self.sample_tokens(None)

        quant_cfg = getattr(mtq, quant_config["quant_cfg"]) if quant_config["quant_cfg"] else {}
        quant_kv_cfg = (
            getattr(mtq, quant_config["kv_quant_cfg"]) if quant_config["kv_quant_cfg"] else {}
        )

        if hasattr(model, "unwrap"):
            model = model.unwrap()

        # Check if model has MLA and update KV config accordingly
        if quant_kv_cfg:
            quant_kv_cfg["quant_cfg"] = update_kv_cfg_for_mla(model, quant_kv_cfg["quant_cfg"])

        if quant_kv_cfg:
            quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(
                quant_cfg, quant_kv_cfg["quant_cfg"]
            )

        with disable_compilation(model):
            print("quantizing model...")
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        quantizer_file_path = quant_config["quant_file_path"]
        if quantizer_file_path:
            self.model_runner._dummy_run(1)
            print(f"Loading quantizer values from {quantizer_file_path}")
            # Load on CPU to avoid failures when the checkpoint was saved from a different
            # GPU mapping
            saved_quant_dict = torch.load(quantizer_file_path, map_location="cpu")
            # convert quant keys to vLLM format
            if hasattr(self.model_runner.model, "hf_to_vllm_mapper"):
                saved_quant_dict = self.model_runner.model.hf_to_vllm_mapper.apply_dict(
                    saved_quant_dict
                )
                saved_quant_dict = {
                    key.replace("quantizer_", "quantizer._"): value
                    for key, value in saved_quant_dict.items()
                    if key.endswith("quantizer_")
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
                    raise ValueError(
                        f"Key {key} not found in checkpoint state dict, but exists in model"
                    )

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

            model.load_state_dict(current_state_dict)

            # Only barrier if distributed is actually initialized (avoids deadlocks).
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    mtq.fold_weight(model)
    for name, module in model.named_modules():
        if name.endswith("weight_quantizer"):
            assert not module.is_enabled, f"quantizer {name} is still enabled"


class FakeQuantWorker(BaseWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        with disable_compilation(model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> None:
        if (
            quant_config["quant_cfg"]
            or quant_config["kv_quant_cfg"]
            or quant_config["modelopt_state_path"]
        ):
            _fakequant_run_prolog_worker(self)
        super().compile_or_warm_up_model()
