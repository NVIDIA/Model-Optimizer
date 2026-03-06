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


import os
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from vllm_ptq_utils import calibrate_fun, get_quant_config
from vllm_reload_utils import (
    convert_dict_to_vllm,
    convert_modelopt_state_to_vllm,
    load_state_dict_from_path,
    restore_from_modelopt_state_vllm,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.vllm import disable_compilation
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

quant_config: dict[str, Any] = {
    "dataset": os.environ.get("QUANT_DATASET", "cnn_dailymail"),
    "calib_size": int(os.environ.get("QUANT_CALIB_SIZE", 512)),
    "quant_cfg": os.environ.get("QUANT_CFG", None),
    "kv_quant_cfg": os.environ.get("KV_QUANT_CFG", None),
    "quant_file_path": os.environ.get("QUANT_FILE_PATH", None),
    "modelopt_state_path": os.environ.get("MODELOPT_STATE_PATH", None),
    "calib_batch_size": int(os.environ.get("CALIB_BATCH_SIZE", 1)),
}


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
            quant_config["modelopt_state_path"], weights_only=True, map_location="cpu"
        )
        modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
        map_fun = (
            self.model_runner.model.hf_to_vllm_mapper.apply_dict
            if hasattr(self.model_runner.model, "hf_to_vllm_mapper")
            else None
        )
        # convert modelopt state to vllm format
        modelopt_state = convert_modelopt_state_to_vllm(modelopt_state, map_fun=map_fun)
        # restore model from modelopt state
        restore_from_modelopt_state_vllm(model, modelopt_state)

        if modelopt_weights is not None:
            # convert quantizer state values to vllm format
            modelopt_weights = convert_dict_to_vllm(modelopt_weights, map_fun=map_fun)
            # set quantizer state to model's state_dict
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

        calibrate_loop = calibrate_fun(calib_dataloader, self)

        if hasattr(model, "unwrap"):
            model = model.unwrap()

        quant_cfg = get_quant_config(quant_config, model)

        # quantize model
        with disable_compilation(model):
            print("Quantizing model...")
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        quantizer_file_path = quant_config["quant_file_path"]
        if quantizer_file_path:
            # Get amax and other quantizer state from the quantizer file
            current_state_dict = load_state_dict_from_path(self, quantizer_file_path, model)
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
