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
import warnings
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
    shard_pre_quant_scale_for_tp,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.plugins.vllm_fakequant_hf import is_weight_quantizer_state_key
from modelopt.torch.quantization.plugins.vllm import (
    disable_compilation,
    post_restore_vllm_parallel_linears,
)
from modelopt.torch.utils import safe_load
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

def _parse_dataset_env(value: str) -> str | list[str]:
    # Comma-separated → list (mix calibration, matches hf_ptq.py:502).
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


quant_config: dict[str, Any] = {
    "dataset": _parse_dataset_env(os.environ.get("QUANT_DATASET", "cnn_dailymail")),
    "calib_size": int(os.environ.get("QUANT_CALIB_SIZE", 512)),
    "max_sample_length": int(os.environ.get("QUANT_MAX_SAMPLE_LEN", 512)),
    "quant_cfg": os.environ.get("QUANT_CFG", None),
    "kv_quant_cfg": os.environ.get("KV_QUANT_CFG", None),
    "quant_file_path": os.environ.get("QUANT_FILE_PATH", None),
    "modelopt_state_path": os.environ.get("MODELOPT_STATE_PATH", None),
    "calib_batch_size": int(os.environ.get("CALIB_BATCH_SIZE", 1)),
    "recipe_path": os.environ.get("RECIPE_PATH", None),
    # Where to dump per-rank vllm-calibrated quantizer state for later HF export
    "vllm_state_dir": os.environ.get("MODELOPT_VLLM_STATE_DIR", None),
    "vllm_state_tag": os.environ.get("MODELOPT_VLLM_STATE_TAG", "vllm_calib"),
}


def _save_vllm_calibrated_state(model: torch.nn.Module) -> None:
    """Dump per-rank modelopt state + quantizer state dict for later HF export.

    Each rank writes its own file under MODELOPT_VLLM_STATE_DIR; the export
    script reads all rank files and merges TP-sharded amax / pre_quant_scale
    values across ranks to build the canonical HF state.

    No-op if MODELOPT_VLLM_STATE_DIR is unset.
    """
    state_dir = quant_config.get("vllm_state_dir")
    if not state_dir:
        return

    import modelopt.torch.opt as _mto
    from modelopt.torch.utils import safe_save as _safe_save

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    tag = quant_config.get("vllm_state_tag", "vllm_calib")

    os.makedirs(state_dir, exist_ok=True)

    modelopt_state = _mto.modelopt_state(model)
    quantizer_state_dict = mtq.utils.get_quantizer_state_dict(model)

    out = {
        "modelopt_state": modelopt_state,
        "quantizer_state_dict": quantizer_state_dict,
        "rank": rank,
        "world_size": world_size,
        "quant_config_summary": {
            "qformat": quant_config.get("quant_cfg"),
            "dataset": quant_config.get("dataset"),
            "calib_size": quant_config.get("calib_size"),
            "max_sample_length": quant_config.get("max_sample_length"),
        },
    }

    out_path = os.path.join(state_dir, f"{tag}-rank{rank:02d}-of{world_size:02d}.pth")
    _safe_save(out, out_path)
    print(
        f"[vllm-calib] rank{rank}/{world_size}: saved quantizer state -> {out_path}",
        flush=True,
    )


def _fakequant_run_prolog_worker(self) -> None:
    trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "false").lower() == "true"

    tokenizer = AutoTokenizer.from_pretrained(
        self.model_runner.model_config.tokenizer, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = self.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    if quant_config["modelopt_state_path"]:
        print(f"Loading modelopt state from {quant_config['modelopt_state_path']}")
        # Load on CPU to avoid failures when the checkpoint was saved from a different GPU mapping.
        modelopt_state = safe_load(quant_config["modelopt_state_path"], map_location="cpu")
        modelopt_weights = modelopt_state.pop("modelopt_state_weights", None)
        map_fun = (
            self.model_runner.model.hf_to_vllm_mapper.apply_dict
            if hasattr(self.model_runner.model, "hf_to_vllm_mapper")
            else None
        )
        modelopt_state = convert_modelopt_state_to_vllm(modelopt_state, map_fun=map_fun)
        restore_from_modelopt_state_vllm(model, modelopt_state)

        if modelopt_weights is not None:
            modelopt_weights = convert_dict_to_vllm(modelopt_weights, map_fun=map_fun)
            mtq.utils.set_quantizer_state_dict(model, modelopt_weights)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                from modelopt.torch.quantization.nn import TensorQuantizer
                from modelopt.torch.utils import get_unwrapped_name

                loaded_keys = {
                    get_unwrapped_name(n, model)
                    for n, m in model.named_modules()
                    if isinstance(m, TensorQuantizer)
                }
                # Same namespace as ``loaded_keys``: checkpoint keys may include DDP/FSDP
                # prefixes that ``convert_dict_to_vllm`` does not strip.
                pqs_in_weights = {
                    get_unwrapped_name(k, model)
                    for k, v in modelopt_weights.items()
                    if isinstance(v, dict) and "_pre_quant_scale" in v
                }
                unmatched_pqs = pqs_in_weights - loaded_keys
                if unmatched_pqs:
                    sample = sorted(unmatched_pqs)[:20]
                    warnings.warn(
                        f"{len(unmatched_pqs)} checkpoint pre_quant_scale key(s) have no "
                        f"matching TensorQuantizer in the model (showing up to 20): {sample}",
                        stacklevel=2,
                    )
            # set_quantizer_state_dict does not run modelopt_post_restore (unlike restore_quantizer_state).
            post_restore_vllm_parallel_linears(model)
            # Must follow post_restore: shard_pre_quant_scale_for_tp uses weight H_in vs pqs length.
            shard_pre_quant_scale_for_tp(model)

    else:
        if quant_config["quant_file_path"]:
            print("Will load quant, so only do a single sample calibration")
            quant_config["calib_size"] = 1

        # When dataset is a list (mix calibration), num_samples must be a
        # parallel list. Mirror hf_ptq.py:506-507's extension convention:
        # a single calib_size applies per-dataset (e.g. 512 with two datasets
        # → 512 samples from each, 1024 total).
        ds = quant_config["dataset"]
        n_samples = quant_config["calib_size"]
        if isinstance(ds, list) and isinstance(n_samples, int):
            n_samples = [n_samples] * len(ds)

        calib_dataloader = get_dataset_dataloader(
            dataset_name=ds,
            tokenizer=tokenizer,
            batch_size=quant_config["calib_batch_size"],
            num_samples=n_samples,
            max_sample_length=quant_config["max_sample_length"],
            device=self.device,
        )

        calibrate_loop = calibrate_fun(calib_dataloader, self)

        quant_cfg = get_quant_config(quant_config, model)

        with disable_compilation(model):
            print("Quantizing model...")
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        quantizer_file_path = quant_config["quant_file_path"]
        if quantizer_file_path:
            self.model_runner._dummy_run(1)
            current_state_dict = load_state_dict_from_path(self, quantizer_file_path, model)
            model.load_state_dict(current_state_dict)

            # Only barrier if distributed is actually initialized (avoids deadlocks).
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.distributed.barrier()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(model)

    # Dump per-rank quantizer state BEFORE fold_weight (folding is irreversible
    # for export — once amax is folded into weights we can't reconstruct the
    # state needed by the unified-HF export path).
    _save_vllm_calibrated_state(model)

    mtq.fold_weight(model)
    for name, module in model.named_modules():
        if is_weight_quantizer_state_key(name) and module.is_enabled:
            raise RuntimeError(
                f"Weight quantizer {name!r} is still enabled after fold_weight — "
                "double-quantization would corrupt activations."
            )


class FakeQuantWorker(BaseWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        with disable_compilation(model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> float:
        if (
            quant_config["quant_cfg"]
            or quant_config["kv_quant_cfg"]
            or quant_config["modelopt_state_path"]
            or quant_config["recipe_path"]
        ):
            _fakequant_run_prolog_worker(self)
        # Must return the base worker's compilation time (seconds). Returning None
        # breaks vLLM V1 executor: initialize_from_config does max(compilation_times)
        # across TP workers.
        return super().compile_or_warm_up_model()
