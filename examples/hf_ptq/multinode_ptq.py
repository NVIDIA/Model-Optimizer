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

"""Multi-node PTQ (Post-Training Quantization) with FSDP2 support."""

import argparse
import copy
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from example_utils import build_quant_cfg, get_tokenizer
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import get_model_type
from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format
from modelopt.torch.export.unified_export_hf import _export_transformers_checkpoint
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader, get_supported_datasets

RAND_SEED = 1234


def _nvfp4_max_cfg(*, layerwise: bool) -> dict[str, Any]:
    """NVFP4 quant config with explicit max calibration and a layerwise toggle."""
    cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "max", "layerwise": layerwise}
    return cfg


QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_max": _nvfp4_max_cfg(layerwise=False),
    "nvfp4_max_layerwise": _nvfp4_max_cfg(layerwise=True),
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
    "nvfp4_experts_only": mtq.NVFP4_EXPERTS_ONLY_CFG,
    "nvfp4_omlp_only": mtq.NVFP4_OMLP_ONLY_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
}


mto.enable_huggingface_checkpointing()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-node post-training quantization with FSDP2")

    parser.add_argument("--pyt_ckpt_path", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument(
        "--qformat", default="fp8", choices=QUANT_CFG_CHOICES.keys(), help="Quantization format"
    )
    parser.add_argument(
        "--kv_cache_qformat",
        default="fp8",
        choices=list(KV_QUANT_CFG_CHOICES.keys()),
        help="KV cache quantization format",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for calibration")
    parser.add_argument(
        "--calib_size",
        type=str,
        default="512",
        help="Comma-separated list of calibration sizes per dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            f"name of a dataset, or a comma separated list of datasets. "
            f"dataset choices are {get_supported_datasets()}"
        ),
    )
    parser.add_argument(
        "--export_path", default="exported_model", help="Directory to export the quantized model"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for HuggingFace models",
    )
    parser.add_argument("--awq_block_size", default=0, type=int)
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        default=None,
        help=(
            "Override auto-detect by transformer layer class name "
            "(e.g. LlamaDecoderLayer). Auto-detected when omitted."
        ),
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Keep FSDP2 sharded params on CPU; gather to GPU per layer forward.",
    )

    args = parser.parse_args()

    args.dataset = args.dataset.split(",") if args.dataset else None
    args.calib_size = [int(x) for x in args.calib_size.split(",")]

    return args


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize torch.distributed from torchrun env vars and pin the CUDA device."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    return rank, world_size, local_rank, device


def _resolve_decoder_layers(model: nn.Module, override_cls_name: str | None):
    """Return the list of decoder layers to apply ``fully_shard`` to."""
    if override_cls_name:
        layers = [m for m in model.modules() if type(m).__name__ == override_cls_name]
        if not layers:
            raise RuntimeError(f"No modules of class {override_cls_name!r} found in model")
        return layers
    layers = LayerActivationCollector.get_decoder_layers(model)
    if layers is None:
        raise RuntimeError(
            "Could not auto-detect decoder layers; pass "
            "--fsdp_transformer_layer_cls_to_wrap <ClassName> explicitly."
        )
    return layers


def fsdp2_wrap(
    model: nn.Module,
    override_cls_name: str | None = None,
    cpu_offload: bool = False,
) -> nn.Module:
    """Apply FSDP2 ``fully_shard`` to each decoder layer, then to the root module."""
    offload_policy: OffloadPolicy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
    for layer in _resolve_decoder_layers(model, override_cls_name):
        fully_shard(layer, reshard_after_forward=True, offload_policy=offload_policy)
    fully_shard(model, reshard_after_forward=True, offload_policy=offload_policy)
    return model


def load_and_prepare_model(
    model_path: str,
    device: torch.device,
    rank: int,
    trust_remote_code: bool = False,
    override_cls_name: str | None = None,
    cpu_offload: bool = False,
) -> tuple[nn.Module, str, list[str]]:
    """Load model and shard it with FSDP2 using rank-0-only CPU realization.

    Only rank 0 reads real weights from disk; every other rank instantiates the
    model on the ``meta`` device. After ``fully_shard`` sets up the sharded
    DTensor layout and ``to_empty`` allocates per-rank shard storage, rank 0's
    full state dict is broadcast into the sharded structure.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    dtype = getattr(config, "torch_dtype", None) or torch.bfloat16

    if rank == 0:
        src_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        src_model.eval()
        cpu_state_dict = src_model.state_dict()
    else:
        src_model = None
        cpu_state_dict = {}

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
    model.eval()

    model_type = get_model_type(model)
    original_architectures = model.config.architectures

    fsdp2_wrap(model, override_cls_name=override_cls_name, cpu_offload=cpu_offload)

    # For CPU offload: FSDP2 requires its managed params on CPU at lazy_init,
    # so materialize the whole model on CPU. Otherwise materialize on GPU.
    materialize_device = torch.device("cpu") if cpu_offload else device
    model.to_empty(device=materialize_device)

    set_model_state_dict(
        model,
        cpu_state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )

    # With CPU offload, FSDP-managed params stay on CPU but buffers (e.g. MoE
    # router corrections, RoPE caches) must live on GPU for layer forwards.
    if cpu_offload:
        for b in model.buffers():
            b.data = b.data.to(device, non_blocking=True)
        torch.cuda.synchronize()

    # Freeze every param so patch_fsdp_mp_dtypes' trainable-only check skips the
    # uniform-dtype assertion (e.g. Nemotron-H ships mixed bf16/fp32 weights).
    for p in model.parameters():
        p.requires_grad_(False)

    del cpu_state_dict, src_model

    return model, model_type, original_architectures


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_names: list[str],
    calib_sizes: list[int],
    batch_size: int,
) -> DataLoader:
    """Create calibration dataloader from dataset."""
    return get_dataset_dataloader(
        dataset_name=dataset_names,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_sizes,
        device=None,
        include_labels=False,
    )


def shard_dataloader(loader: DataLoader, rank: int, world_size: int) -> DataLoader:
    """Wrap a DataLoader with a DistributedSampler so each rank sees a unique shard."""
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def create_fsdp2_calibration_loop(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
):
    """Calibration loop that forwards through the FSDP-wrapped model."""

    def calibrate(unwrapped_model):
        for batch in tqdm(dataloader, desc="Calibrating"):
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
            # Use outer (FSDP-wrapped) model, not the unwrapped parameter passed by mtq.quantize.
            model(**batch)

    return calibrate


class _Fsdp2StateDictAdapter:
    """Shim exposing ``.get_state_dict(model)`` to ``_export_transformers_checkpoint``."""

    def get_state_dict(self, model: nn.Module):
        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )


def export_model(
    model: nn.Module,
    rank: int,
    export_path: str | Path,
    architectures: list[str],
):
    """Export the quantized model to HuggingFace format on rank 0."""
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    adapter = _Fsdp2StateDictAdapter()
    post_state_dict, hf_quant_config = _export_transformers_checkpoint(
        model, torch.bfloat16, accelerator=adapter
    )

    if rank == 0:
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        hf_quant_config = convert_hf_quant_config_format(hf_quant_config)

        model.save_pretrained(export_dir, state_dict=post_state_dict, save_modelopt_state=False)

        original_config = f"{export_dir}/config.json"
        with open(original_config) as file:
            config_data = json.load(file)

        config_data["quantization_config"] = hf_quant_config
        config_data["architectures"] = architectures

        with open(original_config, "w") as file:
            json.dump(config_data, file, indent=4)

    dist.barrier()


def main(args):
    """Main quantization workflow."""
    if not torch.cuda.is_available():
        raise OSError("GPU is required for quantization.")

    if args.qformat not in QUANT_CFG_CHOICES:
        raise ValueError(
            f"Quantization format {args.qformat} not supported. Choose from: {QUANT_CFG_CHOICES.keys()}"
        )

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    rank, world_size, _, device = setup_distributed()

    tokenizer = get_tokenizer(args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code)
    default_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if args.dataset is None:
        args.dataset = ["cnn_dailymail", "nemotron-post-training-dataset-v2"]
        warnings.warn(
            "No dataset specified. Defaulting to cnn_dailymail and nemotron-post-training-dataset-v2."
        )
        args.calib_size = (args.calib_size + [args.calib_size[-1]] * len(args.dataset))[
            : len(args.dataset)
        ]

    calib_dataloader = create_calibration_dataloader(
        tokenizer=tokenizer,
        dataset_names=args.dataset,
        calib_sizes=args.calib_size,
        batch_size=args.batch_size,
    )
    calib_dataloader = shard_dataloader(calib_dataloader, rank, world_size)

    model, model_type, original_architectures = load_and_prepare_model(
        model_path=args.pyt_ckpt_path,
        device=device,
        rank=rank,
        trust_remote_code=args.trust_remote_code,
        override_cls_name=args.fsdp_transformer_layer_cls_to_wrap,
        cpu_offload=args.cpu_offload,
    )

    quant_cfg = QUANT_CFG_CHOICES[args.qformat]
    quant_cfg = build_quant_cfg(args.qformat, quant_cfg, args.awq_block_size, model_type)

    enable_quant_kv_cache = args.kv_cache_qformat != "none"
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

    if enable_quant_kv_cache:
        quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
            quant_cfg,
            getattr(mtq, KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])["quant_cfg"],
        )

    if rank == 0:
        print("Starting quantization...")

    start_time = time.time()

    if need_calibration(quant_cfg):
        calibrate_fn = create_fsdp2_calibration_loop(model, calib_dataloader, device)
    else:
        calibrate_fn = None
        warnings.warn("Dynamic quantization. Calibration skipped.")

    with torch.no_grad():
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)

    elapsed = time.time() - start_time

    if rank == 0:
        print(f"Quantization completed in {elapsed:.2f}s")
        mtq.print_quant_summary(model)

    start_time = time.time()
    export_model(model, rank, args.export_path, original_architectures)
    elapsed = time.time() - start_time

    if rank == 0:
        if tokenizer is not None:
            tokenizer.padding_side = default_padding_side
            tokenizer.save_pretrained(args.export_path)
        print(f"Export completed in {elapsed:.2f}s")
        print(f"Model exported to {args.export_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    with patch_fsdp_mp_dtypes():
        main(args)
