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

"""Minimal FSDP2 launcher for ``mtq.auto_quantize``.

Loads a HuggingFace causal LM in BF16 across data-parallel ranks via FSDP2,
runs ``mtq.auto_quantize`` with the standard calibration dataloader, and
persists the searcher state to disk so per-layer sensitivities can be
analyzed and re-solved offline.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.utils import core_utils as _modelopt_core_utils
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

# Workaround for a modelopt FSDP2 bug: ``_get_fsdp2_mesh`` returns ``None`` when
# ``post_forward_mesh_info`` is cleared (e.g. between calibration phases of a
# multi-format auto_quantize), and the caller then crashes on ``.ndim``. Fall
# back to the pre-forward ``mesh_info`` which is set at FSDP2 wrap time.
_orig_get_fsdp2_mesh = _modelopt_core_utils._get_fsdp2_mesh


def _patched_get_fsdp2_mesh(module):
    mesh = _orig_get_fsdp2_mesh(module)
    if mesh is not None:
        return mesh
    try:
        from torch.distributed._composable_state import _get_module_state
        state = _get_module_state(module)
        pg = getattr(state, "_fsdp_param_group", None)
        info = getattr(pg, "mesh_info", None) if pg else None
        if info is not None:
            return info.mesh
    except Exception:
        pass
    return mesh


_modelopt_core_utils._get_fsdp2_mesh = _patched_get_fsdp2_mesh

# Local map of the qformats this script supports. Kept inline to avoid
# importing ``hf_ptq`` (which pulls heavy dependencies and example-local
# modules) just for this dict.
QUANT_CFG_CHOICES: dict[str, dict] = {
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "nvfp4_mse": mtq.NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyt_ckpt_path", required=True, help="HF model id or local path.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for search_state_rank*.pt, auto_quantize_search.pth, modelopt_state.pt.",
    )
    parser.add_argument("--dataset", default="cnn_dailymail")
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--calib_seq", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--qformat",
        default="nvfp4,fp8",
        help=f"Comma-separated qformats. Supported: {','.join(QUANT_CFG_CHOICES)}.",
    )
    parser.add_argument("--auto_quantize_bits", type=float, default=4.8)
    parser.add_argument(
        "--auto_quantize_method", choices=("gradient", "kl_div"), default="gradient"
    )
    parser.add_argument("--auto_quantize_score_size", type=int, default=128)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def _find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Locate the decoder-layer ``ModuleList`` on common HF causal-LM topologies."""
    candidates = (
        "model.layers",        # Llama / Mistral / Qwen / Gemma family
        "backbone.layers",     # NemotronH (Mamba-Transformer hybrid)
        "transformer.h",       # GPT-2 / GPT-J / Falcon
        "layers",              # raw Transformer top-level
    )
    for path in candidates:
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList):
            print(f"[fsdp2_auto_quantize] wrapping {len(obj)} decoder layers at `{path}`")
            return obj
    # Fallback: scan named_modules for a ModuleList whose children share a class
    # (covers exotic architectures with unusual attribute paths).
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
            child_classes = {type(c).__name__ for c in mod}
            if len(child_classes) == 1:
                print(f"[fsdp2_auto_quantize] wrapping {len(mod)} decoder layers at `{name}` (fallback scan)")
                return mod
    raise RuntimeError(
        f"Could not locate decoder-layer ModuleList; tried {candidates} and a fallback scan."
    )


def _wrap_with_distributed_sampler(loader: DataLoader) -> DataLoader:
    """Rebuild ``loader`` with a non-shuffled ``DistributedSampler`` over its dataset."""
    # NOTE: dataset is already on GPU per get_dataset_dataloader(device=...) --
    # sampler reduces iteration, not GPU footprint.
    sampler = DistributedSampler(loader.dataset, shuffle=False, drop_last=False)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        collate_fn=loader.collate_fn,
    )


def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[rank 0] world_size={world_size} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_config = AutoConfig.from_pretrained(
        args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code
    )

    # Meta-init: structure only, no I/O. All ranks build the same graph.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            hf_config,
            trust_remote_code=args.trust_remote_code,
            dtype=torch.bfloat16,
        )
    # Some HF modeling code (e.g. NemotronH norms) hard-casts specific params
    # to fp32 even when dtype=bf16. fully_shard requires uniform dtype per
    # shard group, so force everything to bf16 here.
    model = model.to(torch.bfloat16)
    model.eval()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    for layer in _find_decoder_layers(model):
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    # Allocate sharded GPU storage; values are still uninitialised here.
    model.to_empty(device=device)

    # Rank-0 reads the full BF16 checkpoint to CPU (HF lazy-load via
    # ``low_cpu_mem_usage``); ``set_model_state_dict`` then scatters the CPU
    # tensors to the matching sharded GPU params on every rank.
    if rank == 0:
        print(f"[rank 0] loading {args.pyt_ckpt_path} to CPU via low_cpu_mem_usage=True")
        cpu_model = AutoModelForCausalLM.from_pretrained(
            args.pyt_ckpt_path,
            dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        full_sd = cpu_model.state_dict()
        del cpu_model
    else:
        cpu_model = None
        full_sd = {}

    set_model_state_dict(
        model,
        full_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=True,
        ),
    )
    if rank == 0:
        del full_sd
    torch.cuda.empty_cache()
    dist.barrier()

    include_labels = args.auto_quantize_method == "gradient"
    calib_dataloader = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_sample_length=args.calib_seq,
        device=device,
        include_labels=include_labels,
    )
    # Always shard calibration across ranks: FSDP2 already grad-reduces every
    # backward, so disjoint per-rank shards give the same global signal in
    # 1/world_size of the steps. Each rank therefore sees calib_size/world_size
    # batches and total samples consumed globally still equals calib_size.
    calib_dataloader = _wrap_with_distributed_sampler(calib_dataloader)

    def loss_func(output, data):
        return output.loss

    if args.auto_quantize_method == "gradient":

        def forward_step(model, batch):
            return model(**batch)

    else:  # kl_div

        def forward_step(model, batch):
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            return model(**inputs).logits

    checkpoint_path = os.path.join(args.output_dir, "auto_quantize_search.pth")
    qformat_list = args.qformat.split(",")
    unknown = [f for f in qformat_list if f not in QUANT_CFG_CHOICES]
    if unknown:
        raise ValueError(f"Unknown qformat(s): {unknown}. Supported: {list(QUANT_CFG_CHOICES)}")

    if rank == 0:
        print(
            f"[rank 0] running auto_quantize: bits={args.auto_quantize_bits} "
            f"qformats={qformat_list} method={args.auto_quantize_method}"
        )

    model, search_state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": args.auto_quantize_bits},
        data_loader=calib_dataloader,
        forward_step=forward_step,
        loss_func=loss_func if args.auto_quantize_method == "gradient" else None,
        quantization_formats=[QUANT_CFG_CHOICES[f] for f in qformat_list],
        # num_calib_steps is the per-rank lockstep batch count after
        # DistributedSampler sharding; each step covers world_size*batch_size
        # global samples via FSDP2 grad-reduce.
        num_calib_steps=len(calib_dataloader),
        # auto_quantize_score_size is the *global* sample budget; divide by
        # (batch_size * world_size) to get per-rank step count under
        # DistributedSampler, matching how num_calib_steps is already per-rank.
        num_score_steps=min(
            len(calib_dataloader),
            max(args.auto_quantize_score_size // args.batch_size // world_size, 1),
        ),
        verbose=True,
        disabled_layers=[
            entry["quantizer_name"]
            for entry in _default_disabled_quantizer_cfg
            if "parent_class" not in entry
        ],
        method=args.auto_quantize_method,
        checkpoint=checkpoint_path,
    )

    # Save each rank's local searcher state so later analysis can inspect /
    # aggregate per-rank sensitivity scores. Under FSDP2 + method="gradient",
    # the searcher's per-layer scores are computed against locally-sharded
    # gradients, so each rank holds scores only for the parameters it owns.
    per_rank_path = os.path.join(args.output_dir, f"search_state_rank{rank}.pt")
    torch.save(search_state, per_rank_path)

    if rank == 0:
        modelopt_state_path = os.path.join(args.output_dir, "modelopt_state.pt")
        torch.save(mto.modelopt_state(model), modelopt_state_path)
        print(
            f"[rank 0] wrote {modelopt_state_path}; per-rank search states at search_state_rank*.pt"
        )

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
