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

"""Calibrate replicated staged GSQ with torchrun and ordered token tensor files.

Example::

    torchrun --standalone --nproc_per_node=4 gsq_distributed.py \
        --model /path/to/llama --initializer-data initializer.pt \
        --training-data training.pt --checkpoint-dir /shared/gsq-checkpoints \
        --output /shared/gsq.pth

Each data file contains a torch.long tensor shaped [global_sequences, sequence_length].
Sequences must be full length without padding. For multiple nodes, supply torchrun's
matching rendezvous settings and shared checkpoint/output paths on every node.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.utils.distributed import DistributionPlan, broadcast_from, set_distribution_plan


def main():
    """Initialize WORLD, round robin shard each split, calibrate and save on rank zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--initializer-data", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--initializer", choices=("gptq", "max", "mse"), default="gptq")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        datasets = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in (args.initializer_data, args.training_data)
        ]
        if any(
            not isinstance(data, torch.Tensor)
            or data.dtype != torch.long
            or data.ndim != 2
            or min(data.shape) == 0
            for data in datasets
        ):
            raise ValueError("Data files must contain nonempty [sequences, tokens] long tensors")
        torch.manual_seed(0)
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
        )
        set_distribution_plan(model, DistributionPlan(world_size=world))
        config = load_recipe(f"general/ptq/nvfp4_gsq-{args.initializer}").quantize.model_dump()
        config["algorithm"].update(
            initializer_samples=len(datasets[0]),
            train_samples=len(datasets[1]),
        )
        config["algorithm"]["layerwise"].update(
            checkpoint_dir=args.checkpoint_dir,
        )

        def forward_loop(module):
            for data in datasets:
                for index in range(rank, len(data), world):
                    tokens = data[index : index + 1].to(device)
                    module(tokens, use_cache=False, attention_mask=torch.ones_like(tokens))

        mtq.quantize(model, config, forward_loop)

        def save():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            mto.save(model, args.output)

        broadcast_from("save GSQ checkpoint", save)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
