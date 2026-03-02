# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse

import torch
import torch.distributed as dist
from eagle_utils import DataCollatorWithPadding, make_eagle_supervised_data_module
from trainer.distill_trainer import EagleSGLTrainer, EagleTPTrainer
from transformers import AutoTokenizer

torch.manual_seed(0)


def _check_args(args):
    """Sanity check for arguments."""
    # TODO: (hg)


def _setup_pgroups(args):
    """Initialize student/teacher pgroups and set devices."""
    rank = dist.get_rank()
    args.teacher_ranks = list(range(len(args.teacher_devices)))
    args.student_ranks = list(
        range(len(args.teacher_devices), len(args.teacher_devices) + len(args.student_devices))
    )
    if rank in args.teacher_ranks:
        torch.cuda.set_device(args.teacher_devices[rank])
    else:
        torch.cuda.set_device(args.student_devices[rank - len(args.teacher_ranks)])
    print(
        f"Starting process rank={rank}, device={torch.cuda.current_device()}, world_size={dist.get_world_size()}"
    )
    args.teacher_pgroup = dist.new_group(ranks=args.teacher_ranks)
    args.student_pgroup = dist.new_group(ranks=args.student_ranks)


def train(args):
    """Entrance for training."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.training_seq_len
    )
    args.use_offline_training = False
    args.vlm_processor = None
    args.offline_data_path = None
    data_module = make_eagle_supervised_data_module(tokenizer, args)

    # Ensure different ranks load the same data
    g = torch.Generator()
    g.manual_seed(0)

    train_dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(max_length=args.training_seq_len),
        drop_last=True,
        generator=g,
    )
    trainer_cls = {
        "sglang": EagleSGLTrainer,
        "hf": EagleTPTrainer,
    }[args.teacher_backend]
    trainer = trainer_cls(dist.get_rank(), args, tokenizer, train_dataloader)
    trainer.train()
    trainer.save(args.out_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU distributed two-stage forward example")

    # Training args
    parser.add_argument("--model_path", type=str, required=True, help="Target model path.")
    parser.add_argument("--data_path", type=str, required=True, help="Training dataset.")
    parser.add_argument("--training_seq_len", type=str, default=1024)
    parser.add_argument("--eagle_config_path", type=str, default="eagle_config.json")
    parser.add_argument("--out_path", type=str, default="ckpts/fast-trained")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8, help="Total bs across all ranks.")

    # Trainer args
    parser.add_argument("--teacher_backend", type=str, choices=["sglang", "hf"], default="sglang")
    parser.add_argument(
        "--teacher_ep_size",
        type=int,
        default=1,
        help="Teacher EP size, only used for sglang backend.",
    )
    parser.add_argument("--teacher_devices", type=list, default=[0, 1, 2, 3])
    parser.add_argument("--student_devices", type=list, default=[4, 5, 6, 7])
    parser.add_argument(
        "--lazy_preprocess", type=bool, default=True, help="Whether to use lazy preprocessing."
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=20000)
    parser.add_argument(
        "--total_steps", type=int, default=60000, help="Total number of steps for debugging."
    )
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12357")

    args = parser.parse_args()

    dist.init_process_group("nccl")

    _check_args(args)
    _setup_pgroups(args)
    train(args)


if __name__ == "__main__":
    main()
