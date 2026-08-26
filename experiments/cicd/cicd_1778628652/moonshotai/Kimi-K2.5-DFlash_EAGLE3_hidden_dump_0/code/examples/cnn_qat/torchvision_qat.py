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
import os
import random
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_train_loader_and_sampler, get_val_loader_and_sampler, train, validate

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.utils import print_rank_0

# Suppress known ModelOpt PTQ warnings
warnings.filterwarnings(
    "ignore",
    message="Distributed training is initialized but no parallel_state is set",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN QAT using ModelOpt")
    # Data paths
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to the training dataset (ImageNet-style, with subfolders per class)",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        required=True,
        help="Path to the validation dataset (ImageNet-style, with subfolders per class)",
    )
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU/process")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers per GPU/process"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id for single-GPU training, ignored if using torchrun for multi-GPU",
    )
    # Distributed
    parser.add_argument(
        "--dist-url",
        type=str,
        default="tcp://127.0.0.1:23456",
        help="URL for DDP initialization, used if not using torchrun or for custom multi-node setups",
    )
    parser.add_argument(
        "--dist-backend", type=str, default="nccl", help="Distributed backend (e.g., nccl, gloo)"
    )
    parser.add_argument("--print-freq", type=int, default=100, help="Print frequency")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Number of GPUs
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeError("No CUDA devices found. At least one GPU is required.")

    # Detect torchrun environment
    is_torchrun = "LOCAL_RANK" in os.environ
    if is_torchrun:
        args.multi_gpu = True
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.rank = int(os.environ.get("RANK", args.gpu))
        args.world_size = int(os.environ.get("WORLD_SIZE", ngpus))
    else:
        args.multi_gpu = False
        args.gpu = args.gpu or 0
        args.rank = 0
        args.world_size = 1

    # Seed and cuDNN settings
    random.seed(args.seed + args.gpu)
    torch.manual_seed(args.seed + args.gpu)
    cudnn.benchmark = True

    # Create output directory if it doesn't exist
    if args.rank == 0 and args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Call worker (torchrun or single-GPU)
    main_worker(args)


def main_worker(args: argparse.Namespace) -> None:
    # Setup device
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed_all(args.seed + args.gpu)

    # Initialize DDP if multi-gpu
    if args.multi_gpu:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=(None if "LOCAL_RANK" in os.environ else args.dist_url),
            world_size=args.world_size,
            rank=args.rank,
        )

    # Load model and initialize loss
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Validation loaders: one for calibration (0 workers), one for eval
    val_calib_loader, _ = get_val_loader_and_sampler(args, num_workers=0)
    val_loader, _ = get_val_loader_and_sampler(args)

    # FP32 baseline evaluation
    print_rank_0("Evaluating FP32 baseline...")
    orig_acc = validate(model, val_loader, criterion, args)
    print_rank_0(f"Original FP32 model accuracy: {orig_acc:.2f}%")

    # PTQ calibration with 512 images
    print_rank_0("Starting PTQ...")
    quant_cfg = mtq.INT8_DEFAULT_CFG

    def calibrate(m: nn.Module):
        m.eval()
        seen = 0
        with torch.no_grad():
            for images, _ in val_calib_loader:
                m(images.cuda(args.gpu, non_blocking=True))
                seen += images.size(0)
                if seen >= 512:
                    break

    model = mtq.quantize(model, quant_cfg, calibrate)
    ptq_acc = validate(model, val_loader, criterion, args)
    print_rank_0(f"PTQ model accuracy: {ptq_acc:.2f}%")

    # Wrap model for QAT
    if args.multi_gpu:
        dist.barrier()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        print_rank_0("DDP initialized")

    # Initialize QAT data, optimizer, scheduler
    train_loader, _ = get_train_loader_and_sampler(args)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    best_acc = 0.0
    start = time.time()
    print_rank_0(f"Starting QAT for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print_rank_0(f"Epoch {epoch}/{args.epochs}")
        train(model, train_loader, criterion, optimizer, epoch - 1, args)
        scheduler.step()
        acc = validate(model, val_loader, criterion, args)
        if args.rank == 0:
            print(f"Epoch {epoch} QAT accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                ckpt_filename = f"resnet50_qat_epoch_{epoch}.pth"
                ckpt_path = os.path.join(args.output_dir, ckpt_filename)
                to_save = model.module if hasattr(model, "module") else model
                mto.save(to_save, ckpt_path)
                print(f"Saved best QAT model: {ckpt_path} (Acc: {best_acc:.2f}%)")

    print_rank_0(f"Training complete in {time.time() - start:.2f}s")
    if args.multi_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
