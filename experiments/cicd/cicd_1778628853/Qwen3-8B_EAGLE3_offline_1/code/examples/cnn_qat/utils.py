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

# Adapted from https://github.com/pytorch/examples/blob/c67bbaba018d438447036c114133c6ad6c4f78b1/imagenet/main.py

# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import time
from enum import Enum

import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ":f", summary_type: Summary = Summary.AVERAGE) -> None:
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def all_reduce(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count if self.count != 0 else 0.0

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        if self.summary_type is Summary.NONE:
            return ""
        fmtmap = {
            Summary.AVERAGE: "{name} {avg:.3f}",
            Summary.SUM: "{name} {sum:.3f}",
            Summary.COUNT: "{name} {count:.3f}",
        }
        if self.summary_type not in fmtmap:
            raise ValueError(f"invalid summary type {self.summary_type}")
        return fmtmap[self.summary_type].format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: list[AverageMeter], prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self) -> None:
        parts = [m.summary() for m in self.meters if m.summary()]
        print(" * " + "  ".join(parts))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)
) -> list[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: list[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Train: "
    )

    model.train()

    if args.multi_gpu and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if hasattr(args, "print_freq") and args.print_freq > 0 and i % args.print_freq == 0:
            progress.display(i + 1)


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    args: argparse.Namespace,
) -> float:
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

    total_batches = len(val_loader)
    if (
        args.multi_gpu
        and hasattr(val_loader, "sampler")
        and hasattr(val_loader.sampler, "num_replicas")
        and (len(val_loader.sampler) * val_loader.sampler.num_replicas) < len(val_loader.dataset)
    ):
        num_aux_samples = len(val_loader.dataset) - (
            len(val_loader.sampler) * val_loader.sampler.num_replicas
        )
        num_aux_batches = (num_aux_samples + val_loader.batch_size - 1) // val_loader.batch_size
        total_batches += num_aux_batches

    progress = ProgressMeter(total_batches, [batch_time, losses, top1, top5], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if hasattr(args, "print_freq") and args.print_freq > 0 and i % args.print_freq == 0:
                progress.display(i + 1)

    if args.multi_gpu:
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if args.rank == 0:
        progress.display_summary()

    return top1.avg


def get_train_loader_and_sampler(
    args: argparse.Namespace,
) -> tuple[DataLoader, Sampler | None]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_ds = datasets.ImageFolder(
        args.train_data_path,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_ds) if args.multi_gpu else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    return train_loader, train_sampler


def get_val_loader_and_sampler(
    args: argparse.Namespace,
    use_distributed_sampler: bool = True,
    num_workers: int = -1,
) -> tuple[DataLoader, Sampler | None]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_ds = datasets.ImageFolder(
        args.val_data_path,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_sampler = None
    if args.multi_gpu and use_distributed_sampler:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if num_workers == -1 else num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    return val_loader, val_sampler
