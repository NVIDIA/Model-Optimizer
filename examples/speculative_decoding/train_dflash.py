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

"""Standalone DFlash training script using SpecForge's data pipeline.

Uses SpecForge's tokenizer template + offset-mapping loss mask for data
preprocessing, and ModelOpt's DFlash module for the draft model. This
isolates data pipeline differences from model architecture differences.

Usage:
    torchrun --nproc_per_node=8 train_dflash.py \
        --model /path/to/Qwen3-8B \
        --data /path/to/train.jsonl \
        --chat-template qwen \
        --block-size 16 \
        --num-draft-layers 5 \
        --num-epochs 3 \
        --lr 1e-4 \
        --output-dir /path/to/output
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DFlash training with SpecForge data pipeline")
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument("--data", type=str, required=True, help="Training data JSONL path")
    parser.add_argument("--chat-template", type=str, default="qwen", help="Chat template name")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-draft-layers", type=int, default=5)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=0, help="0 = save at end only")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ar-samples", type=int, default=20, help="AR validation samples")
    return parser.parse_args()


def is_rank0():
    """Check if current process is rank 0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(msg):
    """Print only on rank 0."""
    if is_rank0():
        print(msg, flush=True)


def build_dataset(tokenizer, data_path, chat_template_name, max_length):
    """Build dataset using SpecForge's data pipeline.

    Uses SpecForge's GeneralParser to tokenize conversations with the
    proper chat template and compute offset-mapping-based loss masks.
    """
    from specforge.data.parse import GeneralParser
    from specforge.data.template import TEMPLATE_REGISTRY

    template = TEMPLATE_REGISTRY.get(chat_template_name)
    parser = GeneralParser(tokenizer, template)

    raw_dataset = load_dataset("json", data_files=data_path)["train"]

    processed = {"input_ids": [], "loss_mask": []}
    skipped = 0
    for sample in raw_dataset:
        convs = sample.get("conversations", sample.get("messages", []))
        if not convs:
            skipped += 1
            continue
        try:
            input_ids, loss_mask = parser.parse(convs, max_length=max_length)
            processed["input_ids"].append(input_ids)
            processed["loss_mask"].append(loss_mask)
        except Exception:
            skipped += 1

    print_rank0(f"Processed {len(processed['input_ids'])} samples, skipped {skipped}")
    return processed


class DFlashDataset(torch.utils.data.Dataset):
    """Simple dataset wrapping tokenized input_ids and loss_mask."""

    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.loss_mask = data["loss_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "loss_mask": self.loss_mask[idx],
        }


def collate_fn(batch):
    """Collate batch of samples."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    loss_mask = torch.stack([b["loss_mask"] for b in batch])
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def train(args):
    """Main training loop."""
    # Init distributed
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    mto.enable_huggingface_checkpointing()

    # Load model
    print_rank0(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Detect mask_token_id
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            mask_token_id = tokenizer.mask_token_id
        elif hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            mask_token_id = tokenizer.pad_token_id
        else:
            mask_token_id = tokenizer.eos_token_id
    print_rank0(f"mask_token_id: {mask_token_id}")

    # Convert to DFlash
    config = {
        "dflash_block_size": args.block_size,
        "dflash_use_torch_compile": False,
        "dflash_architecture_config": {
            "num_hidden_layers": args.num_draft_layers,
            "mask_token_id": mask_token_id,
        },
    }
    mtsp.convert(model, [("dflash", config)])
    print_rank0(
        f"DFlash module created: {sum(p.numel() for p in model.dflash_module.parameters()):,} params"
    )

    # Build dataset using SpecForge pipeline
    print_rank0("Building dataset with SpecForge pipeline...")
    data = build_dataset(tokenizer, args.data, args.chat_template, args.max_length)

    # Filter samples with too few loss tokens
    min_loss_tokens = 2 * args.block_size
    filtered_ids = []
    filtered_masks = []
    for i in range(len(data["input_ids"])):
        if data["loss_mask"][i].sum() >= min_loss_tokens:
            filtered_ids.append(data["input_ids"][i])
            filtered_masks.append(data["loss_mask"][i])
    print_rank0(f"After filtering: {len(filtered_ids)} samples (min {min_loss_tokens} loss tokens)")
    data = {"input_ids": filtered_ids, "loss_mask": filtered_masks}

    dataset = DFlashDataset(data)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Wrap with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )
    raw_model = model.module

    # Optimizer — only train dflash_module
    optimizer = torch.optim.AdamW(
        [p for p in raw_model.dflash_module.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    # LR scheduler
    steps_per_epoch = len(dataloader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print_rank0(f"Training: {total_steps} steps, {warmup_steps} warmup, {steps_per_epoch}/epoch")

    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # Create labels from loss_mask: -100 for masked positions
            labels = input_ids.clone()
            labels[loss_mask == 0] = -100

            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=labels,
            )

            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % args.log_interval == 0:
                acc = output.train_acc[0][0] if hasattr(output, "train_acc") else 0.0
                lr = scheduler.get_last_lr()[0]
                print_rank0(
                    f"Step {global_step} | loss={loss.item():.4f} | acc={acc:.4f} | lr={lr:.2e}"
                )

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                if is_rank0():
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    raw_model.save_pretrained(save_path)
                    print_rank0(f"Saved checkpoint: {save_path}")

    # Save final model
    if is_rank0():
        os.makedirs(args.output_dir, exist_ok=True)
        raw_model.save_pretrained(args.output_dir)
        print_rank0(f"Saved final model: {args.output_dir}")

    dist.barrier()

    # AR validation on rank 0
    if is_rank0() and args.num_ar_samples > 0:
        print_rank0("\n=== AR Validation ===")
        model.eval()
        from modelopt.torch.speculative.plugins.transformers import HFARValidation

        validator = HFARValidation(raw_model, tokenizer)
        ds = load_dataset("/hf-local/HuggingFaceH4/mt_bench_prompts")["train"]

        ars = []
        for i in range(min(args.num_ar_samples, len(ds))):
            prompt = ds[i]["prompt"][0]
            chat = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt").input_ids.to(device)
            try:
                _, ar = validator.validate(osl=32, input_ids=inp, steps=3)
                ars.append(ar)
                print_rank0(f"  AR={ar:.2f} | {prompt[:60]}")
            except Exception as e:
                print_rank0(f"  ERROR | {prompt[:60]}... | {e}")

        if ars:
            avg = sum(ars) / len(ars)
            print_rank0("\n==== DFlash AR Results ====")
            print_rank0(f"Average AR: {avg:.4f}")
            print_rank0(f"Min: {min(ars):.4f}, Max: {max(ars):.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    train(args)
