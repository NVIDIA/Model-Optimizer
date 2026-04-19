# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.nn as nn
from dataset_utils import build_blend_dataset, load_blend_config
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.recipe import ModelOptPTQRecipe, load_recipe


def get_dataloader(args, tokenizer):
    config = load_blend_config("configs/dataset/blend.yaml")
    ds = build_blend_dataset(config, tokenizer, args.max_length)

    train_dataset = ds["train"]
    if 0 < args.train_size < len(train_dataset):
        train_dataset = train_dataset.select(range(args.train_size))

    calib_dataset = ds["eval"]
    if 0 < args.calib_size < len(calib_dataset):
        calib_dataset = calib_dataset.select(range(args.calib_size))

    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([item["input_ids"] for item in batch]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
            "labels": torch.tensor([item["labels"] for item in batch]),
        }

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    calib_dataloader = DataLoader(
        calib_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_dataloader, calib_dataloader


def train(model, optimizer, train_dataloader, tokenizer, epochs, output_dir, device):
    for epoch in (pbar := tqdm(range(epochs))):
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
        for batch in (pbar_batch := tqdm(train_dataloader)):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=inputs, attention_mask=attention_mask, labels=batch["labels"].to(device)
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar_batch.set_description(f"loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1} completed | Loss: {loss.item():.4f}")

    if output_dir:
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QAT Training Script")
    # Data paths
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--train-size", type=int, default=2048, help="Train size")
    parser.add_argument("--calib-size", type=int, default=512, help="Calib size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max length")
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--recipe",
        type=str,
        default="general/ptq/nvfp4_default-fp8_kv",
        help="Path to a quantization recipe YAML (built-in or custom)",
    )
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--print-freq", type=int, default=100, help="Print frequency")
    parser.add_argument(
        "--output-dir", type=str, default="qat_model", help="Directory to save the checkpoints"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Enable automatic save/load of modelopt state huggingface checkpointing
    # modelopt state will be saved automatically to "modelopt_state.pt"
    mto.enable_huggingface_checkpointing()

    # Load model and initialize loss
    model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Get dataloaders
    train_dataloader, calib_dataloader = get_dataloader(args, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calibrate the model
    def calibrate(m: nn.Module):
        for batch in calib_dataloader:
            m(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

    # Load recipe and quantize the model
    recipe = load_recipe(args.recipe)
    if not isinstance(recipe, ModelOptPTQRecipe):
        raise ValueError(f"Expected PTQ recipe, but got {type(recipe).__name__} from {args.recipe}")
    model = mtq.quantize(model, recipe.quantize, calibrate)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Train the model
    model.train()
    model.to(device)
    train(model, optimizer, train_dataloader, tokenizer, args.epochs, args.output_dir, device)


if __name__ == "__main__":
    main()
