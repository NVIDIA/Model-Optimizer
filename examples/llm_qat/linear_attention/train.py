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

"""Fine-tune KDA attention parameters with recurrent-state fake quantization."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from fla.layers.kda import KimiDeltaAttention
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase


def main():
    """Quantize, train on local text, and save a ModelOpt-aware Hugging Face checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--train-data", type=Path, required=True, help="Parquet file with a text column"
    )
    parser.add_argument(
        "--quant-config",
        type=Path,
        default=Path(__file__).with_name("configs") / "kda_decode_state_int8.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    options = parser.parse_args()
    if options.train_steps < 1 or options.length < 2:
        parser.error("train-steps must be positive and length must be at least two")
    if not 0 <= options.prefill_tokens < options.length:
        parser.error("prefill-tokens must leave at least one suffix label")
    torch.manual_seed(options.seed)
    mto.enable_huggingface_checkpointing()
    tokenizer = AutoTokenizer.from_pretrained(
        options.model, local_files_only=True, trust_remote_code=options.trust_remote_code
    )
    text = "\n\n".join(pq.read_table(options.train_data, columns=["text"])["text"].to_pylist())
    tokens = tokenizer.encode(text, add_special_tokens=False)
    needed = options.train_steps * options.length
    if len(tokens) < needed:
        parser.error(f"Training data has {len(tokens)} tokens; {needed} are required")
    blocks = torch.tensor(tokens[:needed]).reshape(options.train_steps, options.length)
    model = AutoModelForCausalLM.from_pretrained(
        options.model,
        local_files_only=True,
        trust_remote_code=options.trust_remote_code,
        torch_dtype=torch.float32,
    )
    layers = [module for module in model.modules() if isinstance(module, KimiDeltaAttention)]
    if not layers:
        raise ValueError("The model must contain FLA KimiDeltaAttention layers")
    # Keep attention parameters and optimizer moments in FP32; freeze the rest in BF16.
    model.requires_grad_(False)
    for layer in layers:
        layer.requires_grad_(True)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.cuda()
    mtq.quantize(model, json.loads(options.quant_config.read_text()))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=options.learning_rate, weight_decay=0)
    model.train()
    for step, block in enumerate(blocks, 1):
        ids = block.unsqueeze(0).cuda()
        labels = ids.clone()
        labels[:, : options.prefill_tokens] = -100
        optimizer.zero_grad(set_to_none=True)
        # Keep phase metadata active during activation-checkpoint recomputation.
        with linear_attention_training_phase(model, [options.prefill_tokens]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(f"step={step} loss={loss.item():.4f}", flush=True)
    model.save_pretrained(options.output)
    tokenizer.save_pretrained(options.output)


if __name__ == "__main__":
    main()
