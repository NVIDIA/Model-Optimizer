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

"""Fine-tune Megatron GDN/KDA attention with recurrent-state fake quantization."""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
from megatron.core import parallel_state
from megatron.core.ssm import gated_delta_net

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.linear_attention import linear_attention_training_phase
from modelopt.torch.utils.plugins.mbridge import load_mbridge_model_from_hf


def main():
    """Load a local model through Megatron Bridge and save a Megatron QAT checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--train-data", type=Path, required=True, help="Parquet file with a text column"
    )
    parser.add_argument(
        "--quant-config",
        type=Path,
        default=Path(__file__).with_name("configs") / "decode_state_int8.json",
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
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error(
            "This example uses one GPU; use a Megatron training schedule for multiple GPUs"
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.distributed.init_process_group("nccl")
    try:
        bridge, _, models, model, tokenizer = load_mbridge_model_from_hf(
            hf_model_name_or_path=str(options.model),
            trust_remote_code=options.trust_remote_code,
            provider_overrides={
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "sequence_parallel": False,
                "seq_length": options.length,
                "gradient_accumulation_fusion": False,
                "bf16": False,
                "params_dtype": torch.float32,
                "pipeline_dtype": torch.float32,
            },
        )
        torch.manual_seed(options.seed)
        layer_types = (gated_delta_net.GatedDeltaNet,)
        if hasattr(gated_delta_net, "KimiDeltaAttention"):
            layer_types += (gated_delta_net.KimiDeltaAttention,)
        layers = [module for module in model.modules() if isinstance(module, layer_types)]
        if not layers:
            raise ValueError("The model must contain Megatron GatedDeltaNet or KimiDeltaAttention")
        # Keep attention parameters and optimizer moments in FP32; freeze the rest in BF16.
        model.requires_grad_(False)
        for layer in layers:
            layer.requires_grad_(True)
        for parameter in model.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)
        mtq.quantize(model, json.loads(options.quant_config.read_text()))
        text = "\n\n".join(pq.read_table(options.train_data, columns=["text"])["text"].to_pylist())
        tokens = tokenizer.encode(text, add_special_tokens=False)
        needed = options.train_steps * (options.length + 1)
        if len(tokens) < needed:
            parser.error(f"Training data has {len(tokens)} tokens; {needed} are required")
        blocks = torch.tensor(tokens[:needed]).reshape(options.train_steps, options.length + 1)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=options.learning_rate, weight_decay=0)
        model.train()
        for step, block in enumerate(blocks, 1):
            ids, labels = block[:-1].unsqueeze(0).cuda(), block[1:].unsqueeze(0).cuda()
            positions = torch.arange(options.length, device=ids.device).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            # Megatron takes shifted labels and returns per-token losses, unlike HF models.
            with linear_attention_training_phase(model, [options.prefill_tokens]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = model(
                        input_ids=ids, position_ids=positions, attention_mask=None, labels=labels
                    )
                    loss = losses[:, options.prefill_tokens :].float().mean()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            print(f"step={step} loss={loss.item():.4f}", flush=True)
        bridge.save_megatron_model(
            models,
            str(options.output),
            hf_tokenizer_path=str(options.model),
            hf_tokenizer_kwargs={"trust_remote_code": options.trust_remote_code},
        )
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
