# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
  # Load from HuggingFace model and generate text:
  python -m mbridge_distillation.generate_text --model llama --prompt="Hello, how are you?"
  
  # Load from local checkpoint:
  python -m mbridge_distillation.generate_text --model llama --checkpoint /path/to/checkpoint --prompt="Hello, how are you?"

  # Use specific parallelism:
  python -m mbridge_distillation.generate_text --model llama --tp 1 --pp 1 --prompt="Hello"
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Model ID and AnyModel converter name per --model flag
MODEL_MAP = {
    "gpt": ("openai/gpt-oss-20b", "gpt_oss_20b"),
    "nemo": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron_h_v2"),
    "llama": ("meta-llama/Llama-3.2-3B-Instruct", "llama"),
    "qwen": ("Qwen/Qwen3-8B", "qwen3"),
}

class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, and attention mask, then raises StopIteration.
    Used for single-step inference in the forward pass.
    """

    def __init__(self, input_ids, position_ids):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for text generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, and attention mask.

    Args:
        data_iterator: Iterator providing batches of input data
        model: The Megatron model to run forward pass on
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (model_output, loss_function)
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def _ensure_deps():
    """Ensure transformers, anymodel, and megatron.bridge are importable."""
    try:
        from transformers import AutoConfig
    except ImportError as e:
        logger.error("transformers is required: %s", e)
        sys.exit(1)
    try:
        from modelopt.torch.puzzletron.anymodel import ConverterFactory
    except ImportError as e:
        logger.error("ModelOpt anymodel is required (install Model-Optimizer): %s", e)
        sys.exit(1)
    try:
        from megatron.bridge import AutoBridge
    except ImportError as e:
        logger.error("Megatron-Bridge is required: %s", e)
        sys.exit(1)


def load_hf_config(model_id: str, trust_remote_code: bool = True):
    """Load HuggingFace config only (no weights)."""
    from transformers import AutoConfig
    logger.info("Loading config for %s (config only, no weights)", model_id)
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    return config


def create_block_library(config, converter_name: str):
    """Build block_configs from HF config using the AnyModel Converter."""
    from modelopt.torch.puzzletron.anymodel import ConverterFactory
    
    logger.info(f"Creating block library for converter: {converter_name}")
    converter_cls = ConverterFactory.get(converter_name)
    if converter_cls is None:
        raise ValueError(
            f"Unknown converter '{converter_name}'. "
            f"Valid: gpt_oss_20b, nemotron_h_v2, llama, qwen3."
        )
    
    logger.info(f"Using converter class: {converter_cls}")
    block_configs = converter_cls.create_block_configs_from_main_config(config)
    
    # Normalize to list of dicts for JSON and mbridge
    out = []
    for bc in block_configs:
        if hasattr(bc, "to_dict"):
            out.append(bc.to_dict())
        elif isinstance(bc, dict):
            out.append(bc)
        else:
            import dataclasses
            out.append(dataclasses.asdict(bc))
    logger.info("Created block library: %d block configs (converter=%s)", len(out), converter_name)
    return out


def main(args) -> None:
    """Main function for text generation from HuggingFace models via MBridge and AnyModel."""
    
    # Setup parallelism
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp
    
    logger.info(f"Starting text generation with parallelism: TP={tp}, PP={pp}, EP={ep}, ETP={etp}")

    _ensure_deps()

    # Determine model ID and converter
    if args.model not in MODEL_MAP:
        raise ValueError(f"Unknown model type: {args.model}. Available: {list(MODEL_MAP.keys())}")
        
    model_id, converter_name = MODEL_MAP[args.model]
    load_path = args.checkpoint if args.checkpoint else model_id
    
    logger.info(f"Selected model type: {args.model}")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Converter Name: {converter_name}")
    logger.info(f"Load Path: {load_path}")

    # 1. Load HF Config
    logger.info("Step 1: Loading HF Config...")
    config = load_hf_config(load_path, trust_remote_code=args.trust_remote_code)
    logger.info("HF Config loaded successfully.")

    # 2. Create Block Library (AnyModel)
    # This ensures we have block configs even for homogeneous models
    logger.info("Step 2: Creating Block Library via AnyModel...")
    
    block_configs = None
    # Check if block_configs.json exists in checkpoint
    if args.checkpoint and os.path.isfile(os.path.join(args.checkpoint, "block_configs.json")):
        block_configs_json = os.path.join(args.checkpoint, "block_configs.json")
        logger.info("Found block_configs.json in checkpoint directory: %s", block_configs_json)
        with open(block_configs_json) as f:
            data = json.load(f)
        block_configs = data.get("block_configs", data) if isinstance(data, dict) else data
        logger.info("Loaded block library from checkpoint (%d blocks)", len(block_configs))
    else:
        logger.info("No block_configs.json found, creating from config using converter...")
        block_configs = create_block_library(config, converter_name)
        logger.info(f"Block library created with {len(block_configs)} blocks.")

    # 3. Apply MBridge Patch
    logger.info("Step 3: Applying MBridge Patch...")
    try:
        from .patch_mbridge_provider import (
            apply_patch,
            set_provider_block_configs,
        )
    except ImportError:
        # If running as script from example dir
        try:
            from patch_mbridge_provider import (
                apply_patch,
                set_provider_block_configs,
            )
        except ImportError:
             # Fallback if module path is different
             sys.path.append(os.path.dirname(__file__))
             from patch_mbridge_provider import (
                apply_patch,
                set_provider_block_configs,
            )

    apply_patch()
    logger.info("MBridge patch applied.")

    # 4. Load Model with AnyModel Patcher
    logger.info("Step 4: Loading Model with AnyModel Patcher...")
    from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
    from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
    
    descriptor = ModelDescriptorFactory.get(converter_name)
    logger.info("Using AnyModel patcher with descriptor: %s", converter_name)
    
    with deci_x_patcher(model_descriptor=descriptor):
        logger.info(f"Loading HF model from {load_path} inside patcher context...")
        bridge = AutoBridge.from_hf_pretrained(
            load_path,
            trust_remote_code=args.trust_remote_code,
        )
        logger.info("HF model loaded into bridge.")

    # 5. Convert to Megatron Provider
    logger.info("Step 5: Converting to Megatron Provider...")
    provider = bridge.to_megatron_provider(load_weights=True)
    
    # Set parallelism
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.pipeline_dtype = torch.bfloat16
    
    # Set block configs
    set_provider_block_configs(provider, block_configs=block_configs)
    logger.info(f"Provider configured with {len(block_configs)} block configs.")

    # Finalize provider
    logger.info("Finalizing provider...")
    provider.finalize()
    
    # Initialize model parallel
    logger.info("Initializing model parallel...")
    provider.initialize_model_parallel(seed=0)

    # 6. Build Mcore Model
    logger.info("Step 6: Building Mcore Model...")
    model_list = provider.provide_distributed_model(wrap_with_ddp=False)
    model = model_list
    
    # TEMP FIX for inference failure when mtp_num_layers is not None
    for m in model:
        if hasattr(m, 'config') and hasattr(m.config, 'mtp_num_layers'):
            m.config.mtp_num_layers = None
    
    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        
    nparams = sum(p.numel() for p in model[0].parameters())
    logger.info("Built mcore model: %d parameters", nparams)

    # 7. Initialize Tokenizer
    logger.info("Step 7: Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer initialized.")

    # 8. Generate Text
    logger.info("Step 8: Starting Generation...")
    prompt = args.prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()
    stop_tokens = [tokenizer.eos_token_id]

    start_time = time.time()
    
    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            if step % 5 == 0:
                logger.info(f"Generation step {step}/{args.max_new_tokens}")

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(input_ids, position_ids)

            output = fwd_bwd_function(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # Debug: print token information for first few steps
                if step < 3:
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    logger.info(f"Step {step} Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    logger.info(f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})")
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            if next_token_ids.item() in stop_tokens:
                logger.info("Stop token generated.")
                break

    end_time = time.time()
    generation_time = end_time - start_time
    tokens_generated = generated_ids.size(1) - tokenizer.encode(prompt, return_tensors="pt").size(1)
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(list(generated_ids[0]))
    
    print_rank_0("\n" + "="*40)
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0("-" * 20)
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("="*40)
    print_rank_0(f"Generation Stats: {tokens_generated} tokens in {generation_time:.2f}s ({tokens_generated/generation_time:.2f} tokens/s)")
    print_rank_0("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation from HuggingFace Models via MBridge and AnyModel")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model choice: gpt (gpt-oss-20b), nemo (Nemotron-3-Nano), llama (Llama-3.2-3B), qwen (Qwen3-8B)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint (HF format) to load weights/config from. Overrides --model for loading, but --model is still used for converter selection.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt for text generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for HF models")
    
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
