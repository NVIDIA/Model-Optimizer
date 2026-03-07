# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Generate text from a HuggingFace model via Megatron Bridge with AnyModel (heterogeneous) support.

This script demonstrates the full pipeline for loading an AnyModel/Puzzletron heterogeneous
checkpoint into Megatron-Core via Megatron Bridge and running greedy text generation.

Key differences from a standard Megatron Bridge workflow
---------------------------------------------------------
Standard Bridge (homogeneous models):
    bridge = AutoBridge.from_hf_pretrained(...)
    provider = bridge.to_megatron_provider()
    model = provider.provide_distributed_model(...)

This script (heterogeneous AnyModel models):
    # 1. Apply the provider patch (one-time setup):
    apply_patch()

    # 2. Load HF model + convert to Bridge (deci_x_patcher needed to load heterogeneous HF model):
    with deci_x_patcher(model_descriptor=descriptor):
        bridge = AutoBridge.from_hf_pretrained(...)
    provider = bridge.to_megatron_provider()

    # 3. Attach block_configs to the provider (read from hf_config.block_configs, which is
    #    set automatically when the checkpoint was saved by AnyModel):
    set_provider_block_configs(provider, block_configs)

    # 4. Build the MCore model — mbridge_patcher is now activated automatically inside
    #    provide_distributed_model(), injecting per-layer config before each layer is built:
    model = provider.provide_distributed_model(wrap_with_ddp=False)

Block config sources (in priority order)
-----------------------------------------
1. hf_config.block_configs — the canonical source. Set by AnyModel when saving a
   heterogeneous checkpoint. Reading from the HF config guarantees the block_configs
   exactly match the per-layer architecture that was used to create the weights.

2. AnyModel ConverterFactory — generate default block_configs from the global model config.
   Used when first converting a model that does not yet have block_configs in its config.

Supported models
----------------
Run with --model <key> where key is one of:
    gpt    → openai/gpt-oss-20b       (GPT-OSS, MoE, all layers are MoE TransformerLayers)
    nemo   → nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  (Nemotron-H, Mamba+MoE hybrid)
    llama  → meta-llama/Llama-3.2-3B-Instruct            (dense GPT, homogeneous)
    qwen   → Qwen/Qwen3-8B                               (dense GPT with GQA)

Usage
-----
    # Single GPU, load from HuggingFace (requires HF_TOKEN for gated models):
    torchrun --nproc_per_node=1 generate_text.py --model llama --prompt "Hello"

    # Load from a local heterogeneous checkpoint saved by AnyModel:
    torchrun --nproc_per_node=1 generate_text.py \\
        --model nemo \\
        --checkpoint /path/to/checkpoint \\
        --trust-remote-code

    # Multi-GPU with tensor parallelism:
    torchrun --nproc_per_node=4 generate_text.py --model llama --tp 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoConfig, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported model registry
# ---------------------------------------------------------------------------

# Maps --model flag → (hf_model_id, anymodel_converter_name)
#   hf_model_id:          Used when --checkpoint is not provided (load from HuggingFace Hub).
#   anymodel_converter:   Key for ConverterFactory / ModelDescriptorFactory.
#                         Also used to generate block_configs when they are not in config.json.
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gpt":   ("openai/gpt-oss-20b",                         "gpt_oss_20b"),
    "nemo":  ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron_h_v2"),
    "llama": ("meta-llama/Llama-3.2-3B-Instruct",           "llama"),
    "qwen":  ("Qwen/Qwen3-8B",                              "qwen3"),
}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


class _SingleBatchIterator:
    """Yields one batch dict then stops — required by MCore's forward_backward_func."""

    def __init__(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> None:
        self._batch = {"tokens": input_ids, "position_ids": position_ids}
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        self._done = True
        return self._batch


def _forward_step(data_iterator, model, **_) -> tuple:
    """Forward step function required by MCore's get_forward_backward_func."""
    batch = next(data_iterator)
    output = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    return output, lambda x, **__: x  # loss_func is identity (we only need logits)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_hf_config(load_path: str, trust_remote_code: bool) -> "PretrainedConfig":
    """Load only the HuggingFace config (no weights) from a path or Hub model ID."""
    logger.info("Loading HF config from %r (no weights)", load_path)
    return AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)


def _get_block_configs(hf_config, converter_name: str) -> Optional[list]:
    """Load block_configs from hf_config (primary) or generate via AnyModel (fallback).

    See module docstring for the priority order and rationale.
    """
    from block_config_utils import load_block_configs
    return load_block_configs(hf_config, converter_name)


def _get_model_descriptor(converter_name: str):
    """Return the AnyModel ModelDescriptor for the given converter name, or None."""
    try:
        from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
        descriptor = ModelDescriptorFactory.get(converter_name)
        if descriptor is None:
            logger.warning("No AnyModel descriptor found for converter '%s'", converter_name)
        return descriptor
    except ImportError:
        logger.warning("ModelOpt AnyModel not installed; cannot obtain model descriptor")
        return None


def _load_bridge(load_path: str, trust_remote_code: bool, descriptor) -> "AutoBridge":
    """Load an HF model into a Megatron Bridge object.

    If an AnyModel descriptor is available, the model is loaded inside ``deci_x_patcher``,
    which patches the HF model's __init__ / from_pretrained path to correctly construct
    heterogeneous layers (different sub-layer types per slot).  This is required for
    AnyModel checkpoints (e.g. Nemotron-H, GPT-OSS heterogeneous).

    For standard homogeneous models (e.g. vanilla Llama) the patcher is a no-op, so
    always using it is safe.
    """
    if descriptor is not None:
        from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
        logger.info("Loading HF model with deci_x_patcher (descriptor=%s)", type(descriptor).__name__)
        with deci_x_patcher(model_descriptor=descriptor):
            return AutoBridge.from_hf_pretrained(load_path, trust_remote_code=trust_remote_code)
    else:
        logger.info("Loading HF model without deci_x_patcher (AnyModel not available)")
        return AutoBridge.from_hf_pretrained(load_path, trust_remote_code=trust_remote_code)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    if args.model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown --model '{args.model}'. Valid choices: {sorted(MODEL_REGISTRY)}"
        )

    hf_model_id, converter_name = MODEL_REGISTRY[args.model]
    load_path = args.checkpoint or hf_model_id

    logger.info("=== mbridge_distillation_v2/generate_text ===")
    logger.info("  model key:        %s", args.model)
    logger.info("  HF load path:     %s", load_path)
    logger.info("  converter:        %s", converter_name)
    logger.info("  parallelism:      TP=%d PP=%d EP=%d ETP=%d", args.tp, args.pp, args.ep, args.etp)

    # ------------------------------------------------------------------
    # Step 1: Load HF config
    #   The HF config (config.json) is the primary source for block_configs.
    #   For AnyModel checkpoints, config.json contains the block_configs attribute
    #   that was set when the checkpoint was saved.  We read it here, before loading
    #   any weights, because block_configs determines the *shape* of each layer.
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading HF config")
    hf_config = _load_hf_config(load_path, trust_remote_code=args.trust_remote_code)

    # ------------------------------------------------------------------
    # Step 2: Obtain block_configs
    #   Priority: hf_config.block_configs → AnyModel converter fallback → None (homogeneous)
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading block_configs")
    block_configs = _get_block_configs(hf_config, converter_name)
    if block_configs:
        logger.info("  block_configs: %d layers", len(block_configs))
    else:
        logger.info("  block_configs: None (homogeneous model — no per-layer overrides)")

    # ------------------------------------------------------------------
    # Step 3: Get AnyModel descriptor (needed for deci_x_patcher)
    # ------------------------------------------------------------------
    logger.info("Step 3: Getting AnyModel model descriptor")
    descriptor = _get_model_descriptor(converter_name)

    # ------------------------------------------------------------------
    # Step 4: Apply provider patch (one-time class-level setup)
    #   Patches ModelProviderMixin.provide() so that any provider with block_configs
    #   attached will automatically run inside mbridge_patcher during model construction.
    # ------------------------------------------------------------------
    logger.info("Step 4: Applying Megatron Bridge provider patch")
    from provider_patch import apply_patch, set_provider_block_configs
    apply_patch()

    # ------------------------------------------------------------------
    # Step 5: Load HF model into Bridge
    #   Uses deci_x_patcher (if descriptor available) to handle heterogeneous HF models.
    #   Bridge loads the HF model, reads its weights, and stores a weight conversion plan.
    # ------------------------------------------------------------------
    logger.info("Step 5: Loading HF model into Megatron Bridge")
    bridge = _load_bridge(load_path, trust_remote_code=args.trust_remote_code, descriptor=descriptor)
    logger.info("  Bridge loaded: %s", type(bridge).__name__)

    # ------------------------------------------------------------------
    # Step 6: Convert to Megatron provider and configure
    #   to_megatron_provider() creates a provider with parallelism settings.
    #   We then attach block_configs so the provider knows about per-layer overrides.
    # ------------------------------------------------------------------
    logger.info("Step 6: Converting Bridge to Megatron provider")
    provider = bridge.to_megatron_provider(load_weights=True)
    logger.info("  Provider type: %s", type(provider).__name__)

    # Configure distributed parallelism.
    provider.tensor_model_parallel_size  = args.tp
    provider.pipeline_model_parallel_size = args.pp
    provider.expert_model_parallel_size   = args.ep
    provider.expert_tensor_parallel_size  = args.etp
    provider.pipeline_dtype               = torch.bfloat16

    # Attach block_configs. This also ensures the provider's provide() is patched,
    # even if the provider subclass overrides provide() (bypassing the class patch).
    set_provider_block_configs(provider, block_configs)
    logger.info(
        "  block_configs attached: %s",
        f"{len(block_configs)} layers" if block_configs else "None",
    )

    # ------------------------------------------------------------------
    # Step 7: Finalize and initialize model-parallel state
    # ------------------------------------------------------------------
    logger.info("Step 7: Finalizing provider and initializing model-parallel state")
    provider.finalize()
    provider.initialize_model_parallel(seed=0)

    # ------------------------------------------------------------------
    # Step 8: Build the MCore model
    #   provide_distributed_model() calls provider.provide() which is now wrapped by
    #   _patched_provide(), which activates mbridge_patcher during construction.
    #   Every TransformerLayer and MambaLayer is intercepted and built with its
    #   per-layer config from block_configs.
    # ------------------------------------------------------------------
    logger.info("Step 8: Building MCore model")
    model_list = provider.provide_distributed_model(wrap_with_ddp=False)

    # Temporary workaround: mtp_num_layers must be None for inference to work correctly.
    for m in model_list:
        if hasattr(m, "config") and hasattr(m.config, "mtp_num_layers"):
            m.config.mtp_num_layers = None

    model_list = [m.cuda() for m in model_list]
    for m in model_list:
        m.eval()

    n_params = sum(p.numel() for p in model_list[0].parameters())
    logger.info("  Built %s with %.2fB parameters", type(model_list[0]).__name__, n_params / 1e9)

    # ------------------------------------------------------------------
    # Step 9: Load tokenizer
    # ------------------------------------------------------------------
    logger.info("Step 9: Loading tokenizer from %r", load_path)
    tokenizer = AutoTokenizer.from_pretrained(
        load_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Step 10: Greedy generation loop
    # ------------------------------------------------------------------
    logger.info("Step 10: Starting generation (max_new_tokens=%d)", args.max_new_tokens)
    prompt = args.prompt

    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )
    generated_ids = input_ids.clone()
    stop_token_ids = set(filter(None, [tokenizer.eos_token_id]))

    fwd_bwd_func = get_forward_backward_func()
    tp_world = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    is_last_pp_stage = parallel_state.is_pipeline_last_stage()

    start = time.monotonic()

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            output = fwd_bwd_func(
                forward_step_func=_forward_step,
                data_iterator=_SingleBatchIterator(input_ids, position_ids),
                model=model_list,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if is_last_pp_stage:
                # All-gather logits across TP ranks (vocab is sharded across TP).
                gathered = [torch.zeros_like(output) for _ in range(tp_world)]
                dist.all_gather(gathered, output, group=tp_group)
                logits = torch.cat(gathered, dim=2)  # [batch, seq, vocab]
                next_token_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)  # [batch, 1]

                if step < 3:
                    top5_vals, top5_ids = torch.topk(logits[0, -1], 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids.tolist()]
                    logger.info(
                        "Step %d top-5: %s  →  selected %r (id=%d)",
                        step,
                        list(zip(top5_tokens, [f"{v:.3f}" for v in top5_vals.tolist()])),
                        tokenizer.decode([next_token_id.item()]),
                        next_token_id.item(),
                    )
            else:
                # Non-last PP stages: placeholder token to keep dist.broadcast shape consistent.
                next_token_id = torch.ones((1, 1), device=input_ids.device, dtype=input_ids.dtype)

            # Broadcast from last rank to synchronize all ranks.
            dist.broadcast(next_token_id, src=get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Advance inputs for next step (full-sequence re-encoding; KV cache not used here).
            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            if next_token_id.item() in stop_token_ids:
                logger.info("Step %d: EOS token generated, stopping.", step)
                break

    elapsed = time.monotonic() - start
    n_new = generated_ids.size(1) - tokenizer.encode(prompt, return_tensors="pt").size(1)
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)

    print_rank_0("\n" + "=" * 60)
    print_rank_0("PROMPT:    " + prompt)
    print_rank_0("-" * 60)
    print_rank_0("GENERATED: " + generated_text)
    print_rank_0("=" * 60)
    print_rank_0(
        f"Stats: {n_new} new tokens in {elapsed:.2f}s "
        f"({n_new / elapsed:.2f} tok/s)"
    )
    print_rank_0("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help=(
            "Model to load. Determines both the default HuggingFace model ID and the "
            "AnyModel converter used to generate block_configs when they are not in config.json."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Local directory of an HF-format checkpoint to load. "
            "If omitted, the model is downloaded from HuggingFace Hub using the default "
            "model ID for --model (requires HF_TOKEN for gated models)."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help="Input text for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        dest="max_new_tokens",
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument("--tp",  type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pp",  type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--ep",  type=int, default=1, help="Expert parallel size.")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace config/tokenizer/model loading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
