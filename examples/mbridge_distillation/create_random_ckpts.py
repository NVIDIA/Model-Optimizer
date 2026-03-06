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
# WITHOUT WARRANTIES OR CONDITIONS FOR A PARTICULAR PURPOSE.  See the
# License for the specific language governing permissions and
# limitations under the License.

"""
Create a small checkpoint from a model by loading block config via create_block_library,
then walking layer-by-layer and replacing dimensions with small values (64/128/256, num_experts 8/16/32).
Optionally inject no_ops when --no_op is set. Saves a full AnyModel-style checkpoint.

Usage:
  python -m mbridge_distillation.create_random_ckpts --model llama --output-dir ./small_ckpt
  python -m mbridge_distillation.create_random_ckpts --model llama --checkpoint /path/to/ckpt --no_op --output-dir ./small_ckpt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

# Add parent so we can import from generate_text
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Heterogeneous small values (cycle by layer for diverse checkpoints)
FFN_SIZES = [64, 128, 256]
EXPERT_DIMS = [64, 128, 256]
NUM_EXPERTS_OPTIONS = [8, 16, 32]
SHARED_EXPERT_DIMS = [64, 128, 256]


def _divisors_of(n: int, min_val: int = 1, max_val: int | None = None) -> list[int]:
    """Return sorted list of divisors of n in [min_val, max_val or n]. Used for GQA head counts."""
    if n <= 0:
        return []
    out = []
    for d in range(1, int(n**0.5) + 1):
        if n % d != 0:
            continue
        for x in (d, n // d):
            if min_val <= x <= (max_val if max_val is not None else n) and x not in out:
                out.append(x)
    return sorted(out)


def create_block_library(config, converter_name: str):
    """Build block_configs from HF config using the AnyModel Converter (same as generate_text)."""
    from modelopt.torch.puzzletron.anymodel import ConverterFactory

    logger.info("Creating block library for converter: %s", converter_name)
    converter_cls = ConverterFactory.get(converter_name)
    if converter_cls is None:
        raise ValueError(
            f"Unknown converter '{converter_name}'. "
            "Valid: gpt_oss_20b, nemotron_h_v2, llama, qwen3."
        )
    block_configs = converter_cls.create_block_configs_from_main_config(config)
    out = []
    for bc in block_configs:
        if hasattr(bc, "to_dict"):
            out.append(bc.to_dict())
        elif isinstance(bc, dict):
            out.append(copy.deepcopy(bc))
        else:
            import dataclasses
            out.append(dataclasses.asdict(bc))
    logger.info("Created block library: %d block configs (converter=%s)", len(out), converter_name)
    return out


def _get_kv_heads_options(num_attention_heads: int) -> list[int]:
    """TP-safe KV head options: divisors of num_attention_heads that are divisible by 4.
    Returns 2–3 distinct values for heterogeneous layers (small, small, next pattern).
    """
    all_divisors = _divisors_of(num_attention_heads)
    valid_kv = [d for d in all_divisors if d >= 1 and (d == 1 or d % 4 == 0)]
    if not valid_kv:
        valid_kv = [4] if num_attention_heads >= 4 else [1]
    # Prefer multiple distinct values for heterogeneity (e.g. [2, 2, 4] or [4, 4, 8])
    if len(valid_kv) >= 2:
        return [valid_kv[0], valid_kv[0], valid_kv[min(1, len(valid_kv) - 1)]]
    return [valid_kv[0]]


def apply_diverse_small_values(
    block_configs: list[dict],
    config,
    descriptor,
    use_no_ops: bool,
) -> list[dict]:
    """Apply heterogeneous small values (64/128/256, 8/16/32) and descriptor-aware no_ops.
    Follows the old codebase: multiple KV head options (TP-safe divisors), cycling FFN/MoE
    sizes per layer, and no_ops only when descriptor supports them (mod/rem pattern).
    """
    num_attention_heads = int(getattr(config, "num_attention_heads", 4) or 4)
    kv_heads_options = _get_kv_heads_options(num_attention_heads)

    attn_no_op_ok = descriptor.attn_no_op_supported() if use_no_ops else False
    mlp_no_op_ok = descriptor.mlp_no_op_supported() if use_no_ops else False
    attn_no_op_mod, attn_no_op_rem = 7, 3
    ffn_no_op_mod, ffn_no_op_rem = 5, 2

    # Experts per token for MoE (from config or default 1)
    cfg = getattr(config, "text_config", None) or config
    num_experts_per_tok = int(getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "experts_per_token", None) or 1)

    result = []
    for layer_idx, block in enumerate(block_configs):
        block = copy.deepcopy(block)
        attn = block.setdefault("attention", {})
        ffn = block.setdefault("ffn", {})

        # No-op: only if use_no_ops and descriptor supports it (same mod/rem as old codebase)
        if use_no_ops:
            if attn_no_op_ok and layer_idx % attn_no_op_mod == attn_no_op_rem:
                attn["no_op"] = True
                attn.pop("num_key_value_heads", None)
            if mlp_no_op_ok and layer_idx % ffn_no_op_mod == ffn_no_op_rem:
                ffn["no_op"] = True
                ffn.pop("intermediate_size", None)
                ffn.pop("moe", None)

        # Attention: heterogeneous num_key_value_heads (only if not no_op and not Mamba)
        # Note: after dataclasses.asdict(), "mamba" key is always present; use .get() to
        # distinguish Mamba slots (mamba is a dict) from standard attention slots (mamba is None).
        if not attn.get("no_op") and attn.get("mamba") is None and "num_key_value_heads" in attn:
            kv = kv_heads_options[layer_idx % len(kv_heads_options)]
            if 1 <= kv <= num_attention_heads:
                attn["num_key_value_heads"] = kv
            else:
                attn["num_key_value_heads"] = kv_heads_options[0]

        # FFN: heterogeneous intermediate_size (cycle through set of values)
        if not ffn.get("no_op"):
            if "intermediate_size" in ffn and ffn["intermediate_size"] is not None:
                ffn["intermediate_size"] = FFN_SIZES[layer_idx % len(FFN_SIZES)]
            if "moe" in ffn and ffn["moe"] is not None:
                moe = ffn["moe"]
                if isinstance(moe, dict):
                    ne = NUM_EXPERTS_OPTIONS[layer_idx % len(NUM_EXPERTS_OPTIONS)]
                    ed = EXPERT_DIMS[layer_idx % len(EXPERT_DIMS)]
                    moe["num_local_experts"] = ne
                    moe["expert_intermediate_dim"] = ed
                    top_k = moe.get("num_experts_per_tok") or num_experts_per_tok
                    if top_k > ne:
                        moe["num_experts_per_tok"] = min(top_k, ne)
                    else:
                        moe["num_experts_per_tok"] = num_experts_per_tok
                    # Shared expert dim (only when the model has shared experts, e.g. Nemotron-H)
                    if moe.get("shared_expert_intermediate_dim") is not None:
                        moe["shared_expert_intermediate_dim"] = SHARED_EXPERT_DIMS[layer_idx % len(SHARED_EXPERT_DIMS)]

        result.append(block)
    return result


def _sanitize_model_id(model_id: str) -> str:
    """Sanitize model id for use as a directory name."""
    return re.sub(r"[^\w\-.]", "-", model_id).strip("-")


def _get_model_source_dir(load_path: str) -> Path | None:
    """Return the directory containing the model (local path or HF snapshot) for copying custom code."""
    path = Path(load_path)
    if path.exists() and path.is_dir():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download
        snapshot_path = snapshot_download(repo_id=load_path)
        return Path(snapshot_path)
    except Exception:
        return None


def _copy_modeling_code_to_checkpoint(source_dir: Path, output_dir: Path) -> int:
    """Copy all .py files from source_dir to output_dir for trust_remote_code."""
    count = 0
    for py_file in source_dir.glob("*.py"):
        if py_file.is_file():
            dest = output_dir / py_file.name
            shutil.copy2(py_file, dest)
            count += 1
    return count


def _copy_tokenizer_files_to_checkpoint(source_dir: Path, output_dir: Path) -> int:
    """Copy tokenizer files (e.g. tekken.json, *.model) so NeMo tokenizers find them."""
    count = 0
    for f in source_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if name == "tekken.json" or name.endswith(".model") or ".model." in name:
            shutil.copy2(f, output_dir / name)
            count += 1
    return count


def load_hf_config(load_path: str, trust_remote_code: bool = True):
    """Load HuggingFace config only (no weights)."""
    from transformers import AutoConfig
    logger.info("Loading config from %s (config only)", load_path)
    return AutoConfig.from_pretrained(load_path, trust_remote_code=trust_remote_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create small block_configs from a model using create_block_library, then override values to 64/128/256 and 8/16/32."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt", "nemo", "llama", "qwen"],
        help="Model type (same as generate_text: gpt, nemo, llama, qwen)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint to load config from. If not set, use default model id for --model.",
    )
    parser.add_argument(
        "--no_op",
        action="store_true",
        default=False,
        help="Inject no_ops in some layers (every 3rd layer attn no_op, every 5th ffn no_op). Default: False.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the full checkpoint (config, weights, block_configs.json, tokenizer). Default: ./small_ckpt_<sanitized_model>",
    )
    parser.add_argument(
        "--blocks-only",
        action="store_true",
        help="Only write block_configs.json to --output-dir (or stdout path); do not build or save the model checkpoint.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="If set, use this many layers for create_block_library (model really small).",
    )
    parser.add_argument(
        "--small_config",
        action="store_true",
        default=False,
        help="Override config to small sizes (hidden_size=256, num_attention_heads=4, intermediate_size=128) before creating block library.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Weight dtype for saved checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for weight initialization.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading HF config.",
    )
    args = parser.parse_args()

    # Same MODEL_MAP as generate_text
    MODEL_MAP = {
        "gpt": ("openai/gpt-oss-20b", "gpt_oss_20b"),
        "nemo": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron_h_v2"),
        "llama": ("meta-llama/Llama-3.2-3B-Instruct", "llama"),
        "qwen": ("Qwen/Qwen3-8B", "qwen3"),
    }
    model_id, converter_name = MODEL_MAP[args.model]
    load_path = args.checkpoint if args.checkpoint else model_id
    logger.info("Model type: %s, converter: %s, load path: %s", args.model, converter_name, load_path)

    config = load_hf_config(load_path, trust_remote_code=args.trust_remote_code)

    if args.small_config:
        config.num_attention_heads = 4
        config.num_key_value_heads = getattr(config, "num_key_value_heads", None) or 2
        config.intermediate_size = 128
        if hasattr(config, "num_local_experts"):
            config.num_local_experts = 8
        if hasattr(config, "moe_intermediate_size"):
            config.moe_intermediate_size = 128
        if hasattr(config, "expert_intermediate_dim"):
            config.expert_intermediate_dim = 128
        if args.num_layers is None:
            args.num_layers = 4
        logger.info("Small config: num_attention_heads=4, intermediate_size=128")

        # Nemotron-H (Mamba hybrid) specific overrides
        if getattr(config, "model_type", "") == "nemotron_h":
            config.hidden_size = 128
            # head_dim for attention = hidden_size // num_attention_heads = 32
            if hasattr(config, "head_dim"):
                config.head_dim = 32
            # Mamba2 constraint: mamba_num_heads * mamba_head_dim <= expand * hidden_size
            # expand=2, hidden_size=128 -> d_inner=256; use 2 heads * 64 dim = 128 <= 256
            if hasattr(config, "mamba_num_heads"):
                config.mamba_num_heads = 2
            if hasattr(config, "mamba_head_dim"):
                config.mamba_head_dim = 64
            # n_groups must divide mamba_num_heads
            if hasattr(config, "n_groups"):
                config.n_groups = 1
            if hasattr(config, "n_group"):
                config.n_group = 1
            # SSM state and chunk sizes
            if hasattr(config, "ssm_state_size"):
                config.ssm_state_size = 8
            if hasattr(config, "chunk_size"):
                config.chunk_size = 16
            # MoE: use n_routed_experts (Nemotron field), not num_local_experts
            if hasattr(config, "n_routed_experts"):
                config.n_routed_experts = 8
            config.intermediate_size = 64
            if hasattr(config, "moe_intermediate_size"):
                config.moe_intermediate_size = 64
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                config.moe_shared_expert_intermediate_size = 128
            # Need at least 6 layers to cover M/E/M/E/M/* slot pattern
            if args.num_layers is None or args.num_layers < 6:
                args.num_layers = 6
            logger.info(
                "Nemotron-H small config: hidden_size=128, mamba_num_heads=2, mamba_head_dim=64, "
                "n_routed_experts=8, num_layers=%d", args.num_layers
            )
    if args.num_layers is not None:
        config.num_hidden_layers = args.num_layers
        logger.info("Using num_hidden_layers=%d", config.num_hidden_layers)

    # Get descriptor early so we can use attn_no_op_supported/mlp_no_op_supported for heterogeneous no_ops
    from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptorFactory
    descriptor = ModelDescriptorFactory.get(converter_name)
    if descriptor is None:
        raise ValueError(f"Unknown descriptor for converter: {converter_name}")

    block_configs = create_block_library(config, converter_name)
    block_configs = apply_diverse_small_values(
        block_configs,
        config=config,
        descriptor=descriptor,
        use_no_ops=args.no_op,
    )

    output_dir = args.output_dir
    if output_dir is None:
        default_subdir = _sanitize_model_id(load_path)
        output_dir = f"./small_ckpt_{default_subdir}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always write block_configs.json into the output dir
    block_configs_path = output_dir / "block_configs.json"
    with open(block_configs_path, "w", encoding="utf-8") as f:
        json.dump({"block_configs": block_configs}, f, indent=2)
    logger.info("Wrote %d block configs to %s (no_op=%s)", len(block_configs), block_configs_path, args.no_op)

    if args.blocks_only:
        print(f"Saved block configs only to {output_dir} (layers={len(block_configs)}, no_op={args.no_op})")
        return

    # Build and save full checkpoint (like the old create_random_ckpts)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
    from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import maybe_cast_block_configs
    from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import force_cache_dynamic_modules, save_checkpoint

    torch.manual_seed(args.seed)

    # Attach block_configs to config; cast to BlockConfig list for deci_x_patcher (same as old codebase)
    setattr(config, "block_configs", maybe_cast_block_configs(block_configs))
    force_cache_dynamic_modules(config, load_path)

    # Get model class (LlamaForCausalLM, etc.) and build from config under patcher
    try:
        from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import _get_model_class_from_config
    except ImportError:
        from transformers import AutoModelForCausalLM as _AM
        def _get_model_class_from_config(_c):
            return _AM

    with deci_x_patcher(model_descriptor=descriptor, block_configs=config.block_configs):
        model_class = _get_model_class_from_config(config)
        if model_class is AutoModelForCausalLM:
            model = model_class.from_config(config, trust_remote_code=args.trust_remote_code)
        else:
            model = model_class._from_config(config)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    model.to(dtype=dtype_map[args.dtype])

    save_checkpoint(model, output_dir, descriptor)
    logger.info("Saved model checkpoint to %s", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved tokenizer to %s", output_dir)

    source_dir = _get_model_source_dir(load_path)
    if source_dir is not None:
        n_copied = _copy_modeling_code_to_checkpoint(source_dir, output_dir)
        if n_copied:
            logger.info("Copied %d .py file(s) from %s into checkpoint.", n_copied, source_dir)
        n_tok = _copy_tokenizer_files_to_checkpoint(source_dir, output_dir)
        if n_tok:
            logger.info("Copied %d tokenizer file(s) into checkpoint.", n_tok)

    print(f"Saved checkpoint to {output_dir} (layers={len(block_configs)}, no_op={args.no_op})")


if __name__ == "__main__":
    main()
