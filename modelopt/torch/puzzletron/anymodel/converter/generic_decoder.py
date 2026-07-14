"""Converter derived entirely from a composable generic decoder contract."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ...block_config import AttentionConfig, BlockConfig, FFNConfig, MLAConfig, MoEConfig
from ...tools.checkpoint_utils_hf import load_model_config, save_model_config
from .base import Converter

__all__ = [
    "GenericDecoderConverter",
    "rewrite_checkpoint_key",
    "rewrite_safetensor_checkpoint_keys",
]


def rewrite_checkpoint_key(
    key: str, rewrites: tuple[tuple[str, str], ...]
) -> str:
    """Apply exactly one descriptor-declared checkpoint-key migration."""
    matches = [(pattern, replacement) for pattern, replacement in rewrites if re.search(pattern, key)]
    if len(matches) > 1:
        raise ValueError(f"checkpoint key {key!r} matches multiple rewrite rules: {matches!r}")
    if not matches:
        return key
    pattern, replacement = matches[0]
    return re.sub(pattern, replacement, key, count=1)


def rewrite_safetensor_checkpoint_keys(
    checkpoint: Path,
    rewrites: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    """Materialize descriptor-declared key migrations without mutating hardlinked sources."""
    checkpoint = Path(checkpoint)
    if not rewrites:
        return {}
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = dict(index.get("weight_map") or {})
        shard_names = sorted(set(weight_map.values()))
    elif (checkpoint / "model.safetensors").is_file():
        index = None
        weight_map = {}
        shard_names = ["model.safetensors"]
    else:
        raise ValueError(
            "descriptor checkpoint-key rewrites require a safetensors checkpoint"
        )

    rewritten: dict[str, str] = {}
    for shard_name in shard_names:
        shard_path = checkpoint / shard_name
        tensors = load_file(shard_path, device="cpu")
        migrated: dict[str, Any] = {}
        for key, tensor in tensors.items():
            new_key = rewrite_checkpoint_key(key, rewrites)
            if new_key in migrated:
                raise ValueError(
                    f"checkpoint-key migration collides at {new_key!r} in {shard_name}"
                )
            migrated[new_key] = tensor
            if new_key != key:
                rewritten[key] = new_key
        if not any(key in rewritten for key in tensors):
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
        temp_path = shard_path.with_name(f".{shard_path.name}.rewrite.tmp")
        temp_path.unlink(missing_ok=True)
        save_file(migrated, temp_path, metadata=metadata)
        os.chmod(temp_path, shard_path.stat().st_mode)
        os.replace(temp_path, shard_path)

    if index is not None:
        migrated_map: dict[str, str] = {}
        for key, shard_name in weight_map.items():
            new_key = rewrite_checkpoint_key(key, rewrites)
            if new_key in migrated_map:
                raise ValueError(f"checkpoint index migration collides at {new_key!r}")
            migrated_map[new_key] = shard_name
        index["weight_map"] = migrated_map
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return rewritten


def _value(config: Any, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _moe_layer(config: Any, layer_idx: int) -> bool:
    enabled = _value(config, "enable_moe_block", None)
    if enabled is False:
        return False
    indices = _value(config, "moe_layer_indices", None)
    if indices is not None:
        return int(layer_idx) in {int(value) for value in indices}
    first_dense = _value(config, "first_k_dense_replace", None)
    if first_dense is not None:
        first_dense = int(first_dense)
        if layer_idx < first_dense:
            return False
        frequency = int(_value(config, "moe_layer_freq", 1) or 1)
        return (layer_idx - first_dense) % frequency == 0
    frequency = _value(config, "moe_layer_freq", None)
    if frequency is not None:
        return layer_idx % int(frequency) == 0
    return enabled is True or any(
        _value(config, field, None) not in (None, 0)
        for field in ("num_experts", "num_local_experts", "n_routed_experts")
    )


class GenericDecoderConverter(Converter):
    """Attach typed block configs using descriptor-owned structural contracts."""

    @staticmethod
    def create_block_configs(descriptor, config: Any) -> list[BlockConfig]:
        contract = descriptor.generic_decoder_contract(config)
        if contract is None:
            raise ValueError(f"{descriptor.__name__} has no generic decoder contract")
        lm_config = descriptor.get_language_model_config(config)
        num_layers = int(_value(lm_config, "num_hidden_layers"))
        layer_types = tuple(_value(lm_config, "layer_types", ()) or ())
        blocks: list[BlockConfig] = []
        for layer_idx in range(num_layers):
            subblocks = []
            if contract.latent_attention is not None:
                latent = contract.latent_attention
                subblocks.append(
                    MLAConfig(
                        num_heads=int(_value(lm_config, "num_attention_heads")),
                        q_lora_rank=int(_value(lm_config, latent.q_lora_rank_field)),
                        kv_lora_rank=int(_value(lm_config, latent.kv_lora_rank_field)),
                    )
                )
            if contract.attention is not None:
                attention = contract.attention
                layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else None
                query_heads, kv_heads, head_dim, k_eq_v, kv_source_layer = (
                    attention.layer_geometry(
                        lm_config,
                        layer_idx=layer_idx,
                        layer_types=layer_types,
                    )
                )
                if (
                    layer_type == "sliding_attention"
                    and _value(lm_config, "sliding_window", None) is not None
                ):
                    sliding_window = int(_value(lm_config, "sliding_window"))
                elif (
                    layer_type == "full_attention"
                    and (
                        contract.explicit_full_attention_window
                        or hasattr(lm_config, "sliding_window")
                    )
                ):
                    sliding_window = "full"
                else:
                    sliding_window = None
                subblocks.append(
                    AttentionConfig(
                        num_kv_heads=kv_heads,
                        num_query_heads=query_heads,
                        qk_head_dim=head_dim,
                        sliding_window_size=sliding_window,
                        k_eq_v=k_eq_v,
                        kv_source_layer=kv_source_layer,
                    )
                )

            use_moe = contract.routed_moe is not None and _moe_layer(
                lm_config, layer_idx
            )
            if contract.dense_ffn is not None and (
                not use_moe or not contract.routed_moe.replaces_dense_ffn
            ):
                subblocks.append(
                    FFNConfig(
                        intermediate_size=contract.dense_ffn.layer_intermediate_size(
                            lm_config,
                            layer_idx=layer_idx,
                        )
                    )
                )
            if use_moe:
                moe = contract.routed_moe
                shared = _value(lm_config, moe.shared_intermediate_field, None)
                subblocks.append(
                    MoEConfig(
                        num_experts=int(_value(lm_config, moe.num_experts_field)),
                        expert_intermediate_size=int(
                            _value(lm_config, moe.intermediate_field)
                        ),
                        shared_expert_intermediate_size=(
                            int(shared) if moe.shared_expert_name and shared is not None else None
                        ),
                        top_k=int(_value(lm_config, moe.top_k_field)),
                    )
                )
            blocks.append(BlockConfig(subblock_configs=tuple(subblocks)))
        return blocks

    @classmethod
    def convert(cls, descriptor, input_dir: Path, output_dir: Path):
        cls.copy_checkpoint_files(input_dir, output_dir)
        trust_remote_code = descriptor.requires_trust_remote_code()
        config = load_model_config(input_dir, trust_remote_code=trust_remote_code)
        contract = descriptor.generic_decoder_contract(config)
        rewrite_safetensor_checkpoint_keys(
            output_dir, contract.checkpoint_key_rewrites
        )
        block_configs = cls.create_block_configs(descriptor, config)
        out_config = copy.deepcopy(config)
        descriptor.set_block_configs(out_config, block_configs)
        save_model_config(out_config, output_dir)

    @staticmethod
    def create_block_configs_from_main_config(config):
        raise TypeError(
            "GenericDecoderConverter requires create_block_configs(descriptor, config)"
        )
