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

"""Offline native model prefixes derived from pinned public checkpoint configs."""

import copy
import importlib
import inspect
import json
from pathlib import Path

import torch


def cases():
    """Return each ranked supported checkpoint plus distinct Qwen release variants."""
    return json.loads((Path(__file__).parent / "data/automodel_bypass_models.json").read_text())


def tiny_config(record):
    """Shrink dimensions while retaining the exact distinct decoder prefix."""
    from transformers import PretrainedConfig

    raw = copy.deepcopy(record["config"])
    decode = getattr(PretrainedConfig, "_decode_special_floats", None)
    if decode is not None:
        raw = decode(raw)
    text = raw.get("text_config", raw.get("llm_config", raw))
    prefix = record["minimum_pattern_prefix"]
    if not prefix:
        raise ValueError("Checkpoint has no registered native architecture")
    original_layers = text.get("num_hidden_layers", len(text.get("layers_block_type", [])))
    selected_layers = record.get("representative_layer_ids", list(range(prefix)))
    # Per-layer discriminators and source identities retain their global indices.
    for key, value in list(text.items()):
        if isinstance(value, list) and len(value) >= original_layers and key != "architectures":
            text[key] = [value[index] for index in selected_layers]
        elif key in ("kv_source_layer_ids", "index_source_layer_ids", "dspark_target_layer_ids"):
            text[key] = [i for i in value if i < prefix]
    dimensions = {
        "vocab_size": 128,
        "hidden_size": 256,
        "intermediate_size": 512,
        "dense_intermediate_size": 512,
        "moe_intermediate_size": 256,
        "shared_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
        "num_hidden_layers": prefix,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "v_head_dim": 64,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "o_lora_rank": 32,
        "o_groups": 1,
        "n_routed_experts": 4,
        "num_local_experts": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "n_shared_experts": 1,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 64,
        "swa_v_head_dim": 64,
        "linear_head_dim": 16,
        "linear_num_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 1024,
        "index_head_dim": 64,
        "index_n_heads": 4,
        "index_topk": 4,
        "index_kpool": 4,
        "hc_mult": 2,
        "hc_count": 2,
        "hc_lowrank": 4,
        "hc_sinkhorn_iters": 3,
        "indexer_budget": 4,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 64,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 4,
        "engram_n_heads": 2,
        "engram_head_dim": 8,
        "engram_vocab_size": 128,
        "engram_compressed_vocab_size": 128,
        "engram_num_embeddings": [4096, 4096],
        "ple_embed_dim": 64,
        "ple_conv_kernel_size": 4,
        "num_nextn_predict_layers": 0,
        "mtp_num_hidden_layers": 0,
        "num_mtp_modules": 0,
        "dspark_block_size": 0,
        "dspark_noise_token_id": 0,
        "moe_latent_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "mamba_head_dim": 64,
        "mamba_num_heads": 8,
        "ssm_state_size": 16,
        "n_groups": 1,
        "chunk_size": 16,
        "conv_kernel": 4,
    }
    if text.get("qk_rope_head_dim") == 0:
        dimensions["qk_rope_head_dim"] = 0
    if "qk_head_dim" in text:
        dimensions["qk_head_dim"] = dimensions["qk_nope_head_dim"] + dimensions["qk_rope_head_dim"]
    if "rotary_dim" in text:
        dimensions["rotary_dim"] = 32
    if "linear_num_value_heads" in text:
        ratio = text["linear_num_value_heads"] // text["linear_num_key_heads"]
        dimensions["linear_num_value_heads"] = 2 * ratio
    architecture = record["architectures"][0]
    if architecture.startswith("DeepseekV4"):
        dimensions["num_key_value_heads"] = 1
    if architecture == "Glm5NextForConditionalGeneration":
        dimensions["num_key_value_heads"] = dimensions["num_attention_heads"]
    if architecture == "MistralLarge4ForConditionalGeneration":
        dimensions["num_key_value_heads"] = dimensions["num_attention_heads"]
        dimensions["head_dim"] = dimensions["qk_nope_head_dim"] + dimensions["qk_rope_head_dim"]
        dimensions["q_lora_rank"] = None
        positions = {layer: index for index, layer in enumerate(selected_layers)}
        text["kv_source_layers"] = [positions[index] for index in text["kv_source_layers"]]
    if "ple_embed_dim" in text:
        dimensions["ple_embed_dim"] = (
            (text.get("ngram_size", 3) - 1) * text.get("heads_per_ngram", 8) * 4
        )
    for key, value in dimensions.items():
        if key in text or key in ("hidden_size", "vocab_size", "num_hidden_layers"):
            text[key] = value
    text.update(
        dtype="bfloat16",
        torch_dtype="bfloat16",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    raw.update(tie_word_embeddings=False)
    raw.pop("quantization_config", None)
    text.pop("quantization_config", None)
    vision = raw.get("vision_config")
    if isinstance(vision, dict):
        for key, value in {
            "depth": 1,
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "num_attention_heads": 2,
            "out_hidden_size": 256,
            "projection_intermediate_size": 64,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "image_size": 8,
            "head_dim": 16,
            "max_img_size": 16,
            "min_num_patches": 4,
            "max_num_patches": 64,
            "num_cls_tokens": 1,
            "num_registers": 0,
        }.items():
            if key in vision:
                vision[key] = value
        for key, value in list(vision.items()):
            if isinstance(value, list) and key == "vit_window_attn_types":
                vision[key] = value[:1]
            elif key == "deepstack_visual_indexes":
                vision[key] = [0]
    if architecture == "MuseSparkForConditionalGeneration":
        text.update(hidden_size=512, num_attention_heads=8, moe_hidden_size=256, sliding_window=16)
        raw.update(
            image_token_id=120,
            patch_token_id=120,
            video_token_id=121,
            audio_token_id=122,
            use_audio_perception=False,
            out_hidden_size=128,
            projector_hidden_size=64,
        )
        vision.update(layer_types=["full_attention"], pos_emb_height=4, pos_emb_width=4)
        raw["audio_config"].update(
            hidden_sizes=[8], depths=[1], patch_sizes=[2], num_hidden_layers=1, out_hidden_size=512
        )
    if architecture == "MuseSpark1xOmniForCausalLM":
        # Retain compressed routing with a complete 256-value IQ input block.
        raw.update(
            hidden_size=512,
            use_audio_perception=False,
            image_token_id=120,
            patch_token_id=121,
            video_token_id=122,
        )
        raw["moe"].update(
            num_experts=4, topk=2, override_routed_hidden_dim=256, compression_factor=2
        )
        raw["global_attn_cfg"] = [16 if window else 0 for window in raw["global_attn_cfg"]]
        vision.update(downsample_factor=2, patch_size=2, patch_temporal=2, projector_input_dim=16)
        vision["navit_config"].update(
            hidden_size=8,
            image_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            patch_size=2,
            patch_temporal=2,
            vision_output_dim=32,
        )
    if architecture == "MistralLarge4ForConditionalGeneration":
        raw["image_token_index"] = 120
        # Native from_dict performs the released composite-to-flat conversion.
        from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config

        config = MistralLarge4Config.from_dict(raw)
        config.vision_config.update(image_break_token_id=121, image_end_token_id=122)
        return config
    if architecture == "NemotronH_Omni_Reasoning_V3":
        # Native AutoModel uses this checkpoint's own RADIO implementation.
        # Only its pinned Python/config files are needed, never checkpoint weights.
        import pytest
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniConfig
        from transformers import NemotronHConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            code = snapshot_download(
                record["model_id"],
                revision=record["revision"],
                allow_patterns=["*.py", "config.json"],
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            pytest.skip(
                "Prefetch this pinned Nemotron config and Python code; weights are unnecessary"
            )
        radio = get_class_from_dynamic_module("configuration_radio.RadioConfig", code)
        raw.update(
            force_image_size=8,
            patch_size=2,
            vit_hidden_size=32,
            projector_hidden_size=64,
            img_context_token_id=120,
            video_context_token_id=121,
            sound_context_token_id=122,
        )
        raw["llm_config"] = NemotronHConfig.from_dict(text)
        vision["summary_idxs"] = [0]
        raw["vision_config"] = radio(**vision)
        raw.pop("model_type", None)
        config = NemotronOmniConfig(**raw)
        config.name_or_path = code
        return config
    sparse = text.get("sparse_attention_config")
    if isinstance(sparse, dict):
        sparse.update(
            sparse_index_dim=64, sparse_num_index_heads=2, sparse_topk_blocks=2, sparse_block_size=4
        )
        for k, v in list(sparse.items()):
            if isinstance(v, list) and len(v) >= original_layers:
                sparse[k] = v[:prefix]
    if record["architectures"][0] == "KimiK3ForCausalLM":
        # The released K3 MLA has equal query/KV head counts, not GQA.
        text["num_key_value_heads"] = text["num_attention_heads"]
        text.update(
            num_experts=4,
            num_experts_per_token=2,
            num_shared_experts=1,
            routed_expert_hidden_size=256,
            attn_res_block_size=2,
        )
        linear = text["linear_attn_config"]
        linear.update(
            num_heads=4,
            head_dim=64,
            kda_layers=[i for i in linear["kda_layers"] if i <= prefix],
            full_attn_layers=[i for i in linear["full_attn_layers"] if i <= prefix],
        )
    # Register native custom configs without importing their model implementations.
    importlib.import_module("nemo_automodel._transformers.registry")
    from transformers import AutoConfig

    config = AutoConfig.for_model(raw.pop("model_type"), **raw)
    return config


def build_model(record, device):
    """Build the real native class with its own initialization and Torch backends."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING
    from nemo_automodel.components.models.common import BackendConfig

    architecture = record["architectures"][0]
    if architecture not in MODEL_ARCH_MAPPING:
        import pytest

        pytest.skip(f"This AutoModel version has no native {architecture} implementation")
    config = tiny_config(record)
    module, name, *_ = MODEL_ARCH_MAPPING[architecture]
    cls = getattr(importlib.import_module(module), name)
    backend = BackendConfig(
        attn=(
            "flex"
            if architecture
            in ("Qwen4ExpForConditionalGeneration", "Qwen3_8_FlashNextForConditionalGeneration")
            else "eager"
            if architecture.startswith("DeepseekV4")
            else "sdpa"
        ),
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    kwargs = {"backend": backend}
    if architecture == "DeepseekV41ForCausalLM":
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        vocab = {"[UNK]": 0, **{f"t{i}": i for i in range(1, 128)}}
        kwargs["tokenizer"] = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")),
            unk_token="[UNK]",
        )
    if architecture in (
        "Qwen4ExpForConditionalGeneration",
        "Qwen3_8_FlashNextForConditionalGeneration",
    ):
        from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
            Qwen3_8_FlashNextEngramTableConfig,
        )

        config.language_model_only = True
        kwargs["engram_table_config"] = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=1024,
            embedding_dim=config.text_config.ple_embed_dim
            // ((config.text_config.ngram_size - 1) * config.text_config.heads_per_ngram),
        )
    constructor = getattr(cls, "from_config", cls)
    model = constructor(config, **kwargs).to(device)
    initialize = getattr(model, "initialize_weights", None) or model.init_weights
    signature = inspect.signature(initialize).parameters
    initialize(
        **{
            key: value
            for key, value in {"buffer_device": device, "dtype": torch.bfloat16}.items()
            if key in signature
        }
    )
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype

    cast_model_to_dtype(model, torch.bfloat16)
    if architecture in (
        "Qwen4ExpForConditionalGeneration",
        "Qwen3_8_FlashNextForConditionalGeneration",
    ):
        from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
            Qwen3_8_FlashNextNGramEmbedding,
        )

        for module in model.modules():
            if isinstance(module, Qwen3_8_FlashNextNGramEmbedding):
                # Shrink the real hash geometry together with its table; retain
                # the native hash function, document resets and multipliers.
                sizes = module.ngram_heads_vocab_sizes.new_tensor(
                    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
                )
                assert sizes.numel() == module.ngram_heads
                module.ngram_heads_vocab_sizes.copy_(sizes)
                module.ngram_heads_offsets.copy_(
                    torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
                )
    return model.eval()
