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

"""Offline random HF prefixes without pretrained weights."""

import copy
import importlib
import inspect
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig
from transformers.core_model_loading import build_glob_alternation

# Canonical HF model type and real task class.
_FAMILIES = {
    "llama": "LlamaForCausalLM",
    "bailing_hybrid": "BailingHybridForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "qwen3_next": "Qwen3NextForCausalLM",
    "qwen3_vl_moe": "Qwen3VLMoeForConditionalGeneration",
    "qwen3_omni_moe": "Qwen3OmniMoeThinkerForConditionalGeneration",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    "qwen4_exp": "Qwen4ExpForConditionalGeneration",
    "deepseek_v3": "DeepseekV3ForCausalLM",
    "deepseek_v32": "DeepseekV32ForCausalLM",
    "deepseek_v4": "DeepseekV4ForCausalLM",
    "glm_moe_dsa": "GlmMoeDsaForCausalLM",
    "glm5_next": "Glm5NextForConditionalGeneration",
    "kimi_k25": "Kimi_K25ForConditionalGeneration",
    "kimi_linear": "KimiLinearForCausalLM",
    "minimax_m3_vl": "MiniMaxM3SparseForConditionalGeneration",
    "mimo_v2_flash": "MiMoV2FlashForCausalLM",
    "nemotron_h": "NemotronHForCausalLM",
    "nemotron_h_omni": "NemotronH_Omni_Reasoning_V3",
    "muse_spark": "MuseSparkForConditionalGeneration",
    "mistral_large4": "MistralLarge4ForConditionalGeneration",
}


def cases(families=None):
    """Use pinned release prefixes when available, official defaults otherwise."""
    records = json.loads(
        (Path(__file__).parent / "data/transformers_bypass_models.json").read_text()
    )
    result = []
    for family, task in _FAMILIES.items():
        if families is not None and family not in families:
            continue
        record = next((r for r in records if r["config"].get("model_type") == family), None)
        if family == "kimi_linear" or (
            record is not None and not record.get("minimum_pattern_prefix")
        ):
            record = None
        if record is not None:
            record = copy.deepcopy(record)
        else:
            raw = AutoConfig.for_model(family).to_dict()
            # Retain every official default pattern, including both dense and routed MLPs.
            depth = 4 if family in ("qwen3_next", "kimi_linear", "nemotron_h") else 2
            if family in ("deepseek_v3", "deepseek_v32", "kimi_k25"):
                depth = 4
            if family == "kimi_linear":
                depth = 5
            record = {"config": raw, "minimum_pattern_prefix": depth, "model_id": family}
        record.update(family=family, architectures=[task])
        if family == "qwen3_omni_moe":
            # Exercise the real thinker task; the optional speech generator is separate.
            record["config"] = AutoConfig.for_model("qwen3_omni_moe_thinker").to_dict()
        result.append(record)
    if families is None or "kimi_k3" in families:
        remote = copy.deepcopy(next(r for r in records if r["model_id"] == "moonshotai/Kimi-K3"))
        remote.update(family="kimi_k3", architectures=["KimiLinearForCausalLM"], remote_code=True)
        result.append(remote)
    return result


def tiny_config(record):
    """Shrink dimensions while retaining the exact distinct decoder prefix."""
    from transformers import AutoConfig, PretrainedConfig

    raw = copy.deepcopy(record["config"])
    raw.pop("auto_map", None)
    decode = getattr(PretrainedConfig, "_decode_special_floats", None)
    if decode is not None:
        raw = decode(raw)
    if "llm_config" in raw:
        raw["text_config"] = raw.pop("llm_config")
    text = raw.get("text_config", raw)
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
        "attn_res_block_size": 2,
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
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "num_expert_group": 1,
        "routed_expert_hidden_size": 256,
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
    if raw["model_type"] == "mimo_v2":
        # Half the release's 192/128 head widths retain an even 32-dimension partial RoPE.
        dimensions.update(head_dim=96, v_head_dim=64, swa_head_dim=96, swa_v_head_dim=64)
    if raw["model_type"] == "mimo_v2" and "swa_num_key_value_heads" in text:
        ratio = text["swa_num_key_value_heads"] // text["num_key_value_heads"]
        dimensions["swa_num_key_value_heads"] = dimensions["num_key_value_heads"] * ratio
    if text.get("qk_rope_head_dim") == 0:
        dimensions["qk_rope_head_dim"] = 0
    if "qk_head_dim" in text:
        dimensions["qk_head_dim"] = dimensions["qk_nope_head_dim"] + dimensions["qk_rope_head_dim"]
    if "rotary_dim" in text:
        dimensions["rotary_dim"] = 32
    if "linear_num_value_heads" in text:
        ratio = text["linear_num_value_heads"] // text["linear_num_key_heads"]
        dimensions["linear_num_value_heads"] = 2 * ratio
    if record.get("remote_code"):
        # Remote K3 forces FA2: retain the released q:k:v padding geometry (192:192:128).
        dimensions.update(qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32)
    if "q_lora_rank" in text and text["q_lora_rank"] is None:
        dimensions["q_lora_rank"] = None
    mla = text.get("model_type") in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa", "kimi_linear")
    if mla:
        dimensions["num_key_value_heads"] = dimensions["num_attention_heads"]
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
    if mla and not record.get("remote_code"):
        # These serialized derived fields otherwise overwrite __post_init__ geometry.
        text.pop("head_dim", None)
        text.pop("qk_head_dim", None)
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
    if raw["model_type"] == "nemotron_h_omni":
        raw.pop("sound_config", None)
        raw["audio_config"] = None
        raw["vision_config"] = {
            "model_type": "radio",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 2,
            "image_size": 8,
            "max_img_size": 16,
            "num_cls_tokens": 1,
            "num_registers": 0,
            "summary_idxs": [0],
            "video_temporal_patch_size": 2,
        }
        raw.update(force_image_size=8, vision_hidden_size=32, projector_hidden_size=64)
        raw.pop("vit_hidden_size", None)
    if raw["model_type"] == "qwen4_exp":
        positions = {old + 1: new + 1 for new, old in enumerate(selected_layers)}
        text["ple_layer_ids"] = [
            positions[index] for index in text.get("ple_layer_ids", []) if index in positions
        ]
        text.update(
            ngram_vocab_size_base=128, make_ngram_vocab_size_divisible_by=1, split_ngram_parts=1
        )
    linear = text.get("linear_attn_config")
    if isinstance(linear, dict):
        linear.update(num_heads=4, head_dim=16)
        for key in ("kda_layers", "full_attn_layers"):
            linear[key] = [i for i in linear.get(key, []) if i <= prefix]
    sparse = text.get("sparse_attention_config")
    if isinstance(sparse, dict):
        sparse.update(
            sparse_index_dim=64, sparse_num_index_heads=2, sparse_topk_blocks=2, sparse_block_size=4
        )
        for key, value in list(sparse.items()):
            if isinstance(value, list):
                sparse[key] = value[:prefix]
    audio = raw.get("audio_config")
    if isinstance(audio, dict) and raw["model_type"] == "qwen3_omni_moe_thinker":
        audio.update(
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            output_dim=256,
            downsample_hidden_size=32,
        )
    if record.get("family") == "deepseek_v41":
        text.update(
            num_key_value_heads=1,
            compress_ratios=[0, 2],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            candidate_source_layer_id=1,
            engram_layer_ids=[1],
            engram_num_embeddings=[18],
            engram_max_ngram_size=2,
            engram_vocab_size=7,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=128,
            engram_token_map=list(range(128)),
            num_nextn_predict_layers=0,
            dspark_target_layer_ids=[],
        )
    if record.get("family") == "step5":
        text.update(
            num_attention_groups=2,
            num_attention_groups_full_attention=2,
            num_attention_groups_sliding_attention=2,
            moe_num_experts=4,
            moe_top_k=2,
            moe_layer_list=[1],
            moe_layers_enum="1",
            share_expert_dim=256,
            partial_rotary_factors=[1.0, 0.5],
            num_nextn_predict_layers=0,
        )
        if isinstance(text.get("sparse_config"), dict):
            text["sparse_config"].update(
                proxy_dim=32,
                sparse_indexer_rope_dim=16,
                sparse_indexer_num_heads=4,
                num_provider_groups=2,
                topk=2,
                region_block_size=4,
            )
    if record.get("family") == "kimi_k31":
        text.update(
            first_k_dense_replace=1,
            num_experts=4,
            num_experts_per_token=2,
            num_shared_experts=1,
            routed_expert_hidden_size=256,
            attn_res_block_size=2,
        )
        text["linear_attn_config"].update(
            num_heads=4, head_dim=64, kda_layers=[1], full_attn_layers=[2]
        )
        text["dsa_config"].update(
            dsa_layers=[2], index_topk=4, index_num_heads=2, index_head_dim=32
        )
    text.update(use_cache=False)
    model_type = raw.pop("model_type")
    if record.get("remote_code"):
        config_class = _remote_class("configuration_kimi_k3.KimiLinearConfig", record)
        config = config_class(**raw)
        config._attn_implementation = "eager"
        return config
    if record.get("modelopt_config_class"):
        extension = importlib.import_module(record["modelopt_module"])
        config_class = getattr(extension, record["modelopt_config_class"])
        fields = dict(raw.get("text_config", raw))
        fields.pop("model_type", None)
        config = config_class(**fields)
    else:
        config = AutoConfig.for_model(model_type, **raw)
    config.get_text_config()._attn_implementation = "eager"
    config.get_text_config()._experts_implementation = "eager"
    return config


def build_model(record, device):
    """Instantiate the actual HF class directly, without network or model weights."""
    extension = (
        importlib.import_module(record["modelopt_module"])
        if record.get("modelopt_module")
        else None
    )
    config = tiny_config(record)
    if record.get("omni_wrapper"):
        config = transformers.Qwen3OmniMoeConfig(thinker_config=config, enable_audio_output=False)
    if record.get("remote_code"):
        cls = _remote_class("modeling_kimi_linear.KimiLinearForCausalLM", record)
        from modelopt.torch.models.kimi_k3 import _remote_fixes

        remote = sys.modules[cls.__module__]
        _remote_fixes._patch_kda_a_log(remote)
        _remote_fixes._patch_kda_beta_sigmoid(remote)
        _remote_fixes._shim_create_causal_mask(remote)
        _remote_fixes._patch_kimi_cache_get_mask_sizes(remote)
    else:
        cls = getattr(extension or transformers, record["architectures"][0])
    with torch.device(device):
        if record.get("native_precision"):
            # Checkpoint loading applies this plan; the random constructor does not.
            model = cls._from_config(config, dtype=torch.float32)
            patterns = list(model._get_dtype_plan(torch.bfloat16))
            policy, _, _ = build_glob_alternation(patterns or [r"(?!)"])
            for name, parameter in model.named_parameters():
                if parameter.is_floating_point() and not policy.search(name):
                    parameter.data = parameter.data.to(torch.bfloat16)
            for owner, module in model.named_modules():
                if isinstance(getattr(module, "config", None), transformers.PretrainedConfig):
                    module.config.dtype = torch.bfloat16
                for name, buffer in module.named_buffers(recurse=False):
                    full_name = ".".join(filter(None, (owner, name)))
                    if (
                        buffer.is_floating_point()
                        and name not in module._non_persistent_buffers_set
                        and not policy.search(full_name)
                    ):
                        module._buffers[name] = buffer.to(torch.bfloat16)
            return model.eval()
        model = cls(config)
    return model.to(dtype=torch.bfloat16).eval()


def _remote_class(name, record):
    # The official checkpoint's optional FLA code has no stock-HF substitute.
    import pytest
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from modelopt.torch.models.kimi_k3._remote_fixes import _install_output_recorder_shim

    path = (
        Path(os.environ["HF_HOME"])
        / "hub/models--moonshotai--Kimi-K3/snapshots"
        / record["revision"]
    )
    if not (path / "modeling_kimi_linear.py").exists():
        pytest.skip("Prefetch pinned Kimi-K3 source files; no weights are required")
    if name.startswith("modeling_"):
        pytest.importorskip("fla", reason="Official Kimi-K3 remote modeling requires fla-core")
    _install_output_recorder_shim()
    return get_class_from_dynamic_module(name, str(path), local_files_only=True)


@contextmanager
def hf_torch_kernels(model):
    """Select official HF Torch references for bounded portable-math tests only."""
    names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
        "mamba2_chunk_scan",
        "mamba2_selective_state_update",
        "mamba2_split_conv1d_scan_combined",
    )
    namespaces = {sys.modules[type(module).__module__] for module in model.modules()}
    replaced = []
    try:
        for namespace in namespaces:
            for name in names:
                function = getattr(namespace, name, None)
                if not callable(function):
                    continue
                original = inspect.unwrap(function)
                if original is not function and original.__module__ == namespace.__name__:
                    replaced.append((namespace, name, function))
                    setattr(namespace, name, original)
        yield
    finally:
        for namespace, name, function in reversed(replaced):
            setattr(namespace, name, function)


def release_cases():
    """Two representative text blocks from AA releases and Qwen 3.5+ families."""
    records = json.loads(
        (Path(__file__).parent / "data/transformers_bypass_models.json").read_text()
    )
    records += json.loads(
        (Path(__file__).parent / "data/transformers_extra_bypass_models.json").read_text()
    )
    requested = {
        "deepseek-ai/DeepSeek-V4.1-Flash-1017": ("deepseek_v41", "DeepseekV41ForCausalLM", (0, 1)),
        "stepfun-ai/Step-5-Preview": ("step5", "Step5ForCausalLM", (0, 3)),
        "moonshotai/Kimi-K3.1-Pro": ("kimi_k31", "KimiK31ForCausalLM", (0, 3)),
        "zai-org/GLM-5.3": ("glm_moe_dsa", "GlmMoeDsaForCausalLM", (0, 3)),
        "zai-org/GLM-5.3-Flash": ("glm5_next", "Glm5NextForConditionalGeneration", (0, 3)),
        "inclusionAI/Ling-3.1-flash-1019": ("bailing_hybrid", "BailingHybridForCausalLM", (0, 7)),
        "MiniMaxAI/MiniMax-M3.1-Flash": (
            "minimax_m3_vl",
            "MiniMaxM3SparseForConditionalGeneration",
            (0, 3),
        ),
        "mistralai/Mistral-Large-4-1T-A52B": (
            "mistral_large4",
            "MistralLarge4ForConditionalGeneration",
            (0, 1),
        ),
        "meta-models/Muse-Spark-Open-816B-A42B": (
            "muse_spark",
            "MuseSparkForConditionalGeneration",
            (0, 1),
        ),
        "XiaomiMiMo/MiMo-V2.6-Pro-RL": ("mimo_v2", "MiMoV2ForCausalLM", (0, 1)),
        "XiaomiMiMo/MiMo-V2.6-Flash-RL": ("mimo_v2", "MiMoV2ForCausalLM", (0, 1)),
    }
    result, seen = [], set()
    for source in records:
        name = source["model_id"]
        if name in seen:
            continue
        if name not in requested and not name.startswith(
            ("Qwen/Qwen3.5-", "Qwen/Qwen3.6-", "Qwen/Qwen3.8-")
        ):
            continue
        record = copy.deepcopy(source)
        raw = record["config"]
        text = raw.get("text_config", raw)
        family, task, selected = (
            requested[name]
            if name in requested
            else (raw["model_type"], source["architectures"][0], (0, 3))
        )
        if family == "qwen4_exp":
            selected = (1, 3)
        record.update(
            family=family,
            architectures=[task],
            minimum_pattern_prefix=2,
            representative_layer_ids=list(selected),
            native_precision=True,
        )
        if "first_k_dense_replace" in text:
            text["first_k_dense_replace"] = 1
        if family in ("mimo_v2", "deepseek_v41", "step5", "kimi_k31"):
            record["modelopt_module"] = "modelopt.torch.models." + family
        if family in ("deepseek_v41", "step5", "kimi_k31"):
            record["modelopt_config_class"] = {
                "deepseek_v41": "DeepseekV41Config",
                "step5": "Step5Config",
                "kimi_k31": "KimiK31Config",
            }[family]
        result.append(record)
        seen.add(name)
    for name, (family, task, selected) in requested.items():
        if name not in seen and family == "muse_spark":
            result.append(
                {
                    "model_id": name,
                    "family": family,
                    "architectures": [task],
                    "config": AutoConfig.for_model(family).to_dict(),
                    "minimum_pattern_prefix": 2,
                    "representative_layer_ids": list(selected),
                    "native_precision": True,
                }
            )
    return result
