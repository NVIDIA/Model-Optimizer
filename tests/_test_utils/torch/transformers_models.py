# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
from pathlib import Path

import pytest
import torch
from _test_utils.torch.misc import set_seed

transformers = pytest.importorskip("transformers")
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForQuestionAnswering,
    AutoProcessor,
    AutoTokenizer,
    BertConfig,
    DeepseekV3Config,
    Gemma3Config,
    GptOssConfig,
    LlamaConfig,
    NemotronConfig,
    PreTrainedModel,
    Qwen3Config,
    Qwen3MoeConfig,
    T5Config,
    T5ForConditionalGeneration,
)

import modelopt.torch.opt as mto

SEED = 1234

TINY_TOKENIZER_PATH = Path(__file__).parent / "tokenizer"


def get_tiny_tokenizer(*, pad_side: str = "left") -> "transformers.PreTrainedTokenizerBase":
    """Returns a tiny tokenizer for tests.

    Default to left padding, which is what decoder-LM calibration/generation expects and what
    ``get_dataset_dataloader`` warns about otherwise. Callers needing right padding can override
    with ``pad_side="right"``.
    """
    tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER_PATH)
    tokenizer.padding_side = pad_side
    return tokenizer


def _pad_vocab_size(vocab_size: int, multiple: int = 128) -> int:
    """Round a vocab size up to a multiple."""
    return ((vocab_size + multiple - 1) // multiple) * multiple


def _create_tiny_llm_dir(
    dir_path: Path | str,
    build_model,
    *,
    with_tokenizer: bool = False,
    return_model: bool = False,
    tokenizer_factory=get_tiny_tokenizer,
    **config_kwargs,
) -> Path | tuple[Path, PreTrainedModel]:
    """Save a tiny model (and, if ``with_tokenizer``, a tokenizer sized to it) to ``dir_path``."""
    dir_path = Path(dir_path)
    if with_tokenizer:
        tokenizer = tokenizer_factory()
        tokenizer.save_pretrained(dir_path)
        config_kwargs["vocab_size"] = tokenizer.vocab_size
    model = build_model(**config_kwargs)
    model.save_pretrained(dir_path)
    return (dir_path, model) if return_model else dir_path


def _get_tiny_vlm_tokenizer(ref_tokenizer: "transformers.PreTrainedTokenizerBase"):
    """Tiny tokenizer re-registering ``ref_tokenizer``'s named vision tokens + chat template, so the
    saved processor and the model config (built from the same tokenizer) share small, matching ids."""
    # transformers' model-agnostic base special-token attributes; anything beyond these on a tokenizer
    # is a model-specific named token (e.g. a VLM's image_token / vision_start_token).
    _base_special_token_attrs = {
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    }

    named = {
        attr: str(getattr(ref_tokenizer, attr))
        for attr in ref_tokenizer.SPECIAL_TOKENS_ATTRIBUTES
        if attr not in _base_special_token_attrs and getattr(ref_tokenizer, attr, None) is not None
    }
    tokenizer = get_tiny_tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": list(named.values())})
    # Register the named tokens so ``<name>`` + ``<name>_id`` resolve and round-trip through save.
    tokenizer._set_model_specific_special_tokens(named)
    tokenizer.chat_template = ref_tokenizer.chat_template
    return tokenizer


def get_tiny_vlm_processor(
    ref_model_id: str, *, trust_remote_code: bool = False
) -> "transformers.ProcessorMixin":
    """Tiny-vocab VLM processor: the real ``ref_model_id`` image/video processor + chat template,
    paired with a tiny tokenizer that carries the ref's vision tokens (so the vocab stays small)."""
    real = AutoProcessor.from_pretrained(ref_model_id, trust_remote_code=trust_remote_code)
    tokenizer = _get_tiny_vlm_tokenizer(real.tokenizer)
    kwargs = {"image_processor": real.image_processor, "tokenizer": tokenizer}
    if getattr(real, "video_processor", None) is not None:
        kwargs["video_processor"] = real.video_processor
    return type(real)(**kwargs)


def _create_tiny_vlm_dir(
    dir_path: Path | str,
    ref_model_id: str,
    build_model,
    *,
    with_processor: bool,
    return_model: bool,
    **config_kwargs,
) -> Path | tuple[Path, PreTrainedModel]:
    """Save a tiny VLM (and, if ``with_processor``, its small-vocab processor) to ``dir_path``."""
    dir_path = Path(dir_path)
    if with_processor:
        get_tiny_vlm_processor(ref_model_id).save_pretrained(dir_path)
    model = build_model(**config_kwargs)
    model.save_pretrained(dir_path)
    return (dir_path, model) if return_model else dir_path


##### Qwen3 #####
def get_tiny_qwen3(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(config_kwargs)
    # NOTE: Use AutoModelForCausalLM.from_config() instead of Qwen3ForCausalLM() for correct dtype handling
    return AutoModelForCausalLM.from_config(Qwen3Config(**kwargs))


def create_tiny_qwen3_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, return_model: bool = False, **config_kwargs
) -> Path | tuple[Path, PreTrainedModel]:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_qwen3",
        get_tiny_qwen3,
        with_tokenizer=with_tokenizer,
        return_model=return_model,
        **config_kwargs,
    )


##### Qwen3 MoE #####
def get_tiny_qwen3_moe(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
    }
    kwargs.update(config_kwargs)
    return AutoModelForCausalLM.from_config(Qwen3MoeConfig(**kwargs))


def create_tiny_qwen3_moe_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path | tuple[Path, PreTrainedModel]:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_qwen3_moe",
        get_tiny_qwen3_moe,
        with_tokenizer=with_tokenizer,
        **config_kwargs,
    )


##### Qwen3-VL #####
def get_tiny_qwen3vl(**config_kwargs) -> PreTrainedModel:
    # Lazy imports — Qwen3VL requires transformers>=4.57
    from transformers import Qwen3VLConfig

    set_seed(SEED)

    # Defaults: hidden_size=num_attention_heads*head_dim (e.g. 4*8=32).
    # Pass config_kwargs to override for multi-GPU tests (e.g. num_attention_heads=num_gpus,
    # num_key_value_heads=num_gpus, hidden_size=num_gpus*head_dim).
    text_kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    text_kwargs.update(config_kwargs)
    # Pass as dicts — transformers 5.3.0 Qwen3VLConfig.__init__ only handles
    # vision_config/text_config when they are dicts or None, not instances.
    vision_kwargs = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 4,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": text_kwargs["hidden_size"],  # must match text hidden_size
        # Single deepstack injection (the bridge requires len <= num language-model layers, so keep
        # this small for tiny models; defaults to [8, 16, 24] which needs >= 3 layers).
        "deepstack_visual_indexes": [0],
    }
    cfg = Qwen3VLConfig(text_config=text_kwargs, vision_config=vision_kwargs, dtype=torch.bfloat16)
    return AutoModelForImageTextToText.from_config(cfg)


def create_tiny_qwen3vl_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, return_model: bool = False, **config_kwargs
) -> Path | tuple[Path, PreTrainedModel]:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_qwen3vl",
        get_tiny_qwen3vl,
        with_tokenizer=with_tokenizer,
        return_model=return_model,
        **config_kwargs,
    )


##### Gemma3-VL #####
# Real tiny Gemma3 reused (via get_tiny_vlm_processor) for the fixture's image processor + chat
# template; the vision geometry below matches it so the processor's pixel_values fit the tiny tower.
GEMMA3_VL_REF = "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration"


def get_tiny_gemma3vl(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    # Vocab + vision token ids derive from the ref tokenizer to match the saved processor.
    tokenizer = _get_tiny_vlm_tokenizer(AutoTokenizer.from_pretrained(GEMMA3_VL_REF))

    # layer_types auto-generates (sliding/full attention) so the pruning code path is exercised.
    text_kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 1024,  # >= 256 image tokens + text for image calibration
        "sliding_window": 16,
        "vocab_size": _pad_vocab_size(len(tokenizer)),
    }
    text_kwargs.update(config_kwargs)
    text_kwargs.setdefault("query_pre_attn_scalar", text_kwargs["head_dim"])
    # Tiny SigLIP vision tower; the multimodal projector maps it to the text hidden size.
    # Match Megatron-Bridge Gemma3VL (and real Gemma3 checkpoints): no SigLIP pooling head.
    vision_kwargs = {
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        # Real patch geometry so the processor's pixel_values feed the tiny vision tower.
        "image_size": 896,
        "patch_size": 14,
        "num_channels": 3,
        "vision_use_head": False,
    }
    cfg = Gemma3Config(
        text_config=text_kwargs,
        vision_config=vision_kwargs,
        mm_tokens_per_image=256,  # 896 / 14 -> 4096 patches pooled to 256 image tokens
        # Token ids (from the tiny tokenizer) must match the processor for vision-token merging.
        image_token_index=tokenizer.image_token_id,
        boi_token_index=tokenizer.boi_token_id,
        eoi_token_index=tokenizer.eoi_token_id,
        dtype=torch.bfloat16,
    )
    return AutoModelForImageTextToText.from_config(cfg)


def create_tiny_gemma3vl_dir(
    tmp_path: Path | str, with_processor: bool = False, return_model: bool = False, **config_kwargs
) -> Path | tuple[Path, PreTrainedModel]:
    return _create_tiny_vlm_dir(
        Path(tmp_path) / "tiny_gemma3vl",
        GEMMA3_VL_REF,
        get_tiny_gemma3vl,
        with_processor=with_processor,
        return_model=return_model,
        **config_kwargs,
    )


##### Qwen3.5-VL (dense, hybrid GatedDeltaNet + gated attention) #####
# Real Qwen3.5-VL reused (via get_tiny_vlm_processor) for the fixture's image processor + chat
# template; the vision patch params below match it so the processor's pixel_values fit the tiny tower.
QWEN3_5_VL_REF = "Qwen/Qwen3.5-0.8B"


def get_tiny_qwen3_5_vl(**config_kwargs) -> PreTrainedModel:
    from transformers import Qwen3_5Config

    set_seed(SEED)

    # Vocab + vision token ids derive from the ref tokenizer to match the saved processor.
    tokenizer = _get_tiny_vlm_tokenizer(AutoTokenizer.from_pretrained(QWEN3_5_VL_REF))

    # Hybrid GatedDeltaNet (linear attention) + gated full-attention (layer_types auto-generated).
    text_kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 32,
        # Pad to a multiple of 128 (Megatron pads the vocab to make_vocab_size_divisible_by=128;
        # matching here avoids an embedding-size mismatch on import). Token ids stay < len(tokenizer).
        "vocab_size": _pad_vocab_size(len(tokenizer)),
        # GatedDeltaNet linear-attention dims (kept small for tiny models).
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
    }
    text_kwargs.update(config_kwargs)
    vision_kwargs = {
        "hidden_size": 16,
        "intermediate_size": 16,
        "depth": 2,
        "num_heads": 2,
        # Real patch params so the processor's pixel_values fit the tiny tower; only hidden dims shrink.
        "patch_size": 16,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "spatial_merge_size": 2,
        "out_hidden_size": text_kwargs["hidden_size"],  # must match text hidden_size
        "deepstack_visual_indexes": [],  # no deepstack injection (unlike Qwen3-VL)
    }
    cfg = Qwen3_5Config(
        text_config=text_kwargs,
        vision_config=vision_kwargs,
        dtype=torch.bfloat16,
        # Token ids (from the tiny tokenizer) must match the processor for vision-token merging.
        image_token_id=tokenizer.image_token_id,
        video_token_id=tokenizer.video_token_id,
        vision_start_token_id=tokenizer.vision_bos_token_id,
        vision_end_token_id=tokenizer.vision_eos_token_id,
    )
    return AutoModelForImageTextToText.from_config(cfg)


def create_tiny_qwen3_5_vl_dir(
    tmp_path: Path | str, with_processor: bool = False, return_model: bool = False, **config_kwargs
) -> Path | tuple[Path, PreTrainedModel]:
    return _create_tiny_vlm_dir(
        Path(tmp_path) / "tiny_qwen3_5_vl",
        QWEN3_5_VL_REF,
        get_tiny_qwen3_5_vl,
        with_processor=with_processor,
        return_model=return_model,
        **config_kwargs,
    )


##### NEMOTRON #####
def get_tiny_nemotron(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    # hidden_size=64, ffn_hidden_size=128: relu2 activation needs non-trivial dims
    # to avoid all-zero activations (scaling factor 0) in NVFP4 quantization.
    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(config_kwargs)
    return AutoModelForCausalLM.from_config(NemotronConfig(**kwargs))


def create_tiny_nemotron_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_nemotron",
        get_tiny_nemotron,
        with_tokenizer=with_tokenizer,
        **config_kwargs,
    )


##### NEMOTRON-H (Mamba + Attention + MoE/MLP hybrid) #####
def get_tiny_nemotron_h(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    # Lazy import — NemotronHConfig only exists in newer transformers builds, and this
    # module is imported broadly (including by the min-transformers CI job).
    from transformers import NemotronHConfig

    # Tiny NemotronH hybrid. hybrid_override_pattern letters: M=Mamba, E=MoE/FFN, *=Attention.
    # "ME*E" matches the NemotronH default and exercises Mamba + MoE + attention layers.
    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "ME*E",
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "mamba_num_heads": 8,
        "mamba_head_dim": 16,
        "ssm_state_size": 32,
        "n_groups": 2,
        "conv_kernel": 4,
        # MoE
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "n_shared_experts": 1,
        "moe_shared_expert_intermediate_size": 32,
        "vocab_size": 32,
        "max_position_embeddings": 32,
    }
    kwargs.update(config_kwargs)
    return AutoModelForCausalLM.from_config(NemotronHConfig(**kwargs))


def create_tiny_nemotron_h_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_nemotron_h",
        get_tiny_nemotron_h,
        with_tokenizer=with_tokenizer,
        **config_kwargs,
    )


##### DeepSeek V3 #####
def get_tiny_deepseek_v3(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)
    kwargs = {
        "dtype": torch.bfloat16,
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 0,
        "kv_lora_rank": 16,
        "q_lora_rank": 32,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "max_position_embeddings": 128,
        # Required so vLLM allocates ``gate.e_score_correction_bias`` (HF saves it unconditionally).
        "topk_method": "noaux_tc",
    }
    kwargs.update(config_kwargs)
    cfg = DeepseekV3Config(**kwargs)
    # Survive transformers versions that drop unknown kwargs from the dataclass.
    cfg.topk_method = kwargs["topk_method"]
    return AutoModelForCausalLM.from_config(cfg)


def create_tiny_deepseek_v3_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_deepseek_v3",
        get_tiny_deepseek_v3,
        with_tokenizer=with_tokenizer,
        **config_kwargs,
    )


##### GPT-OSS #####
def get_tiny_gpt_oss(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
        "num_hidden_layers": 4,
        "num_local_experts": 8,
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 32,
        "head_dim": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }
    kwargs.update(config_kwargs)
    return AutoModelForCausalLM.from_config(GptOssConfig(**kwargs))


def create_tiny_gpt_oss_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_gpt_oss",
        get_tiny_gpt_oss,
        with_tokenizer=with_tokenizer,
        **config_kwargs,
    )


##### LLAMA #####
def get_tiny_llama(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)
    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(config_kwargs)
    return AutoModelForCausalLM.from_config(LlamaConfig(**kwargs))


def create_tiny_llama_dir(
    tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs
) -> Path:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_llama",
        get_tiny_llama,
        with_tokenizer=with_tokenizer,
        **config_kwargs,
    )


##### T5 #####
def get_tiny_t5(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)
    kwargs = {
        "dtype": torch.bfloat16,
        "vocab_size": 32,
        "d_model": 32,
        "d_kv": 32,
        "d_ff": 32,
        "num_layers": 2,
        "num_heads": 16,
        "relative_attention_num_buckets": 8,
        "relative_attention_max_distance": 32,
        "decoder_start_token_id": 0,
    }
    kwargs.update(config_kwargs)
    return T5ForConditionalGeneration(T5Config(**kwargs)).to(torch.bfloat16)


def create_tiny_t5_dir(tmp_path: Path | str, with_tokenizer: bool = False, **config_kwargs) -> Path:
    return _create_tiny_llm_dir(
        Path(tmp_path) / "tiny_t5",
        get_tiny_t5,
        with_tokenizer=with_tokenizer,
        tokenizer_factory=lambda: AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-T5Model"
        ),
        **config_kwargs,
    )


##### BERT #####
def get_tiny_bert(**config_kwargs) -> PreTrainedModel:
    set_seed(SEED)

    kwargs = {
        "dtype": torch.bfloat16,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "max_position_embeddings": 32,
        "vocab_size": 32,
    }
    kwargs.update(config_kwargs)
    return AutoModelForQuestionAnswering.from_config(BertConfig(**kwargs))


def create_tiny_bert_dir(tmp_path: Path | str, **config_kwargs) -> Path:
    return _create_tiny_llm_dir(Path(tmp_path) / "tiny_bert", get_tiny_bert, **config_kwargs)


##### TESTERS #####
def tf_output_tester(model_ref, model_test):
    inputs = model_ref.dummy_inputs
    model_ref.eval()
    model_test.to(model_ref.dtype).eval()
    output_ref = model_ref(**inputs)
    output_test = model_test(**inputs)
    atol = 1e-2 if model_ref.dtype == torch.bfloat16 else 1e-6
    if hasattr(output_ref, "logits"):
        assert torch.allclose(output_ref.logits, output_test.logits, atol=atol)
    else:
        assert torch.allclose(output_ref.start_logits, output_test.start_logits, atol=atol)
        assert torch.allclose(output_ref.end_logits, output_test.end_logits, atol=atol)


def tf_modelopt_state_and_output_tester(model_ref, model_test):
    # Huggingface adds a _is_hf_initialized attribute to the model's modules
    for module in model_test.modules():
        if hasattr(module, "_is_hf_initialized"):
            # AttributeError for PEFT models, PEFT models get `_is_hf_initialized` from model.base_model
            with contextlib.suppress(AttributeError):
                delattr(module, "_is_hf_initialized")

    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)
    assert model_ref_state == model_test_state

    tf_output_tester(model_ref, model_test)
