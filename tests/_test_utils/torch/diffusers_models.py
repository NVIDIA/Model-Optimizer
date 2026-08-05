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

import inspect
from pathlib import Path

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import UNet2DConditionModel

try:
    from diffusers.models.transformers import DiTTransformer2DModel, FluxTransformer2DModel
except Exception:  # pragma: no cover - optional diffusers models
    DiTTransformer2DModel = None
    FluxTransformer2DModel = None

try:
    from diffusers.models.transformers import Flux2Transformer2DModel
except Exception:  # pragma: no cover - optional diffusers models
    Flux2Transformer2DModel = None

try:
    from diffusers.models.transformers import WanTransformer3DModel
except Exception:  # pragma: no cover - optional diffusers models
    WanTransformer3DModel = None

try:
    from diffusers.models.autoencoders import AutoencoderKLWan
except Exception:  # pragma: no cover - optional diffusers models
    AutoencoderKLWan = None

try:
    from diffusers.models.transformers import QwenImageTransformer2DModel
except Exception:  # pragma: no cover - optional diffusers models
    QwenImageTransformer2DModel = None

try:
    from diffusers.models.autoencoders import AutoencoderKLQwenImage
except Exception:  # pragma: no cover - optional diffusers models
    AutoencoderKLQwenImage = None

import modelopt.torch.opt as mto


def get_tiny_unet(**config_kwargs) -> UNet2DConditionModel:
    """Create a tiny UNet2DConditionModel for testing."""
    kwargs = {
        "sample_size": 8,
        "in_channels": 2,
        "out_channels": 2,
        "down_block_types": ("DownBlock2D",),
        "up_block_types": ("UpBlock2D",),
        "block_out_channels": (2,),
        "layers_per_block": 1,
        "cross_attention_dim": 2,
        "attention_head_dim": 1,
        "norm_num_groups": 1,
        "mid_block_type": None,
    }
    kwargs.update(**config_kwargs)
    tiny_unet = UNet2DConditionModel(**kwargs)

    return tiny_unet


def get_tiny_dit(**config_kwargs):
    """Create a tiny DiTTransformer2DModel for testing."""
    if DiTTransformer2DModel is None:
        pytest.skip("DiTTransformer2DModel is not available in this diffusers version.")

    kwargs = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 2,
        "out_channels": 2,
        "num_layers": 1,
        "norm_num_groups": 1,
        "sample_size": 8,
        "patch_size": 2,
        "num_embeds_ada_norm": 10,
    }
    kwargs.update(**config_kwargs)
    return DiTTransformer2DModel(**kwargs)


def get_tiny_flux(**config_kwargs):
    """Create a tiny FluxTransformer2DModel for testing."""
    if FluxTransformer2DModel is None:
        pytest.skip("FluxTransformer2DModel is not available in this diffusers version.")

    kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "out_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 8,
        "pooled_projection_dim": 8,
        "guidance_embeds": False,
        "axes_dims_rope": (2, 2, 4),
    }
    kwargs.update(**config_kwargs)
    return FluxTransformer2DModel(**kwargs)


def get_tiny_flux2(**config_kwargs):
    """Create a tiny Flux2Transformer2DModel for testing."""
    if Flux2Transformer2DModel is None:
        pytest.skip("Flux2Transformer2DModel is not available in this diffusers version.")

    kwargs = {
        "patch_size": 1,
        "in_channels": 16,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "timestep_guidance_channels": 16,
        "mlp_ratio": 3.0,
        "axes_dims_rope": (4, 4, 4, 4),
    }
    kwargs.update(**config_kwargs)
    return Flux2Transformer2DModel(**kwargs)


def create_tiny_unet_dir(tmp_path: Path, **config_kwargs) -> Path:
    """Create and save a tiny UNet model to a directory."""
    tiny_unet = get_tiny_unet(**config_kwargs)
    tiny_unet.save_pretrained(tmp_path / "tiny_unet")
    return tmp_path / "tiny_unet"


def get_unet_dummy_inputs(model: UNet2DConditionModel, batch_size: int = 1):
    """Generate dummy inputs for testing UNet models."""
    latents = torch.randn(
        batch_size, model.config.in_channels, model.config.sample_size, model.config.sample_size
    )
    timestep = torch.tensor([0])
    encoder_hidden_states = torch.randn(batch_size, 1, model.config.cross_attention_dim)

    return {"sample": latents, "timestep": timestep, "encoder_hidden_states": encoder_hidden_states}


def df_output_tester(model_ref, model_test):
    """Test if two diffusers models produce the same output."""
    inputs = get_unet_dummy_inputs(model_ref)
    model_ref.eval()
    model_test.eval()

    with torch.no_grad():
        output_ref = model_ref(**inputs).sample
        output_test = model_test(**inputs).sample

    assert torch.allclose(output_ref, output_test)


def df_modelopt_state_and_output_tester(model_ref, model_test):
    """Test if two diffusers models have the same modelopt state and outputs."""
    model_ref_state = mto.modelopt_state(model_ref)
    model_test_state = mto.modelopt_state(model_test)
    assert model_ref_state == model_test_state

    df_output_tester(model_ref, model_test)


def get_tiny_wan22_transformer(**config_kwargs):
    """Create a tiny WanTransformer3DModel for testing."""
    if WanTransformer3DModel is None:
        pytest.skip("WanTransformer3DModel is not available in this diffusers version.")

    kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 32,
        "freq_dim": 256,
        "ffn_dim": 32,
        "num_layers": 2,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 32,
    }
    kwargs.update(**config_kwargs)
    return WanTransformer3DModel(**kwargs)


def get_tiny_wan22_vae(**config_kwargs):
    """Create a tiny AutoencoderKLWan for testing."""
    if AutoencoderKLWan is None:
        pytest.skip("AutoencoderKLWan is not available in this diffusers version.")

    kwargs = {
        "base_dim": 3,
        "z_dim": 16,
        "dim_mult": [1, 1, 1, 1],
        "num_res_blocks": 1,
        "temperal_downsample": [False, True, True],
    }
    kwargs.update(**config_kwargs)
    return AutoencoderKLWan(**kwargs)


def create_tiny_wan22_pipeline_dir(tmp_path: Path) -> Path:
    """Create and save a tiny Wan 2.2 (14B-style) pipeline to a directory.

    Uses the same tiny config as diffusers' own Wan 2.2 tests:
    - Transformer: 2 heads, 12 head_dim, 2 layers (hidden_dim=24)
    - VAE: base_dim=3, z_dim=16
    - Text encoder: hf-internal-testing/tiny-random-t5 (hidden_size=32)
    - Dual transformer (14B style) with boundary_ratio=0.875

    The saved directory can be loaded with ``WanPipeline.from_pretrained(path)``.
    """
    from diffusers import UniPCMultistepScheduler, WanPipeline
    from transformers import AutoTokenizer, T5EncoderModel

    torch.manual_seed(0)
    vae = get_tiny_wan22_vae()

    torch.manual_seed(0)
    transformer = get_tiny_wan22_transformer()

    torch.manual_seed(0)
    transformer_2 = get_tiny_wan22_transformer()

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
    )
    text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

    pipe = WanPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        boundary_ratio=0.875,
    )

    save_dir = tmp_path / "tiny_wan22"
    pipe.save_pretrained(save_dir)
    return save_dir


def get_tiny_qwen_image_transformer(**config_kwargs):
    """Create a tiny QwenImageTransformer2DModel for testing.

    Scaled down from the real Qwen-Image config (60 layers, 24 heads, head_dim 128,
    joint_attention_dim 3584). Two constraints to keep in mind:
    - ``axes_dims_rope`` must sum to ``attention_head_dim``.
    - ``joint_attention_dim`` must match the text-embedding dim the model is fed. In
      the DMD2 mock-data training path that is the *dataloader's* ``text_embed_dim``
      (the bundled text encoder is bypassed), so pair this with
      ``--data.dataloader.text_embed_dim=<joint_attention_dim>``.
    """
    if QwenImageTransformer2DModel is None:
        pytest.skip("QwenImageTransformer2DModel is not available in this diffusers version.")

    kwargs = {
        "patch_size": 2,
        "in_channels": 64,  # vae z_dim (16) * 2x2 patch
        "out_channels": 16,  # = vae z_dim
        "num_layers": 2,
        "attention_head_dim": 16,
        "num_attention_heads": 2,  # hidden_dim = 32
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "guidance_embeds": False,
        "axes_dims_rope": (8, 4, 4),  # sums to attention_head_dim (16)
    }
    kwargs.update(**config_kwargs)
    # Drop kwargs the installed diffusers QwenImageTransformer2DModel doesn't accept.
    # `pooled_projection_dim` is present in the published config.json but was removed
    # from the constructor in newer diffusers: from_pretrained tolerates the extra
    # config key, but a direct constructor call raises TypeError.
    accepted = set(inspect.signature(QwenImageTransformer2DModel.__init__).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return QwenImageTransformer2DModel(**kwargs)


def get_tiny_qwen_image_vae(**config_kwargs):
    """Create a tiny AutoencoderKLQwenImage for testing (z_dim=16 to match the transformer)."""
    if AutoencoderKLQwenImage is None:
        pytest.skip("AutoencoderKLQwenImage is not available in this diffusers version.")

    kwargs = {
        "base_dim": 8,
        "z_dim": 16,  # = transformer out_channels
        "dim_mult": [1, 2],
        "num_res_blocks": 1,
        "temperal_downsample": [True],  # len == len(dim_mult) - 1
        "attn_scales": [],
        "latents_mean": [0.0] * 16,  # length must == z_dim
        "latents_std": [1.0] * 16,
    }
    kwargs.update(**config_kwargs)
    return AutoencoderKLQwenImage(**kwargs)


def _byte_level_unicode_chars() -> list[str]:
    """The 256 GPT-2/Qwen byte-level BPE unicode characters, in byte order.

    Inlined copy of the standard GPT-2 ``bytes_to_unicode`` mapping (transformers
    v5 removed it from ``transformers.models.gpt2.tokenization_gpt2``).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return [chr(c) for c in cs]


def _build_local_qwen2_tokenizer(out_dir: Path):
    """Build a tiny, fully offline byte-level Qwen2 tokenizer (no Hub access).

    Uses the GPT-2/Qwen byte->unicode mapping for the 256 single-byte tokens plus
    Qwen's core special tokens, with an empty merge table (pure byte-level
    fallback). This is enough to tokenize calibration prompts so the pipeline runs
    end-to-end; it is not meant for high-quality text.
    """
    import json

    import transformers

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {token: idx for idx, token in enumerate(_byte_level_unicode_chars())}
    for special in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        vocab.setdefault(special, len(vocab))
    (out_dir / "vocab.json").write_text(json.dumps(vocab))
    (out_dir / "merges.txt").write_text("#version: 0.2\n")

    special_kwargs = {
        "unk_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    tokenizer_params = inspect.signature(transformers.Qwen2Tokenizer.__init__).parameters
    if "vocab_file" in tokenizer_params:
        # transformers v4: slow tokenizer reads vocab/merges from files.
        return transformers.Qwen2Tokenizer(
            vocab_file=str(out_dir / "vocab.json"),
            merges_file=str(out_dir / "merges.txt"),
            **special_kwargs,
        )
    # transformers v5: tokenizers-backed class takes the vocab dict and merge
    # list directly (``vocab_file=`` would be silently swallowed by **kwargs).
    return transformers.Qwen2Tokenizer(vocab=vocab, merges=[], **special_kwargs)


def create_tiny_qwen_image_pipeline_dir(tmp_path: Path) -> Path:
    """Create and save a tiny, fully offline Qwen-Image pipeline to a directory.

    Mirrors diffusers' ``QwenImagePipelineFastTests.get_dummy_components`` but with
    no Hub access: the Qwen2.5-VL text encoder is built inline from a tiny
    ``Qwen2_5_VLConfig``, and the tokenizer is built locally by
    ``_build_local_qwen2_tokenizer`` (byte-level vocab written to a temp dir). The
    transformer uses ``num_layers=6`` so the first-2/last-2 block-range recipe is
    valid, and its ``joint_attention_dim`` matches the text encoder ``hidden_size``
    (16) so the pipeline runs end-to-end during quantization calibration. The saved
    dir loads with ``QwenImagePipeline.from_pretrained(path)``.
    """
    if QwenImageTransformer2DModel is None or AutoencoderKLQwenImage is None:
        pytest.skip("QwenImage diffusers classes not available in this diffusers version.")
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

    transformers = pytest.importorskip("transformers")

    # Tiny Qwen2.5-VL text encoder, built offline from a tiny config (no Hub model
    # load), mirroring diffusers' QwenImagePipelineFastTests.get_dummy_components.
    qwen_vl_config = transformers.Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {
                "mrope_section": [1, 1, 2],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000.0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 16,
        },
        hidden_size=16,
        vocab_size=152064,
        vision_end_token_id=151653,
        vision_start_token_id=151652,
        vision_token_id=151654,
    )
    text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(qwen_vl_config).eval()

    # Deterministic local byte-level Qwen2 tokenizer (built offline; no Hub, no skip).
    tokenizer = _build_local_qwen2_tokenizer(tmp_path / "qwen_tokenizer")

    torch.manual_seed(0)
    # num_layers=6 so the first-2/last-2 block-range recipe (which needs >=6 blocks)
    # is valid; joint_attention_dim must match the text encoder hidden_size (16).
    transformer = get_tiny_qwen_image_transformer(
        num_layers=6,
        in_channels=16,
        out_channels=4,
        joint_attention_dim=16,
        num_attention_heads=3,
    )
    torch.manual_seed(0)
    vae = get_tiny_qwen_image_vae(z_dim=4, latents_mean=[0.0] * 4, latents_std=[1.0] * 4)

    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=8192,
        max_shift=0.9,
        num_train_timesteps=1000,
        shift=1.0,
        shift_terminal=0.02,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
    )

    pipe = QwenImagePipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    save_dir = tmp_path / "tiny_qwen_image"
    pipe.save_pretrained(save_dir)
    return save_dir
