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

"""Native DSV4 checkpoint loading and real decoder training on small CPU tensors."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

from modelopt.torch.export import export_speculative_decoding
from modelopt.torch.speculative.mtp import NativeMTPCheckpointError, create_native_mtp_boost_model
from modelopt.torch.speculative.plugins.modeling_dsv4_mtp import DeepSeekV4MTPAdapter
from modelopt.torch.utils.plugins.hf_checkpoint_utils import indexed_weight_map


def _write_native(path, *, tied=False, sharded=True, scale_dtype=torch.float8_e8m0fnu):
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "num_nextn_predict_layers": 1,
        "compress_ratios": [128, 0],
        "num_hash_layers": 0,
        "hidden_size": 128,
        "vocab_size": 16,
        "moe_intermediate_size": 64,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "qk_rope_head_dim": 16,
        "q_lora_rank": 32,
        "o_groups": 2,
        "o_lora_rank": 16,
        "num_experts_per_tok": 1,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "max_position_embeddings": 1024,
        "sliding_window": 4,
        "tie_word_embeddings": tied,
        "quantization_config": {
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    shapes = {
        "e_proj.weight": (128, 128),
        "h_proj.weight": (128, 128),
        "enorm.weight": (128,),
        "hnorm.weight": (128,),
        "norm.weight": (128,),
        "attn.attn_sink": (4,),
        "attn.q_norm.weight": (32,),
        "attn.kv_norm.weight": (32,),
        "attn.wq_a.weight": (32, 128),
        "attn.wq_b.weight": (128, 32),
        "attn.wkv.weight": (32, 128),
        "attn.wo_a.weight": (32, 64),
        "attn.wo_b.weight": (128, 32),
        "attn_norm.weight": (128,),
        "ffn_norm.weight": (128,),
        "ffn.gate.weight": (2, 128),
        "ffn.gate.bias": (2,),
        "hc_attn_fn": (8, 256),
        "hc_attn_base": (8,),
        "hc_attn_scale": (3,),
        "hc_ffn_fn": (8, 256),
        "hc_ffn_base": (8,),
        "hc_ffn_scale": (3,),
        "hc_head_fn": (2, 256),
        "hc_head_base": (2,),
        "hc_head_scale": (1,),
    }
    for expert in ["shared_experts", "experts.0", "experts.1"]:
        for proj, shape in [("w1", (64, 128)), ("w2", (128, 64)), ("w3", (64, 128))]:
            shapes[f"ffn.{expert}.{proj}.weight"] = shape
    gen = torch.Generator().manual_seed(2026)
    native, expected = {}, {}
    for suffix, shape in shapes.items():
        key = "mtp.0." + suffix
        if suffix.startswith("ffn.experts."):
            packed = torch.randint(
                0, 256, (shape[0], shape[1] // 2), generator=gen, dtype=torch.uint8
            )
            codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
            magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
            value = (
                magnitudes[(codes & 7).long()] * (1 - 2 * (codes >> 3).float()) / 128
            ).bfloat16()
            native[key] = packed.view(torch.int8)
            native[key.removesuffix(".weight") + ".scale"] = torch.full(
                (shape[0], shape[1] // 32), 120, dtype=torch.uint8
            ).view(scale_dtype)
        elif len(shape) == 2 and not suffix.startswith("hc_") and suffix != "ffn.gate.weight":
            codes = torch.randint(-16, 17, shape, generator=gen).float().to(torch.float8_e4m3fn)
            value = (codes.float() / 128).bfloat16()
            native[key] = codes
            native[key.removesuffix(".weight") + ".scale"] = torch.full(
                ((shape[0] + 127) // 128, (shape[1] + 127) // 128), 120, dtype=torch.uint8
            ).view(scale_dtype)
        else:
            value = torch.randn(shape, generator=gen) * 0.01
            if "norm.weight" in suffix or suffix.endswith("_scale"):
                value = torch.ones(shape)
            if suffix == "ffn.gate.bias":
                value = torch.tensor([0.1000, 0.1001])
            native[key] = (
                value
                if suffix.startswith("hc_") or suffix in ("attn.attn_sink", "ffn.gate.bias")
                else value.bfloat16()
            )
            value = native[key]
        expected[key] = value
    endpoints = {
        key: torch.randn(shape, generator=gen).bfloat16()
        for key, shape in [("embed.weight", (16, 128)), ("norm.weight", (128,))]
    }
    if not tied:
        endpoints["head.weight"] = torch.randn(16, 128, generator=gen).bfloat16()
    expected.update(endpoints)
    if sharded:
        save_file(native, path / "mtp.safetensors")
        save_file(endpoints, path / "target.safetensors")
        weight_map = {
            **dict.fromkeys(native, "mtp.safetensors"),
            **dict.fromkeys(endpoints, "target.safetensors"),
        }
        # Selective loading must never try to open a backbone shard.
        weight_map["layers.0.attn.wq_a.weight"] = "unavailable-backbone.safetensors"
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file({**native, **endpoints}, path / "model.safetensors")
    return expected


@pytest.mark.parametrize(("tied", "sharded"), [(False, True), (True, False)])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_native_load_maps_weights_and_freezes_target(tmp_path, tied, sharded, scale_dtype):
    expected = _write_native(tmp_path, tied=tied, sharded=sharded, scale_dtype=scale_dtype)
    model = create_native_mtp_boost_model(tmp_path)
    mtp = model.eagle_module
    assert model.mtp_config.mtp_adapter == "deepseek_v4"
    assert mtp.state_shape == (2, 128)
    assert not list(model.model.parameters())
    assert torch.equal(model.embed_tokens.weight, expected["embed.weight"])
    assert torch.equal(model.lm_head.weight, expected["embed.weight" if tied else "head.weight"])
    assert torch.equal(model.norm.weight, expected["norm.weight"])
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    router = mtp.decoder.mlp.gate
    _, _, experts = router(torch.zeros(1, 128, dtype=torch.bfloat16))
    assert experts.item() == 1
    for source, target in [
        ("e_proj.weight", mtp.e_proj.weight),
        ("h_proj.weight", mtp.h_proj.weight),
        ("attn.wq_a.weight", mtp.decoder.self_attn.q_a_proj.weight),
        ("hc_attn_fn", mtp.decoder.attn_hc.fn),
        ("hc_head_fn", mtp.hc_head.hc_fn),
        ("ffn.gate.bias", mtp.decoder.mlp.gate.e_score_correction_bias),
    ]:
        torch.testing.assert_close(target, expected["mtp.0." + source], rtol=0, atol=0)
    for i in range(2):
        assert torch.equal(
            mtp.decoder.mlp.experts.gate_up_proj[i],
            torch.cat(
                [
                    expected[f"mtp.0.ffn.experts.{i}.w1.weight"],
                    expected[f"mtp.0.ffn.experts.{i}.w3.weight"],
                ]
            ),
        )
        assert torch.equal(
            mtp.decoder.mlp.experts.down_proj[i], expected[f"mtp.0.ffn.experts.{i}.w2.weight"]
        )
    for name, param in model.named_parameters():
        assert param.requires_grad == (
            name.startswith("eagle_module.") and not name.endswith("e_score_correction_bias")
        )


@pytest.mark.parametrize("mix", [True, False])
def test_native_decoder_trains_through_shared_eagle(tmp_path, mix):
    _write_native(tmp_path)
    model = create_native_mtp_boost_model(tmp_path, mtp_ttt_steps=3, mtp_mix_hidden_states=mix)
    inputs = {
        "input_ids": torch.arange(7).unsqueeze(0),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
        "loss_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
        "base_model_outputs": {
            "draft_hidden_states": torch.randn(1, 7, 2, 128).bfloat16(),
            "base_model_hidden_states": torch.randn(1, 7, 128).bfloat16(),
        },
    }
    output = model(**inputs)
    assert torch.isfinite(output.loss)
    assert len(output.train_acc[0]) == 3
    output.loss.backward()
    assert model.eagle_module.e_proj.weight.grad is not None
    assert model.eagle_module.decoder.mlp.experts.gate_up_proj.grad is not None
    assert all(
        p.grad is None for n, p in model.named_parameters() if not n.startswith("eagle_module.")
    )
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        evaluation = model(**inputs)
    assert torch.isfinite(evaluation.loss)
    assert all(value == 0 for value in evaluation.train_acc[0][1:])
    assert model.eagle_module.config._attn_implementation == "flex_attention"


def test_native_meta_loading_keeps_rotary_buffers_real(tmp_path):
    expected = _write_native(tmp_path)
    meta = DeepSeekV4MTPAdapter.load_mtp(tmp_path, dtype=torch.bfloat16, device="meta")
    assert all(p.is_meta for p in meta.parameters())
    assert all(not b.is_meta for b in meta.rotary_emb.buffers())
    for name, value in DeepSeekV4MTPAdapter._native_tensors(meta.state_dict(), meta.config):
        assert value.dtype == expected[name].dtype


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_weight",
        "missing_scale",
        "extra_mtp",
        "scale_shape",
        "multiple_mtp",
        "compressed_mtp",
    ],
)
def test_invalid_native_checkpoint_fails_before_loading(tmp_path, corruption):
    _write_native(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if corruption == "missing_weight":
        del index["weight_map"]["mtp.0.ffn.experts.1.w3.weight"]
    elif corruption == "missing_scale":
        del index["weight_map"]["mtp.0.e_proj.scale"]
    elif corruption == "extra_mtp":
        index["weight_map"]["mtp.1.e_proj.weight"] = "mtp.safetensors"
    elif corruption == "scale_shape":
        save_file(
            {"mtp.0.e_proj.scale": torch.ones(2, 2).to(torch.float8_e8m0fnu)},
            tmp_path / "bad-scale.safetensors",
        )
        index["weight_map"]["mtp.0.e_proj.scale"] = "bad-scale.safetensors"
    else:
        config_path = tmp_path / "config.json"
        config = json.loads(config_path.read_text())
        if corruption == "multiple_mtp":
            config["num_nextn_predict_layers"] = 2
        else:
            config["compress_ratios"][-1] = 128
        config_path.write_text(json.dumps(config))
    index_path.write_text(json.dumps(index))
    with pytest.raises(NativeMTPCheckpointError):
        create_native_mtp_boost_model(tmp_path)


@pytest.mark.parametrize("sharded", [True, False])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
def test_native_export_reloads_trained_mtp_without_backbone(tmp_path, sharded, scale_dtype):
    source = tmp_path / "native"
    source.mkdir()
    _write_native(source, tied=not sharded, sharded=sharded, scale_dtype=scale_dtype)
    model = create_native_mtp_boost_model(source, mtp_use_torch_compile=False)
    if sharded:
        # FSDP2 saves trainable master weights in FP32 and keeps frozen state in BF16.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    with torch.no_grad():
        # Representable changes exercise both FP8 and MXFP4 export without rounding ambiguity.
        model.eagle_module.e_proj.weight.mul_(2)
        model.eagle_module.decoder.mlp.experts.gate_up_proj.mul_(2)
        if sharded:
            model.eagle_module.hc_head.hc_scale.add_(1e-6)
    checkpoint = tmp_path / "trained"
    model.save_pretrained(checkpoint, max_shard_size="32KB" if sharded else "5GB")
    if sharded:
        # Export must not need frozen target tensors, even to copy or link their shards.
        (source / "target.safetensors").unlink()
    output = tmp_path / "export"
    export_speculative_decoding(
        checkpoint if sharded else str(checkpoint), base_model=source, export_dir=output
    )
    restored = DeepSeekV4MTPAdapter.load_mtp(output, dtype=torch.bfloat16, device="cpu")
    for name, value in model.eagle_module.state_dict().items():
        actual = restored.state_dict()[name]
        torch.testing.assert_close(actual, value.to(actual.dtype), rtol=0, atol=0)
    keys = []
    for shard in output.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as handle:
            shard_keys = handle.keys()
            keys.extend(shard_keys)
            for key in shard_keys:
                if key.endswith(".scale"):
                    assert handle.get_tensor(key).dtype == scale_dtype
    weight_map = indexed_weight_map(output)
    scale_key = "mtp.0.hc_head_scale"
    with safe_open(output / weight_map[scale_key], framework="pt") as handle:
        torch.testing.assert_close(
            handle.get_tensor(scale_key),
            model.eagle_module.hc_head.hc_scale.detach(),
            rtol=0,
            atol=0,
        )
    assert len(keys) == len(set(keys))
    assert set(keys) == set(weight_map)
    assert all(key.startswith("mtp.0.") for key in keys)
    assert (output / "config.json").read_bytes() == (source / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_speculative_decoding(checkpoint, base_model=source, export_dir=output)
    for kwargs in ({}, {"base_model": source, "dtype": torch.bfloat16}):
        with pytest.raises(ValueError):
            export_speculative_decoding(checkpoint, export_dir=tmp_path / "invalid", **kwargs)
        assert not (tmp_path / "invalid").exists()
    incomplete_checkpoint = tmp_path / "incomplete-training"
    incomplete_checkpoint.mkdir()
    (incomplete_checkpoint / "config.json").write_bytes((checkpoint / "config.json").read_bytes())
    saved = {"eagle_module.e_proj.weight": model.eagle_module.e_proj.weight.detach()}
    save_file(saved, incomplete_checkpoint / "model.safetensors")
    with pytest.raises(NativeMTPCheckpointError):
        export_speculative_decoding(
            incomplete_checkpoint, base_model=source, export_dir=tmp_path / "incomplete"
        )
    assert not (tmp_path / "incomplete").exists()
