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

"""Exercise Qwen conditional-generation loading and fusion through public FSDP2 export."""

import copy
import gc
import json
import math
import traceback
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.transformers_models import create_tiny_qwen3_5_vl_offline_dir
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForImageTextToText

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import patch_fsdp_mp_dtypes
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector
from modelopt.torch.utils.distributed import is_fsdp2_model
from modelopt.torch.utils.plugins.model_load_utils import parallel_load_and_prepare_fsdp2

pytestmark = [pytest.mark.usefixtures("need_2_gpus"), pytest.mark.timeout(300)]

_PROJECTION_PAIRS = (("in_proj_qkv", "in_proj_z"), ("in_proj_b", "in_proj_a"))
_TEXT_ONLY_DISABLED = [
    "*visual*",
    "*linear_attn.conv1d*",
    "*linear_attn.in_proj_a*",
    "*linear_attn.in_proj_b*",
    "*mlp.gate.*",
    "*shared_expert_gate*",
]


def _create_qwen_checkpoint(tmp_path, moe):
    try:
        AutoConfig.for_model("qwen3_5_moe" if moe else "qwen3_5")
    except ValueError:
        pytest.skip("Qwen3.5 requires a recent Transformers")
    return create_tiny_qwen3_5_vl_offline_dir(
        tmp_path,
        moe=moe,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        vocab_size=128,
        max_position_embeddings=128,
    )


def _read_checkpoint(directory):
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    filenames = set(weight_map.values()) if weight_map else {"model.safetensors"}
    tensors = {}
    for filename in sorted(filenames):
        shard = load_file(str(directory / filename))
        assert not tensors.keys() & shard.keys(), "duplicate checkpoint tensor"
        if weight_map is not None:
            assert all(weight_map[key] == filename for key in shard), "incorrect shard mapping"
        tensors.update(shard)
    if weight_map is not None:
        assert set(tensors) == set(weight_map), "index and shard contents disagree"
    return tensors


def _check_on_rank_zero(rank, check):
    error = [None]
    if rank == 0:
        try:
            check()
        except Exception:
            error[0] = traceback.format_exc()
    dist.broadcast_object_list(error, src=0)
    assert error[0] is None, error[0]


def _assert_vision_preserved(checkpoint_dir, export_dir):
    source_config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    exported_config = json.loads((Path(export_dir) / "config.json").read_text())
    assert exported_config["architectures"] == source_config["architectures"]
    assert exported_config["architectures"][0].endswith("ForConditionalGeneration")
    assert exported_config["model_type"] == source_config["model_type"]
    for subconfig in ("text_config", "vision_config"):
        assert exported_config[subconfig]["model_type"] == source_config[subconfig]["model_type"]
    for field in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "layer_types",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "vocab_size",
        "max_position_embeddings",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        if field in source_config["text_config"]:
            assert exported_config["text_config"][field] == source_config["text_config"][field], (
                field
            )
    assert exported_config["vision_config"] == source_config["vision_config"]
    source = _read_checkpoint(checkpoint_dir)
    exported = _read_checkpoint(export_dir)
    source_vision = {key for key in source if key.startswith("model.visual.")}
    assert source_vision, "fixture has no vision weights"
    assert source_vision == {key for key in exported if key.startswith("model.visual.")}
    for key in source_vision:
        assert exported[key].dtype == source[key].dtype
        assert torch.equal(exported[key], source[key]), key


def _load_qwen_models(rank, size, checkpoint_dir, device):
    reference = (
        AutoModelForImageTextToText.from_pretrained(
            checkpoint_dir, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    model = parallel_load_and_prepare_fsdp2(
        checkpoint_dir, device, rank, size, attn_implementation="eager"
    )
    assert is_fsdp2_model(model)
    assert model.config.architectures == reference.config.architectures
    assert hasattr(model.model, "visual")
    layers = LayerActivationCollector.get_decoder_layers(model)
    assert len(layers) == 2
    assert any(isinstance(parameter, DTensor) for parameter in layers[0].parameters())
    assert all(not parameter.is_meta for parameter in model.parameters())
    batches = [
        torch.arange(5 + offset, 21 + offset, device=device).unsqueeze(0) for offset in (0, 16)
    ]
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=batches[0], use_cache=False).logits,
            reference(input_ids=batches[0], use_cache=False).logits,
            rtol=1e-2,
            atol=1e-2,
        )
    return model, reference, batches


def _run_qwen_autoquant_export(rank, size, checkpoint_dir, search_dir, export_dir, local_rank=None):
    device = torch.device(f"cuda:{rank if local_rank is None else local_rank}")
    with patch_fsdp_mp_dtypes():
        model, reference, batches = _load_qwen_models(rank, size, checkpoint_dir, device)
        del reference
        model, state = mtq.auto_quantize(
            model,
            constraints={
                "effective_bits": 8.0,
                "cost": {"excluded_module_name_patterns": ["*visual*"]},
            },
            quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
            disabled_layers=_TEXT_ONLY_DISABLED,
            data_loader=batches,
            forward_step=lambda m, tokens: m(input_ids=tokens, labels=tokens, use_cache=False),
            loss_func=lambda output, _: output.loss,
            num_calib_steps=2,
            num_score_steps=2,
            method="gradient",
            checkpoint=search_dir,
        )
        scores = [
            value for stat in state["candidate_stats"].values() for value in stat["raw_scores"]
        ]
        assert scores and all(math.isfinite(value) and value >= 0 for value in scores)
        assert any(value > 0 for value in scores)
        assert state["best"]["is_satisfied"]
        assert state["best"]["constraints"]["effective_bits"] <= 8.0 + 1e-6
        selections = [None] * size
        dist.all_gather_object(
            selections, {key: str(value) for key, value in state["best"]["recipe"].items()}
        )
        assert all(selection == selections[0] for selection in selections)
        export_hf_checkpoint(model, export_dir=export_dir, max_shard_size="512KB")
        dist.barrier()
        try:
            _check_on_rank_zero(rank, lambda: _assert_vision_preserved(checkpoint_dir, export_dir))
        finally:
            del model, state
            gc.collect()
            dist.barrier()


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_fsdp2_qwen_conditional_autoquant_export(dist_workers, tmp_path, moe):
    checkpoint = _create_qwen_checkpoint(tmp_path, moe)
    dist_workers.run(
        partial(
            _run_qwen_autoquant_export,
            checkpoint_dir=str(checkpoint),
            search_dir=str(tmp_path / "search"),
            export_dir=str(tmp_path / "export"),
        )
    )


def _mixed_projection_config(qkvz_nvfp4):
    quant_cfg = [{"quantizer_name": "*", "enable": False}]
    presets = (mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG)
    if not qkvz_nvfp4:
        presets = presets[::-1]
    for projections, preset in zip(_PROJECTION_PAIRS, presets):
        for quantizer in ("input_quantizer", "weight_quantizer"):
            numerics = next(
                entry["cfg"]
                for entry in preset["quant_cfg"]
                if entry.get("quantizer_name") == f"*{quantizer}"
            )
            quant_cfg.extend(
                {
                    "quantizer_name": f"*linear_attn.{projection}.{quantizer}",
                    "cfg": copy.deepcopy(numerics),
                }
                for projection in projections
            )
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def _assert_mixed_projection_export(checkpoint_dir, export_dir, reference_dir, qkvz_nvfp4):
    actual = _read_checkpoint(export_dir)
    expected = _read_checkpoint(reference_dir)
    assert set(actual) == set(expected)
    for key in expected:
        assert (
            actual[key].dtype == expected[key].dtype and actual[key].shape == expected[key].shape
        ), key
        assert torch.equal(actual[key].float(), expected[key].float()), key
    for filename, key in (
        ("hf_quant_config.json", "quantization"),
        ("config.json", "quantization_config"),
    ):
        assert (
            json.loads((Path(export_dir) / filename).read_text())[key]
            == json.loads((Path(reference_dir) / filename).read_text())[key]
        )
    quantization = json.loads((Path(export_dir) / "hf_quant_config.json").read_text())[
        "quantization"
    ]
    assert quantization["quant_algo"] == "MIXED_PRECISION"
    layers = quantization["quantized_layers"]
    source = _read_checkpoint(checkpoint_dir)
    projection_names = set()
    for pair_index, projections in enumerate(_PROJECTION_PAIRS):
        names = [
            next(name for name in layers if name.endswith(f".linear_attn.{projection}"))
            for projection in projections
        ]
        projection_names.update(names)
        nvfp4 = (pair_index == 0) == qkvz_nvfp4
        assert all(layers[name]["quant_algo"] == ("NVFP4" if nvfp4 else "FP8") for name in names)
        for name in names:
            weight = actual[f"{name}.weight"]
            assert weight.dtype == (torch.uint8 if nvfp4 else torch.float8_e4m3fn)
            source_shape = source[f"{name}.weight"].shape
            expected_shape = (*source_shape[:-1], source_shape[-1] // 2) if nvfp4 else source_shape
            assert weight.shape == expected_shape
        if nvfp4:
            for suffix in ("input_scale", "weight_scale_2"):
                assert torch.equal(
                    actual[f"{names[0]}.{suffix}"], actual[f"{names[1]}.{suffix}"]
                ), suffix
    assert set(layers) == projection_names
    _assert_vision_preserved(checkpoint_dir, export_dir)


def _run_qwen_mixed_export(
    rank, size, checkpoint_dir, export_dir, reference_dir, qkvz_nvfp4=True, local_rank=None
):
    device = torch.device(f"cuda:{rank if local_rank is None else local_rank}")
    with patch_fsdp_mp_dtypes():
        model, reference, batches = _load_qwen_models(rank, size, checkpoint_dir, device)

        def calibrate(m):
            for tokens in batches:
                m(input_ids=tokens, use_cache=False)

        config = _mixed_projection_config(qkvz_nvfp4)
        mtq.quantize(model, config, calibrate)
        mtq.quantize(reference, config, calibrate)
        export_hf_checkpoint(model, export_dir=export_dir, max_shard_size="512KB")
        dist.barrier()

        def check_export(reference_model):
            export_hf_checkpoint(reference_model, export_dir=reference_dir, max_shard_size="512KB")
            _assert_mixed_projection_export(checkpoint_dir, export_dir, reference_dir, qkvz_nvfp4)

        try:
            _check_on_rank_zero(rank, partial(check_export, reference))
        finally:
            del model, reference
            gc.collect()
            dist.barrier()


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("qkvz_nvfp4", [True, False], ids=["nvfp4_qkvz", "nvfp4_ba"])
def test_fsdp2_qwen_mixed_projection_export(dist_workers, tmp_path, moe, qkvz_nvfp4):
    checkpoint = _create_qwen_checkpoint(tmp_path, moe)
    dist_workers.run(
        partial(
            _run_qwen_mixed_export,
            checkpoint_dir=str(checkpoint),
            export_dir=str(tmp_path / "export"),
            reference_dir=str(tmp_path / "reference"),
            qkvz_nvfp4=qkvz_nvfp4,
        )
    )
