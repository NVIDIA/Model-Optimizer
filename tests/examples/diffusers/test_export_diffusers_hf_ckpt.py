# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from pathlib import Path
from typing import NamedTuple

import pytest
from _test_utils.examples.models import FLUX_SCHNELL_PATH, SDXL_PATH
from _test_utils.examples.run_command import run_example_command
from _test_utils.torch.misc import minimum_sm


class DiffuserHfExportModel(NamedTuple):
    name: str
    path: str
    dtype: str
    format_type: str
    quant_algo: str
    collect_method: str
    model_dtype: str = "Half"

    def quantize_and_export_hf(self, tmp_path: Path) -> Path:
        hf_ckpt_dir = tmp_path / f"{self.name}_{self.format_type}_hf_ckpt"
        cmd_args = [
            "python",
            "quantize.py",
            "--model",
            self.name,
            "--override-model-path",
            self.path,
            "--calib-size",
            "4",
            "--batch-size",
            "2",
            "--n-steps",
            "2",
            "--percentile",
            "1.0",
            "--alpha",
            "0.8",
            "--format",
            self.format_type,
            "--quant-algo",
            self.quant_algo,
            "--collect-method",
            self.collect_method,
            "--model-dtype",
            self.model_dtype,
            "--trt-high-precision-dtype",
            self.dtype,
            "--hf-ckpt-dir",
            str(hf_ckpt_dir),
        ]
        run_example_command(cmd_args, "diffusers/quantization")
        return hf_ckpt_dir


@pytest.mark.parametrize(
    "model",
    [
        DiffuserHfExportModel(
            name="sdxl-1.0",
            path=SDXL_PATH,
            dtype="Half",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
        ),
        DiffuserHfExportModel(
            name="flux-schnell",
            path=FLUX_SCHNELL_PATH,
            dtype="BFloat16",
            format_type="int8",
            quant_algo="smoothquant",
            collect_method="min-mean",
            model_dtype="BFloat16",
        ),
        pytest.param(
            DiffuserHfExportModel(
                name="sdxl-1.0",
                path=SDXL_PATH,
                dtype="Half",
                format_type="fp8",
                quant_algo="max",
                collect_method="default",
            ),
            marks=minimum_sm(89),
        ),
        pytest.param(
            DiffuserHfExportModel(
                name="flux-schnell",
                path=FLUX_SCHNELL_PATH,
                dtype="BFloat16",
                format_type="fp4",
                quant_algo="max",
                collect_method="default",
                model_dtype="BFloat16",
            ),
            marks=minimum_sm(89),
        ),
    ],
    ids=[
        "sdxl_1.0_int8_smoothquant_min_mean",
        "flux_schnell_int8_smoothquant_min_mean",
        "sdxl_1.0_fp8_max_default",
        "flux_schnell_fp4_max_default",
    ],
)
def test_diffusers_hf_ckpt_export(model: DiffuserHfExportModel, tmp_path: Path) -> None:
    hf_ckpt_dir = model.quantize_and_export_hf(tmp_path)

    assert hf_ckpt_dir.exists(), f"HF checkpoint directory was not created: {hf_ckpt_dir}"

    config_files = list(hf_ckpt_dir.rglob("config.json"))
    assert len(config_files) > 0, f"No config.json found in {hf_ckpt_dir}"

    weight_files = list(hf_ckpt_dir.rglob("*.safetensors")) + list(hf_ckpt_dir.rglob("*.bin"))
    assert len(weight_files) > 0, f"No weight files (.safetensors or .bin) found in {hf_ckpt_dir}"


class QwenHfExportModel(NamedTuple):
    format_type: str
    quant_algo: str
    is_svdquant: bool

    def quantize_and_export_hf(self, tiny_qwen_image_path: str, tmp_path: Path) -> Path:
        hf_ckpt_dir = tmp_path / f"qwen_{self.format_type}_{self.quant_algo}_hf_ckpt"
        cmd_args = [
            "python",
            "quantize.py",
            "--model",
            "qwen-image",
            "--override-model-path",
            str(tiny_qwen_image_path),
            "--format",
            self.format_type,
            "--quant-algo",
            self.quant_algo,
            "--collect-method",
            "default",
            "--model-dtype",
            "BFloat16",
            "--trt-high-precision-dtype",
            "BFloat16",
            "--calib-size",
            "2",
            "--batch-size",
            "1",
            "--n-steps",
            "2",
            "--hf-ckpt-dir",
            str(hf_ckpt_dir),
        ]
        if self.is_svdquant:
            cmd_args.extend(["--lowrank", "8"])
        run_example_command(cmd_args, "diffusers/quantization")
        return hf_ckpt_dir


def _module_prefixes(keys: set[str], suffix: str) -> set[str]:
    """Module paths (key minus suffix) for every key ending in ``suffix``."""
    return {k[: -len(suffix)] for k in keys if k.endswith(suffix)}


def _block_indices(prefixes: set[str]) -> set[int]:
    """transformer_blocks indices referenced by a set of module prefixes."""
    import re

    indices = set()
    for prefix in prefixes:
        match = re.search(r"transformer_blocks\.(\d+)\.", prefix)
        if match:
            indices.add(int(match.group(1)))
    return indices


# Tiny Qwen fixture has 6 transformer blocks; the recipe excludes the first 2 and
# last 2, so only blocks 2 and 3 are quantized.
_QWEN_QUANTIZED_BLOCKS = {2, 3}
_QWEN_LORA_RANK = 8
# Per quantized block: image-stream linears keep full SVDQuant (low-rank branch +
# pre_quant_scale), while the text-stream and modulation linears match the
# "svdquant_skip_layers" patterns for qwen-image in
# examples/diffusers/quantization/models_utils.py and are exported as plain NVFP4.
_QWEN_SVDQUANT_PROMOTED_SUFFIXES = (
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    ".attn.to_out.0",
    ".img_mlp.net.0.proj",
    ".img_mlp.net.2",
)
_QWEN_SVDQUANT_SKIPPED_SUFFIXES = (
    ".attn.add_q_proj",
    ".attn.add_k_proj",
    ".attn.add_v_proj",
    ".attn.to_add_out",
    ".txt_mlp.net.0.proj",
    ".txt_mlp.net.2",
    ".img_mod.1",
    ".txt_mod.1",
)


@pytest.mark.parametrize(
    "qwen_model",
    [
        pytest.param(QwenHfExportModel("fp8", "max", False), marks=minimum_sm(89)),
        pytest.param(QwenHfExportModel("fp4", "max", False), marks=minimum_sm(89)),
        pytest.param(QwenHfExportModel("fp4", "svdquant", True), marks=minimum_sm(89)),
    ],
    ids=["qwen_fp8_max", "qwen_nvfp4_max", "qwen_nvfp4_svdquant"],
)
def test_qwen_image_hf_ckpt_export(
    qwen_model: QwenHfExportModel, tiny_qwen_image_path: str, tmp_path: Path
) -> None:
    from safetensors import safe_open

    hf_ckpt_dir = qwen_model.quantize_and_export_hf(tiny_qwen_image_path, tmp_path)
    assert hf_ckpt_dir.exists(), f"HF checkpoint directory was not created: {hf_ckpt_dir}"

    # The transformer is the quantized component.
    transformer_dir = hf_ckpt_dir / "transformer"
    config_path = transformer_dir / "config.json"
    assert config_path.exists(), f"no transformer/config.json in {hf_ckpt_dir}"
    quant_config = json.loads(config_path.read_text()).get("quantization_config")
    assert quant_config is not None, "missing quantization_config"
    assert quant_config.get("quant_method") == "modelopt"

    keys: set[str] = set()
    lora_tensors: dict[str, object] = {}
    safetensors_files = sorted(transformer_dir.rglob("*.safetensors"))
    assert safetensors_files, f"no safetensors in {transformer_dir}"
    for path in safetensors_files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                keys.add(key)
                if key.endswith((".svdquant_lora_a", ".svdquant_lora_b")):
                    lora_tensors[key] = handle.get_tensor(key)

    # No live quantizer state should leak into the exported checkpoint.
    assert not any("weight_quantizer" in k for k in keys), "quantizer keys leaked into export"
    assert not any("input_quantizer._amax" in k for k in keys)

    # Recipe: only the middle transformer blocks are quantized — first-2/last-2 of
    # transformer_blocks are excluded, and nothing outside transformer_blocks.
    weight_scale_prefixes = _module_prefixes(keys, ".weight_scale")
    assert weight_scale_prefixes, "no quantized linears found in export"
    assert all("transformer_blocks." in p for p in weight_scale_prefixes), (
        f"a non-transformer_blocks module was quantized: {weight_scale_prefixes}"
    )
    assert _block_indices(weight_scale_prefixes) == _QWEN_QUANTIZED_BLOCKS, (
        f"expected only blocks {_QWEN_QUANTIZED_BLOCKS} quantized"
    )

    if qwen_model.is_svdquant:
        a_prefixes = _module_prefixes(keys, ".svdquant_lora_a")
        b_prefixes = _module_prefixes(keys, ".svdquant_lora_b")
        pqs_prefixes = _module_prefixes(keys, ".pre_quant_scale")
        assert a_prefixes, "no promoted svdquant_lora_a keys"
        # Every promoted linear carries lora_a, lora_b, and pre_quant_scale. The
        # svdquant_skip_layers (text-stream + modulation linears) are quantized
        # plain NVFP4: they have weight_scale but no low-rank/pre_quant_scale
        # tensors, so the promoted and quantized sets differ by exactly them.
        expected_promoted = {
            f"transformer_blocks.{block}{suffix}"
            for block in _QWEN_QUANTIZED_BLOCKS
            for suffix in _QWEN_SVDQUANT_PROMOTED_SUFFIXES
        }
        expected_skipped = {
            f"transformer_blocks.{block}{suffix}"
            for block in _QWEN_QUANTIZED_BLOCKS
            for suffix in _QWEN_SVDQUANT_SKIPPED_SUFFIXES
        }
        assert a_prefixes == b_prefixes == pqs_prefixes == expected_promoted
        assert weight_scale_prefixes == expected_promoted | expected_skipped
        # Rank-consistent shapes; lora_a=[rank, in], lora_b=[out, rank], rank == --lowrank.
        for key, tensor in lora_tensors.items():
            if key.endswith(".svdquant_lora_a"):
                assert tensor.shape[0] == _QWEN_LORA_RANK
            else:
                assert tensor.shape[1] == _QWEN_LORA_RANK
        # NVFP4 secondary scales are present.
        assert any(k.endswith(".weight_scale_2") for k in keys)
        # config schema (modeled on nvfp4_awq).
        assert quant_config.get("quant_algo") == "NVFP4_SVD"
        group = next(iter(quant_config.get("config_groups", {}).values()), {})
        assert group.get("lora_rank") == _QWEN_LORA_RANK
        assert group.get("pre_quant_scale") is True
        assert group.get("has_zero_point") is False
        assert quant_config.get("ignore"), "expected excluded modules in 'ignore'"
    else:
        # Plain FP8/NVFP4: weight scales present, no SVDQuant tensors.
        assert weight_scale_prefixes, "no weight_scale in export"
        assert not any(k.endswith(".svdquant_lora_a") for k in keys)
        if qwen_model.format_type == "fp4":
            assert any(k.endswith(".weight_scale_2") for k in keys)


class Wan22HfExportModel(NamedTuple):
    model: str
    backbone: str | None
    format_type: str
    quant_algo: str
    collect_method: str

    def _suffix(self) -> str:
        stem = self.model.replace("wan2.2-t2v-", "")
        parts = [stem, *([self.backbone] if self.backbone else []), self.format_type]
        return "_".join(parts)

    def quantize_and_export_hf(self, tiny_wan22_path: str, tmp_path: Path) -> Path:
        hf_ckpt_dir = tmp_path / f"wan22_{self._suffix()}_hf_ckpt"
        cmd_args = [
            "python",
            "quantize.py",
            "--model",
            self.model,
            "--override-model-path",
            tiny_wan22_path,
            "--format",
            self.format_type,
            "--quant-algo",
            self.quant_algo,
            "--collect-method",
            self.collect_method,
            "--model-dtype",
            "BFloat16",
            "--trt-high-precision-dtype",
            "BFloat16",
            "--calib-size",
            "2",
            "--batch-size",
            "1",
            "--n-steps",
            "2",
            # Tiny video dims — override MODEL_DEFAULTS for fast CI.
            "--extra-param",
            "height=16",
            "--extra-param",
            "width=16",
            "--extra-param",
            "num_frames=5",
            "--hf-ckpt-dir",
            str(hf_ckpt_dir),
        ]
        if self.backbone is not None:
            cmd_args.extend(["--backbone", self.backbone])
        run_example_command(cmd_args, "diffusers/quantization")
        return hf_ckpt_dir


@pytest.mark.parametrize(
    "wan_model",
    [
        Wan22HfExportModel("wan2.2-t2v-14b", None, "int8", "smoothquant", "min-mean"),
        pytest.param(
            Wan22HfExportModel("wan2.2-t2v-14b", None, "fp8", "max", "default"),
            marks=minimum_sm(89),
        ),
    ],
    ids=[
        "wan22_14b_transformer_int8_smoothquant",
        "wan22_14b_transformer_fp8_max",
    ],
)
def test_wan22_hf_ckpt_export(
    wan_model: Wan22HfExportModel, tiny_wan22_path: str, tmp_path: Path
) -> None:
    hf_ckpt_dir = wan_model.quantize_and_export_hf(tiny_wan22_path, tmp_path)

    assert hf_ckpt_dir.exists(), f"HF checkpoint directory was not created: {hf_ckpt_dir}"

    config_files = list(hf_ckpt_dir.rglob("config.json"))
    assert len(config_files) > 0, f"No config.json found in {hf_ckpt_dir}"

    weight_files = list(hf_ckpt_dir.rglob("*.safetensors")) + list(hf_ckpt_dir.rglob("*.bin"))
    assert len(weight_files) > 0, f"No weight files (.safetensors or .bin) found in {hf_ckpt_dir}"
