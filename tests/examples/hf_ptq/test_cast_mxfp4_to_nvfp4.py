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
"""Unit tests for ``examples/hf_ptq/cast_mxfp4_to_nvfp4.py``.

The module lives next to the example script (not inside the ``modelopt`` package),
so we add ``examples/hf_ptq/`` to ``sys.path`` before importing it.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_LLM_PTQ_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"
if str(_LLM_PTQ_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_PTQ_DIR))

import cast_mxfp4_to_nvfp4 as cast

# ---------- quantizer_name_from_blocks_key ----------------------------------


def test_quantizer_name_from_blocks_key():
    assert (
        cast.quantizer_name_from_blocks_key("model.layers.0.mlp.experts.gate_up_proj_blocks")
        == "model.layers.0.mlp.experts.gate_up_proj_weight_quantizer"
    )
    assert (
        cast.quantizer_name_from_blocks_key("model.layers.0.mlp.experts.down_proj_blocks")
        == "model.layers.0.mlp.experts.down_proj_weight_quantizer"
    )


def test_quantizer_name_from_blocks_key_rejects_non_blocks_key():
    with pytest.raises(AssertionError):
        cast.quantizer_name_from_blocks_key("model.layers.0.mlp.experts.gate_up_proj_scales")


# ---------- _collect_keys_with_suffix + build_amax_map (synthetic ckpt) ------


def _write_synthetic_mxfp4_checkpoint(
    tmp_path: Path,
    layer_names: list[str],
    e8m0_per_layer: dict[str, torch.Tensor],
    blocks_per_layer: dict[str, torch.Tensor],
) -> Path:
    """Write a tiny safetensors + index.json mimicking the OpenAI MXFP4 layout.

    Each ``layer_names[i]`` becomes ``<name>_blocks`` + ``<name>_scales`` keys.
    Returns the checkpoint directory.
    """
    ckpt_dir = tmp_path / "fake_mxfp4"
    ckpt_dir.mkdir()
    state = {}
    for name in layer_names:
        state[f"{name}_blocks"] = blocks_per_layer[name]
        state[f"{name}_scales"] = e8m0_per_layer[name]
    shard_name = "model-00001-of-00001.safetensors"
    save_file(state, str(ckpt_dir / shard_name))
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": dict.fromkeys(state, shard_name),
    }
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return ckpt_dir


def test_collect_keys_with_suffix(tmp_path):
    name = "model.layers.0.mlp.experts.gate_up_proj"
    ckpt_dir = _write_synthetic_mxfp4_checkpoint(
        tmp_path,
        [name],
        e8m0_per_layer={name: torch.zeros(4, dtype=torch.uint8)},
        blocks_per_layer={name: torch.zeros((4, 16), dtype=torch.uint8)},
    )
    scales_keys = cast._collect_keys_with_suffix(ckpt_dir, "_scales")
    blocks_keys = cast._collect_keys_with_suffix(ckpt_dir, "_blocks")
    assert set(scales_keys.keys()) == {f"{name}_scales"}
    assert set(blocks_keys.keys()) == {f"{name}_blocks"}


def test_build_amax_map(tmp_path):
    name1 = "model.layers.0.mlp.experts.gate_up_proj"
    name2 = "model.layers.0.mlp.experts.down_proj"
    e8m0 = {
        name1: torch.tensor([130, 128, 125], dtype=torch.uint8),  # ks: 3, 1, -2; spread 5
        name2: torch.tensor([135, 120], dtype=torch.uint8),  # ks: 8, -7; spread 15
    }
    blocks = {
        name1: torch.zeros((3, 16), dtype=torch.uint8),
        name2: torch.zeros((2, 16), dtype=torch.uint8),
    }
    ckpt_dir = _write_synthetic_mxfp4_checkpoint(tmp_path, [name1, name2], e8m0, blocks)

    amax_map = cast.build_amax_map(ckpt_dir)
    assert set(amax_map.keys()) == {f"{n}_weight_quantizer" for n in (name1, name2)}

    e1 = amax_map[f"{name1}_weight_quantizer"]
    assert e1["k_min"] == -2 and e1["k_max"] == 3 and e1["m"] == -5
    assert e1["global_amax"] == pytest.approx(6.0 * 448.0 * 2.0**-5)
    assert e1["pct_lossless"] == pytest.approx(100.0)

    e2 = amax_map[f"{name2}_weight_quantizer"]
    assert e2["k_min"] == -7 and e2["k_max"] == 8 and e2["m"] == 0
    assert e2["pct_lossless"] == pytest.approx(100.0)


def test_build_amax_map_no_scales_raises(tmp_path):
    """A directory without ``*_scales`` tensors should error."""
    empty = tmp_path / "empty"
    empty.mkdir()
    save_file(
        {"model.layers.0.weight": torch.zeros(4)},
        str(empty / "model-00001-of-00001.safetensors"),
    )
    (empty / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"},
            }
        )
    )
    with pytest.raises(SystemExit, match="No '\\*_scales'"):
        cast.build_amax_map(empty)


# ---------- apply_to_model end-to-end (mock model) ---------------------------


class _FakeStaticQuantizer(torch.nn.Module):
    """Stand-in for NVFP4StaticQuantizer.

    Carries a per-block ``_amax`` buffer and a ``global_amax`` property whose
    setter writes ``_global_amax`` — matches the contract apply_to_model relies
    on. Subclasses ``cast.NVFP4StaticQuantizer`` so the isinstance check passes.
    """

    def __init__(self, num_blocks: int):
        super().__init__()
        self.register_buffer("_amax", torch.zeros(num_blocks, dtype=torch.float32))
        self.register_buffer("_global_amax", torch.zeros((), dtype=torch.float32))

    @property
    def global_amax(self) -> torch.Tensor:
        return self._global_amax

    @global_amax.setter
    def global_amax(self, value: torch.Tensor) -> None:
        self._global_amax = value


# Inherit at runtime so isinstance(NVFP4StaticQuantizer) is True.
_FakeStaticQuantizer.__bases__ = (cast.NVFP4StaticQuantizer,)


class _FakeExperts(torch.nn.Module):
    """Mimics a HF GptOssExperts module: a ``*_weight_quantizer`` child."""

    def __init__(self, num_blocks: int):
        super().__init__()
        # Quantizer attribute name must match the source key after stripping
        # ``_blocks`` and appending ``_weight_quantizer``.
        self.gate_up_proj_weight_quantizer = _FakeStaticQuantizer(num_blocks)


class _FakeModel(torch.nn.Module):
    """Single MLP-like submodule path: ``model.layers.0.mlp.experts.gate_up_proj_*``."""

    def __init__(self, num_blocks: int):
        super().__init__()
        self.experts = _FakeExperts(num_blocks)


def test_apply_to_model_writes_global_and_per_block_amax(tmp_path):
    """Happy path: cast overrides _amax + global_amax on the matching quantizer."""
    # Build a synthetic MXFP4 source: 4 in-range MXFP4 blocks => 8 NVFP4 blocks.
    name = "experts.gate_up_proj"
    e8m0 = torch.tensor([130, 128, 125, 132], dtype=torch.uint8)  # ks: 3, 1, -2, 5
    blocks = torch.zeros((4, 16), dtype=torch.uint8)
    ckpt_dir = _write_synthetic_mxfp4_checkpoint(
        tmp_path,
        [name],
        e8m0_per_layer={name: e8m0},
        blocks_per_layer={name: blocks},
    )

    # 8 NVFP4 blocks (each MXFP4 block of 32 splits into two NVFP4 blocks of 16).
    model = _FakeModel(num_blocks=8)
    cast.apply_to_model(model, ckpt_dir)

    quantizer = model.experts.gate_up_proj_weight_quantizer
    # k_max = 5 -> m = -3 -> global_amax = 6 * 448 * 2^-3 = 336.
    assert float(quantizer.global_amax.item()) == pytest.approx(6.0 * 448.0 * 2.0**-3)
    # All in-range -> per-block _amax = 6 * 2^k_j, repeat-interleaved by 2.
    expected_per_mxfp4 = 6.0 * torch.exp2(torch.tensor([3.0, 1.0, -2.0, 5.0]))
    expected_per_nvfp4 = expected_per_mxfp4.repeat_interleave(2)
    assert torch.allclose(quantizer._amax.float(), expected_per_nvfp4)


def test_apply_to_model_raises_on_missing_blocks_pair(tmp_path):
    """If a *_scales tensor has no paired *_blocks tensor, raise ValueError."""
    ckpt_dir = tmp_path / "missing_blocks"
    ckpt_dir.mkdir()
    # Write only the _scales tensor.
    save_file(
        {"experts.gate_up_proj_scales": torch.zeros(2, dtype=torch.uint8)},
        str(ckpt_dir / "model-00001-of-00001.safetensors"),
    )
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"experts.gate_up_proj_scales": "model-00001-of-00001.safetensors"},
            }
        )
    )
    model = _FakeModel(num_blocks=4)
    with pytest.raises(AssertionError, match=r"no paired '.*_blocks' tensor"):
        cast.apply_to_model(model, ckpt_dir)


def test_apply_to_model_raises_on_wrong_quantizer_type(tmp_path):
    """If the matching attribute isn't an NVFP4StaticQuantizer, raise RuntimeError."""
    name = "experts.gate_up_proj"
    e8m0 = torch.tensor([130, 128], dtype=torch.uint8)
    blocks = torch.zeros((2, 16), dtype=torch.uint8)
    ckpt_dir = _write_synthetic_mxfp4_checkpoint(tmp_path, [name], {name: e8m0}, {name: blocks})

    class _NotAQuantizer(torch.nn.Module):
        pass

    class _Wrong(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.Module()
            self.experts.gate_up_proj_weight_quantizer = _NotAQuantizer()

    with pytest.raises(AssertionError, match="expected NVFP4StaticQuantizer"):
        cast.apply_to_model(_Wrong(), ckpt_dir)


# ---------- force_weight_quantizers_static ----------------------------------


def test_force_weight_quantizers_static():
    """Only ``*weight_quantizer`` entries with a ``block_sizes`` dict flip to ``type='static'``;
    input quantizers and entries without block_sizes are left untouched."""
    quant_cfg = [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic"}},
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic"}},
        },
        {"quantizer_name": "*router*", "enable": False},  # no cfg / block_sizes
    ]

    cast.force_weight_quantizers_static(quant_cfg)

    assert quant_cfg[1]["cfg"]["block_sizes"]["type"] == "static"  # weight quantizer forced
    assert quant_cfg[1]["cfg"]["block_sizes"][-1] == 16  # other block_sizes keys preserved
    assert quant_cfg[2]["cfg"]["block_sizes"]["type"] == "dynamic"  # input quantizer untouched
    assert "cfg" not in quant_cfg[3]  # entry without block_sizes untouched
