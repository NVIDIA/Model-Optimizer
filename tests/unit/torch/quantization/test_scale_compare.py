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

from collections import Counter
from collections.abc import Mapping

import pytest
import torch
from safetensors.torch import save_file

from modelopt.torch.quantization.utils.scale_compare import compare_scales, main


class _LoadCounting(Mapping):
    """Mapping that counts tensor loads, like the lazy checkpoint view does on disk reads."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.loads = Counter()

    def __getitem__(self, name):
        self.loads[name] += 1
        return self.tensors[name]

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self):
        return len(self.tensors)


@pytest.mark.parametrize(("rtol", "input_scale_outside"), [(0.0, 1), (0.02, 0)])
def test_compare_scales(rtol, input_scale_outside):
    fp8 = lambda *values: torch.tensor(values).to(torch.float8_e4m3fn)  # noqa: E731
    a = {
        "l0.q.input_scale": torch.tensor(0.5),
        "l1.q.input_scale": torch.tensor(1.0),
        "l0.q.weight_scale": fp8(1.0, 2.0, 4.0, 8.0),
        "gone.input_scale": torch.tensor(1.0),
        "reshaped.input_scale": torch.zeros(2),
    }
    b = {
        "l0.q.input_scale": torch.tensor(0.505),  # 1% off
        "l1.q.input_scale": torch.tensor(1.0),
        "l0.q.weight_scale": fp8(1.0, 2.0, 4.0, 9.0),  # one fp8 step off
        "new.input_scale": torch.tensor(1.0),
        "reshaped.input_scale": torch.zeros(3),
    }

    diff = compare_scales(a, b, rtol=rtol)

    assert diff.only_in_a == ["gone.input_scale"]
    assert diff.only_in_b == ["new.input_scale"]
    assert diff.shape_mismatch == ["reshaped.input_scale"]
    by_name = {t.name: t for t in diff.tensors}
    assert by_name["l1.q.input_scale"].outside_tolerance == 0
    assert by_name["l0.q.input_scale"].outside_tolerance == input_scale_outside
    assert by_name["l0.q.input_scale"].max_rel == pytest.approx(0.005 / 0.505, rel=1e-4)
    assert by_name["l0.q.weight_scale"].outside_tolerance == 1
    assert by_name["l0.q.weight_scale"].max_rel == pytest.approx(1 / 9)
    assert not diff.within_tolerance


@pytest.mark.parametrize(("args", "exit_code"), [([], 1), (["--rtol", "0.02"], 0)])
def test_main_compares_checkpoint_directories(tmp_path, capsys, args, exit_code):
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    # Run A is sharded by layer, run B is one file; only the scales are compared, not the weights.
    save_file(
        {"l0.q.input_scale": torch.tensor(0.5), "l0.q.weight": torch.ones(2, 2)},
        run_a / "model-1.safetensors",
    )
    save_file({"l1.q.input_scale": torch.tensor(1.0)}, run_a / "model-2.safetensors")
    save_file(
        {
            "l0.q.input_scale": torch.tensor(0.5),
            "l1.q.input_scale": torch.tensor(1.01),
            "l0.q.weight": torch.zeros(2, 2),
        },
        run_b / "model.safetensors",
    )

    assert main([str(run_a), str(run_b), *args]) == exit_code

    assert "input_scale" in capsys.readouterr().out


def test_compare_scales_loads_each_tensor_once():
    a = _LoadCounting({f"l{i}.q.input_scale": torch.tensor(float(i + 1)) for i in range(3)})
    b = _LoadCounting(dict(a.tensors))

    assert compare_scales(a, b).within_tolerance

    assert set(a.loads.values()) == set(b.loads.values()) == {1}


def test_nothing_to_compare_is_not_a_pass(tmp_path):
    assert not compare_scales({}, {}).within_tolerance

    save_file({"l0.q.weight": torch.ones(2, 2)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="No quantization scale tensors"):
        main([str(tmp_path), str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        main([str(tmp_path / "missing"), str(tmp_path)])


def test_a_scale_in_two_shards_is_an_error(tmp_path):
    save_file({"l0.q.input_scale": torch.ones(1)}, tmp_path / "model-1.safetensors")
    save_file({"l0.q.input_scale": torch.ones(1)}, tmp_path / "model-2.safetensors")

    with pytest.raises(ValueError, match="more than one file"):
        main([str(tmp_path), str(tmp_path)])


def test_weight_scale_inv_is_compared(tmp_path):
    save_file({"l0.q.weight_scale_inv": torch.tensor(1.0)}, tmp_path / "a.safetensors")
    save_file({"l0.q.weight_scale_inv": torch.tensor(1.1)}, tmp_path / "b.safetensors")

    assert main([str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")]) == 1
