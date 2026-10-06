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

"""Real Gloo KV searches must match a single-process search over all rank data."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import DistributedWorkerPool
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.quantization as mtq


def _formats():
    return [
        (
            {
                "quant_cfg": [
                    {
                        "quantizer_name": "*[kv]_bmm_quantizer",
                        "cfg": {"num_bits": (4, 3), **({"constant_amax": 1.0} if cast else {})},
                    }
                ],
                "effective_bits": 8.0,
                "algorithm": None if cast else "max",
            },
            "cast_fp8" if cast else "calibrated_fp8",
        )
        for cast in (False, True)
    ]


def _batches():
    generator = torch.Generator().manual_seed(19)
    return [torch.randint(0, 32, (1, length), generator=generator) for length in (4, 8, 6, 10)]


def _logits(model, batch):
    return model(input_ids=batch, use_cache=False).logits


def _search(batches, steps, checkpoint=None, forward_step=_logits, bits=8.0):
    model = get_tiny_llama(num_attention_heads=2, num_key_value_heads=1)
    return mtq.auto_quantize(
        model,
        constraints={"effective_bits": bits, "cost_model": "kv_cache"},
        quantization_formats=_formats(),
        data_loader=batches,
        forward_step=forward_step,
        num_calib_steps=steps,
        num_score_steps=steps,
        method="kl_div",
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        verbose=False,
    )[1]


def _distributed_search(rank, world_size, directory, expected):
    torch.set_num_threads(1)
    batches = _batches()[rank::world_size]
    state = _search(batches, 2, directory)
    assert state["num_scored_tokens"] == expected["num_scored_tokens"] == 28
    assert state["search_signature"]["distributed_world_size"] == world_size
    assert state["best"]["recipe"] == expected["best"]["recipe"]
    assert state["best"]["constraints"] == expected["best"]["constraints"]
    assert state["best"]["is_satisfied"]
    for layer, stats in state["layers"].items():
        assert stats["scores"] == pytest.approx(expected["layers"][layer]["scores"], abs=1e-7)
        for candidate, quantizers in state["quantizer_state"][layer].items():
            for attr, buffers in quantizers.items():
                torch.testing.assert_close(
                    buffers["_amax"],
                    expected["quantizer_state"][layer][candidate][attr]["_amax"],
                    rtol=0,
                    atol=0,
                )
    gathered = [None] * world_size
    dist.all_gather_object(gathered, (state["layers"], state["best"]))
    assert all(result == gathered[0] for result in gathered)
    assert (directory / f"rank{rank}.pth").is_file()

    def no_forward(*_args):
        pytest.fail("Resume must reuse both calibration and sensitivity scores.")

    resumed = _search(batches, 2, directory, forward_step=no_forward)
    assert resumed["layers"] == state["layers"]
    assert resumed["best"] == state["best"]
    with pytest.raises(ValueError, match=r"solver failed on rank 0.*could not satisfy"):
        _search(batches, 2, directory, forward_step=no_forward, bits=7.0)

    with pytest.raises(ValueError, match="consistent per-rank checkpoint"):
        _search(batches, 2, directory if rank else directory / "missing")
    with pytest.raises(ValueError, match="same number of batches"):
        _search(batches[:1] if rank == 0 else batches, 2)

    def nonfinite(model, batch):
        output = _logits(model, batch)
        return output * float("nan") if rank == 0 else output

    with pytest.raises(ValueError, match="NaN or Inf logits"):
        _search(batches, 2, forward_step=nonfinite)

    for coverage in ("missing", "shape"):

        def inconsistent_scales(model, batch):
            quantizer = model.model.layers[0].self_attn.k_bmm_quantizer
            if rank == 0 and coverage == "missing":
                quantizer.disable_calib()
            output = _logits(model, batch)
            if rank == 0 and coverage == "shape" and quantizer._calibrator._calib_amax is not None:
                quantizer._calibrator._calib_amax = quantizer._calibrator._calib_amax.reshape(1)
            return output

        with pytest.raises(
            ValueError, match="matching candidate scale presence, shapes and dtypes"
        ):
            _search(batches[:1], 1, forward_step=inconsistent_scales)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("backend", ["gloo", "cpu:gloo"])
def test_distributed_kv_search_and_resume_matches_global_data(tmp_path, backend):
    expected = _search(_batches(), 4)
    pool = DistributedWorkerPool(world_size=2, backend=backend)
    try:
        pool.run(partial(_distributed_search, directory=tmp_path / "search", expected=expected))
    finally:
        pool.shutdown()
