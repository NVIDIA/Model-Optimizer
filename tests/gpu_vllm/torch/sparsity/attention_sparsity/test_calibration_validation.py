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

"""Validation records the counters from the paged serving kernels themselves."""

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.sparsity.attention_sparsity.plugins.vllm import _forward_modelopt


@pytest.mark.parametrize("decode", [False, True])
def test_serving_measurement_does_not_change_output(decode):
    torch.manual_seed(10)
    impl = SimpleNamespace()
    impl.scale, impl.num_kv_heads, impl.head_size = 0.125, 2, 64
    impl.sparse_kw = {"skip_softmax_threshold": 0.5}
    q_len, kv_len = (1 if decode else 256), 512
    q = torch.randn(q_len, 4, 64, device="cuda", dtype=torch.bfloat16).abs()
    k = torch.randn(32, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    k[0] = 10  # A dominant first page makes later tiles skippable.
    v = torch.randn_like(k)
    args = {
        "query": q,
        "key_cache": k,
        "value_cache": v,
        "layer": SimpleNamespace(),
        "cu_seqlens_q": torch.tensor([0, q_len], device="cuda", dtype=torch.int32),
        "seq_lens": torch.tensor([kv_len], device="cuda", dtype=torch.int32),
        "block_table": torch.arange(32, device="cuda", dtype=torch.int32)[None],
        "num_actual_tokens": q_len,
        "max_query_len": q_len,
        "max_seq_len": kv_len,
        "p_qdq": None,
        "p_qdq_amax": 1.0,
        "v_qdq": None,
        "v_qdq_amax": None,
        "quant_active": False,
        "is_causal": not decode,
        "dense_fallback": lambda: pytest.fail("unexpected dense fallback"),
    }
    baseline = _forward_modelopt(impl, output=torch.empty_like(q), **args)
    impl._sparse_validation_stats = {}
    measured = _forward_modelopt(impl, output=torch.empty_like(q), **args)
    torch.testing.assert_close(measured, baseline, rtol=0, atol=0)
    phase = "decode" if decode else "prefill"
    record = impl._sparse_validation_stats[phase]
    assert record["total"] > 0
    assert 0 <= record["skipped"] < record["total"]
    assert record["skipped"] > 0
    if decode:
        assert record["total"] == q.shape[1] * (kv_len // 128)
    assert record["launches"] == 1
    assert record["unmeasured_launches"] == 0
    # A calibrated threshold outside the serving domain must not silently pass.
    impl.sparse_kw = {
        "target_sparse_ratio": 0.5,
        "threshold_scale_factor": {phase: {"a": 2 * kv_len, "b": 0.0}},
    }
    args["dense_fallback"] = lambda: torch.zeros_like(q)
    with pytest.warns(UserWarning, match="Disabling calibrated skip-softmax"):
        _forward_modelopt(impl, output=torch.empty_like(q), **args)
    assert impl._sparse_validation_stats[phase]["unmeasured_launches"] == 1
