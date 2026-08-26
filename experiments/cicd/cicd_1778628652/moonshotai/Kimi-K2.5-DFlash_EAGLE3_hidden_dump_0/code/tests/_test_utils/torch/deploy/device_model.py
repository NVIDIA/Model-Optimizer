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

from contextlib import nullcontext
from unittest import mock

import pytest
import torch
from _test_utils.torch.deploy.lib_test_models import BaseDeployModel

from modelopt.torch._deploy import compile
from modelopt.torch._deploy.utils.torch_onnx import _to_expected_onnx_type
from modelopt.torch.utils import flatten_tree, standardize_model_args


def device_model_tester(model: BaseDeployModel, deployment: dict[str, str]):
    deployment = {"runtime": "ORT", "verbose": "false"}

    # Helper
    def _compare(out_torch, out_local):
        # flatten
        ot_vals, ot_spec = flatten_tree(out_torch)
        ol_vals, ol_spec = flatten_tree(out_local)

        # ensure ot is also torch ...
        print(ot_vals, ol_vals)
        ot_vals = [_to_expected_onnx_type(v) for v in ot_vals]

        # compare tree specs
        assert ot_spec == ol_spec

        # compare values
        print(ot_vals, ol_vals)
        assert all(torch.allclose(ot, ol) for ot, ol in zip(ot_vals, ol_vals))

    # try it for all potential numeric types
    for active in range(model.get.num_choices):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            device_model = compile(model, args, deployment)

        if model.compile_fail:
            continue

        # use an args/kwargs-style forward here
        args_kw = standardize_model_args(model, args, use_kwargs=True)

        # run regular forward
        out_torch = model(*args_kw[:-1], **args_kw[-1])

        # run device forward with args/kwargs and args-only
        out_local = device_model(*args_kw[:-1], **args_kw[-1])
        out_local2 = device_model(*standardize_model_args(model, args))

        if model.invalid_device_input:
            continue

        # compare outputs
        _compare(out_torch, out_local)
        _compare(out_torch, out_local2)

        # profile model
        with mock.patch(
            "modelopt.torch._deploy._runtime.ort_client.ORTLocalClient._profile_defaults",
            new_callable=mock.PropertyMock,
        ) as mock_defaults:
            mock_defaults.return_value = {"iterations": 1, "warm_up": 0.0, "duration": 0.0}
            latency, detailed_results = device_model.profile()
        assert latency > 0.0, "Latency must be positive"
        assert isinstance(detailed_results, dict) and len(detailed_results) > 0, (
            "Detailed results must be a non-empty dict"
        )
