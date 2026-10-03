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

"""Run a reproducible, synthetic head-only search without downloading a model."""

import argparse
import copy
import json

import torch
from torch import nn

import modelopt.torch.quantization as mtq
from experimental.softmax_reparameterization import head_kl, search_head


def main():
    """Compare selected and unshifted heads on disjoint synthetic test states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--algorithm", choices=["max", "gptq"], default="max")
    args = parser.parse_args()
    torch.manual_seed(0)
    head = nn.Linear(128, 512, bias=False).to(args.device).eval()
    with torch.no_grad():
        head.weight.add_(torch.randn(128, device=args.device) * 0.1)
    fit, validation, test = torch.randn(3, 64, 128, device=args.device).unbind()
    config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    config["algorithm"] = args.algorithm
    result = search_head(head, fit, validation, config)
    baseline = search_head(head, fit, validation, config, coefficients=(0,))
    print(
        json.dumps(
            {
                "coefficient": result.coefficient,
                "validation_kl": result.validation_kl,
                "test_kl_unshifted": head_kl(head, baseline.model, test),
                "test_kl_selected": head_kl(head, result.model, test),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
