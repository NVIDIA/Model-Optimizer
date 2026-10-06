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

"""CPU schedule/grouping demonstration; does not execute native vLLM kernels."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import modelopt
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.linear_attention import (
    LinearAttentionDecodeConfig,
    recurrent_decode,
)
from modelopt.torch.quantization.nn import TensorQuantizer


def make_quantizer():
    return TensorQuantizer(
        QuantizerAttributeConfig(
            num_bits=8,
            type="dynamic",
            axis=(0, 1),
            unsigned=False,
            narrow_range=True,
            pass_through_bwd=True,
        )
    )


def reduce_keys(x):
    # Same reduction order isolates QDQ timing/grouping from arithmetic differences.
    while x.shape[-2] > 1:
        half = x.shape[-2] // 2
        x = x[..., :half, :] + x[..., half:, :]
    return x[..., 0, :]


def run(kda, block_v, readout):
    torch.manual_seed(514)
    length, heads, keys, values = 16, 2, 16, 128
    q, k = [
        torch.nn.functional.normalize(torch.randn(length, heads, keys), dim=-1) for _ in range(2)
    ]
    v = torch.randn(length, heads, values)
    g = -torch.rand((length, heads, keys) if kda else (length, heads)) * 0.04
    beta = torch.rand(length, heads) * 0.4
    initial = torch.randn(heads, keys, values)
    initial[..., :64] *= 0.05
    initial[..., 64:] *= 4
    train_q, serve_q = make_quantizer(), make_quantizer()
    config = LinearAttentionDecodeConfig(readout=readout, quantize_initial=True)
    carry, raw_state = None, initial.clone()
    max_output_error = max_consumed_state_error = 0.0
    for t in range(length):
        args = [x[t : t + 1] for x in (q, k, v, g, beta)]
        output, carry = recurrent_decode(
            *args,
            config=config,
            state_quantizer=train_q,
            block_v=block_v,
            initial_state=initial if carry is None else None,
            carry=carry,
        )
        consumed = serve_q(raw_state.unsqueeze(0)).squeeze(0)
        decay = g[t].exp().unsqueeze(-1)
        if not kda:
            decay = decay.unsqueeze(-1)
        decayed = consumed * decay
        residual = v[t] - reduce_keys(k[t].unsqueeze(-1) * decayed)
        update = beta[t].unsqueeze(-1) * residual
        raw_state = decayed + k[t].unsqueeze(-1) * update.unsqueeze(-2)
        served_output = reduce_keys(q[t].unsqueeze(-1) * raw_state) * keys**-0.5
        # Compare at a common semantic boundary, not raw cache representations.
        next_consumed = serve_q(raw_state.unsqueeze(0)).squeeze(0)
        max_output_error = max(max_output_error, (output[0] - served_output).abs().max().item())
        max_consumed_state_error = max(
            max_consumed_state_error, (carry.anchor.values - next_consumed).abs().max().item()
        )
    return {
        "kind": "KDA" if kda else "GDN",
        "block_v": block_v,
        "readout": readout,
        "max_output_error": max_output_error,
        "max_consumed_state_error": max_consumed_state_error,
    }


def main():
    """Compare training state writes with a CPU reference of serving state reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()
    rows = [
        run(kda, block_v, readout)
        for kda in (False, True)
        for block_v, readout in ((128, "working"), (128, "stored"), (64, "working"))
    ]
    source_paths = (
        "torch/quantization/linear_attention/config.py",
        "torch/quantization/linear_attention/decode.py",
        "torch/quantization/linear_attention/utils.py",
        "torch/quantization/nn/modules/tensor_quantizer.py",
        "torch/quantization/tensor_quant.py",
    )
    report = {
        "scope": "CPU INT8 QDQ schedule reference; no native vLLM execution",
        "device": "cpu",
        "dtype": "float32",
        "torch_version": torch.__version__,
        "seed": 514,
        "shape": {"T": 16, "H": 2, "K": 16, "V": 128},
        "source_sha256": {
            "modelopt/" + name: hashlib.sha256(
                (Path(modelopt.__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in source_paths
        },
        "results": rows,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
