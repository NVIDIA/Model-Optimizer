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

import torch


# Models
class ToyModel(torch.nn.Module):
    def __init__(self, dims=[10, 10, 10, 10], bias=True):
        super().__init__()
        assert len(dims) >= 2
        if len(dims) == 2:
            self.linears = torch.nn.Linear(dims[0], dims[1], bias=bias)
        else:
            linears = [
                torch.nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(len(dims) - 1)
            ]
            self.linears = torch.nn.Sequential(*linears)

    def forward(self, x):
        return self.linears(x)


class SmallLinearModelwithCustomWeight(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.q_proj = torch.nn.Linear(weights[0].shape[1], weights[0].shape[0], bias=False)
        self.k_proj = torch.nn.Linear(weights[1].shape[1], weights[1].shape[0], bias=False)
        self.v_proj = torch.nn.Linear(weights[2].shape[1], weights[2].shape[0], bias=False)
        self.o_proj = torch.nn.Linear(weights[3].shape[1], weights[3].shape[0], bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(weights[0])
            self.k_proj.weight.copy_(weights[1])
            self.v_proj.weight.copy_(weights[2])
            self.o_proj.weight.copy_(weights[3])

    def forward(self, x):
        x = self.q_proj(x)
        x = self.k_proj(x)
        x = self.v_proj(x)
        x = self.o_proj(x)
        return x


class SmallQKVModel(torch.nn.Module):
    def __init__(self, dim=4, device="cuda", apply_embed=False, bias=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(2, dim)
        self.q_proj = torch.nn.Linear(dim, dim, bias=bias)
        self.k_proj = torch.nn.Linear(dim, dim, bias=bias)
        self.v_proj = torch.nn.Linear(dim, dim, bias=bias)
        self.o_proj = torch.nn.Linear(dim, dim, bias=bias)
        self.device = device
        self.config = None
        self.apply_embed = apply_embed
        # TODO: Debug why fsdp2 modifies bias of layernorm for awq
        self.input_layernorm = torch.nn.LayerNorm(dim, bias=bias)

    def forward(self, x):
        if self.apply_embed:
            x = self.embedding(x)

        x = self.input_layernorm(x)
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        x = torch.matmul(attn, v_proj)
        o_proj = self.o_proj(x)
        return o_proj


# Quantization configs
partial_fp8_config = {
    "quant_cfg": {
        "*.1.weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.1.input_quantizer": {"num_bits": (4, 3), "axis": None},
        "default": {"num_bits": 8, "enable": False},
    },
    "algorithm": "max",
}

partial_w4a8_config = {
    "quant_cfg": {
        "*.2.weight_quantizer": [
            {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}, "enable": True},
            {"num_bits": (4, 3), "axis": None, "enable": True},
        ],
        "*.2.input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "default": {"num_bits": 8, "enable": False},
    },
    "algorithm": "awq_lite",
}

partial_nvfp4_config = {
    "quant_cfg": {
        "*.1.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*.1.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

partial_nvfp4_awq_config = {
    "quant_cfg": {
        "*.2.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*.2.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*.1.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        },
        "*.1.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        },
        "default": {"enable": False},
    },
    "algorithm": "awq_lite",
}

partial_int4_awq_config = {
    "quant_cfg": {
        "*.2.weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {-1: 128, "type": "static"},
            "enable": True,
        },
        "*.2.input_quantizer": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
    # "algorithm": {"method": "awq_full", "alpha_step": 0.1, "max_co_batch_size": 1024},
    # "algorithm": {"method": "awq_clip", "max_co_batch_size": 2048},
}

partial_fp8_kv_cache_config = {
    "quant_cfg": {
        "*.1.weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.1.input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

partial_int8_kv_cache_config = {
    "quant_cfg": {
        "*.1.weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.1.input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"num_bits": 8, "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

partial_nvfp4_kv_cache_config = {
    "quant_cfg": {
        "*.1.weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*.1.input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*[kv]_bmm_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

only_weight_quantizer_fp8_config = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": False},
        "*output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}
only_input_quantizer_fp8_config = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "enable": False},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "*output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}
only_output_quantizer_fp8_config = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "enable": False},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "enable": False},
        "*output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "max",
}
