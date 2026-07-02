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

"""End-to-end GPU test: unified HF export produces original-hub-aligned tensor names.

Uses a tiny Mixtral, whose transformers ``conversion_mapping`` fuses/renames MoE
experts (``block_sparse_moe.experts.*.w{1,2,3}`` <-> in-memory
``mlp.experts.gate_up_proj``) — the same machinery larger MoE VLMs (e.g. MiniMax-M3)
use. The exported quantized checkpoint's tensor names must match the canonical hub
names obtained from transformers' own ``revert_weight_conversion`` on the reference
(unquantized) model.
"""

import glob
import os
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".weight_scale_inv", ".input_scale")


def _tiny_mixtral_config():
    from transformers import MixtralConfig

    cfg = MixtralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=320,
        max_position_embeddings=128,
    )
    cfg.architectures = ["MixtralForCausalLM"]
    return cfg


def test_export_tensor_names_match_hub_after_conversion_reverse():
    pytest.importorskip("transformers")
    from transformers import MixtralForCausalLM

    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import revert_weight_conversion
    except ImportError:
        pytest.skip("transformers build has no conversion_mapping API")
    if not get_checkpoint_conversion_mapping("mixtral"):
        pytest.skip("transformers build has no mixtral conversion_mapping")

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    cfg = _tiny_mixtral_config()

    # Canonical hub names: transformers' own reverse on the unquantized reference.
    ref = MixtralForCausalLM(cfg)
    hub_names = set(revert_weight_conversion(ref, ref.state_dict()).keys())
    # sanity: reference really is fused/renamed in memory
    assert any(".block_sparse_moe.experts.0.w1.weight" in n for n in hub_names)

    model = MixtralForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")

    def forward_loop(m):
        for _ in range(4):
            m(ids)

    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    with tempfile.TemporaryDirectory() as export_dir:
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=export_dir)
        exported = set()
        for f in glob.glob(os.path.join(export_dir, "*.safetensors")):
            from safetensors import safe_open

            with safe_open(f, framework="pt") as sf:
                exported.update(sf.keys())

    non_scale = {k for k in exported if not any(k.endswith(s) for s in _SCALE_SUFFIXES)}
    # Every exported weight carries its original hub name; nothing renamed/left in-memory.
    assert non_scale == hub_names, (
        f"missing={sorted(hub_names - non_scale)[:5]} extra={sorted(non_scale - hub_names)[:5]}"
    )
    # Experts specifically use the hub layout, not the fused in-memory names.
    assert any(".block_sparse_moe.experts.0.w1.weight" in k for k in non_scale)
    assert not any(".mlp.experts.gate_up_proj" in k for k in exported)
