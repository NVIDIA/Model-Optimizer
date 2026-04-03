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
"""Test parallel vs sequential export produces identical outputs."""

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_vllm_fq_checkpoint


def _quantize_model(tmp_path, suffix=""):
    """Create and quantize a tiny LLaMA model. Returns (model, export_dir)."""
    tiny_model_dir = create_tiny_llama_dir(tmp_path / f"model{suffix}", num_hidden_layers=4)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    model = model.cuda()
    model.eval()

    def forward_loop(model):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128)).cuda()
        with torch.no_grad():
            model(input_ids)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    return model


@pytest.mark.parametrize("quant_cfg", [mtq.FP8_DEFAULT_CFG])
def test_parallel_vs_sequential_identical(tmp_path, quant_cfg):
    """Verify parallel export produces bitwise identical output to sequential."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        pytest.skip("Need >= 2 GPUs for parallel export test")

    # Create a tiny model and spread across GPUs.
    tiny_model_dir = create_tiny_llama_dir(tmp_path / "model", num_hidden_layers=4)
    model = AutoModelForCausalLM.from_pretrained(
        tiny_model_dir, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()

    def forward_loop(model):
        first_device = next(model.parameters()).device
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128)).to(first_device)
        with torch.no_grad():
            model(input_ids)

    model = mtq.quantize(model, quant_cfg, forward_loop)

    # Export sequentially.
    seq_dir = tmp_path / "export_sequential"
    export_hf_vllm_fq_checkpoint(model, export_dir=seq_dir, parallel=False)

    # Re-enable weight quantizers (export disables them — need to restore for second export).
    # The function already re-enables them at the end, so we can just call it again.

    # Export in parallel.
    par_dir = tmp_path / "export_parallel"
    export_hf_vllm_fq_checkpoint(model, export_dir=par_dir, parallel=True)

    # Compare HF weights.
    seq_model = AutoModelForCausalLM.from_pretrained(seq_dir)
    par_model = AutoModelForCausalLM.from_pretrained(par_dir)
    seq_sd = seq_model.state_dict()
    par_sd = par_model.state_dict()

    assert seq_sd.keys() == par_sd.keys(), "Key mismatch between sequential and parallel export"
    for key in seq_sd:
        assert torch.equal(seq_sd[key], par_sd[key]), (
            f"Weight mismatch for {key}: max diff={torch.abs(seq_sd[key] - par_sd[key]).max()}"
        )

    # Compare full modelopt state payload (weights_only=False: modelopt_state contains
    # Python objects — dicts, strings, quantizer configs — that require unpickling).
    seq_state = torch.load(seq_dir / "vllm_fq_modelopt_state.pth", weights_only=False)
    par_state = torch.load(par_dir / "vllm_fq_modelopt_state.pth", weights_only=False)

    # Compare modelopt_state_dict (quantizer metadata including quantizer_state).
    seq_msd = seq_state.get("modelopt_state_dict", [])
    par_msd = par_state.get("modelopt_state_dict", [])
    assert len(seq_msd) == len(par_msd), "modelopt_state_dict length mismatch"
    for (seq_mode, seq_ms), (par_mode, par_ms) in zip(seq_msd, par_msd):
        assert seq_mode == par_mode, f"Mode mismatch: {seq_mode} vs {par_mode}"

    # Compare modelopt_state_weights (per-quantizer tensor state).
    seq_qsd = seq_state["modelopt_state_weights"]
    par_qsd = par_state["modelopt_state_weights"]
    assert seq_qsd.keys() == par_qsd.keys(), "Quantizer state dict key mismatch"
    for key in seq_qsd:
        seq_val = seq_qsd[key]
        par_val = par_qsd[key]
        if isinstance(seq_val, dict):
            for k in seq_val:
                if isinstance(seq_val[k], torch.Tensor):
                    assert torch.equal(seq_val[k], par_val[k]), (
                        f"Quantizer state mismatch for {key}.{k}"
                    )
                else:
                    assert seq_val[k] == par_val[k], f"Quantizer state mismatch for {key}.{k}"
        else:
            assert seq_val == par_val, f"Quantizer state mismatch for {key}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_gpu_fallback(tmp_path):
    """Verify parallel=True gracefully falls back to sequential on single GPU."""
    tiny_model_dir = create_tiny_llama_dir(tmp_path / "model", num_hidden_layers=2)
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    model = model.cuda()  # All on cuda:0
    model.eval()

    def forward_loop(model):
        input_ids = torch.randint(0, model.config.vocab_size, (1, 128)).cuda()
        with torch.no_grad():
            model(input_ids)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    # parallel=True but single GPU → should fall back to sequential without error.
    export_dir = tmp_path / "export"
    export_hf_vllm_fq_checkpoint(model, export_dir=export_dir, parallel=True)

    assert (export_dir / "vllm_fq_modelopt_state.pth").exists()
    reloaded = AutoModelForCausalLM.from_pretrained(export_dir)
    assert reloaded is not None
