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

import ast
import importlib
import inspect
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_qwen3
from transformers import AutoModelForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml.common import pinned_importance, pinned_packed_weight

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


def _context_manager_calls(func):
    """Return the dotted names of every context manager entered by ``with`` in ``func``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                expr = item.context_expr
                target = expr.func if isinstance(expr, ast.Call) else expr
                if isinstance(target, (ast.Attribute, ast.Name)):
                    names.append(ast.unparse(target))
    return names


def test_export_quantized_does_not_use_inference_mode(monkeypatch):
    """``export_quantized`` must not run under ``torch.inference_mode()``.

    On the FSDP2 path (``--use_fsdp2``, multi-node) the export gathers the full params
    via ``get_model_state_dict(full_state_dict=True)`` inside this context. Tensors
    allocated under ``inference_mode`` are inference tensors whose version counter
    cannot be set, so the subsequent ``state_dict()`` -> ``param.detach()`` fails with
    ``RuntimeError: Cannot set version_counter for inference tensor``. ``torch.no_grad()``
    disables autograd just the same but keeps the gathered params as normal tensors.
    """
    hf_ptq = _import_hf_ptq(monkeypatch)
    contexts = _context_manager_calls(hf_ptq.export_quantized)

    assert "torch.inference_mode" not in contexts
    assert "torch.no_grad" in contexts


@pytest.mark.parametrize(("num_bits", "gptq"), [(8, False), ("iq2_xxs", False), ("iq2_xxs", True)])
def test_output_qdq_weight_hf_roundtrip(monkeypatch, tmp_path, num_bits, gptq):
    """PTQ writes restorable QDQ weights; the first QAD forward and no-op update retain them."""
    hf_ptq = _import_hf_ptq(monkeypatch)
    model = get_tiny_qwen3(
        num_hidden_layers=1,
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    ).eval()
    source = tmp_path / "source"
    model.save_pretrained(source, state_dict={**model.state_dict(), "mtp.weight": torch.ones(2, 2)})
    model._modelopt_source_checkpoint = str(source)
    model._modelopt_unplaced_source_keys = ["mtp.weight"]
    inputs = torch.arange(8).reshape(1, -1)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "enable": True,
                "cfg": {
                    "num_bits": num_bits,
                    **(
                        {"backend": "ggml", "block_sizes": {-1: 256}}
                        if isinstance(num_bits, str)
                        else {"axis": 0}
                    ),
                },
            },
        ],
        "algorithm": {
            "method": "gptq",
            "block_size": 256,
            "perc_damp": 0.3,
            "importance_weighted": True,
        }
        if gptq
        else "max",
    }
    mtq.quantize(model, config, lambda m: m(inputs))
    if gptq:
        # Embeddings are fake-quantized, but GPTQ only calibrates supported linear layers.
        assert pinned_packed_weight(model.model.embed_tokens.weight_quantizer, num_bits) is None
        linears = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        assert linears
        for layer in linears:
            assert pinned_packed_weight(layer.weight_quantizer, num_bits) is not None
            assert pinned_importance(layer.weight_quantizer, num_bits).dtype == torch.float32
    expected = {
        name: layer.weight_quantizer(layer.weight).detach().clone()
        for name, layer in model.named_modules()
        if hasattr(layer, "weight_quantizer")
    }
    assert any(
        not torch.equal(dict(model.named_modules())[name].weight, weight)
        for name, weight in expected.items()
    )
    before = model(inputs).logits.detach()
    buffers = {
        name: tensor.clone()
        for name, tensor in model.named_buffers()
        if ".weight_quantizer." in name
    }
    args = SimpleNamespace(
        output_qdq_weight=True,
        export_path=tmp_path / "qdq",
        local_checkpoint_path=source,
        dist_state=SimpleNamespace(is_main=True),
    )
    hf_ptq.export_quantized(args, model, model, "qwen3")
    assert args.checkpoint_exported and (args.export_path / "modelopt_state.pth").is_file()
    index = json.loads((args.export_path / "model.safetensors.index.json").read_text())
    assert "mtp.weight" in index["weight_map"]
    restored = AutoModelForCausalLM.from_pretrained(args.export_path).eval()
    for name, weight in expected.items():
        layer = dict(restored.named_modules())[name]
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        assert layer.weight_quantizer.is_enabled
    for name, tensor in buffers.items():
        actual = dict(restored.named_buffers())[name]
        assert actual.dtype == tensor.dtype
        torch.testing.assert_close(actual, tensor, rtol=0, atol=0)
    torch.testing.assert_close(restored(inputs).logits, before, rtol=0, atol=0)
    optimizer = torch.optim.SGD(restored.parameters(), lr=0)
    for parameter in restored.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.testing.assert_close(restored(inputs).logits, before, rtol=0, atol=0)
