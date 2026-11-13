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
import textwrap
from pathlib import Path

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
