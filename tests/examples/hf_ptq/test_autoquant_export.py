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

"""Offline export transactions through the real HF exporter."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _test_utils.torch.transformers_models import create_tiny_llama_dir
from transformers import AutoModelForCausalLM


@pytest.mark.skipif(sys.platform == "win32", reason="Automatic workers require POSIX locks")
def test_export_budget_recovery_and_resume(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/hf_ptq"))
    exports = importlib.import_module("autoquant_export")
    workspace = importlib.import_module("autoquant_workspace").PipelineWorkspace(
        tmp_path / "work", "search", 65536
    )
    source = create_tiny_llama_dir(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(source)
    assignment = workspace.root / "assignment.json"
    workspace.publish(assignment, json.dumps({"source": str(source), "formats": ["BF16"]}))
    args = SimpleNamespace(
        export_path=str(tmp_path / "export"),
        auto_quantize_assignment=str(assignment),
        auto_quantize_resources={
            "export_bytes": 2 * 1024**2,
            "metadata_bytes": 65536,
            "host_bytes": 1024**3,
        },
        auto_quantize_export_available_bytes=2 * 1024**2,
        checkpoint_exported=False,
    )
    with workspace.run(0, 1):
        with pytest.raises(ValueError), workspace.run(1, 1):
            pytest.fail("A second worker exceeded the lease limit")
        with pytest.raises(BlockingIOError), workspace.run(None, 1):
            pytest.fail("Coordinator overlapped a worker")
    # An interrupted export owns a durable reservation and a private, disposable stage.
    with workspace.run(None, 1), exports.export_lease(args, workspace):
        with pytest.raises(BlockingIOError), workspace.run(0, 1):
            pytest.fail("Worker overlapped a coordinator")
        stage = Path(args.auto_quantize_export_stage)
        stage.mkdir()
        (stage / "interrupted.part").write_bytes(b"partial")
        (source / "oversized.txt").write_bytes(b"x" * args.auto_quantize_resources["export_bytes"])
        with pytest.raises(ValueError):
            exports.export_pipeline(args, model)
        assert not stage.exists() and not Path(args.export_path).exists()
        assert not args.checkpoint_exported
        (source / "oversized.txt").unlink()
        stage.mkdir()
        (stage / "interrupted.part").write_bytes(b"partial")
        exports.export_pipeline(args, model)
        assert args.checkpoint_exported and not stage.exists()
    restored = AutoModelForCausalLM.from_pretrained(args.export_path)
    for key, weight in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], weight, rtol=0, atol=0)
    destination = Path(args.export_path)
    original = {
        p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()
    }
    with workspace.run(None, 1), exports.export_lease(args, workspace):
        exports.export_pipeline(args, model)
        assert original == {
            p.relative_to(destination): p.read_bytes()
            for p in destination.rglob("*")
            if p.is_file()
        }
        weights = destination / "model.safetensors"
        with weights.open("r+b") as stream:
            stream.seek(-1, 2)
            previous = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([previous[0] ^ 1]))
        with pytest.raises(ValueError):
            exports.export_pipeline(args, model)
    args.export_path = str(tmp_path / "another-budget")
    with pytest.raises(ValueError), exports.export_lease(args, workspace):
        pytest.fail("A second retained export exceeded the shared storage reservation")
    assert not Path(args.export_path).exists()
