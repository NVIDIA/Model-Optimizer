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

"""A layerwise export, which writes the checkpoint layer by layer as calibration runs, in the
workflow's ModelOptSession."""

import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from _test_utils.torch.export.utils import layerwise_fp8_config
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.quantization as mtq
from modelopt.torch.export import layerwise_export, unified_export_hf
from modelopt.torch.export.layerwise_export import LAYERWISE_EXPORTER_ATTR, LayerwiseExporter
from modelopt.torch.export.registry import ExportContext
from modelopt.torch.opt import ModelOptSession
from modelopt.torch.utils.plugins.hf_checkpoint_utils import SourceCheckpoint


def test_a_layerwise_export_runs_in_the_workflows_modelopt_session(tmp_path):
    model = get_tiny_llama(num_hidden_layers=1)
    with ModelOptSession().bind(model) as session:
        exporter = LayerwiseExporter(model, tmp_path)
        exporter.bind(list(model.model.layers))

        # From the binding, the workflow's session holds the layerwise export's context, and the
        # model's type.
        assert ExportContext.current().model is model
        assert session.model_type == "llama"

    # A calibration resumed in another session, of a part of the model, binds it again: that
    # session takes the root the checkpoint describes.
    with ModelOptSession().bind(model.model) as resumed:
        exporter.bind(list(model.model.layers))
        assert resumed.model is model


def test_a_layerwise_export_that_cannot_bind_leaves_no_export_in_progress(tmp_path, monkeypatch):
    """The exporter starts its export, and binds the workflow's session to its model, once nothing
    else in binding it can fail: a session bound to the part calibration was handed stays there."""

    def unreadable(*args, **kwargs):
        raise RuntimeError("unreadable quantizer state")

    monkeypatch.setattr(layerwise_export, "get_quant_config", unreadable)
    model = get_tiny_llama(num_hidden_layers=1)
    with ModelOptSession().bind(model.model) as session:
        with pytest.raises(RuntimeError, match="unreadable"):
            LayerwiseExporter(model, tmp_path).bind(list(model.model.layers))
        assert session.runtime.export_context is None
        assert session.model is model.model


def test_a_layerwise_export_finishes_outside_any_workflow_session(tmp_path):
    """A pipeline that establishes no ModelOpt session calibrates layerwise, then finalizes the
    exporter calibration left on the model, which runs in a session of its own. Until then the
    model, which holds the pending export, can still be copied."""
    model = get_tiny_llama(num_hidden_layers=2).eval()
    model.config.architectures = ["LlamaForCausalLM"]
    mtq.quantize(model, layerwise_fp8_config(tmp_path), lambda m: m(torch.randint(0, 32, (1, 16))))

    copy.deepcopy(model)
    getattr(model, LAYERWISE_EXPORTER_ATTR).finalize()
    assert {"model-tail.safetensors", "model.safetensors.index.json"} <= {
        path.name for path in tmp_path.iterdir()
    }


def test_a_layerwise_export_reads_the_type_of_the_model_it_writes(tmp_path):
    """The exporter's model is the root its checkpoint describes, which a pipeline may calibrate
    through a part, as a VLM through its language model. Calibrating the part outside any workflow
    session reads the root's model type, as the export of the root does: here it refuses, before
    calibrating, a block of the root that only its type's spec names an MoE block, whose experts no
    export handler takes."""

    class DeepseekV3MoE(nn.Module):  # an MoE block by deepseek_v3's spec, not by its name
        def __init__(self):
            super().__init__()
            self.experts = nn.Linear(4, 4)

    root = get_tiny_llama(num_hidden_layers=1).eval()
    root.config.architectures = ["LlamaForCausalLM"]
    # The calibrated part keeps a llama config of its own; the root's type is another.
    root.model.config = copy.deepcopy(root.config)
    root.config.model_type = "deepseek_v3"
    root.moe = DeepseekV3MoE()
    setattr(root.model, LAYERWISE_EXPORTER_ATTR, LayerwiseExporter(root, tmp_path))

    with pytest.raises(NotImplementedError, match="experts type 'Linear'"):
        mtq.quantize(
            root.model,
            layerwise_fp8_config(tmp_path),
            lambda m: m(torch.randint(0, 32, (1, 16))),
        )


def test_a_layerwise_export_of_a_part_is_refused_for_the_whole_model(tmp_path):
    """A workflow bound to a VLM that calibrates only its language model layerwise, building no
    exporter on the VLM first, holds an exporter of the language model: the VLM's export refuses it
    rather than write the language model's checkpoint as the VLM's."""
    language_model = get_tiny_llama(num_hidden_layers=1).eval()
    language_model.config.architectures = ["LlamaForCausalLM"]
    vlm = nn.Module()
    vlm.language_model = language_model
    vlm.vision_tower = nn.Linear(4, 4)
    with ModelOptSession().bind(vlm):
        mtq.quantize(
            language_model,
            layerwise_fp8_config(tmp_path / "export"),
            lambda m: m(torch.randint(0, 32, (1, 16))),
        )
        with pytest.raises(ValueError, match="neither, nor a part"):
            unified_export_hf.export_hf_checkpoint(vlm, export_dir=tmp_path / "vlm")
    assert not (tmp_path / "export" / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("fails_at", ["binding", "calibration"])
def test_a_layerwise_export_that_fails_is_not_the_workflows(tmp_path, fails_at):
    """A layerwise export that refuses the model as calibration binds it, or whose calibration
    fails, is not held by the workflow's session, so calibrating again, with an exporter built on
    the model first as hf_ptq does, writes through that one."""
    model = get_tiny_llama(num_hidden_layers=1).eval()
    model.config.architectures = ["LlamaForCausalLM"]
    cfg = layerwise_fp8_config(tmp_path / "failed")

    def calibrate(m):
        m(torch.randint(0, 32, (1, 16)))

    def fail(m):
        raise RuntimeError("the calibration failed")

    if fails_at == "binding":  # INT8, which the layerwise export does not write
        cfg = {**copy.deepcopy(mtq.INT8_DEFAULT_CFG), "algorithm": cfg["algorithm"]}
        first, error = calibrate, NotImplementedError
    else:
        first, error = fail, RuntimeError
    with ModelOptSession() as session:
        with pytest.raises(error):
            mtq.quantize(model, cfg, first)
        assert session.runtime.layerwise_exporter is None
        LayerwiseExporter(model, tmp_path / "export")
        mtq.quantize(model, layerwise_fp8_config(tmp_path / "export"), calibrate)
        unified_export_hf.export_hf_checkpoint(model, export_dir=tmp_path / "export")
    assert (tmp_path / "export" / "model.safetensors.index.json").exists()


def test_a_layerwise_export_resumes_in_a_later_calibration_outside_any_session(
    tmp_path, monkeypatch
):
    """A layerwise calibration that failed part-way resumes from its checkpoint in a later call,
    outside any workflow session too: the export the first call started goes on in the session of
    the second, and the exporter left on the model finalizes it."""
    model = get_tiny_llama(num_hidden_layers=3).eval()
    model.config.architectures = ["LlamaForCausalLM"]
    cfg = layerwise_fp8_config(tmp_path / "export")
    cfg["algorithm"]["layerwise"]["checkpoint_dir"] = str(tmp_path / "checkpoint")
    export_layer = LayerwiseExporter.export_layer

    def run_out_of_memory_at_layer_1(self, layer_idx, *args, **kwargs):
        if layer_idx == 1:
            monkeypatch.setattr(LayerwiseExporter, "export_layer", export_layer)
            raise RuntimeError("out of memory")
        return export_layer(self, layer_idx, *args, **kwargs)

    monkeypatch.setattr(LayerwiseExporter, "export_layer", run_out_of_memory_at_layer_1)
    with pytest.raises(RuntimeError, match="out of memory"):
        mtq.quantize(model, cfg, lambda m: m(torch.randint(0, 32, (1, 16))))
    mtq.calibrate(model, cfg["algorithm"], lambda m: m(torch.randint(0, 32, (1, 16))))
    getattr(model, LAYERWISE_EXPORTER_ATTR).finalize()
    assert (tmp_path / "export" / "model.safetensors.index.json").exists()


def test_a_layerwise_export_takes_the_source_its_workflow_recorded(tmp_path):
    """The source checkpoint the workflow's session holds wins over the one the model names, as it
    does for the whole-model export: its non-model files are copied, though the model names a Hub
    ID."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text("{}")
    model = get_tiny_llama(num_hidden_layers=1).eval()
    model.config.architectures = ["LlamaForCausalLM"]
    model.config._name_or_path = "org/some-model"
    with ModelOptSession(source=SourceCheckpoint(str(source))) as session:
        mtq.quantize(
            model,
            layerwise_fp8_config(tmp_path / "export"),
            lambda m: m(torch.randint(0, 32, (1, 16))),
        )
        getattr(model, LAYERWISE_EXPORTER_ATTR).finalize()
        # Finalized, the exporter is let go of, by the session too.
        assert session.runtime.layerwise_exporter is None
    assert (tmp_path / "export" / "tokenizer_config.json").exists()
