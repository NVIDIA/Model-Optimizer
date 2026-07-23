# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
import transformers
from transformers import LlamaForCausalLM

from examples.puzzletron.main import _stage_output_patterns
from modelopt.torch.puzzletron.anymodel.models.llama.llama_model_descriptor import (
    LlamaModelDescriptor,
)
from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.dataset import DataLayout, PuzzletronBatch, batch_from_automodel
from modelopt.torch.puzzletron.diagnostics import width_slice_equivalence as width_slice_module
from modelopt.torch.puzzletron.diagnostics.width_slice_equivalence import (
    WidthSliceCase,
    _runtime_context,
    _RuntimeRecipeAdapter,
    _validate_case_record,
    compare_width_slice_outputs,
    evaluate_width_slice_equivalence,
    normalize_width_slice_batch,
    validate_width_slice_artifacts,
)
from modelopt.torch.puzzletron.identity import canonicalize, stable_hash
from modelopt.torch.puzzletron.manifest import StageManifest, write_stage_manifest
from modelopt.torch.puzzletron.stages import DEFAULT_HANDLERS
from modelopt.torch.puzzletron.stages import diagnostics as diagnostics_stage_module
from modelopt.torch.puzzletron.stages.diagnostics import width_slice_equivalence_stage
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config
from modelopt.torch.puzzletron.utils.data import dataloaders as dataloader_module
from tests._test_utils.torch.transformers_models import (
    create_tiny_llama_dir,
    create_tiny_qwen3_5_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


def _block_configs(num_layers: int) -> list[BlockConfig]:
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=16, num_kv_heads=2),
            FFNConfig(intermediate_size=32),
        )
    )
    return [block for _ in range(num_layers)]


def _tiny_sorted_llama(tmp_path: Path) -> Path:
    checkpoint = create_tiny_llama_dir(tmp_path)
    config = load_model_config(checkpoint)
    LlamaModelDescriptor.set_block_configs(config, _block_configs(config.num_hidden_layers))
    config.block_configs = [block.to_dict() for block in config.block_configs]
    config.save_pretrained(checkpoint)
    return checkpoint


def _text_batch(layout: DataLayout = DataLayout.FIXED):
    collated = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    if layout is DataLayout.PACKED_VARLEN:
        collated.update(
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
            seq_idx=torch.tensor([[0, 0, 1, 1]]),
            max_seqlen=2,
        )
    return batch_from_automodel(
        collated,
        sample_ids=("sample-0",),
        source_metadata={"dataset": "fixture", "revision": "v1"},
        layout=layout,
    )


def _tiny_generic_vlm_checkpoint(tmp_path: Path, *, packed: bool) -> Path:
    checkpoint = _tiny_sorted_llama(tmp_path)
    config = load_model_config(checkpoint)
    config.architectures = ["TinyGenericVLMForCausalLM"]
    config.require_media = True
    config.require_packing = packed
    config.block_configs = [block.to_dict() for block in config.block_configs]
    config.save_pretrained(checkpoint)
    return checkpoint


class _FFNOnlyLlamaDescriptor(LlamaModelDescriptor):
    @classmethod
    def puzzletron_capabilities(cls, config):
        capabilities = super().puzzletron_capabilities(config)
        return replace(
            capabilities,
            descriptor_name="llama-width-equivalence-test",
            axes={"ffn_intermediate": capabilities.axes["ffn_intermediate"]},
        )


class _TinyGenericVLMForCausalLM(LlamaForCausalLM):
    """Tiny generic media/packing-aware model used to prove forwarded batch semantics."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        cu_seqlens=None,
        seq_idx=None,
        max_seqlen=None,
        **kwargs,
    ):
        if getattr(self.config, "require_media", False):
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError("generic VLM media tensors were not forwarded")
        if getattr(self.config, "require_packing", False):
            if cu_seqlens is None or seq_idx is None or max_seqlen is None:
                raise RuntimeError("generic packed metadata was not forwarded")
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )


def _rehash_artifact_json(artifact_dir: Path, record: dict) -> None:
    record["record_hash"] = stable_hash(
        canonicalize({key: value for key, value in record.items() if key != "record_hash"}),
        prefix="width_slice_record",
    )
    case_path = next((artifact_dir / "cases").rglob("*.json"))
    case_path.write_text(json.dumps(record))
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cases"] = [
        record if item["case_identity"] == record["case_identity"] else item
        for item in summary["cases"]
    ]
    summary["case_hashes"][record["case_identity"]] = record["record_hash"]
    summary["passed"] = all(item["passed"] for item in summary["cases"])
    summary["artifact_identity"] = stable_hash(
        canonicalize({key: value for key, value in summary.items() if key != "artifact_identity"}),
        prefix="width_slice_summary",
    )
    summary_path.write_text(json.dumps(summary))
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["case_hashes"] = summary["case_hashes"]
    manifest["summary_identity"] = summary["artifact_identity"]
    manifest["artifact_identity"] = summary["artifact_identity"]
    manifest["passed"] = summary["passed"]
    manifest["manifest_hash"] = stable_hash(
        canonicalize({key: value for key, value in manifest.items() if key != "manifest_hash"}),
        prefix="width_slice_manifest",
    )
    manifest_path.write_text(json.dumps(manifest))


def _rehash_summary_and_manifest(artifact_dir: Path, summary: dict, manifest: dict) -> None:
    summary["artifact_identity"] = stable_hash(
        canonicalize({key: value for key, value in summary.items() if key != "artifact_identity"}),
        prefix="width_slice_summary",
    )
    (artifact_dir / "summary.json").write_text(json.dumps(summary))
    manifest["summary_identity"] = summary["artifact_identity"]
    manifest["manifest_hash"] = stable_hash(
        canonicalize({key: value for key, value in manifest.items() if key != "manifest_hash"}),
        prefix="width_slice_manifest",
    )
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest))


def test_generic_case_factory_covers_loaded_checkpoint_capabilities_and_two_layers(
    tmp_path: Path,
):
    checkpoint = _tiny_sorted_llama(tmp_path)
    config = load_model_config(checkpoint)
    capabilities = LlamaModelDescriptor.puzzletron_capabilities(config)

    cases = LlamaModelDescriptor.width_slice_equivalence_operations(config, checkpoint)

    expected_axes = {
        axis_id
        for axis_id, axis in capabilities.axes.items()
        if axis.materialize_impl and axis.runtime_slice_impl
    }
    assert {case.axis_id for case in cases.values()} == expected_axes
    hidden = [case for case in cases.values() if case.axis_id == "hidden_width"]
    assert len(hidden) == 1
    assert hidden[0].scope == "global"
    assert hidden[0].layers == (0, 1)
    for axis_id in expected_axes - {"hidden_width"}:
        local = [case for case in cases.values() if case.axis_id == axis_id]
        assert {case.layers for case in local} == {(0,), (1,)}
        assert all(case.target_value < case.source_value for case in local)
        assert all(case.expected_structure["requires_tensor_shape_change"] for case in local)


def test_qwen_dense_inherits_concrete_generic_operations_from_model_descriptor(tmp_path: Path):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_converter import Qwen3P5Converter
    from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
        Qwen3P5TextModelDescriptor,
    )

    checkpoint = create_tiny_qwen3_5_dir(
        tmp_path,
        layer_types=["linear_attention", "linear_attention", "full_attention", "full_attention"],
    )
    config = load_model_config(checkpoint)
    Qwen3P5TextModelDescriptor.set_block_configs(
        config,
        Qwen3P5Converter.create_block_configs_from_main_config(config),
    )
    config.block_configs = [block.to_dict() for block in config.block_configs]
    config.save_pretrained(checkpoint)

    cases = Qwen3P5TextModelDescriptor.width_slice_equivalence_operations(config, checkpoint)

    assert cases
    assert {case.axis_id for case in cases.values()} == {
        axis_id
        for axis_id, axis in Qwen3P5TextModelDescriptor.puzzletron_capabilities(config).axes.items()
        if axis.materialize_impl and axis.runtime_slice_impl
    }
    assert all(not callable(case) for case in cases.values())
    assert {case.layers for case in cases.values() if case.axis_id == "gdn_key_groups"} == {
        (0,),
        (1,),
    }


def test_tiny_qwen_checkpoint_executes_inherited_materialize_and_runtime_hooks(tmp_path: Path):
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_converter import Qwen3P5Converter
    from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
        Qwen3P5TextModelDescriptor,
    )

    checkpoint = create_tiny_qwen3_5_dir(
        tmp_path,
        layer_types=["linear_attention", "linear_attention", "full_attention", "full_attention"],
    )
    config = load_model_config(checkpoint)
    Qwen3P5TextModelDescriptor.set_block_configs(
        config,
        Qwen3P5Converter.create_block_configs_from_main_config(config),
    )
    config.block_configs = [block.to_dict() for block in config.block_configs]
    config.save_pretrained(checkpoint)

    class FFNOnlyQwenDescriptor(Qwen3P5TextModelDescriptor):
        @classmethod
        def puzzletron_capabilities(cls, loaded_config):
            capabilities = super().puzzletron_capabilities(loaded_config)
            return replace(
                capabilities,
                descriptor_name="qwen-width-equivalence-test",
                axes={"ffn_intermediate": capabilities.axes["ffn_intermediate"]},
            )

    batch = normalize_width_slice_batch(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        descriptor=FFNOnlyQwenDescriptor,
        checkpoint_config=config,
        layout=DataLayout.FIXED,
        sample_ids=("sample-0",),
        source_metadata={"dataset": "fixture", "revision": "v1"},
    )

    summary = evaluate_width_slice_equivalence(
        descriptor=FFNOnlyQwenDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=batch,
        artifact_dir=tmp_path / "qwen-artifacts",
        tolerances={
            "loss_atol": 2.0e-3,
            "loss_rtol": 2.0e-3,
            "output_atol": 2.0e-3,
            "output_rtol": 2.0e-3,
        },
    )

    assert summary["passed"] is True
    assert len(summary["cases"]) == 2
    assert all(case["target_applied"] for case in summary["cases"])
    assert all(case["runtime_hook_executions"] > 0 for case in summary["cases"])


def test_real_materialized_slice_matches_runtime_hook_from_same_tiny_checkpoint(
    tmp_path: Path,
):
    checkpoint = _tiny_sorted_llama(tmp_path)

    summary = evaluate_width_slice_equivalence(
        descriptor=_FFNOnlyLlamaDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=_text_batch(),
        artifact_dir=tmp_path / "artifacts",
        tolerances={
            "loss_atol": 1.0e-5,
            "loss_rtol": 1.0e-5,
            "output_atol": 1.0e-5,
            "output_rtol": 1.0e-5,
        },
    )

    assert summary["passed"] is True
    assert len(summary["cases"]) == 2
    assert {case["layer_idx"] for case in summary["cases"]} == {0, 1}
    for case in summary["cases"]:
        assert case["target_applied"] is True
        assert case["runtime_hook_count"] > 0
        assert case["runtime_hook_executions"] > 0
        assert case["structural_evidence"]["changed_tensors"]
        assert case["lineage"]["physical_source_identity"] == summary["checkpoint_identity"]
        assert case["lineage"]["runtime_source_identity"] == summary["checkpoint_identity"]
        assert case["implementation_provenance"]


@pytest.mark.parametrize("layout", [DataLayout.FIXED, DataLayout.PACKED_VARLEN])
@pytest.mark.parametrize("multimodal", [False, True])
def test_real_batch_adapter_semantics_feed_generic_descriptor_cases(
    tmp_path: Path, layout: DataLayout, multimodal: bool
):
    checkpoint = _tiny_sorted_llama(tmp_path)
    collated = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    if layout is DataLayout.PACKED_VARLEN:
        collated.update(
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
            seq_idx=torch.tensor([[0, 0, 1, 1]]),
            max_seqlen=2,
        )
    if multimodal:
        collated.update(
            pixel_values=torch.ones(1, 3, 2, 2),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
            n_images_per_sample=torch.tensor([1]),
        )

    batch = normalize_width_slice_batch(
        collated,
        descriptor=LlamaModelDescriptor,
        checkpoint_config=load_model_config(checkpoint),
        layout=layout,
        sample_ids=("sample-0",),
        source_metadata={"dataset": "fixture", "revision": "v1"},
    )
    cases = LlamaModelDescriptor.width_slice_equivalence_operations(
        load_model_config(checkpoint), checkpoint
    )

    assert batch.layout is layout
    assert (batch.modality.value == "multimodal") is multimodal
    assert ("pixel_values" in batch.model_kwargs) is multimodal
    assert (batch.sequence.global_cu_seqlens is not None) is (layout is DataLayout.PACKED_VARLEN)
    assert cases


@pytest.mark.parametrize("layout", [DataLayout.FIXED, DataLayout.PACKED_VARLEN])
def test_tiny_generic_vlm_executes_real_automodel_collator_batch_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: DataLayout,
):
    from nemo_automodel.components.datasets.vlm.collate_fns import (
        neat_packed_vlm_collater,
        pad_collate_fn,
    )

    monkeypatch.setattr(
        transformers,
        "TinyGenericVLMForCausalLM",
        _TinyGenericVLMForCausalLM,
        raising=False,
    )
    checkpoint = _tiny_generic_vlm_checkpoint(
        tmp_path / layout.value,
        packed=layout is DataLayout.PACKED_VARLEN,
    )
    example = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([1, 2, 3, 4]),
        "attention_mask": torch.tensor([1, 1, 2, 2]),
        "position_ids": torch.arange(4),
        "pixel_values": torch.ones(1, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
        "n_images": 1,
    }
    if layout is DataLayout.PACKED_VARLEN:
        collated = neat_packed_vlm_collater(
            [example],
            padding_idx=0,
            attn_implementation="flash_attention_2",
        )
    else:
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        collated = pad_collate_fn([example], processor)
    batch = normalize_width_slice_batch(
        collated,
        descriptor=_FFNOnlyLlamaDescriptor,
        checkpoint_config=load_model_config(checkpoint),
        layout=layout,
        sample_ids=("vlm-0",),
        source_metadata={"dataset": "processor-collator-fixture", "revision": "v1"},
    )

    summary = evaluate_width_slice_equivalence(
        descriptor=_FFNOnlyLlamaDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=batch,
        artifact_dir=tmp_path / f"artifacts-{layout.value}",
    )

    assert summary["passed"] is True
    assert summary["batch_modality"] == "multimodal"
    assert summary["batch_layout"] == layout.value


def test_loss_uses_explicit_absolute_and_relative_tolerances():
    metrics = compare_width_slice_outputs(
        physical_loss=torch.tensor(100.0),
        runtime_loss=torch.tensor(100.5),
        physical_output=torch.tensor([1.0, 2.0]),
        runtime_output=torch.tensor([1.0, 2.0]),
        tolerances={
            "loss_atol": 0.1,
            "loss_rtol": 0.01,
            "output_atol": 0.0,
            "output_rtol": 0.0,
        },
    )

    assert metrics["loss_delta"] == pytest.approx(0.5)
    assert metrics["loss_allowed_delta"] == pytest.approx(1.105)
    assert metrics["loss_close"] is True
    assert metrics["passed"] is True


def test_resume_rejects_tampered_case_and_recomputes_it(tmp_path: Path):
    checkpoint = _tiny_sorted_llama(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    kwargs = {
        "descriptor": _FFNOnlyLlamaDescriptor,
        "sorted_checkpoint_dir": checkpoint,
        "batch": _text_batch(),
        "artifact_dir": artifact_dir,
    }
    first = evaluate_width_slice_equivalence(**kwargs)
    case_path = next((artifact_dir / "cases").rglob("*.json"))
    tampered = json.loads(case_path.read_text())
    tampered["passed"] = not tampered["passed"]
    case_path.write_text(json.dumps(tampered))

    with pytest.raises(RuntimeError, match=r"hash|schema|passed"):
        validate_width_slice_artifacts(artifact_dir)

    second = evaluate_width_slice_equivalence(**kwargs)
    assert second == first
    assert validate_width_slice_artifacts(artifact_dir)["passed"] is True


@pytest.mark.parametrize("mutation", ["metrics", "target", "provenance"])
def test_validation_rebuilds_cases_and_rejects_coordinated_rehash(tmp_path: Path, mutation: str):
    checkpoint = _tiny_sorted_llama(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    evaluate_width_slice_equivalence(
        descriptor=_FFNOnlyLlamaDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=_text_batch(),
        artifact_dir=artifact_dir,
    )
    case_path = next((artifact_dir / "cases").rglob("*.json"))
    record = json.loads(case_path.read_text())
    if mutation == "metrics":
        record["metrics"].update(
            physical_loss=123.0,
            runtime_loss=123.0,
            loss_delta=0.0,
            loss_allowed_delta=0.00124,
            loss_close=True,
            output_close=True,
            passed=True,
        )
        record["passed"] = True
    elif mutation == "target":
        record["target_value"] += 1
    else:
        record["implementation_provenance"]["source_hash"] = "forged-current-source"
    _rehash_artifact_json(artifact_dir, record)

    with pytest.raises(RuntimeError, match=r"metrics|target|case|provenance|implementation"):
        validate_width_slice_artifacts(artifact_dir, descriptor=_FFNOnlyLlamaDescriptor)


@pytest.mark.parametrize("mutation", ["summary_record", "manifest_identity"])
def test_validation_rejects_coordinated_container_rehash(tmp_path: Path, mutation: str):
    checkpoint = _tiny_sorted_llama(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    evaluate_width_slice_equivalence(
        descriptor=_FFNOnlyLlamaDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=_text_batch(),
        artifact_dir=artifact_dir,
    )
    summary = json.loads((artifact_dir / "summary.json").read_text())
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if mutation == "summary_record":
        summary["cases"][0]["metrics"]["physical_loss"] += 10.0
    else:
        manifest["artifact_identity"] = "forged-artifact-identity"
    _rehash_summary_and_manifest(artifact_dir, summary, manifest)

    with pytest.raises(RuntimeError, match=r"summary|artifact|case"):
        validate_width_slice_artifacts(artifact_dir, descriptor=_FFNOnlyLlamaDescriptor)


def test_runtime_rejects_layer_execution_when_installed_target_hook_does_not_execute(
    monkeypatch: pytest.MonkeyPatch,
):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = torch.nn.Linear(4, 4)
            self.unused = torch.nn.Linear(4, 4)

        def forward(self, value):
            return self.active(value)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer()])

    class Descriptor:
        @staticmethod
        def layer_block_name(index):
            return f"model.layers.{index}"

        @staticmethod
        def adapt_module_name_for_model(name, model):
            del model
            return name

        @staticmethod
        def get_language_model_config(config):
            return config

    @contextmanager
    def unused_hook_context(adapter, targets):
        del targets
        handle = (
            adapter.model_parts[0]
            .model.layers[0]
            .unused.register_forward_pre_hook(lambda module, args: None)
        )
        try:
            yield
        finally:
            handle.remove()

    monkeypatch.setattr(_RuntimeRecipeAdapter, "architecture_context", unused_hook_context)
    source = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=2, num_kv_heads=1),
            FFNConfig(intermediate_size=8),
        )
    )
    target = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=2, num_kv_heads=1),
            FFNConfig(intermediate_size=4),
        )
    )
    case = WidthSliceCase(
        case_identity="unused-hook",
        axis_id="ffn_intermediate",
        axis_class="block::hook",
        scope="layer",
        layers=(0,),
        source_value=8,
        target_value=4,
        subblock_kind="ffn",
        field="intermediate_size",
        source_block_config=source,
        target_block_config=target,
        expected_structure={},
        implementation_provenance={},
    )
    model = Model()

    with (
        pytest.raises(RuntimeError, match=r"axis.*hook.*did not execute"),
        _runtime_context(
            model,
            Descriptor,
            SimpleNamespace(hidden_size=4, num_attention_heads=2),
            case,
        ),
    ):
        model.model.layers[0](torch.ones(1, 4))


def test_axis_structure_rejects_unrelated_tensor_shape_change():
    case = WidthSliceCase(
        case_identity="shape",
        axis_id="ffn_intermediate",
        axis_class="block::hook",
        scope="layer",
        layers=(0,),
        source_value=8,
        target_value=4,
        subblock_kind="ffn",
        field="intermediate_size",
        source_block_config=None,
        target_block_config=None,
        expected_structure={},
        implementation_provenance={},
    )

    before = {"model.layers.0.self_attn.q_proj.weight": [8, 8]}
    after = {"model.layers.0.self_attn.q_proj.weight": [4, 8]}
    axis_filter = getattr(width_slice_module, "_axis_specific_changed_shapes", None)
    evidence = (
        width_slice_module._changed_shapes(before, after)
        if axis_filter is None
        else axis_filter(
            before,
            after,
            descriptor=LlamaModelDescriptor,
            case=case,
            num_layers=2,
        )
    )

    assert evidence == {}


def test_artifact_validation_detects_deleted_case(tmp_path: Path):
    checkpoint = _tiny_sorted_llama(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "width_slice_equivalence"
    evaluate_width_slice_equivalence(
        descriptor=_FFNOnlyLlamaDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=_text_batch(),
        artifact_dir=artifact_dir,
    )
    next((artifact_dir / "cases").rglob("*.json")).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        validate_width_slice_artifacts(artifact_dir)


def test_stage_is_reachable_from_the_generic_handler_registry():
    assert DEFAULT_HANDLERS["slicing_sanity"] is width_slice_equivalence_stage


def test_dag_resume_inventories_manifest_summary_and_every_case():
    assert _stage_output_patterns({}, "slicing_sanity") == (
        "artifacts/width_slice_equivalence/manifest.json",
        "artifacts/width_slice_equivalence/summary.json",
        "artifacts/width_slice_equivalence/cases/**/*.json",
        "artifacts/width_slice_equivalence/comparisons/*.safetensors",
    )


def test_dag_resume_inventories_distributed_parent_sweep_summary():
    config = {"slicing_sanity": {"backend": "distributed_parent_sweep"}}

    assert _stage_output_patterns(config, "slicing_sanity") == (
        "artifacts/slicing_sanity/summary.json",
    )


def test_validation_rechecks_authoritative_checkpoint_content(tmp_path: Path):
    checkpoint = _tiny_sorted_llama(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    kwargs = {
        "descriptor": _FFNOnlyLlamaDescriptor,
        "sorted_checkpoint_dir": checkpoint,
        "batch": _text_batch(),
        "artifact_dir": artifact_dir,
    }
    evaluate_width_slice_equivalence(
        **kwargs,
    )
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config["tampered"] = True
    config_path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match=r"checkpoint.*identity|content"):
        validate_width_slice_artifacts(artifact_dir)

    refreshed = evaluate_width_slice_equivalence(**kwargs)
    assert (
        validate_width_slice_artifacts(artifact_dir)["artifact_identity"]
        == refreshed["artifact_identity"]
    )


@pytest.mark.parametrize("mutation", ["lineage", "nan_metric"])
def test_case_schema_rejects_semantically_forged_evidence(tmp_path: Path, mutation: str):
    checkpoint = _tiny_sorted_llama(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    evaluate_width_slice_equivalence(
        descriptor=_FFNOnlyLlamaDescriptor,
        sorted_checkpoint_dir=checkpoint,
        batch=_text_batch(),
        artifact_dir=artifact_dir,
    )
    case_path = next((artifact_dir / "cases").rglob("*.json"))
    record = json.loads(case_path.read_text())
    if mutation == "lineage":
        record["lineage"]["runtime_source_identity"] = "different-checkpoint"
    else:
        record["metrics"]["loss_delta"] = float("nan")
    record["record_hash"] = stable_hash(
        canonicalize({key: value for key, value in record.items() if key != "record_hash"}),
        prefix="width_slice_record",
    )

    with pytest.raises(RuntimeError, match=r"lineage|finite"):
        _validate_case_record(record)


def test_stage_uses_shared_dataloader_and_publishes_semantic_artifact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sorted_dir = tmp_path / "ckpts" / "sorted_teacher"
    sorted_dir.mkdir(parents=True)
    (sorted_dir / "config.json").write_text("{}")
    checkpoint_config = SimpleNamespace(num_hidden_layers=2)

    class Descriptor:
        @classmethod
        def get_language_model_config(cls, config):
            return config

        @classmethod
        def width_slice_equivalence_tolerances(cls):
            return {
                "loss_atol": 1e-5,
                "loss_rtol": 1e-5,
                "output_atol": 1e-5,
                "output_rtol": 1e-5,
            }

        @classmethod
        def position_id_axes(cls, config):
            del config
            return 1

    raw_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    captured = {}
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.pipeline_config.load_runtime_hydra_config",
        lambda config: SimpleNamespace(descriptor="fixture", puzzle_dir=str(tmp_path)),
    )
    monkeypatch.setattr(
        diagnostics_stage_module.ModelDescriptorFactory,
        "get",
        lambda name: Descriptor,
    )
    monkeypatch.setattr(
        diagnostics_stage_module, "load_model_config", lambda path: checkpoint_config
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.utils.data.dataloaders.prepare_validation_dataloader",
        lambda config, tokenizer, data_layout: [raw_batch],
    )

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {
            "passed": True,
            "cases": [],
            "tolerances": kwargs["tolerances"],
            "artifact_identity": "width-artifact-v2",
        }

    monkeypatch.setattr(diagnostics_stage_module, "evaluate_width_slice_equivalence", evaluate)
    monkeypatch.setattr(
        diagnostics_stage_module,
        "validate_width_slice_artifacts",
        lambda artifact_dir, **kwargs: {"passed": True},
    )

    result = width_slice_equivalence_stage(
        {
            "puzzle_dir": str(tmp_path),
            "experiment": {"dir": str(tmp_path)},
            "data": {"layout": "fixed", "modality": "text"},
            "scoring": {"eval_samples": 1},
        },
        StageManifest(stage="width_slice_equivalence"),
    )

    assert result.status == "success"
    assert isinstance(captured["batch"], PuzzletronBatch)
    assert captured["sampled_layers"] is None
    stage_manifest = json.loads(
        (tmp_path / "manifests" / "width_slice_equivalence.json").read_text()
    )
    assert stage_manifest["outputs"]["artifact_identity"] == "width-artifact-v2"


def test_stage_selects_generic_multimodal_processor_collator_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sorted_dir = tmp_path / "ckpts" / "sorted_teacher"
    sorted_dir.mkdir(parents=True)
    (sorted_dir / "config.json").write_text("{}")
    checkpoint_config = SimpleNamespace(num_hidden_layers=2)

    class Descriptor:
        @classmethod
        def get_language_model_config(cls, config):
            return config

        @classmethod
        def width_slice_equivalence_tolerances(cls):
            return {}

        @classmethod
        def position_id_axes(cls, config):
            del config
            return 1

    raw_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.ones(1, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
        "n_images_per_sample": torch.tensor([1]),
    }
    selected = {}
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.pipeline_config.load_runtime_hydra_config",
        lambda config: SimpleNamespace(descriptor="fixture", puzzle_dir=str(tmp_path)),
    )
    monkeypatch.setattr(
        diagnostics_stage_module.ModelDescriptorFactory, "get", lambda name: Descriptor
    )
    monkeypatch.setattr(
        diagnostics_stage_module, "load_model_config", lambda path: checkpoint_config
    )
    monkeypatch.setattr(
        dataloader_module,
        "prepare_validation_dataloader",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("text-only tokenizer loader selected for VLM")
        ),
    )

    def multimodal_loader(args, *, checkpoint_dir, data_layout):
        selected.update(args=args, checkpoint_dir=checkpoint_dir, data_layout=data_layout)
        return [raw_batch]

    monkeypatch.setattr(
        dataloader_module,
        "prepare_multimodal_validation_dataloader",
        multimodal_loader,
        raising=False,
    )
    monkeypatch.setattr(
        diagnostics_stage_module,
        "evaluate_width_slice_equivalence",
        lambda **kwargs: {
            "passed": True,
            "cases": [],
            "tolerances": kwargs["tolerances"],
            "artifact_identity": "vlm-artifact",
        },
    )
    monkeypatch.setattr(
        diagnostics_stage_module,
        "validate_width_slice_artifacts",
        lambda artifact_dir, **kwargs: {"passed": True},
    )

    result = width_slice_equivalence_stage(
        {
            "puzzle_dir": str(tmp_path),
            "experiment": {"dir": str(tmp_path)},
            "data": {
                "layout": "fixed",
                "modality": "multimodal",
                "path": str(tmp_path / "vlm-data"),
            },
            "scoring": {"eval_samples": 1},
        },
        StageManifest(stage="width_slice_equivalence"),
    )

    assert result.status == "success"
    assert selected["checkpoint_dir"] == sorted_dir
    assert selected["data_layout"] == "fixed"


def test_distributed_slicing_verdict_warns_without_failing_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    summary_path = tmp_path / "artifacts" / "slicing_sanity" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    finding = {
        "stage": "slicing_sanity",
        "severity": "warning",
        "message": "runtime and physical slices differ",
        "evidence": {},
    }
    summary_path.write_text(
        json.dumps(
            {
                "passed": False,
                "axes": ["moe_experts"],
                "rows": [{"axis": "moe_experts", "method": "physical"}],
                "findings": [finding],
            }
        )
    )
    monkeypatch.setattr(
        "modelopt.torch.puzzletron.pipeline_config.load_runtime_hydra_config",
        lambda config: SimpleNamespace(puzzle_dir=str(tmp_path)),
    )
    manifest = StageManifest(stage="slicing_sanity")

    result = width_slice_equivalence_stage(
        {
            "puzzle_dir": str(tmp_path),
            "experiment": {"dir": str(tmp_path)},
            "slicing_sanity": {"backend": "distributed_parent_sweep"},
        },
        manifest,
    )

    assert result.status == "success"
    assert manifest.status == "success"
    assert manifest.outputs["passed"] is False
    assert manifest.outputs["findings"] == [finding]
