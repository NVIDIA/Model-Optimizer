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
"""End-to-end test for Quantization Aware Distillation (QAD): quantize + distill + export."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import yaml
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command
from _test_utils.torch.export.unified_checkpoint import (
    assert_exported_checkpoint_matches,
    assert_per_expert_experts_complete,
)
from _test_utils.torch.megatron.modelopt_state import (
    assert_has_modelopt_state,
    assert_no_quantizers_matching,
)
from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_5_moe_vl_dir,
    create_tiny_qwen3_dir,
)
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata


def _load_checkpoint_tensors(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Model tensors of a Megatron distributed checkpoint (optimizer state excluded)."""
    reader = FileSystemReader(str(checkpoint_path))
    metadata = reader.read_metadata().state_dict_metadata
    tensors = {
        name: torch.empty(meta.size, dtype=meta.properties.dtype)
        for name, meta in metadata.items()
        if not name.startswith("optimizer") and hasattr(meta, "size")
    }
    dcp.load(tensors, storage_reader=reader)
    return tensors


# LSQ scale learning needs NVIDIA-NeMo/Megatron-Bridge#6699, and resuming it #6698.
# TODO: drop this probe once the minimum Megatron-Bridge carries both.
try:
    from megatron.bridge.training.checkpointing import _param_group_lr_bounds  # noqa: F401
    from megatron.bridge.training.post_training.lsq import freeze_all_but_lsq_scales  # noqa: F401

    HAS_BRIDGE_LSQ = True
except ImportError:
    HAS_BRIDGE_LSQ = False


def _write_calibration_data(tmp_path: Path) -> str:
    """Write a small calibration set, so the test needs no dataset download on the GPU node."""
    calib_path = tmp_path / "calibration.jsonl"
    calib_path.write_text(
        "\n".join(
            json.dumps(
                {"text": f"Sample {i}: " + "The quick brown fox jumps over the lazy dog. " * 4}
            )
            for i in range(8)
        ),
        encoding="utf-8",
    )
    return str(calib_path)


@pytest.mark.timeout(720)  # Multiple steps in one test hence takes longer than the default timeout
@pytest.mark.parametrize(
    ("create_student", "lora_rank"),
    [
        (lambda tmp_path: create_tiny_qwen3_dir(tmp_path, with_tokenizer=True), 0),
        (
            lambda tmp_path: create_tiny_qwen3_dir(
                tmp_path,
                with_tokenizer=True,
                hidden_size=128,
                intermediate_size=256,
            ),
            4,
        ),
        pytest.param(
            lambda tmp_path: create_tiny_qwen3_5_moe_vl_dir(
                tmp_path,
                with_processor=True,
                # Cover both Qwen3.5 decoder kinds at the same layer count
                num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"],
            ),
            0,
        ),
    ],
    ids=["qwen3", "qwen3_lora_nvfp4", "qwen3_5_moe_vl"],
)
def test_qad(tmp_path: Path, num_gpus, create_student, lora_rank):
    """Quantize a tiny model, run QAD from the quantized student, and export the result.

    Covers what only QAD exercises: that the ModelOpt state survives distillation. Per-architecture
    export is covered more cheaply by test_quantize_export.py, so keep this to one LLM and one VLM.
    """
    hf_model_path = create_student(tmp_path)
    is_vlm = "vision_config" in (hf_model_path / "config.json").read_text()
    # mrope rotary embeddings CP-shard themselves, so the VLM exercises the path where the batch's
    # position ids must stay full-length. The LLM case stays on tensor parallelism.
    cp_size = num_gpus if is_vlm else 1
    tp_size = 1 if cp_size > 1 else num_gpus
    quantized_megatron_path = tmp_path / "quantized_megatron"
    distill_output_dir = tmp_path / "qad_output"
    train_iters = 3
    early_exit_iter = 2
    calib_dataset = "cnn_dailymail"
    if lora_rank:
        # Keep the LoRA smoke test independent of dataset downloads on the GPU node.
        calib_dataset = _write_calibration_data(tmp_path)

    # Step 1: PTQ the model and save its quantizers and optional adapters.
    quantize_cmd = extend_cmd_parts(
        # QAD below must load this checkpoint at the same TP, so size the PTQ run to tp_size.
        ["torchrun", f"--nproc_per_node={tp_size}", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        recipe="general/ptq/nvfp4_default-kv_fp8"
        if lora_rank
        else "general/ptq/fp8_default-kv_fp8",
        lora_rank=lora_rank,
        tp_size=tp_size,
        pp_size=1,
        calib_dataset_name=calib_dataset,  # text dataset -> (for VLMs) text-only LM calibration
        calib_num_samples=8,
        calib_batch_size=2,
        seq_length=16,
        export_megatron_path=quantized_megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert_has_modelopt_state(quantized_megatron_path)
    # Megatron names these differently from HF, so the recipe's patterns must have aliases.
    assert_no_quantizers_matching(quantized_megatron_path, "conv1d", "mlp.router", "output_layer")

    # Step 2: QAD -- load the quantized student from the Megatron checkpoint (restoring the ModelOpt
    # quantizers) and distill from the (unquantized) HF teacher. The distilled checkpoint must keep the
    # ModelOpt state so the quantizers survive distillation.
    # The LoRA case trains only adapters, so its base weights would not exercise the cache's gradient.
    distill_cmd = extend_cmd_parts(
        ["torchrun", f"--nproc_per_node={num_gpus}", "distill.py", "--use_mock_data"]
        + ([] if lora_rank else ["--fake_quant_weight_cache"]),
        student_hf_path=hf_model_path,
        student_megatron_path=quantized_megatron_path,
        teacher_hf_path=hf_model_path,
        output_dir=distill_output_dir,
        tp_size=tp_size,
        pp_size=1,
        cp_size=cp_size,
        seq_length=16,
        mbs=1,
        gbs=4,
        logit_kl_top_k=8,
        train_iters=train_iters,
        lr_warmup_iters=2,
        eval_interval=early_exit_iter,
        eval_iters=1,
        save_interval=1,
        log_interval=1,
        exit_interval=early_exit_iter,
        exit_duration_in_mins=10,
        recompute_granularity="full" if lora_rank else None,
        recompute_method="uniform" if lora_rank else None,
        recompute_num_layers=1 if lora_rank else None,
    )
    run_example_command(distill_cmd, example_path="megatron_bridge", setup_free_port=True)
    distilled_megatron_path = distill_output_dir / "checkpoints"
    tracker = distilled_megatron_path / "latest_checkpointed_iteration.txt"
    assert tracker.read_text(encoding="utf-8").strip() == str(early_exit_iter)
    assert (distilled_megatron_path / "iter_0000001").is_dir()
    assert_has_modelopt_state(distilled_megatron_path)

    if not lora_rank and not is_vlm:
        # The weight cache reuses fake-quantized weights within a step; training must not change.
        uncached_dir = tmp_path / "qad_uncached"
        uncached_cmd = [arg for arg in distill_cmd if arg != "--fake_quant_weight_cache"]
        uncached_cmd[uncached_cmd.index("--output_dir") + 1] = str(uncached_dir)
        run_example_command(uncached_cmd, example_path="megatron_bridge", setup_free_port=True)
        cached, uncached = (
            _load_checkpoint_tensors(path / f"iter_{early_exit_iter:07d}")
            for path in (distilled_megatron_path, uncached_dir / "checkpoints")
        )
        assert cached and cached.keys() == uncached.keys()
        for name, value in cached.items():
            torch.testing.assert_close(value, uncached[name], rtol=0, atol=0, msg=name)

    if lora_rank:
        # Reuse the output directory to exercise optimizer and trained-adapter restoration.
        run_example_command(distill_cmd, example_path="megatron_bridge", setup_free_port=True)
        assert tracker.read_text(encoding="utf-8").strip() == str(train_iters)
        # Compare against the same seed, data, schedule, and PTQ checkpoint without a restart.
        reference_dir = tmp_path / "qad_uninterrupted"
        reference_cmd = distill_cmd.copy()
        reference_cmd[reference_cmd.index("--output_dir") + 1] = str(reference_dir)
        reference_cmd[reference_cmd.index("--exit_interval") + 1] = str(train_iters)
        run_example_command(reference_cmd, example_path="megatron_bridge", setup_free_port=True)
        reference_checkpoint = reference_dir / "checkpoints"
        assert (
            reference_checkpoint / "latest_checkpointed_iteration.txt"
        ).read_text().strip() == str(train_iters)
        adapter_states = []
        for checkpoint_path in (distilled_megatron_path, reference_checkpoint):
            reader = FileSystemReader(str(checkpoint_path / f"iter_{train_iters:07d}"))
            metadata = reader.read_metadata().state_dict_metadata
            adapters = {
                name: torch.empty(meta.size, dtype=meta.properties.dtype)
                for name, meta in metadata.items()
                if name.endswith(("lora_A", "lora_B"))
            }
            assert adapters, "Trained checkpoint lost the LoRA factors"
            dcp.load(adapters, storage_reader=reader)
            assert any(
                value.count_nonzero() > 0
                for name, value in adapters.items()
                if name.endswith("lora_B")
            )
            adapter_states.append(adapters)
        resumed, uninterrupted = adapter_states
        assert resumed.keys() == uninterrupted.keys()
        for name, value in resumed.items():
            torch.testing.assert_close(value, uninterrupted[name], rtol=0, atol=0)

    # Step 3: export the distilled quantized checkpoint to a unified HF checkpoint. hf_quant_config.json
    # is only written for a quantized model, so its presence confirms the quantizers survived QAD.
    hf_export_path = tmp_path / "qad_hf"
    export_cmd = extend_cmd_parts(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "export_quantized_megatron_to_hf.py",
        ],
        hf_model_name_or_path=hf_model_path,
        megatron_path=distilled_megatron_path,
        export_unified_hf_path=hf_export_path,
        pp_size=num_gpus,
    )
    run_example_command(export_cmd, example_path="megatron_bridge", setup_free_port=True)
    assert (hf_export_path / "config.json").exists()
    assert (hf_export_path / "hf_quant_config.json").exists()
    if lora_rank:
        assert not (hf_export_path / "adapter_config.json").exists()
        index = json.loads((hf_export_path / "model.safetensors.index.json").read_text())
        assert not any("lora_" in key for key in index["weight_map"])
        quant_config = json.loads((hf_export_path / "hf_quant_config.json").read_text())
        assert quant_config["quantization"]["quant_algo"] == "NVFP4"
    # A quantized export writes routed experts one per expert while the BF16 reference packs
    # them, so both sides of that expansion differ from the reference.
    text_config = json.loads((hf_model_path / "config.json").read_text())
    is_moe = bool(text_config.get("text_config", text_config).get("num_experts"))
    # QAD trains the student, so language-model weights drift from the reference; the vision
    # tower is never trained and must still come through byte for byte.
    assert_exported_checkpoint_matches(
        hf_export_path,
        hf_model_path,
        check_values=False,
        allow_missing=("mlp.experts.gate_up_proj", "mlp.experts.down_proj") if is_moe else (),
        allow_unexpected=("mlp.experts.",) if is_moe else (),
        bit_exact_prefixes=("model.visual.",) if is_vlm else (),
    )
    if is_moe:
        # allow_unexpected waives the expert names wholesale; this re-tightens it.
        assert_per_expert_experts_complete(hf_export_path)

    # Step 4: export for vLLM fake-quant evaluation (test_quantize_export.py covers the rest).
    # Only QAD saves a VLM's language model alone; the export must still carry the vision tower.
    if is_vlm:
        fakequant_path = tmp_path / "qad_vllm_fakequant"
        fakequant_cmd = extend_cmd_parts(
            [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                "export_quantized_megatron_to_hf.py",
                "--vllm_fakequant_export",
            ],
            hf_model_name_or_path=hf_model_path,
            megatron_path=distilled_megatron_path,
            export_unified_hf_path=fakequant_path,
            tp_size=num_gpus,
        )
        run_example_command(fakequant_cmd, example_path="megatron_bridge", setup_free_port=True)
        assert_exported_checkpoint_matches(
            fakequant_path,
            hf_model_path,
            check_values=False,
            bit_exact_prefixes=("model.visual.",),
        )
        recipe = yaml.safe_load((fakequant_path / "quant_recipe.yaml").read_text())
        state = torch.load(fakequant_path / "quantizer_state.pth", weights_only=True)
        assert any(name.endswith("k_bmm_quantizer") for name in recipe["quantizers"])
        assert {key.rpartition(".")[0] for key in state} <= recipe["quantizers"].keys()


def _optimizer_param_groups(iteration_dir: Path) -> list[dict]:
    """Param groups saved by Megatron's distributed optimizer in a checkpoint iteration."""
    reader = FileSystemReader(str(iteration_dir))
    state = {
        name: None
        for name, meta in reader.read_metadata().state_dict_metadata.items()
        if isinstance(meta, BytesStorageMetadata)
        and name.startswith("optimizer.")
        and ".optimizer/" in name
    }
    dcp.load(state, storage_reader=reader)
    return [
        group for entries in state.values() for entry in entries for group in entry["param_groups"]
    ]


@pytest.mark.timeout(720)
@pytest.mark.skipif(
    not HAS_BRIDGE_LSQ,
    reason="Megatron-Bridge lacks LSQ scale learning (NVIDIA-NeMo/Megatron-Bridge#6699, #6698)",
)
def test_qad_lsq_scale_only(tmp_path: Path, num_gpus):
    """Scale-only LSQ QAD across a restart: only the scales train, in their own param group."""
    hf_model_path = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    quantized_megatron_path = tmp_path / "quantized_megatron"
    # Static-block NVFP4, and so LSQ, needs tp_size 1.
    quantize_cmd = extend_cmd_parts(
        ["torchrun", "--nproc_per_node=1", "quantize.py", "--skip_generate"],
        hf_model_name_or_path=hf_model_path,
        recipe="general/qad/nvfp4_lsq-mse_init-fp8_kv",
        tp_size=1,
        pp_size=1,
        calib_dataset_name=_write_calibration_data(tmp_path),
        calib_num_samples=8,
        calib_batch_size=2,
        seq_length=16,
        export_megatron_path=quantized_megatron_path,
    )
    run_example_command(quantize_cmd, example_path="megatron_bridge", setup_free_port=True)

    # The scales get their own peak learning rate; the frozen weights would move at --lr.
    lr, scale_lr, train_iters = 1e-3, 2e-3, 4
    distill_output_dir = tmp_path / "qad_output"
    distill_cmd = extend_cmd_parts(
        [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "distill.py",
            "--use_mock_data",
            "--lsq_scale_only",
        ],
        student_hf_path=hf_model_path,
        student_megatron_path=quantized_megatron_path,
        teacher_hf_path=hf_model_path,
        output_dir=distill_output_dir,
        tp_size=1,
        pp_size=1,
        seq_length=16,
        mbs=1,
        gbs=4,
        logit_kl_top_k=8,
        train_iters=train_iters,
        lr=lr,
        min_lr=lr / 10,
        lsq_scale_lr=scale_lr,
        lr_warmup_iters=1,
        eval_interval=train_iters,
        eval_iters=1,
        save_interval=1,
        log_interval=1,
        exit_interval=2,
        exit_duration_in_mins=10,
    )
    # Stop at iteration 2, then resume from it to the end.
    run_example_command(distill_cmd, example_path="megatron_bridge", setup_free_port=True)
    run_example_command(distill_cmd, example_path="megatron_bridge", setup_free_port=True)
    checkpoints = distill_output_dir / "checkpoints"
    tracker = checkpoints / "latest_checkpointed_iteration.txt"
    assert tracker.read_text(encoding="utf-8").strip() == str(train_iters)

    ptq = _load_checkpoint_tensors(next(quantized_megatron_path.glob("iter_*")))
    resumed_from, final = (
        _load_checkpoint_tensors(checkpoints / f"iter_{iteration:07d}")
        for iteration in (2, train_iters)
    )
    scales = [name for name in ptq if name.endswith(("_amax_pre", "_amax_post"))]
    assert scales, "The LSQ recipe produced no learnable scales"
    for name, value in ptq.items():
        if name not in scales:
            torch.testing.assert_close(resumed_from[name], value, rtol=0, atol=0, msg=name)
            torch.testing.assert_close(final[name], value, rtol=0, atol=0, msg=name)
    assert any(not torch.equal(resumed_from[name], ptq[name]) for name in scales)
    assert any(not torch.equal(final[name], resumed_from[name]) for name in scales)
    # The scales are the only trainable parameters; after the restart their group keeps its own
    # learning rate instead of falling back to --lr.
    for iteration in (2, train_iters):
        groups = _optimizer_param_groups(checkpoints / f"iter_{iteration:07d}")
        assert groups
        for group in groups:
            assert group["max_lr"] == scale_lr and group["wd_mult"] == 0.0, group
