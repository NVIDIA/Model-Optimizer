# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-rank released-AutoModel DCP-to-full-state export proof for the PDD example."""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_DIR = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
for path in (_REPO_ROOT, _REPO_ROOT / "tests", _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _test_utils.torch.diffusers_models import create_tiny_qwen_image_pipeline_dir
from export_pdd_qwen_image import collective_export_memory_preflight
from inference_pdd_qwen_image import build_pdd_student
from pdd_export import inspect_pdd_export, write_pdd_export
from pdd_recipe import build_pdd_export_setup, resolve_pdd_recipe_config


def _raw_config(model_dir: pathlib.Path, checkpoint_dir: pathlib.Path) -> dict:
    return {
        "model": {
            "pretrained_model_name_or_path": str(model_dir),
            "torch_dtype": "float32",
            "device": "cpu",
            "transformer_engine_linear": False,
            "peft": None,
            "guidance_embeds": False,
            "fuse_qkv_projections": False,
        },
        "pdd": {
            "pred_type": "flow",
            "num_train_timesteps": None,
            "guidance_scale": 4.0,
            "student_sample_steps": 2,
            "student_sample_type": "ode",
            "grid_size": 4,
            "flow_shift": 5.0,
            "block_size_min": 1,
            "block_size_max": 4,
            "teacher_integrator": "euler",
            "inference_blocks": [2, 2],
            "data_free": False,
        },
        "optim": {"learning_rate": 2.0e-5, "weight_decay": 0.01},
        "fsdp": {
            "dp_size": 2,
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "activation_checkpointing": False,
        },
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": str(checkpoint_dir),
            "model_save_format": "torch_save",
            "save_consolidated": False,
        },
    }


def _full_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def main() -> None:
    dist.init_process_group("gloo")
    payload = [tempfile.mkdtemp(prefix="modelopt-pdd-export-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    root = pathlib.Path(payload[0])
    model_root = root / "model"
    model_dir = model_root / "tiny_qwen_image"
    try:
        if dist.get_rank() == 0:
            assert create_tiny_qwen_image_pipeline_dir(model_root) == model_dir
        dist.barrier()
        config = resolve_pdd_recipe_config(_raw_config(model_dir, root / "checkpoints"))
        source = build_pdd_export_setup(config)
        expected = _full_state(source.student)
        source.checkpointer.save_model(source.student, str(root / "dcp"))
        source.checkpointer.close()

        destination = build_pdd_export_setup(config)
        destination.checkpointer.load_model(destination.student, str(root / "dcp" / "model"))
        full_state_bytes, largest_tensor_bytes = collective_export_memory_preflight(
            destination.student,
            max_shard_bytes=4 * 1024 * 1024,
            headroom=1.0,
            device=torch.device("cpu"),
        )
        actual = _full_state(destination.student)
        status = None
        if dist.get_rank() == 0:
            try:
                assert expected and actual
                assert expected.keys() == actual.keys()
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                assert full_state_bytes == sum(
                    tensor.numel() * tensor.element_size() for tensor in actual.values()
                )
                assert largest_tensor_bytes == max(
                    tensor.numel() * tensor.element_size() for tensor in actual.values()
                )
                identity = {
                    "schema_version": 1,
                    "model": {
                        "id": "Qwen/Qwen-Image",
                        "revision": "3" * 40,
                        "dtype": "float32",
                    },
                    "pdd_metadata": destination.metadata.to_dict(),
                    "guidance": {"scale": 4.0, "rescale": 1.0, "eps": 1e-5},
                    "automodel": {
                        key: destination.automodel_snapshot[key]
                        for key in (
                            "distribution",
                            "version",
                            "package_tree_sha256",
                            "wheel_sha256",
                            "runtime_versions",
                        )
                    },
                    "topology": {"world_size": 2, "pure_data_parallel": True},
                }
                output = write_pdd_export(
                    root / "export",
                    actual,
                    metadata=destination.metadata,
                    transformer_config=destination.transformer_config,
                    identity=identity,
                    source_checkpoint={
                        "name": "step_00000001",
                        "manifest_sha256": "1" * 64,
                        "completed_steps": 1,
                    },
                    modelopt_source={"commit": "2" * 40, "dirty": False},
                    max_shard_bytes=4 * 1024 * 1024,
                )
                descriptor = inspect_pdd_export(output)
                assert descriptor.metadata == destination.metadata
                restored, restored_descriptor, _dtype = build_pdd_student(output)
                assert restored_descriptor.metadata == destination.metadata
                restored_state = restored.state_dict()
                assert restored_state.keys() == actual.keys()
                for key in actual:
                    torch.testing.assert_close(restored_state[key], actual[key], rtol=0, atol=0)
                status = {"ok": True}
            except BaseException as error:
                status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        else:
            assert not expected and not actual
        payload = [status]
        dist.broadcast_object_list(payload, src=0)
        if not payload[0]["ok"]:
            raise RuntimeError(payload[0]["error"])
        destination.checkpointer.close()
        dist.barrier()
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(root)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
