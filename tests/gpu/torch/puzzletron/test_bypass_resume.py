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

"""GPU integration test for the bypass-distillation resume path.

The existing ``test_bypass.py`` covers the save side: a fresh bypass run
produces a checkpoint and a ``ckpts/<id>`` symlink. What it doesn't cover is
the *resume* side: a re-launched job calling ``find_latest_run_dir`` against
a real experiment directory and loading optimizer / state via ``load_local_state``.

That contract — between what training writes (``saving_completed`` marker,
``args.json``, ``stitched/*.pth``) and what the resume helpers read — is
exactly the kind of thing that quietly diverges as the save format evolves.
A unit test can pin the regex; only an integration test pins the byte-level
agreement between writer and reader.

Single dense family (Llama-3.2-3B-Instruct) is enough — the resume code path
is family-agnostic.
"""

from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.misc import set_seed
from _test_utils.torch.puzzletron.utils import setup_test_model_and_data
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.activation_scoring.score_pruning_activations as score_pruning_activations
import modelopt.torch.puzzletron.bypass_distillation as bypass_distillation
import modelopt.torch.puzzletron.pruning.pruning_ckpts as pruning_ckpts
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel import convert_model
from modelopt.torch.puzzletron.bypass_distillation.bypass_checkpoint_utils import (
    find_latest_run_dir,
)
from modelopt.torch.puzzletron.bypass_distillation.bypass_utils import set_experiment_id
from modelopt.torch.puzzletron.tools.hydra_utils import initialize_hydra_config_for_dir

# Match the constants in test_bypass.py so the run completes in two steps.
SEED = 1234
TRAINING_TOKENS = 128
BLOCK_SIZE = 64
PRUNED_INTERMEDIATE_SIZE = 256
PRUNED_NUM_KV_HEADS = 4

# One dense family — resume path is family-agnostic, so a second parametrize
# row would only add runtime, not coverage.
HF_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CONVERTER = "llama"


def _bypass_cfg_dict(*, find_last_ckpt_for_resume: bool) -> dict:
    """Minimal bypass config — derived from test_bypass.py's _make_bypass_cfg_dict
    for a dense family with FFN+KV pruning."""
    return {
        "dtype": "bf16",
        "seed": 42,
        "experiment_id": None,
        "experiment_dir": None,
        "iter_num": 1,
        "step_num": 1,
        "token_count": 0,
        "data": {
            "data_column": "conversation",
            "block_size": BLOCK_SIZE,
            "bos_rate": 0.5,
            "fim_rate": 0,
            "fim_spm_rate": 0,
            "source_datasets_to_discard": [],
            "load_from_disk": True,
            "keep_in_memory": False,
            "val_dataset_name": "valid",
            "max_eval_samples": 1,
            "eval_samples_per_process": None,
            "shuffle_train_data_seed": 42,
        },
        "training": {
            "learning_rate": 1e-4,
            "training_tokens": TRAINING_TOKENS,
            "micro_batch_size": 1,
            "val_micro_batch_size": 1,
            "warmup_ratio": 0.05,
            "warmup_steps": None,
            "min_lr_factor": 1e-5,
            "grad_accumulation_steps": 1,
            "skip_first_batches": 0,
            "weight_decay": 0.1,
            "decay_lr": True,
            "beta1": 0.9,
            "beta2": 0.95,
            "use_grad_scaling": False,
            "grad_clip": 1.0,
            "grad_clip_type": "norm",
            "clipping_count": 0,
            "log_interval": 5,
            "eval_interval": 100,
        },
        "resume_checkpoint_path": None,
        "find_last_ckpt_for_resume": find_last_ckpt_for_resume,
        "parameter_count": None,
        "init_checkpoint_path": None,
        "model": {
            "student_weights_dtype": "bf16",
            "model_overrides": {
                "delete_old_checkpoints": True,
                "save_interval_seconds": None,
                "save_interval": 1_000_000_000,
                "save_checkpoint_when_done": True,
            },
            "model_config_overrides": {
                "ffn": [{"intermediate_size": PRUNED_INTERMEDIATE_SIZE, "no_op": None}],
                "attention": [{"num_key_value_heads": PRUNED_NUM_KV_HEADS, "no_op": None}],
            },
        },
        "model_factory": {
            "factory": "bypass_factory_fn",
            "block_loss_func": "normalized_mse_loss",
            "gqa_init_mode": "AverageKV",
            "mlp_init_mode": "Truncate",
            "mlp_init_config": {"activations_log_dir": None},
            "linear_init_mode": "FromTeacher",
            "submodule_for_loss_calculation": None,
            "keys_to_learn": "entire_block",
        },
        "disable_initial_validate": True,
        "validate_teacher_model": False,
        "validate_student_model": False,
        "disable_validation": True,
        "best_val_loss": 1e9,
        "compile": False,
        "disable_fa2": False,
        "teacher_model_load_on_cpu": False,
        "save_checkpoint_before_training": False,
        "disable_checkpoint_save": False,
        "save_best_ckpt": True,
        "kill_after_first_save": False,
        "realize_best_or_latest": "best",
        "wandb_log": False,
        "wandb": {"project": None, "entity": None},
    }


def _expected_experiment_dir(puzzle_dir: Path, bypass_cfg_dict: dict) -> Path:
    """Compute the experiment directory the runtime will choose."""
    cfg = OmegaConf.create({"bypass": dict(bypass_cfg_dict)})
    set_experiment_id(cfg)
    return puzzle_dir / "bypass/bypass_runs" / cfg.bypass.experiment_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_bypass_resume_finds_latest_checkpoint(project_root_path: Path, tmp_path: Path):
    """Run bypass once, verify ``find_latest_run_dir`` locates the saved
    checkpoint, then re-launch with ``find_last_ckpt_for_resume=True`` and
    verify the second run resumes from the saved iter_num.
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(_resume_job, project_root_path, tmp_path),
        backend="nccl",
    )


def _resume_job(project_root_path: Path, tmp_path: Path, rank: int, size: int):
    set_seed(SEED)
    dist.setup(timeout=timedelta(10))

    puzzle_dir, hf_checkpoint_path, dataset_path = setup_test_model_and_data(
        tmp_path, rank, HF_MODEL_NAME, hybrid_override_pattern=None
    )

    hydra_config_dir = str(project_root_path / "tests/gpu/torch/puzzletron/resources/configs")
    hydra_config_name = f"{HF_MODEL_NAME}/{Path(HF_MODEL_NAME).name}"

    if rank == 0:
        convert_model(
            input_dir=str(hf_checkpoint_path),
            output_dir=str(puzzle_dir / "ckpts/teacher"),
            converter=CONVERTER,
        )
    dist.barrier()

    import hydra

    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[f"puzzle_dir={puzzle_dir}", f"dataset_path={dataset_path}"],
    )
    hydra_cfg = hydra.utils.instantiate(hydra_cfg)

    score_pruning_activations.launch_score_activations(hydra_cfg)
    if rank == 0:
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    dist.barrier()

    # First bypass run — produces a real checkpoint.
    cfg_dict = _bypass_cfg_dict(find_last_ckpt_for_resume=False)
    OmegaConf.update(hydra_cfg, "bypass", cfg_dict, merge=True)
    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    experiment_dir = _expected_experiment_dir(puzzle_dir, cfg_dict)
    if rank == 0:
        # The save side wrote what the resume side expects.
        assert experiment_dir.exists(), f"Expected experiment dir at {experiment_dir}"
        latest = find_latest_run_dir(experiment_dir)
        assert latest is not None, f"find_latest_run_dir returned None for {experiment_dir}"
        assert (Path(latest) / "saving_completed").exists(), (
            f"Resume target {latest} missing saving_completed marker"
        )
        assert (Path(latest) / "args.json").exists(), (
            f"Resume target {latest} missing args.json — load path would crash"
        )
    dist.barrier()

    # Second bypass run — re-uses the same experiment_dir, finds the latest
    # checkpoint via ``find_last_ckpt_for_resume=True``, and resumes.
    # Reset cfg.bypass to a fresh dict (experiment_id back to None so
    # set_experiment_id recomputes the same id from model_config_overrides).
    cfg_dict_resume = _bypass_cfg_dict(find_last_ckpt_for_resume=True)
    cfg_dict_resume["training"]["training_tokens"] = TRAINING_TOKENS * 2
    OmegaConf.update(hydra_cfg, "bypass", cfg_dict_resume, merge=True)
    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        # After the second run, iter_num must have advanced past 1 — proving
        # the run picked up state from the first run rather than starting fresh.
        # (The resume code path overwrites iter_num from args.json on line 826.)
        assert hydra_cfg.bypass.iter_num > 1, (
            f"Resume failed: iter_num={hydra_cfg.bypass.iter_num} suggests fresh start, "
            f"not a resume from the saved checkpoint"
        )

    dist.cleanup()
