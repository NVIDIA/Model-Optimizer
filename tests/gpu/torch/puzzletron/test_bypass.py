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

"""GPU integration tests for bypass distillation (blockwise local distillation).

Each test is parametrized over the same model families covered by ``test_puzzletron.py``
(see ``PUZZLETRON_FAMILIES`` in ``tests/_test_utils/torch/puzzletron/utils.py``).

Tiny model dimensions used throughout (set by ``setup_test_model_and_data``):
  - hidden_size: 256, intermediate_size: 512, num_layers: max(2, world_size)
  - num_attention_heads: 32, num_key_value_heads: 8
  - num_local_experts: 16 (MoE families only, e.g. Qwen3-VL)
  - training_tokens: 128, block_size: 64, micro_batch_size: 1  -> max_steps = 2

Pruning targets (used by all four tests):
  - pruned intermediate_size: 256 (dense) — half of teacher
  - pruned num_local_experts: 8 (MoE)    — half of teacher
  - pruned num_key_value_heads: 4         — half of teacher

mlp_init_mode is family-aware:
  - Dense families use ``Truncate`` (FFN intermediate slicing in the generic path).
  - MoE families use ``ExpertRemoval`` and delegate per-expert weight slicing to the
    ``experts_removal`` mixin registered on the descriptor. ``mlp_init_config`` is
    sourced from the family's pruning YAML (``mlp_init_config_yaml``) — no
    per-family branching needed in this test file.

To add a new model family:
  1. Append one row to PUZZLETRON_FAMILIES in tests/_test_utils/torch/puzzletron/utils.py.
  2. Ensure tests/gpu/torch/puzzletron/resources/configs/<family>/<family>.yaml exists
     and that setup_test_model_and_data() can build a tiny stand-in for it.
  3. For MoE families, ensure the family's descriptor registers ``"kv_heads"`` and
     ``"experts_removal"`` in ``pruning_mixins()`` (see e.g. NemotronH, GPT-OSS,
     Qwen3-VL descriptors).
  4. The four bypass tests below pick up the new row automatically.
"""

import copy
from datetime import timedelta
from functools import partial
from pathlib import Path

import hydra
import pytest
import torch
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.misc import set_seed
from _test_utils.torch.puzzletron.utils import PUZZLETRON_FAMILIES, setup_test_model_and_data
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.activation_scoring.score_pruning_activations as score_pruning_activations
import modelopt.torch.puzzletron.bypass_distillation as bypass_distillation
import modelopt.torch.puzzletron.pruning.pruning_ckpts as pruning_ckpts
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel import convert_model
from modelopt.torch.puzzletron.bypass_distillation.bypass_utils import set_experiment_id
from modelopt.torch.puzzletron.tools.hydra_utils import initialize_hydra_config_for_dir

# ---------------------------------------------------------------------------
# Constants — shared tiny-model dimensions and pruning targets
# ---------------------------------------------------------------------------

SEED = 1234

# Teacher tiny-model dimensions (set uniformly by setup_test_model_and_data)
TEACHER_INTERMEDIATE_SIZE = 512
TEACHER_NUM_KV_HEADS = 8
TEACHER_NUM_LOCAL_EXPERTS = 16

# Pruned targets (half of teacher)
PRUNED_INTERMEDIATE_SIZE = 256
PRUNED_NUM_KV_HEADS = 4
PRUNED_NUM_LOCAL_EXPERTS = 8

# Training budget: 128 tokens / (64 block * 1 mbs) = 2 steps — completes fast
TRAINING_TOKENS = 128
BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block_override(has_moe_layers: bool, pruned: bool = True) -> dict:
    """Return a single FFN-block override entry, family-aware.

    When ``pruned=True`` the override compresses the block (halves intermediate size for
    dense or halves num_local_experts for MoE). When ``pruned=False`` it pins the block
    to teacher size — used by tests that exercise attention pruning while keeping the FFN
    side fixed.
    """
    if has_moe_layers:
        n_experts = PRUNED_NUM_LOCAL_EXPERTS if pruned else TEACHER_NUM_LOCAL_EXPERTS
        return {"moe": {"num_local_experts": n_experts}, "no_op": None}
    intermediate = PRUNED_INTERMEDIATE_SIZE if pruned else TEACHER_INTERMEDIATE_SIZE
    return {"intermediate_size": intermediate, "no_op": None}


def _mlp_init_settings(has_moe_layers: bool, hydra_cfg) -> tuple[str, dict]:
    """Return ``(mlp_init_mode, mlp_init_config)`` for the family.

    Dense families use ``Truncate`` (FFN intermediate slicing). MoE families use
    ``ExpertRemoval``, which delegates per-expert weight slicing to the
    ``experts_removal`` mixin registered on the descriptor. The expert-scores
    metadata (``expert_scores_key``, ``layer_prefix_template``) is read directly
    from the family's pruning YAML — no per-family branching here.
    """
    if not has_moe_layers:
        return "Truncate", {"activations_log_dir": None}

    mlp_init_config = (
        OmegaConf.to_container(
            hydra_cfg.pruning.get("mlp_init_config_yaml", OmegaConf.create({})),
            resolve=True,
        )
        or {}
    )
    mlp_init_config["activations_log_dir"] = str(hydra_cfg.pruning.activations_log_dir)
    return "ExpertRemoval", mlp_init_config


def _make_bypass_cfg_dict(
    has_moe_layers: bool,
    hydra_cfg,
    *,
    include_block_override: bool = True,
    block_pruned: bool = True,
    include_attention_override: bool = True,
    attention_pruned: bool = True,
    configs_list: list | None = None,
) -> dict:
    """Return a plain-dict bypass config suitable for OmegaConf.update injection.

    Args:
        has_moe_layers: Whether the model family is MoE (dispatches FFN override shape
            and the mlp_init_mode).
        hydra_cfg: The post-pruning hydra config — used to source the family's
            ``mlp_init_config_yaml`` and ``activations_log_dir`` for MoE expert removal.
        include_block_override / block_pruned: Whether to override the per-block FFN
            sub-component, and whether to prune (vs. pin to teacher).
        include_attention_override / attention_pruned: Same for the attention sub-component.
        configs_list: If provided, populates bypass.configs for a multi-config sweep.
    """
    overrides: dict = {}
    if include_block_override:
        overrides["ffn"] = [_block_override(has_moe_layers, pruned=block_pruned)]
    if include_attention_override:
        kv = PRUNED_NUM_KV_HEADS if attention_pruned else TEACHER_NUM_KV_HEADS
        overrides["attention"] = [{"num_key_value_heads": kv, "no_op": None}]

    mlp_init_mode, mlp_init_config = _mlp_init_settings(has_moe_layers, hydra_cfg)

    cfg = {
        "dtype": "bf16",
        "seed": 42,
        "experiment_id": None,
        "experiment_dir": None,
        "iter_num": 1,
        "step_num": 1,
        "token_count": 0,
        "data": {
            # The dummy test dataset stores conversations under the "conversation" column.
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
            # Large eval_interval so validation is skipped during this short run.
            # Validation is fully disabled anyway (disable_validation=True below).
            "eval_interval": 100,
        },
        "resume_checkpoint_path": None,
        "find_last_ckpt_for_resume": False,
        "parameter_count": None,
        "init_checkpoint_path": None,
        "model": {
            "student_weights_dtype": "bf16",
            "model_overrides": {
                "delete_old_checkpoints": True,
                "save_interval_seconds": None,
                # Effectively disable step-interval saving; rely on save_checkpoint_when_done.
                "save_interval": 1_000_000_000,
                "save_checkpoint_when_done": True,
            },
            "model_config_overrides": overrides,
        },
        "model_factory": {
            "factory": "bypass_factory_fn",
            "block_loss_func": "normalized_mse_loss",
            "gqa_init_mode": "AverageKV",
            "mlp_init_mode": mlp_init_mode,
            "mlp_init_config": mlp_init_config,
            "linear_init_mode": "FromTeacher",
            "submodule_for_loss_calculation": None,
            "keys_to_learn": "entire_block",
        },
        # Disable all validation to keep tests fast.
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
        # Do NOT use kill_after_first_save — it raises RuntimeError which becomes sys.exit(1).
        # Instead let the short training run (2 steps) complete naturally.
        "kill_after_first_save": False,
        "realize_best_or_latest": "best",
        "wandb_log": False,
        "wandb": {"project": None, "entity": None},
    }

    if configs_list is not None:
        cfg["configs"] = configs_list

    return cfg


def _expected_experiment_id(bypass_cfg_dict: dict) -> str:
    """Compute the experiment_id that ``set_experiment_id`` will assign.

    Avoids duplicating the formula in tests — uses the same function the runtime uses.
    """
    cfg = OmegaConf.create({"bypass": copy.deepcopy(bypass_cfg_dict)})
    set_experiment_id(cfg)
    return cfg.bypass.experiment_id


def _setup_hydra_cfg_and_pruning(
    project_root_path: Path,
    tmp_path: Path,
    rank: int,
    size: int,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
) -> tuple:
    """Set up the tiny model, convert it, score activations, and create pruning ckpts.

    Returns ``(puzzle_dir, dataset_path, hydra_cfg)``.

    Steps performed:
    1. Create a small HF model and dummy dataset via ``setup_test_model_and_data``.
    2. Convert the HF checkpoint to AnyModel/DeciLM format (rank 0 only).
    3. Load the per-family Hydra config with ``puzzle_dir`` and ``dataset_path`` overrides.
    4. Run ``score_pruning_activations`` (distributed).
    5. Run ``pruning_ckpts`` (rank 0 only) then barrier.
    """
    set_seed(SEED)
    dist.setup(timeout=timedelta(10))

    puzzle_dir, hf_checkpoint_path, dataset_path = setup_test_model_and_data(
        tmp_path, rank, hf_model_name, hybrid_override_pattern
    )

    hydra_config_dir = str(project_root_path / "tests/gpu/torch/puzzletron/resources/configs")
    # Per-family hydra config name follows the layout configs/<family>/<basename>/<basename>.
    hydra_config_name = f"{hf_model_name}/{Path(hf_model_name).name}"

    # Step 0: Convert HF checkpoint to AnyModel/DeciLM format.
    if rank == 0:
        convert_model(
            input_dir=str(hf_checkpoint_path),
            output_dir=str(puzzle_dir / "ckpts/teacher"),
            converter=converter,
        )
    dist.barrier()

    # Step 1: Load Hydra config.
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[
            f"puzzle_dir={puzzle_dir}",
            f"dataset_path={dataset_path}",
        ],
    )
    hydra_cfg = hydra.utils.instantiate(hydra_cfg)

    # Step 2: Score pruning activations (distributed).
    score_pruning_activations.launch_score_activations(hydra_cfg)

    # Step 3: Create pruning checkpoints (rank 0 only).
    if rank == 0:
        pruning_ckpts.launch_prune_ckpt(hydra_cfg)
    dist.barrier()

    return puzzle_dir, dataset_path, hydra_cfg


# ---------------------------------------------------------------------------
# Tests — each parametrized over PUZZLETRON_FAMILIES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    PUZZLETRON_FAMILIES,
)
def test_bypass_block_pruning(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
):
    """Bypass distillation with the per-block sub-component pruned.

    For dense families, prunes FFN intermediate (512 -> 256). For MoE families,
    prunes num_local_experts (16 -> 8). KV heads are also halved (8 -> 4).
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_block_pruning_job,
            project_root_path,
            tmp_path,
            hf_model_name,
            converter,
            hybrid_override_pattern,
            has_moe_layers,
        ),
        backend="nccl",
    )


def _test_bypass_block_pruning_job(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
    rank: int,
    size: int,
):
    puzzle_dir, _, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path,
        tmp_path,
        rank,
        size,
        hf_model_name,
        converter,
        hybrid_override_pattern,
    )

    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, hydra_cfg)
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_experiment_id = _expected_experiment_id(bypass_cfg_dict)
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / expected_experiment_id
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id

        assert experiment_dir.exists(), (
            f"Expected bypass experiment directory to exist: {experiment_dir}"
        )
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_block_pruning[{hf_model_name}] completed. "
        f"Puzzle directory: {puzzle_dir}"
    )


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    PUZZLETRON_FAMILIES,
)
def test_bypass_kv_head_compression(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
):
    """Bypass distillation with KV heads halved (8 -> 4) and FFN block pinned to teacher.

    For dense, the experiment_id will be ``bypass_ffn_512_heads_4`` (FFN at teacher size,
    attention halved). For MoE, ``bypass_experts_16_heads_4``.
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_kv_head_compression_job,
            project_root_path,
            tmp_path,
            hf_model_name,
            converter,
            hybrid_override_pattern,
            has_moe_layers,
        ),
        backend="nccl",
    )


def _test_bypass_kv_head_compression_job(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
    rank: int,
    size: int,
):
    puzzle_dir, _, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path,
        tmp_path,
        rank,
        size,
        hf_model_name,
        converter,
        hybrid_override_pattern,
    )

    bypass_cfg_dict = _make_bypass_cfg_dict(
        has_moe_layers,
        hydra_cfg,
        block_pruned=False,  # keep FFN/experts at teacher
        attention_pruned=True,  # halve KV heads
    )
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_experiment_id = _expected_experiment_id(bypass_cfg_dict)
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / expected_experiment_id
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id

        assert experiment_dir.exists(), (
            f"Expected bypass experiment directory to exist: {experiment_dir}"
        )
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_kv_head_compression[{hf_model_name}] completed. "
        f"Puzzle directory: {puzzle_dir}"
    )


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    PUZZLETRON_FAMILIES,
)
def test_bypass_multi_config_sequential(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
):
    """Bypass distillation sweep: two configs run sequentially via bypass.configs list.

    Config 0: block pruned + attention pruned
    Config 1: block at teacher + attention pruned
    Both checkpoint symlinks must exist after the sweep completes.
    """
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_multi_config_sequential_job,
            project_root_path,
            tmp_path,
            hf_model_name,
            converter,
            hybrid_override_pattern,
            has_moe_layers,
        ),
        backend="nccl",
    )


def _test_bypass_multi_config_sequential_job(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
    rank: int,
    size: int,
):
    puzzle_dir, _, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path,
        tmp_path,
        rank,
        size,
        hf_model_name,
        converter,
        hybrid_override_pattern,
    )

    configs_list = [
        {
            "model_config_overrides": {
                "ffn": [_block_override(has_moe_layers, pruned=True)],
                "attention": [{"num_key_value_heads": PRUNED_NUM_KV_HEADS, "no_op": None}],
            },
            "keys_to_learn": "entire_block",
        },
        {
            "model_config_overrides": {
                "ffn": [_block_override(has_moe_layers, pruned=False)],
                "attention": [{"num_key_value_heads": PRUNED_NUM_KV_HEADS, "no_op": None}],
            },
            "keys_to_learn": "entire_block",
        },
    ]
    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, hydra_cfg, configs_list=configs_list)
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        # Compute expected IDs by running set_experiment_id against each sub-config.
        expected_ids = []
        for sub in configs_list:
            sub_cfg = copy.deepcopy(bypass_cfg_dict)
            sub_cfg["model"]["model_config_overrides"] = sub["model_config_overrides"]
            sub_cfg["experiment_id"] = None
            expected_ids.append(_expected_experiment_id(sub_cfg))

        for experiment_id in expected_ids:
            experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id
            ckpt_symlink = puzzle_dir / "ckpts" / experiment_id

            assert experiment_dir.exists(), (
                f"Expected bypass experiment directory to exist: {experiment_dir}"
            )
            assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
                f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
            )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_multi_config_sequential[{hf_model_name}] completed. "
        f"Puzzle directory: {puzzle_dir}"
    )


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    PUZZLETRON_FAMILIES,
)
def test_bypass_checkpoint_contents(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
):
    """Verify that a bypass checkpoint contains expected HuggingFace model files."""
    spawn_multiprocess_job(
        size=torch.cuda.device_count(),
        job=partial(
            _test_bypass_checkpoint_contents_job,
            project_root_path,
            tmp_path,
            hf_model_name,
            converter,
            hybrid_override_pattern,
            has_moe_layers,
        ),
        backend="nccl",
    )


def _test_bypass_checkpoint_contents_job(
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
    rank: int,
    size: int,
):
    puzzle_dir, _, hydra_cfg = _setup_hydra_cfg_and_pruning(
        project_root_path,
        tmp_path,
        rank,
        size,
        hf_model_name,
        converter,
        hybrid_override_pattern,
    )

    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, hydra_cfg)
    OmegaConf.update(hydra_cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(hydra_cfg)
    dist.barrier()

    if rank == 0:
        expected_experiment_id = _expected_experiment_id(bypass_cfg_dict)
        ckpt_symlink = puzzle_dir / "ckpts" / expected_experiment_id

        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink: {ckpt_symlink}"
        )

        # The symlink resolves to the latest checkpoint dir; verify HF config exists.
        resolved = ckpt_symlink.resolve()
        config_json = resolved / "config.json"
        assert config_json.exists(), (
            f"Expected HuggingFace config.json inside checkpoint: {config_json}"
        )

        # The saving_completed marker must be present (set by save_bypass_checkpoint).
        saving_completed = resolved / "saving_completed"
        assert saving_completed.exists(), (
            f"Expected saving_completed marker inside checkpoint: {saving_completed}"
        )

    dist.cleanup()

    print(
        f"PYTEST SUMMARY: test_bypass_checkpoint_contents[{hf_model_name}] completed. "
        f"Puzzle directory: {puzzle_dir}"
    )
