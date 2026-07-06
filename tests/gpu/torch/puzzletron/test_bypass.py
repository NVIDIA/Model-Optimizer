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

The tests use representative model families instead of parametrizing every scenario over
the full Puzzletron family matrix:

  - Llama-3.2-3B: dense FFN pruning with ``mlp_init_mode="Truncate"``.
  - GPT-OSS-20B: MoE expert pruning, windowed attention, and attention sinks.

The broader no-bypass family matrix remains covered by ``test_puzzletron.py``.

To keep GPU CI within its time budget, all bypass scenarios for a family run inside a
single spawned multi-process job that performs the expensive one-time setup
(``setup_test_model_and_data`` -> convert -> score activations -> pruning ckpts) *once*
and then executes each scenario sequentially against that shared teacher/pruning state.
This is safe because ``launch_bypass_distillation`` is idempotent across sequential
launches (already exercised by the multi-config sweep) and each scenario writes to a
distinct ``experiment_id`` directory, so runs never clobber each other. Every scenario
helper is a collective operation: it runs on all ranks and gates assertions on rank 0.

Tiny model dimensions used throughout (set by ``setup_test_model_and_data``):
  - hidden_size: 256, intermediate_size: 512, num_layers: max(2, world_size)
  - num_attention_heads: 32, num_key_value_heads: 8
  - num_local_experts: 16 (MoE families only, e.g. Qwen3-VL)
  - training_tokens: 128, block_size: 64, micro_batch_size: 1  -> max_steps = 2

Pruning targets (used by all scenarios):
  - pruned intermediate_size: 256 (dense) — half of teacher
  - pruned num_local_experts: 8 (MoE)    — half of teacher
  - pruned num_key_value_heads: 4         — half of teacher

mlp_init_mode is family-aware:
  - Dense families use ``Truncate`` (FFN intermediate slicing in the generic path).
  - MoE families use ``ExpertRemoval`` and delegate per-expert weight slicing to the
    ``experts_removal`` mixin registered on the descriptor. ``mlp_init_config`` is
    sourced from the family's pruning YAML (``mlp_init_config_yaml``) — no
    per-family branching needed in this test file.

To add a new bypass-specific model family, add a targeted scenario job below instead of
expanding every scenario by default.
"""

import copy
import json
from datetime import timedelta
from pathlib import Path

import hydra
import pytest
import torch
from _test_utils.torch.misc import set_seed
from _test_utils.torch.puzzletron.utils import setup_test_model_and_data
from omegaconf import OmegaConf

import modelopt.torch.puzzletron.activation_scoring.score_pruning_activations as score_pruning_activations
import modelopt.torch.puzzletron.bypass_distillation as bypass_distillation
import modelopt.torch.puzzletron.pruning.pruning_ckpts as pruning_ckpts
import modelopt.torch.puzzletron.replacement_library.build_replacement_library as build_lib
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.anymodel import ModelDescriptorFactory, convert_model
from modelopt.torch.puzzletron.bypass_distillation.bypass_checkpoint_utils import (
    find_latest_run_dir,
)
from modelopt.torch.puzzletron.bypass_distillation.bypass_utils import set_experiment_id
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_state_dict
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config
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

# Static tiny-dims bypass config skeleton loaded by ``_make_bypass_cfg_dict``.
BYPASS_TEST_CONFIG_PATH = Path(__file__).parent / "resources/configs/bypass_test_defaults.yaml"

# Llama-3.2-3B is the smallest dense family and the canonical "FFN bypass" path.
LLAMA_FAMILY = pytest.param(
    "meta-llama/Llama-3.2-3B-Instruct", "llama", None, False, id="llama-3.2-3B"
)
# GPT-OSS adds MoE expert pruning (mlp_init_mode="ExpertRemoval") and windowed
# attention with sinks — different code paths than dense Llama.
GPT_OSS_FAMILY = pytest.param("openai/gpt-oss-20b", "gpt_oss", None, True, id="gpt-oss-20b")


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

    # Load the static tiny-dims skeleton from yaml, then inject the per-scenario
    # dynamic fields below. ``resolve=True`` returns a plain dict, matching the
    # OmegaConf.update injection sites downstream.
    cfg = OmegaConf.to_container(OmegaConf.load(BYPASS_TEST_CONFIG_PATH), resolve=True)

    # Re-apply the values that the test module owns as the single source of truth.
    cfg["data"]["block_size"] = BLOCK_SIZE
    cfg["training"]["training_tokens"] = TRAINING_TOKENS

    # Per-scenario / per-family dynamic fields.
    cfg["model"]["model_config_overrides"] = overrides
    cfg["model_factory"]["mlp_init_mode"] = mlp_init_mode
    cfg["model_factory"]["mlp_init_config"] = mlp_init_config

    if configs_list is not None:
        cfg["configs"] = configs_list

    return cfg


def _expected_experiment_id(hydra_cfg, bypass_cfg_dict: dict) -> str:
    """Compute the experiment_id that ``set_experiment_id`` will assign.

    Avoids duplicating the formula in tests while preserving the top-level
    teacher identity that the runtime includes in the hash.
    """
    cfg = copy.deepcopy(hydra_cfg)
    OmegaConf.update(cfg, "bypass", copy.deepcopy(bypass_cfg_dict), merge=False)
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
    dist.setup(timeout=timedelta(minutes=10))

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
# Scenarios
#
# Each scenario is a collective operation: it runs on all ranks against the
# shared post-setup ``hydra_cfg`` and gates its assertions on rank 0. Scenarios
# deepcopy the base config so their bypass overrides never leak into the next
# scenario, and each pins an explicit ``experiment_id`` (or, for the multi-config
# sweep, relies on the auto architecture-derived IDs) so their output
# directories never collide. Every scenario ends with a ``dist.barrier()`` so all
# ranks re-synchronize before the next collective launch.
# ---------------------------------------------------------------------------


def _scenario_block_pruning(hydra_cfg, puzzle_dir: Path, has_moe_layers: bool, rank: int) -> None:
    """Bypass distillation with the per-block sub-component pruned.

    For dense families, prunes FFN intermediate (512 -> 256). For MoE families,
    prunes num_local_experts (16 -> 8). KV heads are also halved (8 -> 4).
    """
    cfg = copy.deepcopy(hydra_cfg)
    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, cfg)
    bypass_cfg_dict["experiment_id"] = "bypass_scn_block_pruning"
    OmegaConf.update(cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    if rank == 0:
        experiment_id = bypass_cfg_dict["experiment_id"]
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id
        ckpt_symlink = puzzle_dir / "ckpts" / experiment_id

        assert experiment_dir.exists(), (
            f"Expected bypass experiment directory to exist: {experiment_dir}"
        )
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
        )
        resolved = ckpt_symlink.resolve()
        assert (resolved / "config.json").exists(), (
            f"Expected HuggingFace config.json inside checkpoint: {resolved}"
        )
        assert (resolved / "saving_completed").exists(), (
            f"Expected saving_completed marker inside checkpoint: {resolved}"
        )

    dist.barrier()


def _scenario_kv_head_compression(
    hydra_cfg, puzzle_dir: Path, has_moe_layers: bool, rank: int
) -> None:
    """Bypass distillation with KV heads halved (8 -> 4) and FFN block pinned to teacher."""
    cfg = copy.deepcopy(hydra_cfg)
    bypass_cfg_dict = _make_bypass_cfg_dict(
        has_moe_layers,
        cfg,
        block_pruned=False,  # keep FFN/experts at teacher
        attention_pruned=True,  # halve KV heads
    )
    bypass_cfg_dict["experiment_id"] = "bypass_scn_kv_head"
    OmegaConf.update(cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    if rank == 0:
        experiment_id = bypass_cfg_dict["experiment_id"]
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id
        ckpt_symlink = puzzle_dir / "ckpts" / experiment_id

        assert experiment_dir.exists(), (
            f"Expected bypass experiment directory to exist: {experiment_dir}"
        )
        assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
            f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
        )

    dist.barrier()


def _scenario_multi_config_sequential(
    hydra_cfg, puzzle_dir: Path, has_moe_layers: bool, rank: int
) -> None:
    """Bypass distillation sweep: two configs run sequentially via bypass.configs list.

    Config 0: block pruned + attention pruned
    Config 1: block at teacher + attention pruned
    Both checkpoint symlinks must exist after the sweep completes. The sub-run
    experiment IDs are the auto architecture-derived IDs (``ffn_256_heads_4`` /
    ``ffn_512_heads_4``), which are distinct from the explicit ``bypass_scn_*`` IDs
    the other scenarios use, so nothing collides.
    """
    cfg = copy.deepcopy(hydra_cfg)
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
    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, cfg, configs_list=configs_list)
    OmegaConf.update(cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    if rank == 0:
        # Compute expected IDs by running set_experiment_id against each sub-config.
        expected_ids = []
        for sub in configs_list:
            sub_cfg = copy.deepcopy(bypass_cfg_dict)
            sub_cfg["model"]["model_config_overrides"] = sub["model_config_overrides"]
            sub_cfg["experiment_id"] = None
            expected_ids.append(_expected_experiment_id(cfg, sub_cfg))

        for experiment_id in expected_ids:
            experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id
            ckpt_symlink = puzzle_dir / "ckpts" / experiment_id

            assert experiment_dir.exists(), (
                f"Expected bypass experiment directory to exist: {experiment_dir}"
            )
            assert ckpt_symlink.exists() or ckpt_symlink.is_symlink(), (
                f"Expected bypass checkpoint symlink to exist: {ckpt_symlink}"
            )

    dist.barrier()


def _scenario_resume_from_checkpoint(
    hydra_cfg, puzzle_dir: Path, has_moe_layers: bool, rank: int
) -> None:
    """Two-phase scenario: train + save, then re-launch with resume and verify continuity.

    Phase 1: short bypass run (2 steps), checkpoint saved under
        ``puzzle_dir/bypass/bypass_runs/<exp_id>/step-NNNNNN-ckpt/``.
    Phase 2: same hydra_cfg + ``find_last_ckpt_for_resume=True`` + double the
        training_tokens budget. The resume path in
        ``training_loop.run_bypassed_training:805-840`` must restore
        ``iter_num`` / ``step_num`` / ``token_count`` from the saved
        ``args.json`` and load stitched-module + optimizer state from disk.

    Both phases pin the same explicit ``experiment_id`` so phase 2 discovers phase
    1's checkpoint.

    The GradScaler save/load mechanism added in the recent CodeRabbit-driven
    fix is tested separately in
    ``tests/gpu/torch/puzzletron/test_bypass_checkpoint_utils.py`` because
    GradScaler is fp16-only and the bypass test infrastructure ships bf16,
    which makes ``GradScaler.step()`` raise on the unscale path.
    """
    experiment_id = "bypass_scn_resume"
    experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id

    # ---- Phase 1: train + save ---------------------------------------------
    cfg = copy.deepcopy(hydra_cfg)
    phase1_cfg = _make_bypass_cfg_dict(has_moe_layers, cfg)
    phase1_cfg["experiment_id"] = experiment_id
    phase1_cfg["find_last_ckpt_for_resume"] = False
    OmegaConf.update(cfg, "bypass", phase1_cfg, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    phase1_iter_num = None
    if rank == 0:
        resume_checkpoint = find_latest_run_dir(experiment_dir)
        assert resume_checkpoint is not None, f"Phase 1 missing resume checkpoint: {experiment_dir}"
        args_json_path = Path(resume_checkpoint) / "args.json"
        stitched_dir = Path(resume_checkpoint) / "stitched"
        # Phase 1 must have produced the canonical artifacts.
        assert args_json_path.exists(), f"Phase 1 missing args.json: {args_json_path}"
        with open(args_json_path) as f:
            phase1_state = json.load(f)
        phase1_iter_num = phase1_state["iter_num"]
        assert phase1_iter_num > 1, (
            f"Phase 1 should have advanced past iter 1, got {phase1_iter_num}"
        )

        # Optimizer state must be present (covers the resume path's load).
        assert (stitched_dir / "block_0.optimizer_state.pth").exists(), stitched_dir

    dist.barrier()

    # ---- Phase 2: resume and continue --------------------------------------
    cfg = copy.deepcopy(hydra_cfg)
    phase2_cfg = _make_bypass_cfg_dict(has_moe_layers, cfg)
    phase2_cfg["experiment_id"] = experiment_id
    phase2_cfg["find_last_ckpt_for_resume"] = True
    # Double the budget so the resumed run takes additional steps.
    phase2_cfg["training"]["training_tokens"] = TRAINING_TOKENS * 2
    OmegaConf.update(cfg, "bypass", phase2_cfg, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    if rank == 0:
        phase2_resume_checkpoint = find_latest_run_dir(experiment_dir)
        assert phase2_resume_checkpoint is not None, f"Phase 2 missing checkpoint: {experiment_dir}"
        phase2_args_json_path = Path(phase2_resume_checkpoint) / "args.json"
        assert phase2_args_json_path.exists(), "Phase 2 should have args.json"
        with open(phase2_args_json_path) as f:
            phase2_state = json.load(f)
        phase2_iter_num = phase2_state["iter_num"]
        # The resumed run must have moved past phase 1's last iter — proves
        # both that resume happened (didn't restart at 1) and that further
        # training executed.
        assert phase2_iter_num > phase1_iter_num, (
            f"Resume did not advance: phase1={phase1_iter_num}, phase2={phase2_iter_num}"
        )

    dist.barrier()


def _scenario_subblock_mode(
    hydra_cfg, puzzle_dir: Path, has_moe_layers: bool, keys_to_learn: str, rank: int
) -> None:
    """Verify that ``keys_to_learn`` correctly freezes the right param groups.

    For each keys_to_learn value:
      - Run bypass for 2 steps with that keys_to_learn.
      - After training, load the saved HF-format checkpoints.
      - Compare against the start checkpoint, which holds the post-init pre-train weights:
          * subblock_ffn → only FFN keys differ from init; attention identical.
          * subblock_attention → only attention keys differ; FFN identical.
          * entire_block → both differ.

    GPT-OSS coverage matters because the MoE expert path uses
    ``mlp_init_mode="ExpertRemoval"`` instead of ``"Truncate"``, and GPT-OSS's
    windowed attention adds attention-sink parameters that the freeze must
    correctly include in the "attention" group.
    """
    cfg = copy.deepcopy(hydra_cfg)
    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, cfg)
    bypass_cfg_dict["experiment_id"] = f"bypass_scn_{keys_to_learn}"
    bypass_cfg_dict["model_factory"]["keys_to_learn"] = keys_to_learn
    # Save start-of-training checkpoint so we can diff trained-vs-init.
    bypass_cfg_dict["save_checkpoint_before_training"] = True
    OmegaConf.update(cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    if rank == 0:
        experiment_id = bypass_cfg_dict["experiment_id"]
        experiment_dir = puzzle_dir / "bypass/bypass_runs" / experiment_id
        # `start-step-*` is the pre-training snapshot (saved when
        # save_checkpoint_before_training=True). The post-training snapshot
        # under this short-budget config lives at `final-step-*` (saved by the
        # early-exit branch in training_loop.py); the periodic `step-*` save
        # never fires because the budget is only 2 steps. `latest` is now a
        # resume pointer only, so use the final checkpoint directly.
        start_dirs = sorted(experiment_dir.glob("start-step-*-ckpt"))
        assert start_dirs, f"Expected a start-step-* checkpoint under {experiment_dir}"
        start_dir = start_dirs[0]
        final_dirs = sorted(experiment_dir.glob("final-step-*-ckpt"))
        assert final_dirs, f"Expected a final-step-* checkpoint under {experiment_dir}"
        end_dir = final_dirs[-1].resolve()
        assert end_dir != start_dir.resolve(), (
            f"Final checkpoint still points at the pre-training snapshot {end_dir} - "
            "no post-training checkpoint was written."
        )

        # Diff the HF-format checkpoint tensors. ``stitched/`` stores only
        # optimizer/scaler state; model weights live in the checkpoint root.
        start_state = load_state_dict(start_dir)
        end_state = load_state_dict(end_dir)
        descriptor = ModelDescriptorFactory.get(cfg.descriptor)
        model_config = load_model_config(start_dir)
        lm_config = descriptor.get_language_model_config(model_config)
        weight_groups = descriptor.get_weight_groups(
            start_state.keys() & end_state.keys(),
            lm_config.num_hidden_layers,
        )
        key_kinds = {
            key: "attn" if group_name.endswith("_attention") else "ffn"
            for group_name, keys in weight_groups.items()
            if group_name.endswith(("_attention", "_ffn"))
            for key in keys
        }

        ffn_changed = False
        attn_changed = False
        for key in start_state.keys() & end_state.keys():
            kind = key_kinds.get(key)
            if kind is None:
                continue
            changed = not torch.equal(start_state[key], end_state[key])
            if kind == "ffn" and changed:
                ffn_changed = True
            if kind == "attn" and changed:
                attn_changed = True

        if keys_to_learn == "subblock_ffn":
            assert ffn_changed, "subblock_ffn should change FFN weights"
            assert not attn_changed, "subblock_ffn should leave attention weights bit-identical"
        elif keys_to_learn == "subblock_attention":
            assert attn_changed, "subblock_attention should change attention weights"
            assert not ffn_changed, "subblock_attention should leave FFN weights bit-identical"
        else:  # entire_block
            assert ffn_changed and attn_changed, (
                f"entire_block should change both groups; got ffn={ffn_changed}, attn={attn_changed}"
            )

    dist.barrier()


def _scenario_then_build_library(
    hydra_cfg, puzzle_dir: Path, has_moe_layers: bool, rank: int
) -> None:
    """Run bypass, then build the replacement library; assert bypass entries appear.

    Verifies the wiring between the bypass step and the downstream NAS step:
    - ``realize_bypass_checkpoints`` creates a symlink at ``ckpts/<exp_id>``.
    - ``_get_last_checkpoint_from_each_experiment`` resolves it back to the
      bypass run dir.
    - ``_build_subblocks_df``'s priority sort puts the bypass-rooted path
      before non-bypass ones in the resulting DataFrame.
    - The final ``replacement_library.json`` includes entries pointing at
      the bypass experiment.

    Runs last in the scenario sequence, so discovery also exercises a puzzle_dir
    populated with the other scenarios' bypass runs.
    """
    cfg = copy.deepcopy(hydra_cfg)
    bypass_cfg_dict = _make_bypass_cfg_dict(has_moe_layers, cfg)
    bypass_cfg_dict["experiment_id"] = "bypass_scn_build_lib"
    OmegaConf.update(cfg, "bypass", bypass_cfg_dict, merge=True)

    bypass_distillation.launch_bypass_distillation(cfg)
    dist.barrier()

    if rank == 0:
        experiment_id = bypass_cfg_dict["experiment_id"]
        ckpts_dir = puzzle_dir / "ckpts"

        # 1. The realize step must have created a symlink for this bypass run.
        bypass_symlink = ckpts_dir / experiment_id
        assert bypass_symlink.is_symlink() or bypass_symlink.exists(), (
            f"Expected bypass symlink at {bypass_symlink}"
        )

        # 2. Discovery must find the bypass entry alongside the teacher (and any
        #    pruning-pipeline outputs from the setup helper).
        discovered = build_lib._get_last_checkpoint_from_each_experiment(puzzle_dir)
        bypass_resolved = bypass_symlink.resolve()
        assert bypass_resolved in discovered, (
            f"Bypass run not discovered. Resolved={bypass_resolved}, discovered={discovered}"
        )
        # The resolved bypass path must contain "bypass" + "bypass_runs" in its
        # parts so the priority sort picks it up.
        assert "bypass" in bypass_resolved.parts and "bypass_runs" in bypass_resolved.parts

        # 3. Build the replacement library and verify the bypass entry appears.
        teacher_dir = ckpts_dir / "teacher"
        subblocks_df = build_lib._build_subblocks_df(
            master_puzzle_dir=puzzle_dir,
            teacher_checkpoint_dir=teacher_dir,
            add_ffn_no_ops=False,
            add_attention_no_ops=False,
            trust_remote_code=False,
        )
        # Some subblock row's checkpoint_dir column must reference a bypass run.
        # The sibling scenarios in this merged job produce the same student
        # architecture (ffn=256 / heads=4), so ``_build_subblocks_df``'s
        # ``drop_duplicates(keep="first")`` may award a given subblock row to
        # another bypass run rather than this one — but its priority sort still
        # guarantees the winner is bypass-rooted over any untrained pruned
        # source. Assert that bypass entries flow into the library (the wiring
        # under test); the ``bypass_resolved in discovered`` check above already
        # proves this specific run was realized and discovered.
        # FFN-only rows leave attention_checkpoint_dir as NaN (and vice versa); we
        # drop those before string-casting because pandas' .astype(str) doesn't
        # reliably stringify NaN on object-dtype columns, and 'X' in float('nan')
        # raises TypeError.
        attn_sources = subblocks_df["attention_checkpoint_dir"].dropna().astype(str).tolist()
        ffn_sources = subblocks_df["ffn_checkpoint_dir"].dropna().astype(str).tolist()
        assert any("bypass_runs" in s for s in attn_sources + ffn_sources), (
            f"replacement_library subblocks_df has no bypass-sourced rows. "
            f"attn_sources={set(attn_sources)}, ffn_sources={set(ffn_sources)}"
        )

    dist.barrier()


# ---------------------------------------------------------------------------
# Tests — one spawned job per family, running all scenarios after a single setup
# ---------------------------------------------------------------------------


def _run_llama_scenarios_job(
    rank: int,
    size: int,
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
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

    _scenario_block_pruning(hydra_cfg, puzzle_dir, has_moe_layers, rank)
    _scenario_kv_head_compression(hydra_cfg, puzzle_dir, has_moe_layers, rank)
    _scenario_multi_config_sequential(hydra_cfg, puzzle_dir, has_moe_layers, rank)
    _scenario_subblock_mode(hydra_cfg, puzzle_dir, has_moe_layers, "subblock_ffn", rank)
    _scenario_resume_from_checkpoint(hydra_cfg, puzzle_dir, has_moe_layers, rank)
    # build_library runs last so discovery sees a fully populated puzzle_dir.
    _scenario_then_build_library(hydra_cfg, puzzle_dir, has_moe_layers, rank)

    # NOTE: no dist.cleanup() here — the dist_workers pool owns the process-group
    # lifecycle and reuses it across tests.
    print(
        f"PYTEST SUMMARY: test_bypass_llama_scenarios[{hf_model_name}] completed. "
        f"Puzzle directory: {puzzle_dir}"
    )


def _run_gpt_oss_scenarios_job(
    rank: int,
    size: int,
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
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

    _scenario_block_pruning(hydra_cfg, puzzle_dir, has_moe_layers, rank)
    _scenario_subblock_mode(hydra_cfg, puzzle_dir, has_moe_layers, "subblock_attention", rank)

    # NOTE: no dist.cleanup() here — the dist_workers pool owns the process-group
    # lifecycle and reuses it across tests.
    print(
        f"PYTEST SUMMARY: test_bypass_gpt_oss_scenarios[{hf_model_name}] completed. "
        f"Puzzle directory: {puzzle_dir}"
    )


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    [LLAMA_FAMILY],
)
def test_bypass_llama_scenarios(
    dist_workers,
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
):
    """All dense-family bypass scenarios in one spawned job (single shared setup).

    Covers block pruning, KV-head compression, the multi-config sequential sweep,
    the subblock_ffn freeze mode, resume-from-checkpoint, and the bypass ->
    build-replacement-library wiring. Each scenario writes to its own
    ``experiment_id`` directory so their outputs never collide.
    """
    dist_workers.run(
        _run_llama_scenarios_job,
        project_root_path,
        tmp_path,
        hf_model_name,
        converter,
        hybrid_override_pattern,
        has_moe_layers,
    )


@pytest.mark.parametrize(
    ("hf_model_name", "converter", "hybrid_override_pattern", "has_moe_layers"),
    [GPT_OSS_FAMILY],
)
def test_bypass_gpt_oss_scenarios(
    dist_workers,
    project_root_path: Path,
    tmp_path: Path,
    hf_model_name: str,
    converter: str,
    hybrid_override_pattern: str | None,
    has_moe_layers: bool,
):
    """MoE-family bypass scenarios in one spawned job (single shared setup).

    Covers MoE expert-removal block pruning and the subblock_attention freeze mode
    (which must include GPT-OSS's windowed-attention sink parameters in the
    attention group). Each scenario writes to its own ``experiment_id`` directory.
    """
    dist_workers.run(
        _run_gpt_oss_scenarios_job,
        project_root_path,
        tmp_path,
        hf_model_name,
        converter,
        hybrid_override_pattern,
        has_moe_layers,
    )
