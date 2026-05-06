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

"""
Puzzletron NAS plugin for the Modelopt framework (based on Puzzle algorithm: https://arxiv.org/abs/2411.19146).

It is used by mtn.convert() to convert a model from HF format to Puzzletron heterogeneous format + do pruning scoring
and save pruned checkpoints, and by mtn.search() to perform the MIP-based NAS search.
"""

from pathlib import Path

import hydra
from torch import nn

import modelopt.torch.utils.distributed as dist
from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher, SearchStateDict

from . import bypass_distillation
from .activation_scoring import launch_score_activations
from .anymodel.converter import ConverterFactory
from .anymodel.model_descriptor import ModelDescriptorFactory
from .build_library_and_stats import launch_build_library_and_stats
from .mip import launch_mip_and_realize_model
from .pruning import launch_prune_ckpt
from .scoring import launch_scoring
from .tools.hydra_utils import initialize_hydra_config_for_dir
from .tools.logger import mprint

__all__ = [
    "PuzzletronModel",
    "PuzzletronConfig",
    "PuzzletronDescriptor",
    "PuzzletronSearcher",
    "convert_puzzletron_model",
    "restore_puzzletron_model",
]


class PuzzletronModel(nn.Module):
    pass  # No model implementation is needed for the puzzletron mode


class PuzzletronConfig(ModeloptBaseConfig):
    """Configuration for Puzzletron NAS algorithm."""

    # Input model path to compress in the HF format
    input_model_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Hydra config directory containing the search space definition
    hydra_config_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Hydra config name containing the search space definition
    hydra_config_name: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Directory to save the compressed model and intermediate results
    puzzle_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    # Dataset path to use for scoring in prunining and NAS search
    dataset_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )


_StageName = str

# Canonical stage order. Stages absent from a given run (e.g. "bypass" when
# bypass isn't configured) are skipped, but the rest keep their relative order.
_STAGE_ORDER: tuple[_StageName, ...] = (
    "start",
    "convert",
    "score_activations",
    "prune",
    "bypass",
    "build_library",
    "score_blocks",
    "mip",
    "complete",
)


def _total_steps(hydra_cfg) -> int:
    """Return total pipeline step count: 9 with bypass, 8 without."""
    return 9 if hydra_cfg.get("bypass", None) is not None else 8


def _progress_step(hydra_cfg, stage: _StageName) -> tuple[int, int]:
    """Return ``(step_number, total_steps)`` for a given pipeline stage.

    Single source of truth for the user-facing ``Puzzletron Progress N/T`` strings —
    keeps numbering coherent across ``main.py``, ``convert_puzzletron_model``, and
    ``PuzzletronSearcher.run_search``, and shifts MIP/realize automatically when
    bypass is added or removed.
    """
    has_bypass = hydra_cfg.get("bypass", None) is not None
    total = _total_steps(hydra_cfg)
    step = 0
    for s in _STAGE_ORDER:
        if s == "bypass" and not has_bypass:
            continue
        step += 1
        if s == stage:
            return step, total
    raise ValueError(f"Unknown pipeline stage: {stage!r}")


def convert_puzzletron_model(model: nn.Module, config: PuzzletronConfig) -> ConvertReturnType:
    """1. Convert the model from HF format to AnyModel format.
    2. Score the pruning activations.
    3. Prune the model and save pruned checkpoints.
    4. (Optional) Run bypass distillation.

    The output of this step will be used by mnt.search() to perform the NAS search.
    """
    # Required for mtn.search() to read NAS configuration
    model.hydra_config_dir = config.hydra_config_dir
    model.hydra_config_name = config.hydra_config_name
    model.puzzle_dir = config.puzzle_dir
    model.dataset_path = config.dataset_path

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=config.hydra_config_dir,
        config_name=config.hydra_config_name,
        overrides=[
            f"puzzle_dir={config.puzzle_dir}",
            f"dataset_path={config.dataset_path}",
        ],
    )
    # Instantiate nested Hydra configs (e.g., pruning_mixin, hook_class)
    hydra_cfg = hydra.utils.instantiate(hydra_cfg)

    has_bypass = hydra_cfg.get("bypass", None) is not None
    convert_step, N = _progress_step(hydra_cfg, "convert")
    score_step, _ = _progress_step(hydra_cfg, "score_activations")
    prune_step, _ = _progress_step(hydra_cfg, "prune")

    # Step 2: Convert HuggingFace model to Puzzletron heterogeneous format
    hf_ckpt_teacher_dir = "ckpts/teacher"  # TODO: make it configurable
    teacher_dir = Path(config.puzzle_dir) / hf_ckpt_teacher_dir
    if dist.is_master():
        if (teacher_dir / "config.json").exists():
            mprint(
                f"Puzzletron Progress {convert_step}/{N}: teacher checkpoint already exists, skipping conversion"
            )
        else:
            mprint(
                f"Puzzletron Progress {convert_step}/{N}: converting model to Puzzletron heterogeneous format (single-gpu)"
            )

            # Get descriptor and converter from the hydra config
            descriptor_name = hydra_cfg.descriptor
            descriptor = ModelDescriptorFactory.get(descriptor_name)
            converter = ConverterFactory.get(descriptor_name)

            # Auto-download from HuggingFace if path doesn't exist locally
            input_model_path = config.input_model_path
            if not Path(input_model_path).exists():
                from huggingface_hub import snapshot_download

                if input_model_path.startswith("https://huggingface.co/"):
                    model_id = "/".join(input_model_path.rstrip("/").split("/")[-2:])
                else:
                    model_id = input_model_path  # assume HF model ID like "org/model-name"
                mprint(
                    f"Downloading HuggingFace model '{model_id}' — this may take several minutes "
                    f"for large models. Other ranks are waiting at a barrier."
                )
                input_model_path = snapshot_download(repo_id=model_id)
                mprint(f"Downloaded to: {input_model_path}")

            converter.convert(
                descriptor=descriptor,
                input_dir=Path(input_model_path),
                output_dir=teacher_dir,
            )
    dist.barrier()

    # Step 3: Score pruning activations (distributed processing)
    activations_log_dir = Path(hydra_cfg.pruning.activations_log_dir)
    if activations_log_dir.exists() and any(activations_log_dir.glob("rank_*.pth")):
        mprint(
            f"Puzzletron Progress {score_step}/{N}: pruning activation scores already exist, skipping scoring"
        )
        dist.barrier()
    else:
        mprint(f"Puzzletron Progress {score_step}/{N}: scoring pruning activations (multi-gpu)")
        launch_score_activations(hydra_cfg)

    # Step 4: Prune the model and save pruned checkpoints (single process)
    pruned_ckpts_dir = Path(hydra_cfg.pruning.pruned_ckpts_output_dir)
    if dist.is_master():
        if pruned_ckpts_dir.exists() and any(pruned_ckpts_dir.iterdir()):
            mprint(
                f"Puzzletron Progress {prune_step}/{N}: pruned checkpoints already exist, skipping pruning"
            )
        else:
            mprint(
                f"Puzzletron Progress {prune_step}/{N}: pruning the model and saving pruned checkpoints (single-gpu)"
            )
            launch_prune_ckpt(hydra_cfg)
    dist.barrier()

    # Step 5: Bypass distillation (optional, distributed processing)
    if has_bypass:
        bypass_step, _ = _progress_step(hydra_cfg, "bypass")
        # Skip if a previous run already produced bypass checkpoints. The realize step
        # writes a `latest` symlink under each experiment_dir; if any exists, bypass has
        # completed and rerunning would waste 5-15 min on teacher load + dataloader setup
        # before its own resume-from-checkpoint logic short-circuits.
        bypass_runs_dir = Path(config.puzzle_dir) / "bypass" / "bypass_runs"
        bypass_done = bypass_runs_dir.exists() and any(
            (run_dir / "latest").exists()
            for run_dir in bypass_runs_dir.iterdir()
            if run_dir.is_dir()
        )
        if bypass_done:
            mprint(
                f"Puzzletron Progress {bypass_step}/{N}: bypass distillation already completed, skipping"
            )
        else:
            mprint(
                f"Puzzletron Progress {bypass_step}/{N}: running bypass distillation (multi-gpu)"
            )
            bypass_distillation.launch_bypass_distillation(hydra_cfg)
        dist.barrier()

    return model, {}


def restore_puzzletron_model(
    model: nn.Module, config: PuzzletronConfig, metadata: MetadataDict
) -> nn.Module:
    """Restore is not needed for the puzzletron mode as we are not saving any model state"""
    return model


@NASModeRegistry.register_mode
class PuzzletronDescriptor(ModeDescriptor):
    """Descriptor for the Puzzletron mode."""

    @property
    def name(self) -> str:
        """String identifier for this mode."""
        return "puzzletron"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Configuration class for this mode."""
        return PuzzletronConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Return the associated searcher implementation."""

        return PuzzletronSearcher

    @property
    def convert(self) -> ConvertEntrypoint:
        """Entrypoint to convert a model."""
        return convert_puzzletron_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """Entrypoint to restore a model."""
        return restore_puzzletron_model

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode.
        For now, this will be a no-op as there is no modelopt's concept of search space defined
        for the puzzletron algorithm.
        """
        return "export_nas"


class PuzzletronSearcher(BaseSearcher):
    """Runs NAS search for the Puzzletron mode."""

    @property
    def default_state_dict(self) -> SearchStateDict:
        """Not needed for the puzzletron mode as we are not saving any model state"""
        return {}

    def run_search(self) -> None:
        # Load hydra config
        hydra_cfg = initialize_hydra_config_for_dir(
            config_dir=self.model.hydra_config_dir,
            config_name=self.model.hydra_config_name,
            overrides=[
                f"puzzle_dir={self.model.puzzle_dir}",
                f"dataset_path={self.model.dataset_path}",
            ],
        )
        # Instantiate nested Hydra configs (e.g., pruning_mixin, hook_class)
        hydra_cfg = hydra.utils.instantiate(hydra_cfg)

        library_step, N = _progress_step(hydra_cfg, "build_library")
        scoring_step, _ = _progress_step(hydra_cfg, "score_blocks")
        mip_step, _ = _progress_step(hydra_cfg, "mip")

        # Build replacement library and subblock statistics (single process)
        puzzle_dir = Path(self.model.puzzle_dir)
        replacement_library_path = puzzle_dir / "replacement_library.json"
        subblock_stats_path = puzzle_dir / hydra_cfg.calc_subblock_stats.subblock_stats_filename
        # Detect a stale library: any ckpts/* entry newer than the library file means
        # a new replacement (e.g. bypass-trained subblocks) appeared after the last build
        # and must be picked up. Without this check, our skip-if-done would happily reuse
        # a no-bypass library even after bypass completes.
        ckpts_dir = puzzle_dir / "ckpts"
        library_is_stale = False
        if replacement_library_path.exists() and ckpts_dir.exists():
            library_mtime = replacement_library_path.stat().st_mtime
            for entry in ckpts_dir.iterdir():
                # Resolve symlinks (bypass + pruning checkpoints land here as symlinks
                # to the real directories elsewhere under puzzle_dir).
                resolved = entry.resolve() if entry.is_symlink() else entry
                if resolved.exists() and resolved.stat().st_mtime > library_mtime:
                    library_is_stale = True
                    mprint(
                        f"Replacement library is stale: '{entry.name}' is newer than the existing library, will rebuild."
                    )
                    break
        if dist.is_master():
            if (
                replacement_library_path.exists()
                and subblock_stats_path.exists()
                and not library_is_stale
            ):
                mprint(
                    f"Puzzletron Progress {library_step}/{N}: replacement library and subblock stats already exist, skipping"
                )
            else:
                mprint(
                    f"Puzzletron Progress {library_step}/{N}: building replacement library and subblock statistics (single-gpu)"
                )
                launch_build_library_and_stats(hydra_cfg)
        dist.barrier()

        # Calculate one block scores (distributed processing)
        mprint(f"Puzzletron Progress {scoring_step}/{N}: calculating one block scores (multi-gpu)")
        launch_scoring(hydra_cfg)

        # MIP search and realize models (distributed processing)
        mprint(f"Puzzletron Progress {mip_step}/{N}: running MIP and realizing models (multi-gpu)")
        launch_mip_and_realize_model(hydra_cfg)
