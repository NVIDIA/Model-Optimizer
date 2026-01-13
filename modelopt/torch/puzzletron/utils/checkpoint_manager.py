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

"""Checkpoint manager for activation hook scoring with periodic saves and resume support."""

import json
import time
from pathlib import Path
from typing import Any

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools.logger import aprint, mprint


class ScoringCheckpointManager:
    """Manages checkpointing for activation hook scoring with periodic saves."""

    def __init__(self, checkpoint_dir: str, activation_hooks=None, checkpoint_interval: int = 100):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            activation_hooks: Dictionary of activation hooks to manage
            checkpoint_interval: Save checkpoint every N batches
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.activation_hooks = activation_hooks
        self.checkpoint_interval = checkpoint_interval
        self.rank = dist.rank()
        self.is_main_process = dist.is_master()

        # Debug: Log checkpoint manager initialization
        hook_count = len(activation_hooks) if activation_hooks else 0
        aprint(
            f"[Rank {self.rank}] Checkpoint manager initialized: {hook_count} hooks, dir: {checkpoint_dir}"
        )

        # Checkpoint files
        self.progress_file = self.checkpoint_dir / "scoring_progress.json"
        self.hook_states_file = self.checkpoint_dir / f"hook_states_rank_{self.rank}.pth"

        # Progress tracking
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = time.time()

        # Ensure directory exists
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load existing checkpoint if available, including hook states.

        Returns:
            Dict with checkpoint info or None if no checkpoint exists
        """
        aprint(f"[Rank {self.rank}] Looking for checkpoint at: {self.progress_file}")
        if not self.progress_file.exists():
            aprint(f"[Rank {self.rank}] No checkpoint file found at {self.progress_file}")
            return None

        try:
            with open(self.progress_file) as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint
            if "current_batch" in checkpoint_data and "total_batches" in checkpoint_data:
                self.current_batch = checkpoint_data["current_batch"]
                self.total_batches = checkpoint_data["total_batches"]

                mprint(
                    f"Found checkpoint: batch {self.current_batch}/{self.total_batches} ({checkpoint_data.get('progress', 0.0):.1%})"
                )
                mprint(
                    f"Will resume from batch {self.current_batch}, skipping batches 0-{self.current_batch - 1}"
                )

                # Load hook states if hooks are available
                if self.activation_hooks is not None:
                    success = self.load_hook_states(self.activation_hooks)
                    if success:
                        aprint(
                            f"[Rank {self.rank}] Successfully loaded hook states from checkpoint"
                        )
                    else:
                        aprint(f"[Rank {self.rank}] Failed to load hook states - starting fresh")

                return checkpoint_data
            else:
                aprint(
                    f"[Rank {self.rank}] Invalid checkpoint format (missing current_batch/total_batches): {checkpoint_data}"
                )
                return None

        except (json.JSONDecodeError, KeyError) as e:
            mprint(f"Error loading checkpoint: {e}")

        return None

    def load_hook_states(self, activation_hooks) -> bool:
        """Load hook states from checkpoint files.

        Args:
            activation_hooks: Hook objects to load states into

        Returns:
            bool: True if hook states were successfully loaded, False otherwise
        """
        import os

        # Each rank loads only its own hook states
        current_rank = int(os.environ.get("RANK", 0))
        hook_states_path = self.checkpoint_dir / f"hook_states_rank_{current_rank}.pth"

        if hook_states_path.exists():
            aprint(f"[Rank {current_rank}] Loading hook states from {hook_states_path}")
            try:
                import torch

                hook_states = torch.load(hook_states_path, map_location="cpu")

                # Load states into corresponding hooks
                loaded_count = 0
                for module_name, hook in activation_hooks.items():
                    if module_name in hook_states:
                        hook.load_state(hook_states[module_name])
                        loaded_count += 1

                        # Log progress info if available (only for a few hooks to avoid spam)
                        if loaded_count <= 3:  # Only log first few hooks
                            progress_info = hook.get_progress_info()
                            if progress_info:
                                aprint(f"[Rank {current_rank}]   {module_name}: {progress_info}")
                    else:
                        aprint(
                            f"[Rank {current_rank}] Warning: No saved state found for hook: {module_name}"
                        )

                aprint(
                    f"[Rank {current_rank}] Successfully loaded states for {loaded_count}/{len(activation_hooks)} hooks"
                )
                return True

            except Exception as e:
                aprint(f"[Rank {current_rank}] Error loading hook states: {e}")
                return False
        else:
            aprint(f"[Rank {current_rank}] No hook states file found at {hook_states_path}")
            return False

    def should_skip_batch(self, batch_idx: int) -> bool:
        """Check if we should skip this batch (already processed in previous run)."""
        should_skip = batch_idx < self.current_batch
        if should_skip and batch_idx % 10 == 0:  # Log every 10th skipped batch to avoid spam
            mprint(f"Skipping batch {batch_idx} (resume from batch {self.current_batch})")
        return should_skip

    def update_progress(self, batch_idx: int, total_batches: int):
        """Update progress and potentially save checkpoint.

        Args:
            batch_idx: Current batch index
            total_batches: Total number of batches
        """
        self.current_batch = batch_idx
        self.total_batches = total_batches

        # Save checkpoint periodically or on completion
        should_save = (
            (batch_idx % self.checkpoint_interval == 0)  # Periodic save
            or (batch_idx == total_batches - 1)  # Final batch
        )

        if should_save:
            # All ranks save their hook states
            if self.activation_hooks is not None:
                try:
                    from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import ForwardHook

                    ForwardHook.save_hook_states(self.activation_hooks, self.checkpoint_dir)
                except Exception as e:
                    mprint(f"Warning: Failed to save hook states: {e}")

            # Only main process saves progress info
            if self.is_main_process:
                self.save_checkpoint()

            # Synchronize all ranks after checkpointing
            dist.barrier()

    def save_checkpoint(self):
        """Save current checkpoint to disk (progress info only).
        Hook states are saved separately in update_progress.
        """
        try:
            # Save progress
            progress_data = {
                "current_batch": self.current_batch,
                "total_batches": self.total_batches,
                "progress": self.current_batch / self.total_batches
                if self.total_batches > 0
                else 0.0,
                "timestamp": time.time(),
                "elapsed_time": time.time() - self.start_time,
                "rank": self.rank,
            }

            # Write progress atomically
            temp_file = self.progress_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(progress_data, f, indent=2)
            temp_file.replace(self.progress_file)

            # Hook states are saved at a higher level to ensure all ranks participate

            if self.current_batch % (self.checkpoint_interval) == 0:
                progress_pct = progress_data["progress"] * 100
                elapsed = progress_data["elapsed_time"]
                mprint(
                    f"Checkpoint saved: batch {self.current_batch}/{self.total_batches} ({progress_pct:.1f}%), elapsed: {elapsed:.1f}s"
                )

        except Exception as e:
            mprint(f"Error saving checkpoint: {e}")

    def finalize(self):
        """Mark scoring as completed."""
        # All ranks save their final hook states
        if self.activation_hooks is not None:
            try:
                from modelopt.torch.nas.plugins.megatron_hooks.base_hooks import ForwardHook

                saved_path = ForwardHook.save_hook_states(
                    self.activation_hooks, self.checkpoint_dir
                )
                mprint(f"Final hook states saved to {saved_path}")
            except Exception as e:
                mprint(f"Warning: Failed to save final hook states: {e}")

        # Only main process saves progress info
        if self.is_main_process:
            self.current_batch = self.total_batches
            self.save_checkpoint()
            mprint(f"Scoring completed and finalized: {self.total_batches} batches processed")

        # Synchronize all ranks after finalization
        dist.barrier()
