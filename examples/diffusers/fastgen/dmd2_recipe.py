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

"""DMD2 distillation recipe built on NeMo AutoModel.

This recipe subclasses :class:`nemo_automodel.recipes.diffusion.train.TrainDiffusionRecipe`
so it inherits AutoModel's student + optimizer + dataloader + checkpoint plumbing, then
drives ``modelopt.torch.fastgen.DMDPipeline`` (or a plugin subclass) through the
three-phase DMD2 alternation (student update / fake-score update / EMA step).

Backbone: **Qwen-Image** (``Qwen/Qwen-Image``) — 4D ``image_latents``,
:class:`QwenImageDMDPipeline` handles 2x2 patch packing / img_shapes /
unpacking. Config: ``configs/dmd2_qwen_image.yaml`` — the canonical
real-data run (4-step + CFG + GAN).

Launch::

    torchrun --nproc-per-node=8 \\
        examples/diffusers/fastgen/dmd2_finetune.py \\
        --config examples/diffusers/fastgen/configs/dmd2_qwen_image.yaml

See ``examples/diffusers/fastgen/README.md`` for the three-phase
alternation diagram + troubleshooting notes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

# nemo_automodel is required to run this example (installed via requirements.txt). Wrap
# the import in a clear, actionable error, but still re-raise so it fails loudly with a
# real stack — a previous gate that fell back to ``object`` silently masked missing deps
# and surfaced as a downstream ``TypeError: takes no arguments``.
try:
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe, is_main_process
except ImportError as exc:
    raise ImportError(
        "The DMD2 fastgen example requires `nemo_automodel`. Install the example "
        "dependencies with:\n"
        "    pip install -r examples/diffusers/fastgen/requirements.txt"
    ) from exc
# Local sibling module (this example directory is on ``sys.path`` — see ``dmd2_finetune.py``).
# Provides the FSDP2 partial-load-tolerant optimizer restore so the example does not depend
# on a patched ``nemo_automodel.components.checkpoint.checkpointing``.
from fastgen_checkpoint import make_optimizer_partial_load_tolerant
from torch import nn

import modelopt.torch.fastgen as mtf
from modelopt.torch.fastgen.config import DMDConfig
from modelopt.torch.fastgen.discriminators import Discriminator_ImageDiT
from modelopt.torch.fastgen.methods.dmd import DMDPipeline
from modelopt.torch.fastgen.plugins import qwen_image as qwen_image_plugin

# Keys under the ``dmd2:`` YAML block that shadow fields on :class:`DMDConfig`. The
# recipe deep-merges these on top of the loaded built-in recipe so users can tweak DMD2
# hyperparameters without editing the shared
# ``modelopt_recipes/general/distillation/dmd2_qwen_image.yaml`` file.
_DMD_CONFIG_OVERRIDE_KEYS = frozenset(DMDConfig.model_fields.keys())


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base`` and return a new dict.

    Nested dicts (e.g. the ``sample_t_cfg`` / ``ema`` sub-configs) are merged key-by-key
    rather than replaced wholesale, so a YAML block that overrides a single sub-field
    keeps the recipe's other sub-fields instead of silently resetting them to
    :class:`DMDConfig` defaults.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


# Auto-detect substrings (matched case-insensitively against ``model_id``) that map to
# DMDPipeline plugin subclasses. Keep this list small — adding a new entry is only the
# right move when the model has a non-diffusers transformer signature that requires a
# pack/unpack wrapper. Models with the standard ``(hidden_states, timestep,
# encoder_hidden_states)`` signature work with the base :class:`DMDPipeline`.
_PIPELINE_PLUGIN_BY_MODEL_SUBSTR = (
    ("qwen-image", "qwen_image"),
    ("qwen_image", "qwen_image"),
)

_DMD_COMPLETE_MARKER = "dmd2_complete.marker"


class DMD2DiffusionRecipe(TrainDiffusionRecipe):
    """DMD2 recipe that reuses ``TrainDiffusionRecipe`` for the student path.

    What the superclass gives us (reused unchanged):

    - Student transformer + AdamW optimizer + LR scheduler, loaded via
      :class:`NeMoAutoDiffusionPipeline` with FSDP2 sharding.
    - ``self.dataloader`` / ``self.sampler`` (swapped to AutoModel's mock dataloader
      when ``data.use_mock: true`` — see :meth:`_build_dataloader`).
    - ``self.step_scheduler`` (gradient accumulation + checkpoint cadence).
    - ``self.checkpointer`` (DCP student weights + optimizer).
    - ``self.device`` / ``self.bf16`` / ``self.clip_grad_max_norm`` / etc.

    What this recipe adds:

    - A frozen teacher loaded via a second :meth:`NeMoAutoDiffusionPipeline.from_pretrained`
      call with the same ``parallel_scheme`` so it lands with the same FSDP2 sharding
      as the student.
    - A trainable fake-score transformer loaded the same way (weights identical to the
      teacher on step 0).
    - A separate AdamW optimizer for the fake-score phase.
    - An :class:`mtf.DMDPipeline` driving VSD + DSM + EMA.
    - Sidecar checkpoint save / restore for fake-score weights, fake-score optimizer,
      EMA shadow, and DMDPipeline iteration counters.

    Classifier-free guidance, the GAN discriminator branch, and real-data training are
    configurable via the ``dmd2:`` / ``data:`` YAML blocks — all enabled in the canonical
    ``configs/dmd2_qwen_image.yaml``. See
    ``examples/diffusers/fastgen/README.md`` for details.
    """

    # ------------------------------------------------------------------ #
    #  Setup                                                             #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Build the student via ``super()``, then add teacher / fake_score / DMDPipeline.

        The extras (``_teacher``, ``_fake_score``, ``_fake_score_optimizer``,
        ``_dmd_pipeline``, ``_dmd_config``) are assigned through ``self.__dict__[...]``
        to bypass :meth:`BaseRecipe.__setattr__`'s auto-tracking — otherwise they'd be
        added to ``__state_tracked`` and clobber the superclass's single-model /
        single-optimizer checkpoint loop.
        """
        # 1. Run the parent setup. Builds self.model / self.optimizer / self.lr_scheduler /
        #    self.dataloader / self.step_scheduler / self.checkpointer / etc. The parent's
        #    trailing call to self.load_checkpoint(self.restore_from) runs BEFORE our
        #    extras exist, so it only restores the student — that is intentional and safe.
        super().setup()

        # 2. Load the frozen teacher. Same from_pretrained path, same parallel_scheme, but
        #    ``load_for_training=False`` so the transformer comes back in eval mode with
        #    requires_grad=False. Bypass __setattr__ to stay invisible to the parent's
        #    __state_tracked loop.
        self.__dict__["_teacher"] = self._load_frozen_teacher()

        # 4. Load the trainable fake-score. Third from_pretrained call — weights start
        #    identical to the teacher (both come from the same HF checkpoint).
        self.__dict__["_fake_score"] = self._load_fake_score()

        # 5. Resolve the DMDConfig: load the fastgen built-in recipe, then apply any
        #    inline overrides under the YAML ``dmd2:`` block.
        self.__dict__["_dmd_config"] = self._resolve_dmd_config()

        # 6. Optimizer for the fake-score phase. LR defaults to student LR when
        #    ``dmd2.fake_score_lr`` isn't set.
        self.__dict__["_fake_score_optimizer"] = self._build_fake_score_optimizer()

        # 7. Optional GAN discriminator. Built when ``gan_loss_weight_gen > 0`` so the
        #    DMDPipeline constructor's assert is satisfied; otherwise ``discriminator=None``
        #    and that assert fires if a YAML enables GAN for an unsupported backbone.
        self.__dict__["_discriminator"] = self._build_discriminator()
        self.__dict__["_discriminator_optimizer"] = self._build_discriminator_optimizer()
        if self._discriminator is not None:
            self._attach_gan_feature_capture()

        # 8. DMDPipeline.
        #
        #    Dispatch to a plugin subclass when the backbone needs a non-diffusers call
        #    signature (e.g. Qwen-Image's packed-latents path). Default = base pipeline.
        pipeline_cls = self._resolve_pipeline_cls()
        pipeline_kwargs = self._resolve_pipeline_kwargs(pipeline_cls)
        self.__dict__["_dmd_pipeline"] = pipeline_cls(
            student=self.model,
            teacher=self._teacher,
            fake_score=self._fake_score,
            config=self._dmd_config,
            discriminator=self._discriminator,
            **pipeline_kwargs,
        )

        # 8. Drop the parent's flow_matching_pipeline — we replace the training loop,
        #    so keeping it around is pure deadweight. The attribute is not tracked by
        #    ``__state_tracked`` (FlowMatchingPipeline is a plain class), so ``del`` is
        #    safe.
        if hasattr(self, "flow_matching_pipeline"):
            del self.flow_matching_pipeline

        # 9. Extend the student-only restore that super().setup() already ran: also
        #    restore the fake_score / fake_score_optimizer / EMA / DMD state from the
        #    same checkpoint directory.
        self._restore_dmd_extras(getattr(self, "_dmd2_resolved_restore_from", self.restore_from))

        if is_main_process():
            logging.info("[DMD2] recipe initialized: %s", self._dmd_config_summary())
            logging.info("[DMD2] full configuration:\n%s", self._dmd_full_config_log())

    # ------------------------------------------------------------------ #
    #  Training loop                                                     #
    # ------------------------------------------------------------------ #

    def _rebuild_dataloader_for_resume(self, global_step: int) -> None:
        """Reset the dataloader to the true data position when resuming (no-op if ``global_step==0``).

        On resume the ``StatefulDataLoader``'s restored state does NOT advance past the
        resume point -- re-checkpointing after a resume fails to capture progress, so each
        window re-serves the same data slice (``_num_yielded`` climbs while the served
        sample is identical; verified on production checkpoints and the harness). The one
        reliably-restored counter is ``global_step``, so we discard the stuck loader state:
        rebuild a FRESH ``StatefulDataLoader`` and skip the deterministic sampler to the
        position implied by ``global_step`` -- epoch ``global_step // epoch_len``, skip
        ``(global_step % epoch_len) * grad_acc`` batches. Not wrapped in try/except: the
        inputs are a ``StatefulDataLoader``'s always-present attrs and the sampler's
        ``set_epoch`` / ``_batches_to_skip``, so it cannot fail here, and silently falling
        back to the stuck loader would reintroduce the re-serving bug. Regression test:
        tests/examples/diffusers/fastgen/test_resume_dataloader.py.
        """
        epoch_len = int(getattr(self.step_scheduler, "epoch_len", 0) or 0)
        grad_acc = int(getattr(self.step_scheduler, "grad_acc_steps", 1) or 1)
        if epoch_len <= 0 or self.sampler is None or global_step <= 0:
            return
        cur_epoch = global_step // epoch_len
        skip_batches = (global_step % epoch_len) * grad_acc
        _old = self.dataloader
        _kw = {
            "collate_fn": getattr(_old, "collate_fn", None),
            "num_workers": int(getattr(_old, "num_workers", 0) or 0),
            "pin_memory": bool(getattr(_old, "pin_memory", False)),
        }
        if _kw["num_workers"] > 0:
            _kw["prefetch_factor"] = getattr(_old, "prefetch_factor", 2)
            _kw["persistent_workers"] = bool(getattr(_old, "persistent_workers", False))
        self.dataloader = StatefulDataLoader(_old.dataset, batch_sampler=self.sampler, **_kw)
        self.step_scheduler.epoch = cur_epoch
        self.sampler.set_epoch(cur_epoch)
        self.sampler._batches_to_skip = skip_batches
        if is_main_process():
            logging.info(
                "[DMD2][resume-fix] fresh dataloader + sampler skip: epoch=%d "
                "skip_batches=%d (global_step=%d epoch_len=%d grad_acc=%d)",
                cur_epoch,
                skip_batches,
                global_step,
                epoch_len,
                grad_acc,
            )

    def run_train_validation_loop(self) -> None:
        """Three-phase DMD2 alternation driven by ``step_scheduler``.

        Each outer iteration picks either the student or fake-score phase based on
        ``global_step % student_update_freq``. The student phase runs
        ``compute_student_loss`` + ``update_ema``. The fake-score phase runs
        ``compute_fake_score_loss`` and, when a discriminator is configured
        (``gan_loss_weight_gen > 0``), ``compute_discriminator_loss``.

        Mirrors the gating in ``FastGen/fastgen/methods/distribution_matching/dmd2.py``
        (``_student_update_step`` / ``_fake_score_discriminator_update_step``).
        """
        dmd = self._dmd_pipeline
        cfg = self._dmd_config

        logging.info(
            "[DMD2] Starting DMD2 training on %s (pipeline=%s)",
            self.model_id,
            type(self._dmd_pipeline).__name__,
        )
        # Dataloader target (mock vs real cache) is non-obvious from the per-step
        # logs; surface it explicitly here so §16's "mock or real dataloader
        # target" bullet is checkable from the startup log.
        try:
            dl_target = type(self.dataloader.dataset).__name__
            logging.info("[DMD2] Dataloader dataset class: %s", dl_target)
        except Exception:
            pass
        logging.info(
            "[DMD2] Global batch size: %s; local batch size: %s; DP size: %s",
            self.global_batch_size,
            self.local_batch_size,
            self.dp_size,
        )
        logging.info(
            "[DMD2] student_update_freq=%d; fake_score_pred_type=%s; guidance_scale=%s;"
            " gan_loss_weight_gen=%s",
            cfg.student_update_freq,
            cfg.fake_score_pred_type,
            cfg.guidance_scale,
            cfg.gan_loss_weight_gen,
        )

        global_step = int(self.step_scheduler.step)

        # On resume, discard the StatefulDataLoader's stuck restored state and reset the
        # data position, epoch, and progress bar from the reliably-restored ``global_step``
        # (see ``_rebuild_dataloader_for_resume``; regression-tested in
        # tests/examples/diffusers/fastgen/test_resume_dataloader.py).
        self._rebuild_dataloader_for_resume(global_step)

        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

            # Progress bar: mirror the sampler's pending skip on the resumed (first)
            # epoch; the sampler zeroes it after the first __iter__, so later epochs
            # start at 0 automatically.
            tqdm_initial = int(getattr(self.sampler, "_batches_to_skip", 0) or 0)

            if is_main_process():
                from tqdm import tqdm

                self.step_scheduler.dataloader = tqdm(
                    self.dataloader,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs} (global step {global_step})",
                    initial=tqdm_initial,
                )
            else:
                self.step_scheduler.dataloader = self.dataloader

            epoch_student_loss = 0.0
            epoch_fake_score_loss = 0.0
            student_steps = 0
            fake_score_steps = 0

            for batch_group in self.step_scheduler:
                is_student_phase = (global_step % cfg.student_update_freq) == 0

                if is_student_phase:
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self._fake_score_optimizer.zero_grad(set_to_none=True)

                self._set_grad_requirements(is_student_phase)

                micro_losses: list[float] = []
                micro_vsd_losses: list[float] = []
                micro_disc_losses: list[float] = []
                for micro_batch in batch_group:
                    (
                        latents,
                        noise,
                        text_embeds,
                        text_mask,
                        neg_text_embeds,
                        neg_text_mask,
                    ) = self._prepare_micro_batch(micro_batch)

                    if is_student_phase:
                        # ``compute_student_loss`` reads ``guidance_scale`` from the
                        # DMDConfig when this kwarg is None. We pass the negative
                        # embedding unconditionally — the function ignores it when
                        # CFG is disabled, and raises a clear ValueError when CFG
                        # is enabled but no negative was supplied.
                        losses = dmd.compute_student_loss(
                            latents,
                            noise,
                            encoder_hidden_states=text_embeds,
                            encoder_hidden_states_mask=text_mask,
                            negative_encoder_hidden_states=neg_text_embeds,
                            negative_encoder_hidden_states_mask=neg_text_mask,
                            guidance_scale=None,
                        )
                        micro_vsd_losses.append(float(losses["vsd"].item()))
                    else:
                        losses = dmd.compute_fake_score_loss(
                            latents,
                            noise,
                            encoder_hidden_states=text_embeds,
                            encoder_hidden_states_mask=text_mask,
                        )

                    (losses["total"] / len(batch_group)).backward()
                    micro_losses.append(float(losses["total"].item()))

                    # GAN: in the fake-score phase, also update the discriminator
                    # on the same batch (FastGen pattern:
                    # _fake_score_discriminator_update_step).
                    if (
                        not is_student_phase
                        and self._discriminator is not None
                        and self._discriminator_optimizer is not None
                    ):
                        self._discriminator_optimizer.zero_grad(set_to_none=True)
                        disc_losses = dmd.compute_discriminator_loss(
                            latents,
                            noise,
                            encoder_hidden_states=text_embeds,
                            encoder_hidden_states_mask=text_mask,
                        )
                        (disc_losses["total"] / len(batch_group)).backward()
                        # Manual gradient all-reduce across DP ranks (the
                        # discriminator is replicated, not FSDP-sharded).
                        if dist.is_initialized():
                            for p in self._discriminator.parameters():
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(
                            self._discriminator.parameters(),
                            max_norm=self.clip_grad_max_norm,
                        )
                        self._discriminator_optimizer.step()
                        micro_disc_losses.append(float(disc_losses["total"].item()))

                # Grad clip on whichever module is the active trainable.
                if is_student_phase:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.clip_grad_max_norm
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._fake_score.parameters(), max_norm=self.clip_grad_max_norm
                    )
                grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                # Step.
                if is_student_phase:
                    self.optimizer.step()
                    dmd.update_ema()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler[0].step(1)
                else:
                    self._fake_score_optimizer.step()

                group_loss_mean = float(sum(micro_losses) / len(micro_losses))
                if is_student_phase:
                    epoch_student_loss += group_loss_mean
                    student_steps += 1
                else:
                    epoch_fake_score_loss += group_loss_mean
                    fake_score_steps += 1

                global_step = int(self.step_scheduler.step)

                if (
                    self.log_every
                    and self.log_every > 0
                    and is_main_process()
                    and (global_step % self.log_every == 0)
                ):
                    self._log_step(
                        global_step=global_step,
                        is_student_phase=is_student_phase,
                        group_loss=group_loss_mean,
                        grad_norm=grad_norm,
                        vsd_loss=(sum(micro_vsd_losses) / len(micro_vsd_losses))
                        if micro_vsd_losses
                        else None,
                        disc_loss=(sum(micro_disc_losses) / len(micro_disc_losses))
                        if micro_disc_losses
                        else None,
                    )

                if self.step_scheduler.is_ckpt_step:
                    # Use the group mean of the active phase as the reported train loss.
                    self.save_checkpoint(epoch, global_step, group_loss_mean)

            # End-of-epoch logging.
            if is_main_process():
                avg_student = (
                    (epoch_student_loss / student_steps) if student_steps else float("nan")
                )
                avg_fake = (
                    epoch_fake_score_loss / fake_score_steps if fake_score_steps else float("nan")
                )
                logging.info(
                    "[DMD2] Epoch %d complete. student_avg=%.6f (%d steps) "
                    "fake_score_avg=%.6f (%d steps)",
                    epoch + 1,
                    avg_student,
                    student_steps,
                    avg_fake,
                    fake_score_steps,
                )

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            reserved = torch.cuda.max_memory_reserved() / (1024**3)
            rank = dist.get_rank() if dist.is_initialized() else 0
            logging.info(
                "[DMD2] PEAK_MEM rank=%d max_allocated=%.2fGiB max_reserved=%.2fGiB",
                rank,
                peak,
                reserved,
            )

        if is_main_process():
            logging.info("[DMD2] Training complete. Final step: %s", global_step)

    # ------------------------------------------------------------------ #
    #  Checkpoint save / restore (sidecars next to student DCP)          #
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, restore_from: str | None = None):
        """Load only from checkpoints whose DMD2 sidecars are complete.

        ``TrainDiffusionRecipe.setup()`` calls this before the DMD2-only objects exist,
        so this method only resolves the path and delegates the student restore to the
        parent. The sidecars are restored later by ``_restore_dmd_extras``.
        """
        # Upgrade our checkpointer instance in place so optimizer restores tolerate FSDP2
        # partial shards. This single seam covers BOTH the parent student-optimizer restore
        # (``super().load_checkpoint`` below) and the later fake-score restore in
        # ``_restore_dmd_extras``. Instance-scoped; model-state load stays strict. Replaces the
        # upstream ``Checkpointer.load_optimizer`` ``allow_partial_load`` patch so stock
        # ``nemo_automodel`` can be used unmodified.
        make_optimizer_partial_load_tolerant(self.checkpointer)

        resolved = self._resolve_complete_dmd_checkpoint(restore_from)
        self.__dict__["_dmd2_resolved_restore_from"] = resolved

        if resolved is None:
            if (
                restore_from is not None
                and str(restore_from).upper() == "LATEST"
                and is_main_process()
            ):
                logging.warning(
                    "[DMD2] restore_from=LATEST but no complete DMD2 checkpoint was found in %s. "
                    "Starting fresh.",
                    self.checkpointer.config.checkpoint_dir,
                )
            return

        super().load_checkpoint(resolved)

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ) -> None:
        """Delegate student save to ``super()``, then sidecar the DMD2 extras."""
        # Recover from a partial save from a previous run (e.g. SLURM time
        # limit killed the job between super().save_checkpoint() — which writes
        # the model + step_scheduler + dataloader — and our DMD2 sidecar
        # writes below). The parent's save_checkpoint refuses to overwrite an
        # existing directory and raises FileExistsError, so without this we'd
        # need a manual cleanup every time a SLURM kill landed mid-save.
        path = os.path.join(self.checkpointer.config.checkpoint_dir, f"epoch_{epoch}_step_{step}")
        if is_main_process() and self.checkpointer.config.enabled and os.path.exists(path):
            if not self._is_dmd_checkpoint_complete(path):
                logging.warning(
                    "[DMD2] cleaning up incomplete checkpoint directory left by a previous run: %s",
                    path,
                )
                shutil.rmtree(path)
        if dist.is_initialized():
            dist.barrier()

        previous_complete = None
        if self.checkpointer.config.enabled:
            previous_complete = self._find_latest_complete_dmd_checkpoint(
                self.checkpointer.config.checkpoint_dir
            )

        super().save_checkpoint(epoch, step, train_loss, val_loss, best_metric_key)

        if not self.checkpointer.config.enabled:
            return

        # The parent save updates LATEST before DMD2 sidecars exist. Until the marker is
        # written below, keep LATEST on the previous complete DMD2 checkpoint.
        if is_main_process():
            if previous_complete is not None:
                self._update_latest_symlink(previous_complete)
            else:
                self._remove_checkpoint_pointer("LATEST")
        if dist.is_initialized():
            dist.barrier()

        self._save_dmd_extras(path)

        if dist.is_initialized():
            dist.barrier()

        if is_main_process():
            self._write_dmd_complete_marker(path)
            self._update_checkpoint_symlink("DMD2_LATEST", path)
            self._update_latest_symlink(path)
        if dist.is_initialized():
            dist.barrier()

    def _save_dmd_extras(self, path: str) -> None:
        """Write fake_score DCP + fake_score_optimizer DCP + ema_shadow.pt + dmd_state.pt."""
        # fake_score weights — DCP sharded save via the same Checkpointer the parent uses
        # for the student. Each rank writes its own shard.
        fs_weights_dir = os.path.join(path, "fake_score")
        os.makedirs(fs_weights_dir, exist_ok=True)
        self.checkpointer.save_model(
            model=self._fake_score,
            weights_path=fs_weights_dir,
            peft_config=None,
            tokenizer=None,
        )
        # fake_score optimizer — also DCP sharded. ``save_optimizer`` takes the optimizer
        # and its owning model in order to rebuild the parameter mapping.
        fs_opt_dir = os.path.join(path, "fake_score_optimizer")
        os.makedirs(fs_opt_dir, exist_ok=True)
        self.checkpointer.save_optimizer(
            self._fake_score_optimizer, self._fake_score, fs_opt_dir, None
        )

        # EMA shadow + DMD scalar state — rank-0 torch.save. EMA's ``state_dict`` already
        # materialises full tensors via ``DTensor.full_tensor()`` under FSDP2 full_tensor
        # mode, so this is a single unsharded file.
        if is_main_process():
            logging.info("[DMD2] saved fake_score weights -> %s", fs_weights_dir)
            logging.info("[DMD2] saved fake_score optimizer -> %s", fs_opt_dir)
            if self._dmd_pipeline.ema is not None:
                ema_path = os.path.join(path, "ema_shadow.pt")
                torch.save(self._dmd_pipeline.ema.state_dict(), ema_path)
                logging.info("[DMD2] saved ema_shadow -> %s", ema_path)
            state_path = os.path.join(path, "dmd_state.pt")
            torch.save({"iteration": self._dmd_pipeline._iteration}, state_path)
            logging.info(
                "[DMD2] saved dmd_state (iteration=%d) -> %s",
                int(self._dmd_pipeline._iteration),
                state_path,
            )
            # Discriminator + its optimizer — replicated across ranks (no FSDP),
            # so rank-0 torch.save of the canonical state_dict suffices.
            if self._discriminator is not None:
                disc_path = os.path.join(path, "discriminator.pt")
                torch.save(self._discriminator.state_dict(), disc_path)
                logging.info("[DMD2] saved discriminator -> %s", disc_path)
            if self._discriminator_optimizer is not None:
                disc_opt_path = os.path.join(path, "discriminator_optimizer.pt")
                torch.save(self._discriminator_optimizer.state_dict(), disc_opt_path)
                logging.info("[DMD2] saved discriminator optimizer -> %s", disc_opt_path)

    def _write_dmd_complete_marker(self, path: str) -> None:
        marker_path = os.path.join(path, _DMD_COMPLETE_MARKER)
        payload = {
            "checkpoint": os.path.basename(os.path.realpath(path)),
            "dmd_iteration": int(self._dmd_pipeline._iteration),
        }
        with open(marker_path, "w") as f:
            json.dump(payload, f)
            f.write("\n")
        logging.info("[DMD2] marked checkpoint complete -> %s", marker_path)

    def _remove_checkpoint_pointer(self, link_name: str) -> None:
        ckpt_root = self.checkpointer.config.checkpoint_dir
        for path in (
            os.path.join(ckpt_root, link_name),
            os.path.join(ckpt_root, f"{link_name}.txt"),
        ):
            if os.path.lexists(path):
                os.remove(path)

    def _restore_dmd_extras(self, restore_from: str | None) -> None:
        """Restore fake_score + fake_score optimizer + EMA + DMD scalar state.

        No-op when no checkpoint is being restored. ``load_checkpoint`` resolves
        ``LATEST`` to the latest complete DMD2 checkpoint before this method runs.
        """
        if restore_from is None:
            return

        ckpt_dir = self._resolve_extras_dir(restore_from)
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            return

        fs_weights_dir = os.path.join(ckpt_dir, "fake_score")
        fs_opt_dir = os.path.join(ckpt_dir, "fake_score_optimizer")
        ema_path = os.path.join(ckpt_dir, "ema_shadow.pt")
        state_path = os.path.join(ckpt_dir, "dmd_state.pt")

        # Checkpointer.save_model writes DCP shards to ``<weights_path>/model/``;
        # load_model expects that *inner* ``model/`` dir as ``model_path`` (see
        # ``BaseRecipe.load_checkpoint`` which passes ``os.path.join(ckpt_dir, "model")``).
        # The kwarg name differs between save (``weights_path``) and load (``model_path``).
        fs_weights_model_dir = os.path.join(fs_weights_dir, "model")
        if os.path.isdir(fs_weights_model_dir):
            self.checkpointer.load_model(model=self._fake_score, model_path=fs_weights_model_dir)
            if is_main_process():
                logging.info("[DMD2] restored fake_score weights <- %s", fs_weights_model_dir)
        elif is_main_process():
            logging.info(
                "[DMD2] WARN: fake_score weights dir missing at %s -- skipping",
                fs_weights_model_dir,
            )
        # load_optimizer, in contrast, appends ``optim/`` internally — pass the base dir.
        if os.path.isdir(os.path.join(fs_opt_dir, "optim")):
            self.checkpointer.load_optimizer(
                self._fake_score_optimizer, self._fake_score, fs_opt_dir, None
            )
            if is_main_process():
                logging.info("[DMD2] restored fake_score optimizer <- %s", fs_opt_dir)
        elif is_main_process():
            logging.info(
                "[DMD2] WARN: fake_score optimizer dir missing at %s -- skipping",
                fs_opt_dir,
            )

        if os.path.isfile(ema_path) and self._dmd_pipeline.ema is not None:
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
            self._dmd_pipeline.ema.load_state_dict(ema_state)
            if is_main_process():
                logging.info("[DMD2] restored ema_shadow <- %s", ema_path)
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self._dmd_pipeline._iteration = int(state.get("iteration", 0))
            if is_main_process():
                logging.info(
                    "[DMD2] restored dmd_state (iteration=%d) <- %s",
                    self._dmd_pipeline._iteration,
                    state_path,
                )

        # Discriminator + its optimizer.
        if self._discriminator is not None:
            disc_path = os.path.join(ckpt_dir, "discriminator.pt")
            if os.path.isfile(disc_path):
                disc_state = torch.load(disc_path, map_location="cpu", weights_only=False)
                self._discriminator.load_state_dict(disc_state)
                if is_main_process():
                    logging.info("[DMD2] restored discriminator <- %s", disc_path)
            elif is_main_process():
                logging.info("[DMD2] WARN: discriminator file missing at %s -- skipping", disc_path)
        if self._discriminator_optimizer is not None:
            disc_opt_path = os.path.join(ckpt_dir, "discriminator_optimizer.pt")
            if os.path.isfile(disc_opt_path):
                disc_opt_state = torch.load(disc_opt_path, map_location="cpu", weights_only=False)
                self._discriminator_optimizer.load_state_dict(disc_opt_state)
                if is_main_process():
                    logging.info("[DMD2] restored discriminator optimizer <- %s", disc_opt_path)
            elif is_main_process():
                logging.info(
                    "[DMD2] WARN: discriminator optimizer file missing at %s -- skipping",
                    disc_opt_path,
                )

    def _resolve_extras_dir(self, restore_from: str) -> str | None:
        """Best-effort resolve of the checkpoint dir, matching BaseRecipe's convention.

        For explicit paths we pass through; for ``"LATEST"`` we look under
        ``checkpointer.config.checkpoint_dir``. This keeps resolution simple and delegates
        the hard cases (async symlinks, cross-node shared filesystems) to the user.
        """
        if os.path.isabs(restore_from):
            return restore_from
        # Try the checkpoint_dir-relative form first (matches the parent's symlink
        # naming — "LATEST" or an explicit ``epoch_N_step_M`` subdir).
        candidate = os.path.join(self.checkpointer.config.checkpoint_dir, restore_from)
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
        return None

    def _resolve_complete_dmd_checkpoint(self, restore_from: str | None) -> str | None:
        ckpt_root = self.checkpointer.config.checkpoint_dir

        if restore_from is None or str(restore_from).upper() in {"LATEST", "DMD2_LATEST"}:
            return self._find_latest_complete_dmd_checkpoint(ckpt_root)

        if os.path.isabs(restore_from):
            candidate = restore_from
        else:
            candidate = os.path.join(ckpt_root, restore_from)
        candidate = os.path.realpath(candidate)

        if not os.path.isdir(candidate):
            return candidate
        if not self._is_dmd_checkpoint_complete(candidate):
            raise RuntimeError(
                f"DMD2 checkpoint is incomplete and cannot be restored: {candidate}. "
                "Use a complete older checkpoint or remove the partial directory."
            )
        return candidate

    def _find_latest_complete_dmd_checkpoint(self, ckpt_root: str) -> str | None:
        dmd2_latest = os.path.join(ckpt_root, "DMD2_LATEST")
        for pointer in (dmd2_latest, os.path.join(ckpt_root, "LATEST")):
            resolved = self._resolve_checkpoint_pointer(pointer)
            if resolved is not None and self._is_dmd_checkpoint_complete(resolved):
                return resolved

        candidates = []
        if os.path.isdir(ckpt_root):
            for name in os.listdir(ckpt_root):
                path = os.path.join(ckpt_root, name)
                if (
                    os.path.isdir(path)
                    and "_step_" in name
                    and self._is_dmd_checkpoint_complete(path)
                ):
                    candidates.append(os.path.realpath(path))
        if not candidates:
            return None
        return max(candidates, key=self._checkpoint_step)

    def _resolve_checkpoint_pointer(self, pointer: str) -> str | None:
        resolved = None
        if os.path.islink(pointer):
            try:
                resolved = os.readlink(pointer)
            except OSError:
                return None
        elif os.path.isfile(pointer + ".txt"):
            try:
                with open(pointer + ".txt") as f:
                    resolved = f.read().strip()
            except OSError:
                return None
        if not resolved:
            return None
        if not os.path.isabs(resolved):
            resolved = os.path.abspath(os.path.join(os.path.dirname(pointer), resolved))
        return os.path.realpath(resolved) if os.path.isdir(resolved) else None

    def _is_dmd_checkpoint_complete(self, path: str) -> bool:
        path = os.path.realpath(path)
        if not os.path.isdir(path):
            return False
        if os.path.isfile(os.path.join(path, _DMD_COMPLETE_MARKER)):
            return True

        fs_model_dir = os.path.join(path, "fake_score", "model")
        fs_opt_metadata = os.path.join(path, "fake_score_optimizer", "optim", ".metadata")
        dmd_state = os.path.join(path, "dmd_state.pt")
        complete = (
            self._dir_has_regular_file(fs_model_dir)
            and os.path.isfile(fs_opt_metadata)
            and os.path.isfile(dmd_state)
        )
        if not complete:
            return False

        if self._cfg_gan_enabled():
            return os.path.isfile(os.path.join(path, "discriminator.pt")) and os.path.isfile(
                os.path.join(path, "discriminator_optimizer.pt")
            )
        return True

    def _cfg_gan_enabled(self) -> bool:
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return False
        try:
            return float(cfg.get("dmd2.gan_loss_weight_gen", 0.0) or 0.0) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _dir_has_regular_file(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        try:
            with os.scandir(path) as entries:
                return any(entry.is_file() for entry in entries)
        except OSError:
            return False

    @staticmethod
    def _checkpoint_step(path: str) -> int:
        match = re.search(r"_step_(\d+)$", os.path.basename(os.path.realpath(path)))
        return int(match.group(1)) if match else -1

    # ------------------------------------------------------------------ #
    #  Helpers — teacher / fake_score loading, DMDConfig resolution      #
    # ------------------------------------------------------------------ #

    def _load_frozen_teacher(self) -> nn.Module:
        """Load a second copy of the pretrained transformer, frozen + FSDP2-sharded.

        The same pretrained path + ``parallel_scheme`` as the student. Setting
        ``load_for_training=False`` walks the parameters once and flips
        ``requires_grad=False`` after FSDP2 wrapping; we also call ``.eval()`` on the
        returned module just to be defensive.
        """
        parallel_scheme = self._build_parallel_scheme_snapshot()
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.bf16,
            device=self.device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=False,
            low_cpu_mem_usage=True,
        )
        teacher = pipe.transformer
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _build_discriminator(self) -> nn.Module | None:
        """Construct the Discriminator_ImageDiT when GAN is enabled.

        Returns ``None`` when ``dmd2.gan_loss_weight_gen`` is zero so the
        DMDPipeline runs without a discriminator (any run with the GAN branch disabled).
        """
        gan_weight = float(self.cfg.get("dmd2.gan_loss_weight_gen", 0.0) or 0.0)
        if gan_weight <= 0.0:
            return None

        # GAN-specific knobs read directly from the YAML so callers don't have to
        # touch the built-in DMDConfig recipe just to flip feature indices.
        feature_indices = self.cfg.get(
            "dmd2.gan_feature_indices", [30]
        )  # middle of Qwen-Image's 60 blocks
        num_blocks = int(self.cfg.get("dmd2.gan_num_blocks", 60))
        inner_dim = int(self.cfg.get("dmd2.gan_inner_dim", 3072))

        disc = Discriminator_ImageDiT(
            feature_indices={int(i) for i in feature_indices},
            num_blocks=num_blocks,
            inner_dim=inner_dim,
        )
        disc.to(device=self.device, dtype=self.bf16)
        disc.train()
        for p in disc.parameters():
            p.requires_grad_(True)
        if is_main_process():
            logging.info(
                "[DMD2] Built discriminator: %s | num_features=%d num_blocks=%d inner_dim=%d "
                "params=%d",
                type(disc).__name__,
                disc.num_features,
                num_blocks,
                inner_dim,
                sum(p.numel() for p in disc.parameters()),
            )
        return disc

    def _build_discriminator_optimizer(self) -> torch.optim.Optimizer | None:
        """AdamW on the discriminator. No FSDP wrap — manual grad all-reduce keeps it simple."""
        if self._discriminator is None:
            return None
        lr = float(self.cfg.get("dmd2.discriminator_lr", 1.0e-5) or 1.0e-5)
        opt = torch.optim.AdamW(
            self._discriminator.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),  # FastGen Qwen-Image DMD2 inherits BaseOptimizerConfig betas
        )
        if is_main_process():
            logging.info("[DMD2] Built discriminator optimizer: AdamW lr=%g betas=(0.9, 0.999)", lr)
        return opt

    def _attach_gan_feature_capture(self) -> None:
        """Install Qwen-Image feature-capture hooks on the teacher when GAN is enabled.

        Reads the latent resolution from the dataloader so the hook can reshape
        ``[B, num_image_patches, 3072]`` into ``[B, 3072, H_lat//2, W_lat//2]``.
        Mock dataloader → spatial_h/spatial_w from the YAML. Real dataloader →
        base_resolution / vae_scale.
        """
        feature_indices = list(self.cfg.get("dmd2.gan_feature_indices", [30]))

        # Resolve h_lat / w_lat. Mock has it in the YAML; real cache uses the
        # configured base_resolution divided by the VAE 8x downsample.
        # Convert the dataloader subtree to a plain dict — AutoModel's ConfigNode
        # doesn't expose deep dotted paths like ``data.dataloader.spatial_h``.
        dl_node = self.cfg.get("data.dataloader", None)
        if dl_node is not None and hasattr(dl_node, "to_dict"):
            dl_dict = dl_node.to_dict()
        elif dl_node is not None:
            try:
                dl_dict = dict(dl_node)
            except (TypeError, ValueError):
                dl_dict = {}
        else:
            dl_dict = {}

        spatial_h = dl_dict.get("spatial_h")
        spatial_w = dl_dict.get("spatial_w")
        base_resolution = dl_dict.get("base_resolution")
        if spatial_h is not None and spatial_w is not None:
            h_lat = int(spatial_h)
            w_lat = int(spatial_w)
        elif base_resolution is not None:
            h_lat = int(base_resolution[0]) // 8
            w_lat = int(base_resolution[1]) // 8
        else:
            # Fallback: hope it's 64x64 (512px image). Smoke tests pin this explicitly.
            h_lat, w_lat = 64, 64
            if is_main_process():
                logging.warning(
                    "[DMD2] Could not infer h_lat/w_lat from data.dataloader; defaulting to 64x64."
                )

        qwen_image_plugin.attach_feature_capture(
            self._teacher,
            feature_indices=feature_indices,
            h_lat=h_lat,
            w_lat=w_lat,
        )
        if is_main_process():
            logging.info(
                "[DMD2] Attached GAN feature capture: indices=%s h_lat=%d w_lat=%d",
                feature_indices,
                h_lat,
                w_lat,
            )

    def _load_fake_score(self) -> nn.Module:
        """Load a third copy, trainable. Weights start identical to the teacher."""
        parallel_scheme = self._build_parallel_scheme_snapshot()
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.bf16,
            device=self.device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=True,
            low_cpu_mem_usage=True,
        )
        fake_score = pipe.transformer
        fake_score.train()
        for p in fake_score.parameters():
            p.requires_grad_(True)
        return fake_score

    def _build_parallel_scheme_snapshot(self) -> dict[str, dict[str, Any]]:
        """Reconstruct the FSDP2 manager_args used for the student.

        Mirrors ``build_model_and_optimizer`` in ``nemo_automodel.recipes.diffusion.train``.
        We can't capture the student's ``parallel_scheme`` directly (the parent doesn't
        stash it), so we rebuild it from the same YAML knobs the parent consumed.
        """
        from torch.distributed.fsdp import MixedPrecisionPolicy

        fsdp_cfg = self.cfg.get("fsdp", None) or {}
        ddp_cfg = self.cfg.get("ddp", None)

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if ddp_cfg is not None:
            return {
                "transformer": {
                    "_manager_type": "ddp",
                    "backend": ddp_cfg.get("backend", "nccl"),
                    "world_size": world_size,
                    "activation_checkpointing": ddp_cfg.get("activation_checkpointing", False),
                }
            }

        dp_size = fsdp_cfg.get("dp_size")
        tp_size = fsdp_cfg.get("tp_size", 1)
        cp_size = fsdp_cfg.get("cp_size", 1)
        pp_size = fsdp_cfg.get("pp_size", 1)
        if dp_size is None:
            denom = max(1, tp_size * cp_size * pp_size)
            dp_size = max(1, world_size // denom)

        return {
            "transformer": {
                "_manager_type": "fsdp2",
                "dp_size": dp_size,
                "dp_replicate_size": fsdp_cfg.get("dp_replicate_size", None),
                "tp_size": tp_size,
                "cp_size": cp_size,
                "pp_size": pp_size,
                "backend": "nccl",
                "world_size": world_size,
                "use_hf_tp_plan": fsdp_cfg.get("use_hf_tp_plan", False),
                "activation_checkpointing": fsdp_cfg.get("activation_checkpointing", True),
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=self.bf16,
                    reduce_dtype=torch.float32,
                    output_dtype=self.bf16,
                ),
            }
        }

    def _resolve_pipeline_cls(self) -> type[DMDPipeline]:
        """Pick the DMDPipeline subclass for the current backbone.

        Resolution order:

        1. ``dmd2.pipeline_plugin`` in the YAML (explicit override, ``null`` for base).
        2. Substring match on ``model.pretrained_model_name_or_path``
           (e.g. ``Qwen-Image`` -> ``qwen_image`` plugin).
        3. Fall back to :class:`DMDPipeline`.
        """
        explicit = self.cfg.get("dmd2.pipeline_plugin", None)
        if explicit is None:
            model_id_lc = (self.model_id or "").lower()
            for needle, plugin_name in _PIPELINE_PLUGIN_BY_MODEL_SUBSTR:
                if needle in model_id_lc:
                    explicit = plugin_name
                    break
        if explicit in (None, "base", "DMDPipeline"):
            return DMDPipeline
        if explicit == "qwen_image":
            # Imported lazily so ``base`` users don't pay the import cost.
            from modelopt.torch.fastgen.plugins.qwen_image import QwenImageDMDPipeline

            return QwenImageDMDPipeline
        raise ValueError(
            f"Unknown dmd2.pipeline_plugin={explicit!r}. Supported: null/'base', 'qwen_image'."
        )

    def _resolve_pipeline_kwargs(self, pipeline_cls: type[DMDPipeline]) -> dict[str, Any]:
        """Extra kwargs to forward to the pipeline subclass constructor (plugin-specific)."""
        if pipeline_cls.__name__ == "QwenImageDMDPipeline":
            # Optional ``guidance`` value passed to the transformer's guidance kwarg every
            # call. Independent of DMDConfig.guidance_scale (which drives the negative-
            # prompt CFG path on the teacher). Leave ``None`` to skip the embedding when
            # the transformer was built with ``guidance_embeds=false`` (default for
            # ``Qwen/Qwen-Image``).
            return {"guidance": self.cfg.get("dmd2.qwen_image_guidance", None)}
        return {}

    def _resolve_dmd_config(self) -> DMDConfig:
        """Load the built-in fastgen recipe, then apply inline YAML overrides."""
        dmd_cfg_node = self.cfg.get("dmd2", None)
        if dmd_cfg_node is None:
            raise ValueError(
                "Missing ``dmd2:`` block in the YAML config. Expected at minimum "
                "``dmd2.recipe_path`` pointing at a fastgen DMDConfig recipe "
                "(e.g. ``general/distillation/dmd2_qwen_image``)."
            )
        dmd_dict = (
            dmd_cfg_node.to_dict() if hasattr(dmd_cfg_node, "to_dict") else dict(dmd_cfg_node)
        )

        recipe_path = dmd_dict.pop("recipe_path", None)
        if recipe_path is None:
            raise ValueError(
                "``dmd2.recipe_path`` is required — Phase 1 relies on the built-in "
                "``modelopt_recipes`` path resolver to hydrate the full DMDConfig."
            )
        base_config = mtf.load_dmd_config(recipe_path)

        # Filter overrides to the subset that actually corresponds to DMDConfig fields.
        # Non-matching keys (e.g. ``fake_score_lr``, ``cfg_mode``) are kept as top-level
        # recipe knobs and read via ``self.cfg.get("dmd2.<key>")``.
        overrides = {k: v for k, v in dmd_dict.items() if k in _DMD_CONFIG_OVERRIDE_KEYS}
        if not overrides:
            return base_config
        # Deep-merge so a YAML block that overrides a single ``sample_t_cfg`` / ``ema``
        # sub-field keeps the recipe's other sub-fields — a shallow ``dict.update`` would
        # replace the whole sub-config and silently reset its siblings to defaults.
        # Re-validate the merged dict so the nested blocks become their Pydantic config
        # objects instead of raw dicts.
        merged = _deep_merge_dicts(base_config.model_dump(), overrides)
        return DMDConfig.model_validate(merged)

    def _build_fake_score_optimizer(self) -> torch.optim.Optimizer:
        """AdamW on fake_score params. LR defaults to student LR; overridable via YAML."""
        fs_lr = self.cfg.get("dmd2.fake_score_lr", None)
        if fs_lr is None:
            fs_lr = self.learning_rate
        optimizer_cfg = self.cfg.get("optim.optimizer", {}) or {}
        optimizer_cfg = (
            optimizer_cfg.to_dict() if hasattr(optimizer_cfg, "to_dict") else dict(optimizer_cfg)
        )
        weight_decay = optimizer_cfg.get("weight_decay", 0.01)
        betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))

        trainable_params = [p for p in self._fake_score.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in fake_score.")
        return torch.optim.AdamW(trainable_params, lr=fs_lr, weight_decay=weight_decay, betas=betas)

    # ------------------------------------------------------------------ #
    #  Inner helpers                                                     #
    # ------------------------------------------------------------------ #

    def _set_grad_requirements(self, is_student_phase: bool) -> None:
        """Toggle train/eval + requires_grad across modules for the active phase.

        Mirrors FastGen's ``_setup_grad_requirements`` (``dmd2.py`` lines 67-77),
        INCLUDING the discriminator toggle that was previously omitted.

        Why the discriminator toggle matters: ``compute_student_loss`` calls
        ``self.discriminator(fake_feat)`` for the ``gan_gen`` term, so the
        discriminator is in the student-phase backward graph. With its params
        left at ``requires_grad=True``, ``total.backward()`` allocates and
        fills ``.grad`` for every discriminator parameter — gradients which
        the student optimizer never consumes and which the next discriminator
        ``zero_grad(set_to_none=True)`` simply wipes. Freezing the discriminator
        during the student phase skips that wasted memory + backward compute
        without changing any numerics (the student still receives the GAN
        signal through the discriminator's input-side gradient, which doesn't
        require the discriminator's own params to be in the autograd graph).

        Called every step; cheap enough that we don't bother caching the last state.
        """
        if is_student_phase:
            self.model.train()
            for p in self.model.parameters():
                p.requires_grad_(True)
            self._fake_score.eval()
            for p in self._fake_score.parameters():
                p.requires_grad_(False)
            if self._discriminator is not None:
                self._discriminator.eval()
                for p in self._discriminator.parameters():
                    p.requires_grad_(False)
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._fake_score.train()
            for p in self._fake_score.parameters():
                p.requires_grad_(True)
            if self._discriminator is not None:
                self._discriminator.train()
                for p in self._discriminator.parameters():
                    p.requires_grad_(True)

    def _prepare_micro_batch(
        self, micro_batch: dict[str, Any]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Extract latents, noise, text conditioning, and optional masks from a batch.

        Accepts both 5D ``video_latents`` and 4D ``image_latents``
        (Qwen-Image / Flux / SD3). Mirrors the key dispatch in
        ``nemo_automodel.components.flow_matching.pipeline.FlowMatchingPipeline.step``.

        ``negative_text_embeddings`` is optional — present when the dataloader
        supplies it (mock T2I, real cache with precomputed empty-prompt
        embedding) and consumed by ``compute_student_loss`` only when CFG is
        enabled (``dmd2.guidance_scale is not None``).
        """
        if "image_latents" in micro_batch:
            latents = micro_batch["image_latents"].to(self.device, dtype=self.bf16)
        elif "video_latents" in micro_batch:
            latents = micro_batch["video_latents"].to(self.device, dtype=self.bf16)
        else:
            raise KeyError(
                "Batch must contain either 'image_latents' (4D) or 'video_latents' (5D). "
                f"Got keys: {sorted(micro_batch.keys())}."
            )
        text_embeds = micro_batch["text_embeddings"].to(self.device, dtype=self.bf16)
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(0)
        text_mask = micro_batch.get("text_embeddings_mask")
        if text_mask is not None:
            text_mask = text_mask.to(self.device)
            if text_mask.ndim == 1:
                text_mask = text_mask.unsqueeze(0)
        negative_text_embeds = micro_batch.get("negative_text_embeddings")
        if negative_text_embeds is not None:
            negative_text_embeds = negative_text_embeds.to(self.device, dtype=self.bf16)
            if negative_text_embeds.ndim == 2:
                negative_text_embeds = negative_text_embeds.unsqueeze(0)
        negative_text_mask = micro_batch.get("negative_text_embeddings_mask")
        if negative_text_mask is not None:
            negative_text_mask = negative_text_mask.to(self.device)
            if negative_text_mask.ndim == 1:
                negative_text_mask = negative_text_mask.unsqueeze(0)
        # Fresh noise per micro-batch — DMD2 samples noise independently at each loss call.
        noise = torch.randn_like(latents)
        return latents, noise, text_embeds, text_mask, negative_text_embeds, negative_text_mask

    def _log_step(
        self,
        *,
        global_step: int,
        is_student_phase: bool,
        group_loss: float,
        grad_norm: float,
        vsd_loss: float | None,
        disc_loss: float | None = None,
    ) -> None:
        """Log a single step. Stdout always; wandb when the parent set it up."""
        phase = "student" if is_student_phase else "fake_score"

        # Stdout
        suffix = f" vsd={vsd_loss:.4f}" if vsd_loss is not None else ""
        if disc_loss is not None:
            suffix += f" disc={disc_loss:.4f}"
        logging.info(
            "[STEP %d] phase=%s loss=%.4f grad_norm=%.4f%s lr=%.2e",
            global_step,
            phase,
            group_loss,
            grad_norm,
            suffix,
            self.optimizer.param_groups[0]["lr"],
        )

        # wandb
        try:
            import wandb

            if wandb.run is not None:
                log_dict: dict[str, Any] = {
                    f"{phase}/loss": group_loss,
                    f"{phase}/grad_norm": grad_norm,
                    "global_step": global_step,
                    "lr_student": self.optimizer.param_groups[0]["lr"],
                    "lr_fake_score": self._fake_score_optimizer.param_groups[0]["lr"],
                }
                if vsd_loss is not None:
                    log_dict["student/vsd"] = vsd_loss
                if disc_loss is not None:
                    log_dict["discriminator/loss"] = disc_loss
                wandb.log(log_dict, step=global_step)
        except Exception:
            # wandb not installed or not initialised — silent no-op.
            pass

    def _dmd_config_summary(self) -> str:
        """Compact one-line summary of the active DMDConfig for startup logging."""
        cfg = self._dmd_config
        t_list = cfg.sample_t_cfg.t_list if cfg.sample_t_cfg is not None else None
        return (
            f"pred_type={cfg.pred_type} fake_score_pred_type={cfg.fake_score_pred_type} "
            f"num_train_timesteps={cfg.num_train_timesteps} "
            f"student_update_freq={cfg.student_update_freq} "
            f"student_sample_steps={cfg.student_sample_steps} "
            f"student_sample_type={cfg.student_sample_type} "
            f"backward_simulation={cfg.backward_simulation} "
            f"t_list={t_list} "
            f"gan_loss_weight_gen={cfg.gan_loss_weight_gen} "
            f"guidance_scale={cfg.guidance_scale} ema={'on' if cfg.ema is not None else 'off'}"
        )

    def _dmd_full_config_log(self) -> str:
        """Full multi-line dump of every DMD2 parameter for startup tracing.

        Two sections: the resolved DMDConfig (every Pydantic field, including
        nested ``sample_t_cfg`` and ``ema`` blocks) and the recipe-side keys
        under ``dmd2:`` that aren't DMDConfig fields (e.g. ``fake_score_lr``,
        ``gan_feature_indices``, ``pipeline_plugin``). Combined they cover
        every knob that ends up driving the DMD2 method at runtime.
        """
        cfg = self._dmd_config
        dmd_node = self.cfg.get("dmd2", {}) or {}
        if hasattr(dmd_node, "to_dict"):
            dmd_node = dmd_node.to_dict()
        else:
            dmd_node = dict(dmd_node)
        recipe_extras = {
            k: v
            for k, v in dmd_node.items()
            if k not in _DMD_CONFIG_OVERRIDE_KEYS and k != "recipe_path"
        }
        combined = {
            "DMDConfig_resolved": cfg.model_dump(),
            "recipe_side_extras": recipe_extras,
        }
        return json.dumps(combined, indent=2, default=str)
