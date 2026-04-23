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

"""DMD2 distillation recipe for Wan 2.2 5B built on NeMo AutoModel.

This recipe subclasses :class:`nemo_automodel.recipes.diffusion.train.TrainDiffusionRecipe`
so it inherits AutoModel's student + optimizer + dataloader + checkpoint plumbing, then
drives ``modelopt.torch.fastgen.DMDPipeline`` through the three-phase DMD2 alternation
(student update / fake-score update / EMA step). Phase 1 targets the
``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` checkpoint under FSDP2 multi-GPU and deliberately
disables the discriminator and CFG branches so the end-to-end VSD + DSM + EMA path can
be debugged on a minimal surface.

Launch::

    torchrun --nproc-per-node=8 \\
        examples/diffusers/fastgen/dmd2_finetune.py \\
        --config examples/diffusers/fastgen/configs/dmd2_wan22_5b.yaml

See ``examples/diffusers/fastgen/README.md`` for the full usage guide, the three-phase
alternation diagram, the Phase 2 roadmap (GAN + CFG + real-data path), and the
troubleshooting notes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

# Direct imports — any failure here stops the module load with a real stack, which
# is what we want at runtime. A previous try/except gate made the subclass fall
# back to ``object``, which silently masked missing nemo_automodel deps and
# surfaced as a downstream ``TypeError: takes no arguments``.
from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe, is_main_process
from torch import nn

import modelopt.torch.fastgen as mtf
from modelopt.torch.fastgen.config import DMDConfig

# Keys under the ``dmd2:`` YAML block that shadow fields on :class:`DMDConfig`. The
# recipe applies these as a Pydantic ``model_copy(update=...)`` on top of the loaded
# built-in recipe so users can tweak DMD2 hyperparameters without editing the shared
# ``modelopt_recipes/general/distillation/dmd2_wan22_5b.yaml`` file.
_DMD_CONFIG_OVERRIDE_KEYS = frozenset(DMDConfig.model_fields.keys())


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

    What this recipe adds (Phase 1):

    - A frozen teacher loaded via a second :meth:`NeMoAutoDiffusionPipeline.from_pretrained`
      call with the same ``parallel_scheme`` so it lands with the same FSDP2 sharding
      as the student.
    - A trainable fake-score transformer loaded the same way (weights identical to the
      teacher on step 0).
    - A separate AdamW optimizer for the fake-score phase.
    - An :class:`mtf.DMDPipeline` driving VSD + DSM + EMA.
    - Sidecar checkpoint save / restore for fake-score weights, fake-score optimizer,
      EMA shadow, and DMDPipeline iteration counters.

    Phase 1 scope — NOT implemented here: classifier-free guidance,
    multiscale discriminator + GAN branch, real ``.meta`` dataset path. See the
    ``Phase 2`` section of ``README.md`` for the roadmap.
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
        #
        #    For the Phase 1 smoke, ``data.dataloader._target_`` in the YAML points at
        #    ``nemo_automodel.components.datasets.diffusion.build_mock_dataloader`` so the
        #    parent wires up the mock dataloader for us — no swap needed.
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
        #    ``dmd2.fake_score_lr`` isn't set; FastGen's Wan 2.2 5B config uses 1e-5 for
        #    all three optimizers.
        self.__dict__["_fake_score_optimizer"] = self._build_fake_score_optimizer()

        # 7. DMDPipeline. Phase 1: discriminator=None. Asserts in the constructor would
        #    fire if ``gan_loss_weight_gen > 0`` without a discriminator — the YAML
        #    forces it to 0.0 for Phase 1.
        self.__dict__["_dmd_pipeline"] = mtf.DMDPipeline(
            student=self.model,
            teacher=self._teacher,
            fake_score=self._fake_score,
            config=self._dmd_config,
            discriminator=None,
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
        self._restore_dmd_extras(self.restore_from)

        if is_main_process():
            logging.info("[DMD2] recipe initialized: %s", self._dmd_config_summary())

    # ------------------------------------------------------------------ #
    #  Training loop                                                     #
    # ------------------------------------------------------------------ #

    def run_train_validation_loop(self) -> None:
        """Three-phase DMD2 alternation driven by ``step_scheduler``.

        Each outer iteration picks either the student or fake-score phase based on
        ``global_step % student_update_freq``. The student phase runs
        ``compute_student_loss`` + ``update_ema``. The fake-score phase runs
        ``compute_fake_score_loss``. Phase 1 never enters the discriminator phase
        because ``gan_loss_weight_gen`` is pinned to 0 in the YAML.

        Mirrors the gating in ``FastGen/fastgen/methods/distribution_matching/dmd2.py``
        (``_student_update_step`` / ``_fake_score_discriminator_update_step``).
        """
        dmd = self._dmd_pipeline
        cfg = self._dmd_config

        logging.info("[DMD2] Starting DMD2 training on Wan 2.2 5B")
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

        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

            if is_main_process():
                from tqdm import tqdm

                self.step_scheduler.dataloader = tqdm(
                    self.dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"
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
                for micro_batch in batch_group:
                    latents, noise, text_embeds = self._prepare_micro_batch(micro_batch)

                    if is_student_phase:
                        losses = dmd.compute_student_loss(
                            latents,
                            noise,
                            encoder_hidden_states=text_embeds,
                            # Phase 1: no CFG. guidance_scale=None short-circuits the
                            # negative-conditioning branch inside compute_student_loss.
                            negative_encoder_hidden_states=None,
                            guidance_scale=None,
                        )
                        micro_vsd_losses.append(float(losses["vsd"].item()))
                    else:
                        losses = dmd.compute_fake_score_loss(
                            latents,
                            noise,
                            encoder_hidden_states=text_embeds,
                        )

                    (losses["total"] / len(batch_group)).backward()
                    micro_losses.append(float(losses["total"].item()))

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

        if is_main_process():
            logging.info("[DMD2] Training complete. Final step: %s", global_step)

    # ------------------------------------------------------------------ #
    #  Checkpoint save / restore (sidecars next to student DCP)          #
    # ------------------------------------------------------------------ #

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ) -> None:
        """Delegate student save to ``super()``, then sidecar the DMD2 extras."""
        super().save_checkpoint(epoch, step, train_loss, val_loss, best_metric_key)

        if not self.checkpointer.config.enabled:
            return

        path = os.path.join(self.checkpointer.config.checkpoint_dir, f"epoch_{epoch}_step_{step}")
        self._save_dmd_extras(path)

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
            if self._dmd_pipeline.ema is not None:
                ema_path = os.path.join(path, "ema_shadow.pt")
                torch.save(self._dmd_pipeline.ema.state_dict(), ema_path)
            state_path = os.path.join(path, "dmd_state.pt")
            torch.save({"iteration": self._dmd_pipeline._iteration}, state_path)

    def _restore_dmd_extras(self, restore_from: str | None) -> None:
        """Restore fake_score + fake_score optimizer + EMA + DMD scalar state.

        No-op when no checkpoint is being restored. Uses the superclass's path resolver
        so ``"LATEST"`` and relative names behave the same way they do for the student.
        """
        if restore_from is None:
            # Auto-detect only kicks in if the parent's load_checkpoint chose a dir — here
            # we mirror that logic by peeking at the checkpoint_dir for the latest.
            # For Phase 1 we keep it simple: no auto-detect of extras. Users who need
            # resume must pass ``checkpoint.restore_from`` explicitly.
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
        # load_optimizer, in contrast, appends ``optim/`` internally — pass the base dir.
        if os.path.isdir(os.path.join(fs_opt_dir, "optim")):
            self.checkpointer.load_optimizer(
                self._fake_score_optimizer, self._fake_score, fs_opt_dir, None
            )

        if os.path.isfile(ema_path) and self._dmd_pipeline.ema is not None:
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=False)
            self._dmd_pipeline.ema.load_state_dict(ema_state)
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self._dmd_pipeline._iteration = int(state.get("iteration", 0))

    def _resolve_extras_dir(self, restore_from: str) -> str | None:
        """Best-effort resolve of the checkpoint dir, matching BaseRecipe's convention.

        For explicit paths we pass through; for ``"LATEST"`` we look under
        ``checkpointer.config.checkpoint_dir``. Phase 1 keeps this simple and delegates
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

    def _resolve_dmd_config(self) -> DMDConfig:
        """Load the built-in fastgen recipe, then apply inline YAML overrides."""
        dmd_cfg_node = self.cfg.get("dmd2", None)
        if dmd_cfg_node is None:
            raise ValueError(
                "Missing ``dmd2:`` block in the YAML config. Expected at minimum "
                "``dmd2.recipe_path`` pointing at a fastgen DMDConfig recipe "
                "(e.g. ``general/distillation/dmd2_wan22_5b``)."
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
        return base_config.model_copy(update=overrides)

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

        Mirrors FastGen's ``_setup_grad_requirements`` (``dmd2.py`` lines 67-77). Called
        every step; cheap enough that we don't bother caching the last state.
        """
        if is_student_phase:
            self.model.train()
            for p in self.model.parameters():
                p.requires_grad_(True)
            self._fake_score.eval()
            for p in self._fake_score.parameters():
                p.requires_grad_(False)
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._fake_score.train()
            for p in self._fake_score.parameters():
                p.requires_grad_(True)

    def _prepare_micro_batch(
        self, micro_batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ``(latents, noise, text_embeds)`` from an AutoModel-format batch."""
        latents = micro_batch["video_latents"].to(self.device, dtype=self.bf16)
        text_embeds = micro_batch["text_embeddings"].to(self.device, dtype=self.bf16)
        if text_embeds.ndim == 2:
            text_embeds = text_embeds.unsqueeze(0)
        # Fresh noise per micro-batch — DMD2 samples noise independently at each loss call.
        noise = torch.randn_like(latents)
        return latents, noise, text_embeds

    def _log_step(
        self,
        *,
        global_step: int,
        is_student_phase: bool,
        group_loss: float,
        grad_norm: float,
        vsd_loss: float | None,
    ) -> None:
        """Log a single step. Stdout always; wandb when the parent set it up."""
        phase = "student" if is_student_phase else "fake_score"

        # Stdout
        suffix = f" vsd={vsd_loss:.4f}" if vsd_loss is not None else ""
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
                wandb.log(log_dict, step=global_step)
        except Exception:
            # wandb not installed or not initialised — silent no-op.
            pass

    def _dmd_config_summary(self) -> str:
        """Compact one-line summary of the active DMDConfig for startup logging."""
        cfg = self._dmd_config
        return (
            f"pred_type={cfg.pred_type} fake_score_pred_type={cfg.fake_score_pred_type} "
            f"num_train_timesteps={cfg.num_train_timesteps} "
            f"student_update_freq={cfg.student_update_freq} "
            f"student_sample_steps={cfg.student_sample_steps} "
            f"gan_loss_weight_gen={cfg.gan_loss_weight_gen} "
            f"guidance_scale={cfg.guidance_scale} ema={'on' if cfg.ema is not None else 'off'}"
        )
