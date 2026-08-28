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

"""Forward-only NeMo AutoModel recipe for pruning activation scoring.

Subclasses ``TrainFinetuneRecipeForNextTokenPrediction`` purely to reuse its
``setup()`` (distributed init, device mesh, parallelized model build, dataloader);
it then freezes the model, registers the parallelism-aware scorers, runs a
no-grad calibration pass, and writes the consolidated scores.

This module imports ``nemo_automodel`` at import time (it subclasses a NeMo recipe),
so it is imported lazily by :mod:`.launch` and is *not* re-exported from the package
``__init__`` — importing the ``automodel`` plugin therefore does not require NeMo.

Supports DP / FSDP / TP / CP / PP: additive scorers reduce across the token group
(``dp_cp``) at finalize; the iterative scorer reduces per iteration via
``step_iteration`` (called once per batch). Under pipeline parallel each stage's
modules are hooked on their owning rank and the forward is driven by
``schedule.eval`` (forward-only); the output writer shards per stage.

In-container validation points:
  * the forward batch keys / device movement (``_FORWARD_KEYS``) match the NeMo
    dataloader; packed-sequence forward (extra packing metadata) is a follow-up.
  * the pipeline-parallel path (``schedule.eval`` + per-stage scorer construction on
    each ``model_parts`` entry) is exercised here for the first time.
"""

import inspect
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

from modelopt.torch.utils import safe_load, safe_save

from ...anymodel.model_descriptor import ModelDescriptorFactory
from ...dataset import (
    DataLayout,
    Modality,
    PuzzletronBatch,
    PuzzletronDataSpec,
    batch_from_automodel,
)
from ...tools.common import resolve_torch_dtype
from ...utils.sample_hash import log_batch_hashes, samples_hashing_enabled
from .hooks import HiddenWidthSiteScorer
from .output import write_scores
from .reduction import MeshGroups
from .target_resolver import build_magnitude_scorers, build_scorers

logger = logging.getLogger(__name__)

__all__ = ["ActivationScoringRecipe"]

# Forward inputs to pass through for scoring; the hooks fire regardless of the loss.
_FORWARD_KEYS = ("input_ids", "attention_mask", "position_ids")


def _ensure_packed_qkv_format(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep model-level RoPE and layer-level attention on the same packed layout."""
    if torch.is_tensor(payload.get("cu_seqlens")):
        payload["qkv_format"] = "thd"
    return payload


class _HiddenStatePassthrough(torch.nn.Module):
    """Keep native last-stage control flow without materializing vocabulary logits."""

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states


# Forward-only/cache kwargs the HF decoder-layer forward passes straight into its token-mixer via
# **kwargs. NeMo's CP-aware mixers (e.g. CPAwareGatedDeltaNet) have a FIXED forward signature with
# no **kwargs, so any of these raises "unexpected keyword argument". For a forward-only, no-cache
# scoring pass they are all inert, so we drop them before the layer runs.
_STALE_LAYER_KWARGS = ("past_key_value", "use_cache", "output_attentions")


def _force_forward_only_pp_schedule(
    pp,
    *,
    log_prefix: str,
    n_microbatches: int | None = None,
) -> None:
    """Replace NeMo's training PP schedule with PyTorch's forward-only schedule.

    NeMo builds a training schedule (1F1B by default) whenever ``pp_size > 1``.
    Calling ``schedule.eval(...)`` disables loss computation, but the training
    schedule still runs its backward communication path and can deadlock in a
    no-grad scoring recipe.  The private ``_ScheduleForwardOnly`` is the exact
    executor we need here: same stage/chunking contract, only forward P2P ops.
    """
    info = getattr(pp, "info", None)
    schedule = getattr(info, "schedule", None)
    if info is None or schedule is None:
        return

    from torch.distributed.pipelining.schedules import _ScheduleForwardOnly

    from ...tools.logger import aprint

    if isinstance(schedule, _ScheduleForwardOnly):
        return
    stage = getattr(schedule, "_stage", None)
    if stage is None:
        aprint(
            f"{log_prefix} PP schedule {type(schedule).__name__} is not a single-stage "
            "schedule; keeping it unchanged"
        )
        return

    forward_microbatches = int(n_microbatches or getattr(schedule, "_n_microbatches"))
    info.schedule = _ScheduleForwardOnly(
        stage,
        n_microbatches=forward_microbatches,
        loss_fn=None,
        args_chunk_spec=getattr(schedule, "_args_chunk_spec", None),
        kwargs_chunk_spec=getattr(schedule, "_kwargs_chunk_spec", None),
        output_merge_spec=getattr(schedule, "_output_merge_spec", None),
        scale_grads=getattr(schedule, "scale_grads", False),
    )
    aprint(
        f"{log_prefix} replaced PP schedule {type(schedule).__name__} "
        f"with _ScheduleForwardOnly for no-grad scoring "
        f"(n_microbatches={forward_microbatches})"
    )


def _rank_trace(message: str) -> None:
    """Print a PP diagnostic from every rank; rank-scoped loggers usually hide nonzero ranks."""
    import os

    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[activation/automodel][rank={rank} local={local_rank}] {message}", flush=True)


def _shape_summary(value) -> str:
    if torch.is_tensor(value):
        return f"{tuple(value.shape)}:{value.dtype}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shape_summary(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_shape_summary(v)}" for k, v in value.items()) + "}"
    return type(value).__name__


def _strip_stale_pp_layer_kwargs(module, args, kwargs):
    """Drop stale forward-only kwargs from a decoder-layer call (forward-pre-hook, with_kwargs).

    Needed in two cases: (1) NeMo's manual pipeline_forward passes the pre-4.x singular
    ``past_key_value`` + ``use_cache``; (2) even without PP, the HF model forward passes
    ``use_cache`` (and possibly ``output_attentions``) which the layer forwards via ``**kwargs``
    into its token mixer. NeMo's ``CPAwareGatedDeltaNet`` (used for linear-attention layers when
    cp_size>1) has no ``**kwargs`` and rejects them. Model-agnostic and a no-op for layers/mixers
    that accept these kwargs (they keep them).
    """
    if not kwargs:
        return None

    keys_to_drop = {k for k in _STALE_LAYER_KWARGS if k in kwargs}
    try:
        signature = inspect.signature(module.forward)
        accepts_var_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed = set(signature.parameters)
            keys_to_drop.update(k for k in kwargs if k not in allowed)
    except (TypeError, ValueError):
        pass

    if not keys_to_drop:
        return None
    kwargs = dict(kwargs)
    for k in keys_to_drop:
        kwargs.pop(k, None)
    return args, kwargs


class ActivationScoringRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Forward-only scoring recipe built on the NeMo finetune recipe scaffolding."""

    def __init__(
        self,
        cfg,
        *,
        pruning_mixin,
        hook_kwargs: dict,
        pruning_cfg=None,
        activations_log_dir: str | None = None,
        eval_iters: int | None = None,
        use_puzzletron_dataloader: bool = True,
        extra_scorer_specs: list[dict] | None = None,
        data_cfg: dict | None = None,
        embedding_pruning_cfg: dict | None = None,
    ):
        super().__init__(cfg)
        self._pruning_mixin = pruning_mixin
        self._hook_kwargs = hook_kwargs
        self._pruning_cfg = pruning_cfg
        self._activations_log_dir = activations_log_dir
        self._eval_iters = eval_iters
        self._use_puzzletron_dataloader = use_puzzletron_dataloader
        self._extra_scorer_specs = extra_scorer_specs or []
        self._data_cfg = dict(data_cfg or {})
        self._embedding_pruning_cfg = dict(embedding_pruning_cfg or {})
        self._data_spec = (
            PuzzletronDataSpec.from_mapping(self._data_cfg)
            if self._data_cfg and "layout" in self._data_cfg
            else None
        )
        self._use_vlm_recipe = bool(
            self._data_spec is not None and self._data_spec.modality is Modality.MULTIMODAL
        )
        self._groups: MeshGroups | None = None
        self._scorers: list | None = None
        self._scorer_groups: list | None = None
        # Data-parallel data sharding (resolved in setup): each dp rank scores a disjoint slice.
        self._dp_size: int = 1
        self._dp_rank: int = 0
        self._use_autocast: bool = True
        self._autocast_dtype: torch.dtype = torch.bfloat16
        self._partial_resume_recovery: dict | None = None
        self._resumed_observability_local: dict[str, Any] = {}
        self._last_canonical_labels: torch.Tensor | None = None
        self._last_canonical_ce_mask: torch.Tensor | None = None
        self._last_canonical_kd_mask: torch.Tensor | None = None
        self._last_canonical_hidden_mask: torch.Tensor | None = None
        self._vision_monitors: list[Any] = []
        self._canonical_batch_fingerprints: list[str] = []
        self._model_descriptor = None

    def _resume_dir(self) -> Path | None:
        if not self._activations_log_dir:
            return None
        return Path(self._activations_log_dir) / ".native_resume"

    def _build_hidden_width_scorers(self, part, model_descriptor, hook_kwargs):
        if model_descriptor is None:
            raise ValueError("minitron_hidden_width scoring requires a model descriptor")
        config = getattr(part, "config", None)
        if config is None:
            raise ValueError("minitron_hidden_width scoring could not resolve model config")
        lm_config = model_descriptor.get_language_model_config(config)
        hidden_size = int(lm_config.hidden_size)
        widths = tuple(
            int(value)
            for value in (
                self._embedding_pruning_cfg.get("widths")
                or hook_kwargs.get("widths")
                or (hidden_size,)
            )
        )
        alignment = int(
            self._embedding_pruning_cfg.get(
                "alignment",
                hook_kwargs.get("alignment", 1),
            )
        )
        spec = model_descriptor.embedding_pruning_spec(
            config,
            widths=widths,
            alignment=alignment,
        )
        scorers = []
        for module_name, module in part.named_modules():
            if not module_name:
                continue
            if not any(re.search(pattern, module_name) for pattern in spec.residual_norm_patterns):
                continue
            scorer = HiddenWidthSiteScorer(
                module,
                self._groups,
                hidden_size=spec.hidden_size,
                name=module_name,
            )
            scorer.register()
            scorers.append(scorer)
        return scorers

    def _scorer_identity(self) -> list[tuple[str, str | None, int | None]]:
        return [
            (type(scorer).__name__, scorer.name, scorer.block_idx)
            for scorer in (self._scorers or [])
        ]

    def _load_resume_checkpoint(self, total: int) -> int:
        resume_dir = self._resume_dir()
        if resume_dir is None:
            return 0
        progress_path = resume_dir / "progress.json"
        if not progress_path.is_file():
            return 0
        progress = json.loads(progress_path.read_text())
        if int(progress.get("total", -1)) != int(total):
            raise RuntimeError(
                f"Activation resume total mismatch: {progress.get('total')} != {total}; "
                f"remove {resume_dir} before changing calibration settings"
            )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        state_path = resume_dir / f"rank_{rank}.pth"
        if not state_path.is_file():
            raise RuntimeError(f"Activation resume is missing rank state: {state_path}")
        payload = safe_load(state_path, map_location="cpu")
        self._resumed_observability_local = dict(payload.get("observability") or {})
        expected_identity = self._scorer_identity()
        saved_identity = [tuple(value) for value in payload.get("scorers", [])]
        if saved_identity != expected_identity:
            raise RuntimeError(
                "Activation resume scorer identity mismatch; remove the stale resume directory"
            )
        states = payload.get("states", [])
        if len(states) != len(self._scorers or []):
            raise RuntimeError("Activation resume scorer-state count mismatch")
        local_state_steps = {
            int(state["curr_iter"])
            for state in states
            if isinstance(state, dict) and state.get("curr_iter") is not None
        }
        if len(local_state_steps) != 1:
            raise RuntimeError(
                f"Activation resume has inconsistent iterative scorer steps: {local_state_steps}"
            )
        state_step = next(iter(local_state_steps))
        marker_step = int(progress["next_step"])
        if torch.distributed.is_initialized():
            step_bounds = torch.tensor(
                [state_step, state_step],
                device=torch.device("cuda", torch.cuda.current_device()),
                dtype=torch.int64,
            )
            torch.distributed.all_reduce(step_bounds[:1], op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(step_bounds[1:], op=torch.distributed.ReduceOp.MAX)
            min_step = int(step_bounds[0].item())
            max_step = int(step_bounds[1].item())
            if min_step != max_step:
                duplicate_range = os.environ.get("ACTIVATION_SCORING_DUPLICATE_RANGE", "")
                if not duplicate_range or min_step != marker_step:
                    raise RuntimeError(
                        "Activation resume rank-state generations disagree without a "
                        f"recoverable committed marker: marker={marker_step} "
                        f"min={min_step} max={max_step}"
                    )
                duplicate_start, duplicate_end = (
                    int(value) for value in duplicate_range.split(":", 1)
                )
                if duplicate_end - duplicate_start != max_step - min_step:
                    raise RuntimeError(
                        "Activation duplicate recovery length mismatch: "
                        f"range={duplicate_range} state_gap={max_step - min_step}"
                    )
                ahead = state_step == max_step
                if state_step not in (min_step, max_step):
                    raise RuntimeError(
                        f"Activation resume has an intermediate rank generation {state_step}"
                    )
                if ahead:
                    tp_size = int(self._groups.tp_size if self._groups is not None else 1)
                    peer_payload = None
                    for peer_rank in range(torch.distributed.get_world_size()):
                        if peer_rank == rank or peer_rank % tp_size != rank % tp_size:
                            continue
                        candidate_path = resume_dir / f"rank_{peer_rank}.pth"
                        candidate = safe_load(candidate_path, map_location="cpu")
                        candidate_identity = [
                            tuple(value) for value in candidate.get("scorers", [])
                        ]
                        candidate_steps = {
                            int(state["curr_iter"])
                            for state in candidate.get("states", [])
                            if isinstance(state, dict) and state.get("curr_iter") is not None
                        }
                        if candidate_identity == expected_identity and candidate_steps == {
                            min_step
                        }:
                            peer_payload = candidate
                            break
                    if peer_payload is None:
                        raise RuntimeError(
                            f"Activation resume could not find a healthy same-TP peer for rank {rank}"
                        )
                    peer_states = peer_payload["states"]
                    states = [
                        peer_states[index]
                        if isinstance(state, dict) and state.get("curr_iter") is not None
                        else state
                        for index, state in enumerate(states)
                    ]
                self._partial_resume_recovery = {
                    "duplicate_start": duplicate_start,
                    "duplicate_end": duplicate_end,
                    "ahead": ahead,
                    "rank": rank,
                }
                state_step = min_step
            else:
                state_step = min_step
        for scorer, state in zip(self._scorers or [], states):
            scorer.load_checkpoint_state(state)
        next_step = state_step
        from ...tools.logger import mprint

        if marker_step != state_step:
            mprint(
                "[activation/automodel] recovered an interrupted checkpoint commit: "
                f"shared_marker={marker_step} common_rank_state={state_step}"
            )
        mprint(
            f"[activation/automodel] resumed exact scorer state at iteration "
            f"{next_step}/{total} from {resume_dir}"
        )
        return next_step

    def _run_partial_resume_recovery(self) -> None:
        recovery = self._partial_resume_recovery
        if recovery is None:
            return
        import itertools

        from ...tools.logger import mprint

        additive = [
            scorer for scorer in (self._scorers or []) if "lane_gram" in scorer.checkpoint_state()
        ]
        iterative = [
            scorer for scorer in (self._scorers or []) if "curr_iter" in scorer.checkpoint_state()
        ]
        before = [scorer.checkpoint_state() for scorer in additive]
        for scorer in iterative:
            scorer.enabled = False
        try:
            batches = itertools.islice(
                self.dataloader,
                int(recovery["duplicate_start"]),
                int(recovery["duplicate_end"]),
            )
            for step, batch in enumerate(batches, start=int(recovery["duplicate_start"])):
                self._forward_batch(batch, step=step)
        finally:
            for scorer in iterative:
                scorer.enabled = True

        after = [scorer.checkpoint_state() for scorer in additive]

        def subtract_delta(old, new):
            if old is None:
                return None
            if torch.is_tensor(old):
                return old - (new - old)
            if isinstance(old, int):
                return old - (int(new) - old)
            return old

        for scorer, old_state, new_state in zip(additive, before, after):
            if recovery["ahead"]:
                restored = {
                    key: subtract_delta(value, new_state.get(key))
                    for key, value in old_state.items()
                }
            else:
                restored = old_state
            scorer.load_checkpoint_state(restored)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        mprint(
            "[activation/automodel] repaired partial rank checkpoint: "
            f"duplicate_range={recovery['duplicate_start']}:{recovery['duplicate_end']} "
            f"ahead_rank={recovery['rank'] if recovery['ahead'] else 'peer'}"
        )
        self._partial_resume_recovery = None

    def _save_resume_checkpoint(self, next_step: int, total: int) -> None:
        resume_dir = self._resume_dir()
        if resume_dir is None:
            return
        resume_dir.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        payload = {
            "version": 1,
            "next_step": int(next_step),
            "total": int(total),
            "scorers": self._scorer_identity(),
            "states": [scorer.checkpoint_state() for scorer in (self._scorers or [])],
            "observability": self._local_observability_metadata(),
        }
        state_path = resume_dir / f"rank_{rank}.pth"
        tmp_path = state_path.with_suffix(".tmp")
        safe_save(payload, tmp_path)
        tmp_path.replace(state_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.dist_env.is_main:
            progress_path = resume_dir / "progress.json"
            progress_tmp = progress_path.with_suffix(".tmp")
            progress_tmp.write_text(
                json.dumps(
                    {"version": 1, "next_step": int(next_step), "total": int(total)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            progress_tmp.replace(progress_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _clear_resume_checkpoint(self) -> None:
        resume_dir = self._resume_dir()
        if resume_dir is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.dist_env.is_main and resume_dir.exists():
            shutil.rmtree(resume_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def setup(self):
        """Build the parallel model + dataloader (NeMo), then freeze and attach scorers."""
        from ...tools.logger import aprint, mprint

        mprint("[activation/automodel] setup: entering NeMo recipe setup")
        if self._use_vlm_recipe:
            from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

            FinetuneRecipeForVLM.setup(self)
        else:
            super().setup()
        mprint("[activation/automodel] setup: NeMo recipe setup complete")
        patched_parts = self._patch_descriptor_pipeline_model_parts()
        if patched_parts:
            mprint(
                "[activation/automodel] descriptor pipeline compatibility patch "
                f"applied to {patched_parts} model part(s)"
            )
        # NeMo's TrainFinetuneRecipe unconditionally builds an optimizer during setup even
        # for forward-only passes.  Free it immediately — we never call optimizer.step().
        self.optimizer = None
        if self.pp is not None:
            # The Puzzletron varlen dataloader packs ``micro_batch_size`` samples into one
            # long sequence and returns dataloader batch_size=1. Non-varlen scoring can also
            # become local batch_size=1 after the explicit DP/FSDP split below. NeMo still needs
            # a large enough dummy local_batch_size to build PP, but this no-grad scoring pass
            # should execute each rank-local batch as a single PP microbatch; otherwise a
            # schedule configured for the pre-DP global batch can wait for a second microbatch
            # that no rank will provide.
            # AutoModel's native VLM dataloader returns the configured local
            # batch and its PP metadata is built for ``pp_microbatch_size``.
            # Preserve the original schedule's microbatch count so a local
            # batch of two is split into two static-shape microbatches. The
            # legacy Puzzletron loader already returns one packed row and must
            # continue to execute as one microbatch.
            forward_microbatches = (
                None if self._use_vlm_recipe and not self._use_puzzletron_dataloader else 1
            )
            _force_forward_only_pp_schedule(
                self.pp,
                log_prefix="[activation/automodel]",
                n_microbatches=forward_microbatches,
            )

        # Use the puzzletron tokenized dataloader so the calibration data matches the legacy
        # scorer exactly (same tokens/order). The recipe drives one batch per iteration.
        if self._use_puzzletron_dataloader and self._pruning_cfg is not None:
            from ...utils.data.dataloaders import prepare_validation_dataloader

            # IMPORTANT: pass tokenizer=None so prepare_dataloader builds AutoTokenizer.from_pretrained
            # (teacher_dir) — the SAME tokenizer the legacy scorer uses. Passing NeMo's self.tokenizer
            # instead produces a different tokenization of the same text (different special-token / chat-
            # template handling), which yields a disjoint calibration set and breaks legacy↔automodel
            # parity (the BOS/FIM RNG is seeded, so the tokenizer is the only remaining difference).
            # All ranks need the dataloader (the loop advances in lock-step), but
            # create_validation_dataloader realizes a shared on-disk cache via a tmp-file
            # rename that races when every rank writes it at once. Realize it on the main
            # rank first, barrier, then the rest read the now-existing cache.
            dataloader_msg = (
                f"dataset={self._pruning_cfg.get('dataset_path', None)} "
                f"split={self._pruning_cfg.get('val_dataset_name', None)} "
                f"eval_samples={self._pruning_cfg.get('eval_samples', None)} "
                f"micro_batch_size={self._pruning_cfg.get('micro_batch_size', None)} "
                f"block_size={self._pruning_cfg.get('block_size', None)} "
                f"cache={self._pruning_cfg.get('realized_dataset_cache_dir', None)}"
            )
            if self.dist_env.is_main:
                mprint(
                    f"[activation/automodel] setup: rank0 preparing Puzzletron dataloader ({dataloader_msg})"
                )
                self.dataloader = prepare_validation_dataloader(
                    self._pruning_cfg,
                    None,
                    data_layout=(
                        self._data_spec.layout.value if self._data_spec is not None else None
                    ),
                )
                mprint(
                    "[activation/automodel] setup: rank0 Puzzletron dataloader ready "
                    f"(len={self._safe_len(self.dataloader)})"
                )
            if torch.distributed.is_initialized():
                aprint("[activation/automodel] setup: entering dataloader barrier")
                torch.distributed.barrier()
                aprint("[activation/automodel] setup: dataloader barrier complete")
            if not self.dist_env.is_main:
                aprint(
                    f"[activation/automodel] setup: rank preparing Puzzletron dataloader ({dataloader_msg})"
                )
                self.dataloader = prepare_validation_dataloader(
                    self._pruning_cfg,
                    None,
                    data_layout=(
                        self._data_spec.layout.value if self._data_spec is not None else None
                    ),
                )
                aprint(
                    "[activation/automodel] setup: rank Puzzletron dataloader ready "
                    f"(len={self._safe_len(self.dataloader)})"
                )

        self._groups = MeshGroups.from_device_mesh(self.device_mesh, moe_mesh=self.moe_mesh)
        self._use_autocast, self._autocast_dtype = self._resolve_forward_autocast()
        if self._use_autocast:
            mprint(f"[activation/automodel] forward autocast ON: dtype={self._autocast_dtype}")
        else:
            mprint("[activation/automodel] forward autocast OFF")

        # Data-parallel data sharding: each calibration micro-batch is SPLIT across the dp ranks
        # (each dp rank processes ``mbs / dp_size`` of the batch's samples); the per-iteration /
        # finalize token SUM-reduce over ``dp_cp`` reassembles the full global batch. This is true
        # data parallelism (work split, no redundant compute) AND is bit-identical to dp=1: every
        # greedy step still sees the same ``mbs`` samples, just gathered across ranks. cp/tp/pp peers
        # of the same dp rank get the SAME sub-batch (cp then splits each sample's sequence). NeMo
        # flattens ``dp_cp`` as dp_shard (outer) x cp (inner), so dp_rank = token_rank // cp_size.
        _, _cp_size = self._cp_info()
        _cp_size = _cp_size or 1
        _tok_size, _tok_rank = self._groups.token_size, self._groups.token_rank
        self._dp_size, self._dp_rank = self._resolve_data_parallel_slice(
            getattr(self.cfg, "distributed", None),
            token_size=_tok_size,
            token_rank=_tok_rank,
            cp_size=_cp_size,
        )
        if self._dp_size > 1:
            mprint(
                f"[activation/automodel] data-parallel batch split ON: dp_size={self._dp_size} "
                f"cp_size={_cp_size} — each micro-batch is split across dp ranks and the dp_cp "
                "reduce reassembles it, so scores (incl. the iterative greedy) match dp=1 exactly."
            )

        # Read the teacher's per-layer block_configs from the checkpoint (config.json /
        # block_configs.json), as the reference KD recipes do — the loaded HF config does
        # not carry them (esp. for force_hf / VL models). Passed explicitly to build_scorers
        # so heterogeneous teachers get correct per-layer dims.
        from .patch import auto_detect_block_configs

        teacher_path = self._pruning_cfg.get("model_name_or_path") if self._pruning_cfg else None
        block_configs = auto_detect_block_configs(teacher_path) if teacher_path else None

        # Freeze every stage this rank owns; build scorers only when a pruning mixin is given
        # (activation scoring). Subclasses for other forward-only passes — e.g. replace-1-block
        # solution scoring — reuse this setup with no mixin and attach their own capture hooks.
        method = self._hook_kwargs.get("method")

        # Build the list of all scorer specs: primary (if mixin+method present) + extras.
        all_specs = []
        if method is not None and (
            self._pruning_mixin is not None
            or method in {"minitron_hidden_width", "magnitude_fallback"}
        ):
            all_specs.append(
                {
                    "pruning_mixin": self._pruning_mixin,
                    "hook_kwargs": self._hook_kwargs,
                    "activations_log_dir": self._activations_log_dir,
                }
            )
        all_specs.extend(self._extra_scorer_specs)

        # One inner list per spec, accumulated over all model_parts.
        spec_scorer_lists: list[list] = [[] for _ in all_specs]

        descriptor_name = None
        if self._pruning_cfg is not None:
            descriptor_name = self._pruning_cfg.get("descriptor", None)
        if descriptor_name is None:
            model_cfg = getattr(self.cfg, "model", None)
            descriptor_name = getattr(model_cfg, "anymodel_descriptor", None)
        model_descriptor = ModelDescriptorFactory.get(descriptor_name) if descriptor_name else None
        self._model_descriptor = model_descriptor

        if self._use_vlm_recipe and model_descriptor is not None:
            from .batch_adapter import VisionForwardMonitor

            seen_vision_modules: set[int] = set()
            for part in self.model_parts:
                for canonical_name in model_descriptor.vision_module_names():
                    adapted_name = model_descriptor.adapt_module_name_for_model(
                        canonical_name,
                        part,
                    )
                    for candidate in dict.fromkeys((adapted_name, canonical_name)):
                        try:
                            vision_module = part.get_submodule(candidate)
                        except AttributeError:
                            continue
                        if id(vision_module) in seen_vision_modules:
                            continue
                        seen_vision_modules.add(id(vision_module))
                        monitor = VisionForwardMonitor(vision_module)
                        monitor.__enter__()
                        self._vision_monitors.append(monitor)

        self._scorers = []
        for part in self.model_parts:
            part.eval()
            for param in part.parameters():
                param.requires_grad_(False)
            for i, spec in enumerate(all_specs):
                spec_method = spec["hook_kwargs"].get("method")
                if spec_method is None:
                    continue
                if spec_method == "minitron_hidden_width":
                    spec_scorer_lists[i] += self._build_hidden_width_scorers(
                        part,
                        model_descriptor,
                        spec["hook_kwargs"],
                    )
                elif spec_method == "magnitude_fallback":
                    targets = spec["hook_kwargs"].get("targets") or ()
                    if not targets:
                        raise ValueError("magnitude_fallback requires descriptor-owned targets")
                    spec_scorer_lists[i] += build_magnitude_scorers(
                        part,
                        self._groups,
                        targets,
                        model_descriptor=model_descriptor,
                        target_type=spec["hook_kwargs"].get("target_type", "generic"),
                    )
                else:
                    spec_scorer_lists[i] += build_scorers(
                        part,
                        self._groups,
                        spec["pruning_mixin"],
                        spec_method,
                        model_descriptor=model_descriptor,
                        block_configs=block_configs,
                        optimize_for=spec["hook_kwargs"].get("optimize_for", "memory"),
                        validation_full_iters=spec["hook_kwargs"].get("validation_full_iters"),
                        calibration_method=spec["hook_kwargs"].get("calibration_method"),
                        clear_gpu_memory=spec["hook_kwargs"].get("clear_gpu_memory", False),
                        scored_axes=spec["hook_kwargs"].get("scored_axes"),
                        token_chunk_size=spec["hook_kwargs"].get("token_chunk_size"),
                    )

        self._scorer_groups = [
            (spec_scorer_lists[i], spec["activations_log_dir"]) for i, spec in enumerate(all_specs)
        ]
        self._scorers = [s for scorers, _ in self._scorer_groups for s in scorers]

        # Strip the stale forward-only kwargs (singular past_key_value, use_cache) before they leak
        # into each layer's token mixer. Needed in TWO cases: (1) under PP, NeMo's manual
        # pipeline_forward passes them; (2) even without PP, NeMo swaps in CP-aware mixers (e.g.
        # CPAwareGatedDeltaNet when cp_size>1) whose forward rejects use_cache, while the HF
        # decoder-layer forward still forwards it through **kwargs. Generic across HF architectures
        # (no model-specific code); a no-op for layers/mixers that accept these kwargs.
        n_layers = self._install_pp_layer_kwarg_sanitizers()
        # Forward-only activation and replacement scoring consume internal hook
        # activations, never the model's dense vocabulary logits.  Remove the
        # head for both PP and non-PP execution.  Keeping it in the non-PP path
        # materializes a (B, S, vocab) tensor that can exceed 60 GiB at long
        # context and, after an exact resume, fragment the allocator before the
        # next batch.  The original weight is retained for FlashKLD metrics.
        self._remove_pp_lm_heads()
        if self.pp is not None:
            # Remove lm_head from all model parts that carry one.  The PP schedule merges the
            # return value of the last stage across micro-batches (merge_chunks); when lm_head
            # is present that return value is the full (B, S, vocab) logits tensor — ~9–20 GiB
            # for large vocabularies — causing OOM on merge.  For scoring we only need the
            # intermediate hook activations; lm_head output is never used.  Without lm_head,
            # pipeline_forward_causal_lm returns hidden_states (~21× smaller than logits).
            self._refresh_pp_hidden_output_meta()
            n_pp_materializers = self._install_pp_dtensor_output_materializers()
            if n_pp_materializers:
                mprint(
                    "[activation/automodel] PP DTensor output materializers on "
                    f"{n_pp_materializers} model part(s)"
                )
            n_sp_restorers = self._install_pp_sequence_parallel_input_restorer()
            if n_sp_restorers:
                mprint(
                    "[activation/automodel] PP sequence-parallel input layout restorers on "
                    f"{n_sp_restorers} non-first-stage model part(s)"
                )
            # For PP+CP: non-first-stage ranks don't receive position_ids from outside;
            # pipeline_forward auto-generates arange(0, S_local) identically on every CP
            # rank, so the CP all-gather sees duplicates instead of the required 0..S-1.
            # Install a pre-hook on every CPAwareGatedDeltaNet that overwrites
            # _cached_position_ids with the correct CP-shard range before the module runs.
            cp_rank, cp_size = self._cp_info()
            if cp_size > 1:
                automodel_cfg = (
                    self._pruning_cfg.get("automodel", {}) if self._pruning_cfg is not None else {}
                )
                gdn_cp_backend = str(automodel_cfg.get("gdn_cp_backend", "native_fla"))
                if gdn_cp_backend == "replicated_exact":
                    from .cp_fallback import install_exact_replicated_gdn_cp

                    n_fallbacks = install_exact_replicated_gdn_cp(self.model_parts)
                    mprint(
                        "[activation/automodel] exact replicated CP GDN fallback "
                        f"on {n_fallbacks} module(s)"
                    )
                elif gdn_cp_backend != "native_fla":
                    raise ValueError(f"Unknown automodel.gdn_cp_backend={gdn_cp_backend!r}")
                if bool(automodel_cfg.get("trace_layer_forwards", False)):
                    n_traces = self._install_layer_forward_traces()
                    mprint(f"[activation/automodel] layer forward traces on {n_traces} module(s)")
                n_cp_hooks = self._install_cp_position_overrides(cp_rank, cp_size)
                mprint(
                    f"[activation/automodel] CP position override hooks on {n_cp_hooks} module(s)"
                )
            # Null out the schedule's loss_fn so _maybe_compute_loss is a no-op.
            # After removing lm_head the last stage returns hidden_states; the schedule
            # would still try to compute loss (and crash on None target_mbs) unless we
            # clear it.  Scoring uses hooks only — the loss value is never needed.
            sched = getattr(getattr(self.pp, "info", None), "schedule", None)
            if sched is not None:
                # PyTorch pipelining stores it as _loss_fn (private).
                for attr in ("_loss_fn", "loss_fn"):
                    if getattr(sched, attr, None) is not None:
                        setattr(sched, attr, None)

        # Report the ACTUAL parallel layout NeMo built (so it's obvious it isn't legacy).
        if n_layers:
            mprint(f"[activation/automodel] decoder-layer kwarg sanitizer on {n_layers} layer(s)")

        device_mesh = self.device_mesh
        mprint(
            "[activation/automodel] ACTUAL device mesh: "
            f"dims={getattr(device_mesh, 'mesh_dim_names', None)} "
            f"shape={tuple(device_mesh.shape) if device_mesh is not None else None} | "
            f"groups: token(dp_cp)={self._groups.token_size} tp={self._groups.tp_size} "
            f"ep={self._groups.ep_size} pp={self._groups.pp_size} | "
            f"pp_stages(model_parts)={len(self.model_parts)} method={method or 'none'}"
        )
        aprint(f"[activation/automodel] this rank owns {len(self._scorers)} scored module(s)")

    @staticmethod
    def _safe_len(obj) -> str:
        try:
            return str(len(obj))
        except Exception:  # noqa: BLE001
            return "unknown"

    def _patch_descriptor_pipeline_model_parts(self) -> int:
        """Let the descriptor install transient aliases on local PP chunks."""
        descriptor_name = None
        if self._pruning_cfg is not None:
            descriptor_name = self._pruning_cfg.get("descriptor", None)
        if descriptor_name is None:
            model_cfg = getattr(self.cfg, "model", None)
            descriptor_name = getattr(model_cfg, "anymodel_descriptor", None)
        descriptor = ModelDescriptorFactory.get(descriptor_name) if descriptor_name else None
        if descriptor is None or not hasattr(descriptor, "patch_pipeline_model_part"):
            return 0

        patched = 0
        for part in self.model_parts or []:
            if descriptor.patch_pipeline_model_part(part):
                patched += 1
        return patched

    def _install_pp_layer_kwarg_sanitizers(self) -> int:
        """Register :func:`_strip_stale_pp_layer_kwargs` on every decoder layer this rank owns.

        Decoder layers are found generically as the children of any sub-module attribute named
        ``layers`` (the ModuleList/ModuleDict that NeMo's ``pipeline_forward`` iterates) — true
        for HF causal-LM architectures broadly, not just one model.
        """
        from torch import nn

        count, seen = 0, set()
        for part in self.model_parts:
            for name, module in part.named_modules():
                if name.rsplit(".", 1)[-1] != "layers":
                    continue
                if isinstance(module, nn.ModuleDict):
                    children = module.values()
                elif isinstance(module, (nn.ModuleList, nn.Sequential)):
                    children = module
                else:
                    continue
                for layer in children:
                    if id(layer) in seen:
                        continue
                    seen.add(id(layer))
                    layer.register_forward_pre_hook(_strip_stale_pp_layer_kwargs, with_kwargs=True)
                    count += 1
        return count

    def _remove_pp_lm_heads(self) -> int:
        """Replace each scoring lm_head with a hidden-state passthrough.

        ``pipeline_forward_causal_lm`` checks ``if self.lm_head is not None`` before
        applying the final norm and identifying the last stage.  Setting the head to
        ``None`` avoids logits but also skips the final norm on native Qwen/Nemotron
        models, leaving replacement scoring with no captured targets.  A passthrough
        preserves the native last-stage branch while returning ``hidden_states`` instead
        of the full ``(B, S, vocab_size)`` logits tensor, eliminating the ~9–20 GiB
        allocation that causes OOM in ``merge_chunks``.
        Only the stage that actually holds lm_head (the last PP stage on its ranks)
        is affected; other stages have no lm_head and are unchanged.
        """
        from ...tools.logger import mprint

        removed = 0
        disabled_mtp = 0
        for part in self.model_parts:
            # Forward-only activation/replacement scoring never consumes MTP
            # auxiliary outputs.  Some native PP models propagate an extra MTP
            # embedding between stages and use ``lm_head is not None`` to detect
            # the final stage.  Removing the head below would otherwise turn the
            # final stage into an apparent intermediate stage and change its
            # output arity after PP metadata has been frozen.  Disable only the
            # transient runtime MTP config; checkpoint weights/config remain
            # untouched for the later MTP distillation stage.
            mtp_config = getattr(part, "mtp_config", None)
            if mtp_config is not None and bool(getattr(mtp_config, "enabled", False)):
                mtp_config.num_layers = 0
                mtp_config.layer_pattern = ""
                part._puzzletron_mtp_disabled_for_scoring = True
                disabled_mtp += 1
            lm_head = getattr(part, "lm_head", None)
            if lm_head is not None:
                weight = getattr(lm_head, "weight", None)
                if weight is not None:
                    # Replace-one-block scoring still needs the LM-head weight to
                    # reconstruct logits from captured hidden states. Keep a
                    # transient reference before removing the module that would
                    # otherwise make PP merge full-vocab logits.
                    part._puzzletron_removed_lm_head_weight = weight.detach()
                part.lm_head = _HiddenStatePassthrough()
                part._puzzletron_lm_head_passthrough = True
                removed += 1
        if removed:
            mprint(
                f"[activation/automodel] replaced lm_head with hidden-state passthrough "
                f"on {removed} model part(s)"
            )
        if disabled_mtp:
            mprint(
                "[activation/automodel] disabled transient MTP PP propagation on "
                f"{disabled_mtp} model part(s) — checkpoint MTP weights remain intact"
            )
        return removed

    def _refresh_pp_hidden_output_meta(self) -> int:
        """Keep PP metadata consistent after removing ``lm_head`` from the last stage.

        NeMo precomputes pipeline stage metadata while ``lm_head`` is still attached, so the
        last stage's output meta may be ``(B, S, vocab)`` logits.  Puzzletron removes the head
        before scoring to avoid materializing logits, which makes the actual last-stage output
        ``(B, S, hidden)``.  Reconfigure the local last stage metadata when precompute is active.
        """
        from ...tools.logger import aprint

        info = getattr(self.pp, "info", None)
        stages = getattr(info, "stages", None) or []
        refreshed = 0
        for stage in stages:
            submod = getattr(stage, "submod", None)
            if submod is None:
                continue
            lm_head_passthrough = bool(getattr(submod, "_puzzletron_lm_head_passthrough", False))
            mtp_disabled = bool(getattr(submod, "_puzzletron_mtp_disabled_for_scoring", False))
            if getattr(submod, "lm_head", None) is not None and not lm_head_passthrough:
                continue
            if not getattr(stage, "is_last", False) and not mtp_disabled:
                continue
            inputs_meta = getattr(stage, "inputs_meta", None)
            if not inputs_meta:
                aprint(
                    "[activation/automodel] PP last-stage inputs_meta absent; runtime shape inference will be used"
                )
                continue
            first_meta = inputs_meta[0]
            if not torch.is_tensor(first_meta) or first_meta.dim() < 2:
                continue

            microbatch_size = int(first_meta.shape[0])
            seq_len = int(first_meta.shape[1])
            hidden_size = int(first_meta.shape[2]) if first_meta.dim() >= 3 else None
            if hidden_size is None:
                cfg = getattr(submod, "config", None)
                text_config = getattr(cfg, "text_config", None)
                if text_config is not None and not hasattr(cfg, "hidden_size"):
                    cfg = text_config
                hidden_size = getattr(cfg, "hidden_size", None)
            if hidden_size is None:
                aprint(
                    "[activation/automodel] could not infer hidden_size for PP last-stage output metadata"
                )
                continue

            try:
                dtype = next(submod.parameters()).dtype
            except StopIteration:
                dtype = torch.bfloat16
            # PyTorch's public method intentionally disallows reconfiguration, but Puzzletron
            # removes lm_head after NeMo precomputes metadata so the last stage returns hidden
            # states instead of logits.  Updating the private frozen meta is the narrowest way
            # to keep send/recv shapes consistent for this forward-only scoring path.
            stage._outputs_meta = (
                torch.empty(
                    microbatch_size,
                    seq_len,
                    int(hidden_size),
                    device="meta",
                    dtype=dtype,
                ),
            )
            # Native PP models with MTP precompute an additional inter-stage
            # embedding in both the producer's outputs and the consumer's
            # inputs.  Forward-only scoring disables that auxiliary branch, so
            # the receiving stage must consume only the hidden-state tensor.
            if mtp_disabled and not getattr(stage, "is_first", False) and len(inputs_meta) > 1:
                stage.inputs_meta = (first_meta,)
            refreshed += 1

        if refreshed:
            aprint(
                "[activation/automodel] refreshed PP last-stage output metadata "
                f"for hidden-state scoring ({refreshed} stage(s))"
            )
        return refreshed

    def _update_pp_seq_len_for_scoring(
        self,
        seq_len: int,
        *,
        envelope_batch_size: int | None = None,
    ) -> None:
        """Update variable-length PP shapes and restore the scoring output contract.

        ``AutoPipeline.update_seq_len`` may rebuild the stage metadata. Native model
        metadata hooks see Puzzletron's non-``None`` hidden-state passthrough as an LM
        head and therefore rebuild a vocabulary-sized output. Scoring intentionally
        returns hidden states, so the contract must be restored *after* every possible
        rebuild (the updater itself remains a no-op when the length is unchanged).
        """
        if envelope_batch_size is not None and hasattr(self.pp, "pp_microbatch_size"):
            envelope_batch_size = int(envelope_batch_size)
            if envelope_batch_size <= 0:
                raise ValueError(
                    f"PP envelope batch size must be positive, got {envelope_batch_size}"
                )
            if int(self.pp.pp_microbatch_size) != envelope_batch_size:
                self.pp.pp_microbatch_size = envelope_batch_size
                # ``update_seq_len`` otherwise skips a rebuild when only the physical
                # packed-envelope batch dimension changed.
                self.pp._pp_current_seq_len = None
        self.pp.update_seq_len(seq_len)
        self._refresh_pp_hidden_output_meta()

    def _pp_envelope_batch_size(self, local_batch_size: int) -> int:
        schedule = self.pp.info.schedule
        microbatches = int(getattr(schedule, "_n_microbatches", 1) or 1)
        local_batch_size = int(local_batch_size)
        return (local_batch_size + microbatches - 1) // microbatches

    def _install_cp_position_overrides(self, cp_rank: int, cp_size: int) -> int:
        """Install pre-hooks on CPAwareGatedDeltaNet modules to fix position_ids under PP+CP.

        Non-first-stage PP ranks receive hidden states (not input_ids) so pipeline_forward
        auto-generates ``position_ids = arange(0, S_local)`` identically on every CP rank.
        The CP all-gather then sees duplicate positions instead of the required dense
        ``0..S-1`` range, causing a RuntimeError in ``_undo_attention_load_balancing``.

        The decoder layer always passes ``position_ids`` as an explicit keyword argument, so
        checking ``if position_ids is None`` inside the module never triggers.  This hook uses
        ``with_kwargs=True`` to intercept and replace the ``position_ids`` kwarg directly with
        the correct CP-shard range ``[cp_rank*S_local .. (cp_rank+1)*S_local - 1]`` before
        the module runs.  It also sets ``_cached_position_ids`` for the rare path where
        ``position_ids`` arrives as ``None``.
        """
        count = 0
        for part in self.model_parts or []:
            for m in part.modules():
                if getattr(m, "_cp_mesh", None) is None or m._cp_mesh.size() <= 1:
                    continue

                def _make_hook(rank: int):
                    def _hook(module, args, kwargs):
                        hidden_states = args[0] if args else kwargs.get("hidden_states")
                        if hidden_states is None:
                            return None
                        to_local = getattr(hidden_states, "to_local", None)
                        hidden_local = to_local() if callable(to_local) else hidden_states
                        S_local = hidden_local.shape[1]
                        provided_positions = kwargs.get("position_ids")
                        if callable(to_local):
                            tp_group = hidden_states.device_mesh.get_group()
                            self._actual_tp_group = tp_group
                            tp_rank = torch.distributed.get_rank(tp_group)
                            tp_size = torch.distributed.get_world_size(tp_group)
                            start = (rank * tp_size + tp_rank) * S_local
                            local_positions = torch.arange(
                                start,
                                start + S_local,
                                device=hidden_states.device,
                                dtype=torch.long,
                            )
                        elif (
                            provided_positions is not None
                            and provided_positions.shape[-1] == S_local
                        ):
                            local_positions = provided_positions
                        else:
                            local_positions = torch.arange(
                                rank * S_local,
                                (rank + 1) * S_local,
                                device=hidden_states.device,
                                dtype=torch.long,
                            )
                        module._cached_position_ids = local_positions
                        kwargs = dict(kwargs)
                        kwargs["position_ids"] = local_positions
                        return args, kwargs

                    return _hook

                m.register_forward_pre_hook(_make_hook(cp_rank), with_kwargs=True)
                count += 1
        return count

    def _install_pp_dtensor_output_materializers(self) -> int:
        """Materialize SP DTensors before PyTorch pipeline P2P communication."""
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return 0

        def _materialize(_module, _args, output):
            if isinstance(output, DTensor):
                return output.full_tensor()
            if isinstance(output, tuple):
                return tuple(
                    value.full_tensor() if isinstance(value, DTensor) else value for value in output
                )
            return output

        stages = getattr(getattr(self.pp, "info", None), "stages", None) or []
        modules = [stage.submod for stage in stages if getattr(stage, "submod", None) is not None]
        if not modules:
            modules = list(self.model_parts or [])

        count, seen = 0, set()
        for part in modules:
            if id(part) in seen:
                continue
            seen.add(id(part))
            part.register_forward_hook(_materialize)
            count += 1
        return count

    @staticmethod
    def _replicate_plain_pp_input(hidden_states, tp_mesh):
        """Restore the TP layout erased by pipeline point-to-point transport.

        PyTorch PP communicates plain tensors. A non-first stage followed by a
        ``SequenceParallel`` norm must therefore see that tensor as replicated;
        otherwise the norm's input hook interprets the full sequence as a local
        shard and doubles the logical sequence length at TP=2.
        """
        from torch.distributed.tensor import DTensor, Replicate

        if isinstance(hidden_states, DTensor) or not torch.is_tensor(hidden_states):
            return hidden_states
        return DTensor.from_local(
            hidden_states,
            device_mesh=tp_mesh,
            placements=[Replicate()],
            run_check=False,
        )

    @staticmethod
    def _has_tp_replicated_placement(value, tp_mesh) -> bool:
        """Return whether ``value`` is replicated specifically over the TP mesh."""

        from torch.distributed.tensor import Replicate

        placements = tuple(getattr(value, "placements", ()) or ())
        value_mesh = getattr(value, "device_mesh", None)
        mesh_dim_names = tuple(getattr(value_mesh, "mesh_dim_names", ()) or ())
        if "tp" in mesh_dim_names and len(mesh_dim_names) == len(placements):
            return isinstance(placements[mesh_dim_names.index("tp")], Replicate)
        return (
            value_mesh is tp_mesh and len(placements) == 1 and isinstance(placements[0], Replicate)
        )

    @classmethod
    def _module_expects_sequence_parallel_input(cls, module, tp_mesh) -> bool:
        """Return whether this decoder layer's own norm is TP sequence-parallel.

        A pre-forward FSDP parameter is also a DTensor, but it is sharded over
        the FSDP mesh. Treating that as evidence of TP sequence parallelism
        converts the plain PP activation too early; FSDP then materializes the
        norm weight as a local tensor and the norm receives mixed tensor types.
        """

        from torch.distributed.tensor import DTensor

        norm = getattr(module, "input_layernorm", None)
        weight = getattr(norm, "weight", None)
        return isinstance(weight, DTensor) and cls._has_tp_replicated_placement(
            weight,
            tp_mesh,
        )

    def _install_pp_sequence_parallel_input_restorer(self) -> int:
        """Mark non-first PP activations replicated before TP sequence parallelism."""
        from ...tools.logger import aprint

        local_stages = (
            getattr(getattr(self.pp, "info", None), "stages", None) or []
            if self.pp is not None
            else []
        )
        if self.pp is None:
            has_first_stage = True
        elif local_stages and all(hasattr(stage, "is_first") for stage in local_stages):
            has_first_stage = any(bool(stage.is_first) for stage in local_stages)
        else:
            has_first_stage = bool(self.pp.info.has_first_stage)
        distributed_cfg = getattr(self.cfg, "distributed", None)
        sequence_parallel = bool(getattr(distributed_cfg, "sequence_parallel", False))
        assert self._groups is not None, "call setup() before installing PP restorers"
        tp_size = int(self._groups.tp_size)
        pruning_cfg = getattr(self, "_pruning_cfg", None)
        trace_enabled = bool(
            pruning_cfg.get("automodel", {}).get("trace_layer_forwards", False)
            if pruning_cfg is not None
            else False
        ) or os.environ.get("PUZZLETRON_TRACE_BATCHES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if trace_enabled:
            _rank_trace(
                "PP SP restorer decision: "
                f"local_stages={[(getattr(stage, 'stage_index', '?'), getattr(stage, 'is_first', '?')) for stage in local_stages]} "
                f"has_first_stage={has_first_stage} sequence_parallel={sequence_parallel} "
                f"tp_size={tp_size}"
            )
        aprint(
            "[activation/automodel] PP SP restorer decision: "
            f"has_first_stage={has_first_stage} sequence_parallel={sequence_parallel} "
            f"tp_size={tp_size}"
        )
        if has_first_stage:
            return 0
        if not sequence_parallel:
            return 0
        if tp_size <= 1:
            return 0
        tp_mesh = self.device_mesh["tp"]

        def _restore(_module, args, kwargs):
            expects_dtensor = self._module_expects_sequence_parallel_input(
                _module,
                tp_mesh,
            )
            if trace_enabled:
                norm = getattr(_module, "input_layernorm", None)
                weight = getattr(norm, "weight", None)
                source_value = args[0] if args else None
                if source_value is None:
                    source_value = next(
                        (
                            kwargs.get(key)
                            for key in ("x", "hidden_states", "inputs_embeds", "input_ids")
                            if kwargs.get(key) is not None
                        ),
                        None,
                    )
                _rank_trace(
                    f"PP SP inspect module={type(_module).__name__} "
                    f"input={type(source_value).__name__} "
                    f"input_placements={getattr(source_value, 'placements', None)} "
                    f"direct_norm_weight={type(weight).__name__} "
                    f"weight_placements={getattr(weight, 'placements', None)} "
                    f"expects_dtensor={expects_dtensor}"
                )
            if not expects_dtensor:
                return args, kwargs
            source = None
            if args and torch.is_tensor(args[0]):
                source = "args[0]"
                args = (
                    self._replicate_plain_pp_input(args[0], tp_mesh),
                    *args[1:],
                )
            else:
                for key in ("x", "hidden_states", "inputs_embeds", "input_ids"):
                    value = kwargs.get(key)
                    if not torch.is_tensor(value):
                        continue
                    if key == "input_ids" and not torch.is_floating_point(value):
                        continue
                    source = key
                    kwargs = dict(kwargs)
                    kwargs[key] = self._replicate_plain_pp_input(value, tp_mesh)
                    break
            if trace_enabled and source is not None:
                value = args[0] if source == "args[0]" else kwargs[source]
                _rank_trace(
                    f"PP SP restored {type(_module).__name__}.{source} -> "
                    f"{type(value).__name__} placements={getattr(value, 'placements', None)}"
                )
            return args, kwargs

        roots = [
            stage.submod for stage in local_stages if getattr(stage, "submod", None) is not None
        ]
        roots.extend(self.model_parts or [])
        modules = []
        for root in roots:
            modules.extend(
                module
                for module in root.modules()
                if hasattr(module, "input_layernorm")
                and (
                    hasattr(module, "self_attn")
                    or hasattr(module, "linear_attn")
                    or hasattr(module, "mlp")
                )
            )

        count, seen = 0, set()
        for part in modules:
            if id(part) in seen:
                continue
            seen.add(id(part))
            part.register_forward_pre_hook(_restore, with_kwargs=True, prepend=True)
            count += 1
        aprint(f"[activation/automodel] PP SP restorer registered on {count} decoder layer(s)")
        if trace_enabled:
            _rank_trace(f"PP SP restorer registered on {count} decoder layer(s)")
        return count

    def _install_layer_forward_traces(self) -> int:
        """Log decoder-layer entry/exit on every rank for compute-node diagnostics."""
        count = 0
        for part in self.model_parts or []:
            for name, module in part.named_modules():
                if not (
                    hasattr(module, "input_layernorm")
                    and hasattr(module, "mlp")
                    and (hasattr(module, "self_attn") or hasattr(module, "linear_attn"))
                ):
                    continue

                label = name

                def _make_pre(layer_label: str):
                    def _pre(_module, _args, _kwargs):
                        value = _args[0] if _args else _kwargs.get("x")
                        _rank_trace(
                            f"enter decoder layer {layer_label} input={type(value).__name__} "
                            f"placements={getattr(value, 'placements', None)}"
                        )
                        return _args, _kwargs

                    return _pre

                def _make_post(layer_label: str):
                    def _post(_module, _args, output):
                        _rank_trace(f"exit decoder layer {layer_label}")
                        return output

                    return _post

                module.register_forward_pre_hook(_make_pre(label), with_kwargs=True)
                module.register_forward_hook(_make_post(label))
                count += 1
        return count

    def _cp_info(self):
        """Return (cp_rank, cp_size) using the device_mesh "cp" dim, like NeMo's parallelizer.

        The NeMo parallelizer (parallelizer.py) uses exactly::

            cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None

        We replicate that, then fall back to walking model_parts for any module whose
        ``_cp_mesh`` was set by the parallelizer.
        """
        import torch.distributed as dist

        from ...tools.logger import mprint

        # Primary: same lookup as NeMo parallelizer.py line 353.
        mesh = self.device_mesh
        if mesh is not None:
            dim_names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
            if "cp" in dim_names:
                cp_mesh = mesh["cp"]
                cp_size = cp_mesh.size()
                if cp_size > 1:
                    cp_group = cp_mesh.get_group()
                    cp_rank = dist.get_rank(cp_group)
                    mprint(
                        f"[activation/automodel] CP sharding active: cp_rank={cp_rank} cp_size={cp_size}"
                    )
                    return cp_rank, cp_size
            else:
                mprint(
                    f"[activation/automodel] _cp_info: device_mesh.mesh_dim_names={dim_names!r} — 'cp' not found"
                )
        else:
            mprint("[activation/automodel] _cp_info: self.device_mesh is None")

        # Fallback: walk model_parts for any CPAwareGatedDeltaNet with _cp_mesh set.
        for part in self.model_parts or []:
            for m in part.modules():
                cp_mesh = getattr(m, "_cp_mesh", None)
                if cp_mesh is not None and cp_mesh.size() > 1:
                    grp = cp_mesh.get_group()
                    cp_rank = dist.get_rank(grp)
                    cp_size = dist.get_world_size(grp)
                    mprint(
                        f"[activation/automodel] CP sharding via module._cp_mesh: cp_rank={cp_rank} cp_size={cp_size}"
                    )
                    return cp_rank, cp_size

        mprint("[activation/automodel] _cp_info: CP not detected (size=1 or not configured)")
        return 0, 1

    def tensor_parallel_group(self):
        """Return the process group from the actual TP mesh NeMo constructed."""
        actual_group = getattr(self, "_actual_tp_group", None)
        if actual_group is not None:
            return actual_group
        mesh = self.device_mesh
        if mesh is not None:
            dim_names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
            if "tp" in dim_names:
                return mesh["tp"].get_group()
        return self._groups.tp_group if self._groups is not None else None

    def context_parallel_group(self):
        """Return the CP group attached to the live recurrent token mixers."""
        for part in self.model_parts or []:
            for module in part.modules():
                cp_mesh = getattr(module, "_cp_mesh", None)
                if cp_mesh is not None and cp_mesh.size() > 1:
                    return cp_mesh.get_group()
        return None

    def _shard_seq_for_cp(self, tensor, cp_rank: int, cp_size: int):
        """Slice a (batch, seq, ...) or (batch, seq) tensor to this CP rank's shard."""
        if tensor is None or cp_size <= 1:
            return tensor
        S = tensor.shape[1]
        shard = S // cp_size
        start = cp_rank * shard
        return tensor[:, start : start + shard]

    @staticmethod
    def _load_balanced_cp_sequence_shard(tensor, cp_rank: int, cp_size: int):
        """Apply PyTorch CP's exact load-balanced sequence permutation locally."""
        if tensor is None or cp_size <= 1:
            return tensor
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            _create_default_load_balancer,
        )

        seq_len = int(tensor.shape[1])
        load_balancer = _create_default_load_balancer(seq_len, cp_size, tensor.device)
        indices = load_balancer._generate_indices()
        if indices.shape[0] == 1 and tensor.shape[0] > 1:
            indices = indices.expand(tensor.shape[0], -1)
        for _ in range(2, tensor.ndim):
            indices = indices.unsqueeze(-1)
        indices = indices.expand(tensor.shape)
        balanced = torch.gather(tensor, 1, indices)
        local_seq_len = seq_len // cp_size
        return balanced.narrow(1, cp_rank * local_seq_len, local_seq_len).contiguous()

    def _resolve_forward_autocast(self) -> tuple[bool, torch.dtype]:
        """Mirror the legacy validator's autocast choice for scoring forwards."""
        autocast_dtype = torch.bfloat16
        use_autocast = True
        if self._pruning_cfg is not None:
            autocast_dtype = resolve_torch_dtype(
                self._pruning_cfg.get("autocast_dtype", "torch.bfloat16")
            )
            descriptor_name = self._pruning_cfg.get("descriptor", None)
            descriptor = ModelDescriptorFactory.get(descriptor_name) if descriptor_name else None
            if descriptor is not None:
                use_autocast = descriptor.uses_autocast()
        return use_autocast, autocast_dtype

    def _forward_autocast_context(self):
        if self._use_autocast and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
        return nullcontext()

    def _pp_meta_summary(self) -> str:
        if self.pp is None:
            return "pp=disabled"
        info = getattr(self.pp, "info", None)
        stages = getattr(info, "stages", None) or []
        if not stages:
            return "stages=[]"
        parts = []
        for stage in stages:
            idx = getattr(stage, "stage_index", getattr(stage, "_stage_index", "?"))
            parts.append(
                "stage="
                f"{idx} first={getattr(stage, 'is_first', '?')} last={getattr(stage, 'is_last', '?')} "
                f"in={_shape_summary(getattr(stage, 'inputs_meta', None))} "
                f"out={_shape_summary(getattr(stage, '_outputs_meta', None))}"
            )
        return "; ".join(parts)

    @staticmethod
    def _resolve_data_parallel_slice(
        distributed_cfg,
        *,
        token_size: int,
        token_rank: int,
        cp_size: int,
    ) -> tuple[int, int]:
        """Resolve logical DP ownership from the scorer's token-reduction group."""
        del distributed_cfg
        if cp_size < 1 or token_size < cp_size or token_size % cp_size:
            return 1, 0
        dp_size = token_size // cp_size
        if dp_size <= 1:
            return 1, 0
        return dp_size, (int(token_rank) // int(cp_size)) % dp_size

    def _trace_pp_eval(
        self, phase: str, step: int, input_ids=None, extra: dict | None = None
    ) -> None:
        trace_enabled = bool(
            self._pruning_cfg.get("automodel", {}).get("trace_layer_forwards", False)
            if self._pruning_cfg is not None
            else False
        )
        if not trace_enabled or self.pp is None or step > 0:
            return
        info = self.pp.info
        schedule = getattr(info, "schedule", None)
        tag = "warmup" if step < 0 else f"step={step}"
        extras = extra or {}
        _rank_trace(
            f"{phase} PP eval {tag} "
            f"first={info.has_first_stage} last={info.has_last_stage} "
            f"schedule={type(schedule).__name__ if schedule is not None else None} "
            f"input={_shape_summary(input_ids)} extra={_shape_summary(extras)} "
            f"meta={self._pp_meta_summary()}"
        )

    def _dp_slice_batch(self, batch):
        """Data-parallel: return this dp rank's contiguous slice of the batch's samples (dim 0).

        Each dp rank gets ``mbs / dp_size`` of the global micro-batch; cp/tp peers share dp_rank
        (same samples). The per-iteration / finalize token reduce over ``dp_cp`` reassembles the
        full batch, so scores match dp=1 exactly. Only splits when the batch divides evenly across
        dp (else every dp rank takes the full batch — the dp reduce then double-counts symmetrically,
        which still yields the same mean, just without the speedup).
        """
        if self._dp_size <= 1:
            return batch
        if isinstance(batch, PuzzletronBatch):
            if bool(batch.source_metadata.get("distributed_sampler_sharded", False)):
                return batch
            return batch.dp_slice(dp_rank=self._dp_rank, dp_size=self._dp_size)
        if not isinstance(batch, Mapping):
            return batch
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] % self._dp_size == 0:
                local = v.shape[0] // self._dp_size
                out[k] = v[self._dp_rank * local : (self._dp_rank + 1) * local]
            else:
                out[k] = v
        return out

    def _canonicalize_batch(self, batch, step: int) -> PuzzletronBatch | None:
        if isinstance(batch, PuzzletronBatch):
            return batch
        if self._data_spec is None:
            return None
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"canonical Puzzletron data expected a mapping batch, got {type(batch).__name__}"
            )
        input_ids = batch.get("input_ids")
        if not torch.is_tensor(input_ids):
            raise ValueError("canonical Puzzletron batch has no input_ids")
        assert isinstance(input_ids, torch.Tensor)
        batch_size = int(input_ids.shape[0]) if input_ids.ndim > 1 else 1
        source_metadata = {
            "dataset": self._data_cfg.get("path"),
            "revision": self._data_cfg.get("revision", "materialized-manifest"),
            "processor": self._data_cfg.get("processor_identity", "automodel-auto-processor"),
            "layout": self._data_spec.layout.value,
            "topology": {
                "dp_size": self._dp_size,
                "cp_size": self._cp_info()[1],
            },
            "distributed_sampler_sharded": not self._use_puzzletron_dataloader,
        }
        return batch_from_automodel(
            batch,
            sample_ids=tuple(f"batch-{step:08d}-row-{index}" for index in range(batch_size)),
            source_metadata=source_metadata,
            layout=self._data_spec.layout,
        )

    def _forward_canonical_batch(self, batch: PuzzletronBatch, step: int) -> None:
        """Use AutoModel's native VLM pre-embed, CP, and PP media path unchanged."""
        from nemo_automodel.components.datasets.vlm.pp_media import stage_vlm_media_for_pp
        from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
        from nemo_automodel.components.utils.model_utils import (
            VLM_INPUT_KEYS,
            filter_forward_kwargs,
        )

        from .batch_adapter import canonicalize_position_ids, prepare_native_cp_inputs

        model = self.model_parts[0]
        device = self.pp.device if self.pp is not None else next(model.parameters()).device
        if step >= 0:
            self._canonical_batch_fingerprints.append(batch.fingerprint)
        self._last_unpadded_batch_size = batch.batch_size
        if self.pp is not None:
            schedule = self.pp.info.schedule
            microbatches = int(getattr(schedule, "_n_microbatches", 1) or 1)
            batch = batch.pad_batch_to_multiple(microbatches)
        batch = batch.to(device)
        payload = dict(batch.model_kwargs)
        payload["labels"] = (
            batch.labels if batch.labels is not None else torch.full_like(batch.input_ids, -100)
        )
        sequence_ids = batch.sequence.seq_ids
        if sequence_ids is None:
            sequence_ids = (
                torch.arange(
                    batch.batch_size,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(1)
                .expand(-1, batch.sequence_length)
                .clone()
            )
            if batch.hidden_mask is not None:
                sequence_ids.masked_fill_(~batch.hidden_mask, -1)
        else:
            sequence_ids = sequence_ids.to(device=device, dtype=torch.long)
        valid_sequence_ids = sequence_ids[sequence_ids >= 0]
        num_original_samples = (
            int(valid_sequence_ids.max().item()) + 1 if valid_sequence_ids.numel() else 0
        )
        payload["_puzzletron_sequence_ids"] = sequence_ids

        cp_rank, cp_size = self._cp_info()
        first_stage = self.pp is None or bool(self.pp.info.has_first_stage)
        native_cp_preparer = (
            cp_size > 1
            and first_stage
            and callable(getattr(model, "prepare_model_inputs_for_cp", None))
        )
        if native_cp_preparer:
            # Let the model produce native positions/embeddings before CP.  This
            # applies equally to text-only, packed, and multimodal batches.
            payload = prepare_native_cp_inputs(model, payload)
            if "inputs_embeds" in payload:
                for key in VLM_INPUT_KEYS:
                    if key != "inputs_embeds":
                        payload.pop(key, None)
        elif cp_size > 1 and not first_stage:
            for key in VLM_INPUT_KEYS:
                if key != "input_ids":
                    payload.pop(key, None)

        # Descriptor expansion is the fallback for models that do not expose a
        # native CP-preparation hook.  Native hooks already own this contract;
        # expanding their missing IDs here would destroy model-specific mRoPE.
        if not native_cp_preparer:
            batch = canonicalize_position_ids(
                batch,
                descriptor=self._model_descriptor,
                config=getattr(model, "config", None),
            )
            payload = dict(batch.model_kwargs) | {
                key: value for key, value in payload.items() if key not in batch.model_kwargs
            }

        pre_cp_primary_key = "inputs_embeds" if "inputs_embeds" in payload else "input_ids"
        pre_cp_global_seq_len = int(payload[pre_cp_primary_key].shape[1])
        train_ctx, payload = make_cp_batch_and_ctx(
            self.device_mesh,
            {
                **payload,
                "_puzzletron_ce_mask": (
                    batch.ce_mask if batch.ce_mask is not None else payload["labels"].ne(-100)
                ),
                "_puzzletron_kd_mask": (
                    batch.kd_mask
                    if batch.kd_mask is not None
                    else (
                        batch.ce_mask if batch.ce_mask is not None else payload["labels"].ne(-100)
                    )
                ),
                "_puzzletron_hidden_mask": (
                    batch.hidden_mask
                    if batch.hidden_mask is not None
                    else torch.ones_like(payload["labels"], dtype=torch.bool)
                ),
            },
            loss_mask=batch.hidden_mask,
            auxiliary_sequence_keys={
                "_puzzletron_sequence_ids": -1,
                "_puzzletron_ce_mask": False,
                "_puzzletron_kd_mask": False,
                "_puzzletron_hidden_mask": False,
            },
        )
        _ensure_packed_qkv_format(payload)
        canonical_labels = payload.get("labels")
        local_sequence_ids = payload.pop("_puzzletron_sequence_ids")
        canonical_ce_mask = payload.pop("_puzzletron_ce_mask")
        canonical_kd_mask = payload.pop("_puzzletron_kd_mask")
        canonical_hidden_mask = payload.pop("_puzzletron_hidden_mask")
        payload.pop("labels", None)
        model_input_key = "inputs_embeds" if "inputs_embeds" in payload else "input_ids"
        # VLM pre-embedding is differentiable, so AutoModel load-balances and
        # shards ``inputs_embeds`` eagerly before the CP context.  Labels and
        # diagnostic masks remain context-managed buffers at this point.  Keep
        # exact CP-local metric copies using the same load-balancer indices;
        # the original buffers still belong to ``train_ctx`` for model loss.
        cp_rank, cp_size = self._cp_info()
        cp_divisor = 2 * cp_size
        padded_global_seq_len = (
            (pre_cp_global_seq_len + cp_divisor - 1) // cp_divisor
        ) * cp_divisor
        expected_local_seq_len = padded_global_seq_len // cp_size
        if (
            canonical_labels is not None
            and cp_size > 1
            and int(canonical_labels.shape[1]) > expected_local_seq_len
        ):
            canonical_labels = self._load_balanced_cp_sequence_shard(
                canonical_labels, cp_rank, cp_size
            )
            canonical_ce_mask = self._load_balanced_cp_sequence_shard(
                canonical_ce_mask, cp_rank, cp_size
            )
            canonical_kd_mask = self._load_balanced_cp_sequence_shard(
                canonical_kd_mask, cp_rank, cp_size
            )
            canonical_hidden_mask = self._load_balanced_cp_sequence_shard(
                canonical_hidden_mask, cp_rank, cp_size
            )
        if canonical_labels is not None and cp_size > 1:
            metric_lengths = {
                int(value.shape[1])
                for value in (
                    canonical_labels,
                    canonical_ce_mask,
                    canonical_kd_mask,
                    canonical_hidden_mask,
                )
            }
            if metric_lengths != {expected_local_seq_len}:
                raise RuntimeError(
                    "canonical CP metric tensors do not match the expected local sequence "
                    f"length {expected_local_seq_len}: observed {sorted(metric_lengths)}"
                )

        with train_ctx(), self._forward_autocast_context():
            # ``make_cp_batch_and_ctx`` load-balances and shards these buffers
            # on context entry, then may restore their full-sequence shapes on
            # exit.  Solution metrics consume them after the forward returns,
            # so preserve the exact CP-local token permutation while the
            # context is active.  A later contiguous slice is not equivalent
            # to the load-balanced CP ordering.
            self._last_canonical_labels = (
                canonical_labels.detach().clone() if canonical_labels is not None else None
            )
            self._last_canonical_ce_mask = canonical_ce_mask.detach().clone()
            self._last_canonical_kd_mask = canonical_kd_mask.detach().clone()
            self._last_canonical_hidden_mask = canonical_hidden_mask.detach().clone()
            for scorer in self._scorers or ():
                if callable(getattr(scorer, "set_batch_metadata", None)):
                    scorer.set_batch_metadata(
                        sequence_ids=local_sequence_ids,
                        num_samples=num_original_samples,
                    )
            if self.pp is not None:
                model_input = payload.pop(model_input_key)
                self._update_pp_seq_len_for_scoring(
                    model_input.shape[1],
                    envelope_batch_size=self._pp_envelope_batch_size(model_input.shape[0]),
                )
                from .pp_utils import set_pp_vlm_chunk_specs

                set_pp_vlm_chunk_specs(
                    self.pp.info.schedule,
                    payload,
                    batch_size=int(model_input.shape[0]),
                )
                with stage_vlm_media_for_pp(self.pp, self.model_parts, payload):
                    self._trace_pp_eval("before", step, input_ids=model_input, extra=payload)
                    if self.pp.info.has_first_stage:
                        self.pp.info.schedule.eval(model_input, **payload)
                    else:
                        self.pp.info.schedule.eval(**payload)
                    self._trace_pp_eval("after", step, input_ids=model_input, extra=payload)
            else:
                model(**filter_forward_kwargs(model, payload))

    def _local_observability_metadata(self) -> dict[str, Any]:
        resumed = self._resumed_observability_local
        return {
            "vision_forward_count": int(resumed.get("vision_forward_count", 0))
            + sum(monitor.forward_count for monitor in self._vision_monitors),
            "vision_output_checksums": list(
                dict.fromkeys(
                    [*resumed.get("vision_output_checksums", [])]
                    + [
                        checksum
                        for monitor in self._vision_monitors
                        for checksum in monitor.output_checksums
                    ]
                )
            ),
            "batch_fingerprints": list(
                dict.fromkeys(
                    [*resumed.get("batch_fingerprints", [])] + self._canonical_batch_fingerprints
                )
            ),
        }

    def observability_metadata(self) -> dict[str, Any]:
        local = self._local_observability_metadata()
        if not torch.distributed.is_initialized():
            return local
        gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
        gathered_items = [item for item in gathered if item is not None]
        if len(gathered_items) != len(gathered):
            raise RuntimeError("observability gather returned an empty rank payload")
        return {
            "vision_forward_count": sum(item["vision_forward_count"] for item in gathered_items),
            "vision_output_checksums": sorted(
                checksum for item in gathered_items for checksum in item["vision_output_checksums"]
            ),
            "batch_fingerprints": sorted(
                set(
                    fingerprint
                    for item in gathered_items
                    for fingerprint in item["batch_fingerprints"]
                )
            ),
        }

    def close_observability(self) -> None:
        for monitor in reversed(self._vision_monitors):
            monitor.__exit__(None, None, None)
        self._vision_monitors.clear()

    def _pp_metadata_ready(self) -> bool:
        """Return True when local PP stages already have static input/output metadata.

        NeMo AutoModel can precompute ``PipelineStage`` shape metadata from
        ``distributed.pipeline.pp_seq_len``.  In that case there is no shape-inference
        forward to absorb, so a hook-disabled warmup is pure extra work.  This matters for
        large HF MoE stages: the warmup would still execute every routed expert path even
        though no scorer can use the result.
        """
        if self.pp is None:
            return False
        stages = getattr(getattr(self.pp, "info", None), "stages", None) or []
        if not stages:
            schedule = getattr(getattr(self.pp, "info", None), "schedule", None)
            stage = getattr(schedule, "_stage", None)
            stages = [stage] if stage is not None else []
        if not stages:
            return False
        for stage in stages:
            if getattr(stage, "inputs_meta", None) is None:
                return False
            try:
                outputs_meta = stage.get_outputs_meta()
            except Exception:  # noqa: BLE001
                return False
            if outputs_meta is None:
                return False
        return True

    def _forward_batch(self, batch, step: int = -1) -> None:
        """Run one forward-only pass; hooks accumulate this iteration's contribution."""
        canonical = self._canonicalize_batch(batch, step)
        if canonical is not None:
            canonical = self._dp_slice_batch(canonical)
            assert isinstance(canonical, PuzzletronBatch)
            if step >= 0 and samples_hashing_enabled():
                log_batch_hashes(
                    canonical.input_ids,
                    "automodel",
                    step,
                    extra=f"canonical dp={self._dp_rank} cp={self._cp_info()[0]}",
                )
            self._forward_canonical_batch(canonical, step)
            return
        batch = self._dp_slice_batch(batch)  # data-parallel: this rank's sub-batch
        cp_rank, cp_size = self._cp_info()
        # Debug: record which samples this rank actually feeds (gated by PUZZLE_HASH_SAMPLES).
        if (
            step >= 0
            and samples_hashing_enabled()
            and isinstance(batch, dict)
            and "input_ids" in batch
        ):
            log_batch_hashes(
                batch["input_ids"], "automodel", step, extra=f"dp={self._dp_rank} cp={cp_rank}"
            )

        if self.pp is not None:
            device = self.pp.device
            input_ids = batch["input_ids"].to(device)
            # The puzzletron dataset keys the next-token labels as "targets"; NeMo's own
            # datasets use "labels". The last pipeline stage computes the (discarded) loss
            # whenever loss_fn is set, so a valid target tensor must be provided.
            targets = batch.get("targets", batch.get("labels"))
            if targets is not None:
                targets = targets.to(device)
            extra = {
                key: batch[key].to(device)
                for key in ("attention_mask", "position_ids")
                if key in batch
            }
            # Each CP rank must receive only its sequence shard; our puzzletron dataloader
            # is not CP-aware and returns the full sequence.  Shard here before handing
            # the inputs to the PP schedule (which passes them into the model).
            if cp_size > 1:
                # The puzzletron dataset does not include position_ids.  pipeline_forward
                # would auto-generate them as arange(0, S_local) independently on every CP
                # rank — both would produce [0..S/2-1], and the CP all-gather would see
                # duplicates instead of the required unique [0..S-1].  Synthesize full-range
                # position_ids from the un-sharded sequence length and shard them together
                # with input_ids so each rank gets its correct slice.
                if "position_ids" not in extra:
                    B, S = input_ids.shape
                    extra["position_ids"] = (
                        torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
                    )
                input_ids = self._shard_seq_for_cp(input_ids, cp_rank, cp_size)
                targets = self._shard_seq_for_cp(targets, cp_rank, cp_size)
                extra = {k: self._shard_seq_for_cp(v, cp_rank, cp_size) for k, v in extra.items()}
            tp_group = self.tensor_parallel_group()
            if bool(
                self._pruning_cfg.get("automodel", {}).get("trace_layer_forwards", False)
                if self._pruning_cfg is not None
                else False
            ):
                global_rank = torch.distributed.get_rank()
                position_values = extra.get("position_ids")
                print(
                    "[activation/automodel] pre-PP shard "
                    f"rank={global_rank} cp_rank={cp_rank}/{cp_size} "
                    f"tp_group={None if tp_group is None else torch.distributed.get_process_group_ranks(tp_group)} "
                    f"tp_rank={None if tp_group is None else torch.distributed.get_rank(tp_group)}/"
                    f"{None if tp_group is None else torch.distributed.get_world_size(tp_group)} "
                    f"input_shape={tuple(input_ids.shape)} "
                    f"positions={None if position_values is None else position_values.reshape(-1).tolist()}",
                    flush=True,
                )
            # lm_head is removed in setup() so the last stage returns hidden_states, not logits.
            # target/losses are not passed: no loss computation is needed for scoring.
            with self._forward_autocast_context():
                self._trace_pp_eval("before", step, input_ids=input_ids, extra=extra)
                if self.pp.info.has_first_stage:
                    self.pp.info.schedule.eval(input_ids, **extra)
                else:
                    self.pp.info.schedule.eval()
                self._trace_pp_eval("after", step, input_ids=input_ids, extra=extra)
        else:
            model = self.model_parts[0]
            device = next(model.parameters()).device
            inputs = {key: batch[key].to(device) for key in _FORWARD_KEYS if key in batch}
            if cp_size > 1:
                if "position_ids" not in inputs:
                    B, S = inputs["input_ids"].shape
                    inputs["position_ids"] = (
                        torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
                    )
                inputs = {k: self._shard_seq_for_cp(v, cp_rank, cp_size) for k, v in inputs.items()}
            with self._forward_autocast_context():
                model(**inputs)

    def run_scoring(self) -> dict:
        """Run the no-grad calibration pass and write the consolidated scores."""
        import time

        from ...tools.logger import mprint

        assert self._scorers is not None, "call setup() before run_scoring()"
        assert self._groups is not None, "call setup() before run_scoring()"
        assert self._eval_iters is not None, "setup must resolve evaluation iterations"

        total = self._eval_iters

        # Under pipeline parallel, the first schedule.eval may trigger a one-time shape-inference
        # forward on all-zero tensors (torch.distributed.pipelining stage._shape_inference). That
        # fires our forward hooks once with bogus zero activations, which would corrupt both the
        # additive accumulation (inflated token count) and the iterative greedy state. Run one
        # warmup forward with the hooks disabled only when shape metadata is absent. When NeMo
        # precomputes PP metadata, the warmup is unnecessary and can be prohibitively expensive
        # for HF MoE stages.
        if self.pp is not None:
            if self._pp_metadata_ready():
                mprint(
                    "[activation/automodel] skipping PP warmup: stage shape metadata is precomputed"
                )
            else:
                mprint("[activation/automodel] warmup forward (hooks disabled) to prime PP shapes")
                for scorer in self._scorers:
                    scorer.enabled = False
                try:
                    with torch.no_grad():
                        self._forward_batch(next(iter(self.dataloader)))
                finally:
                    for scorer in self._scorers:
                        scorer.enabled = True

        mprint(f"[activation/automodel] entering calibration loop: target {total} iteration(s)")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loop_start = time.monotonic()
        start_step = self._load_resume_checkpoint(total)
        self._run_partial_resume_recovery()
        checkpoint_interval = max(
            1, int(os.environ.get("ACTIVATION_SCORING_CHECKPOINT_INTERVAL", "50"))
        )
        max_run_minutes = float(os.environ.get("ACTIVATION_SCORING_MAX_MINUTES", "0"))
        # Cycle the dataloader when pruning_iters > len(dataloader) so that the number of
        # greedy scoring steps is determined by pruning_iters, not by dataset size.
        import itertools

        dl_len = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        if total is not None and dl_len is not None and total > dl_len:
            data_iter = itertools.islice(itertools.cycle(self.dataloader), start_step, total)
        else:
            data_iter = itertools.islice(self.dataloader, start_step, total)
        with torch.no_grad():
            for step, batch in enumerate(data_iter, start=start_step):
                if total is not None and step >= total:
                    break
                t0 = time.monotonic()
                self._forward_batch(batch, step=step)
                # One iteration per batch: reduce + advance any stateful (iterative) scorers
                # in lock-step across ranks; a no-op for additive scorers.
                for scorer in self._scorers:
                    scorer.step_iteration()
                if self.dist_env.is_main:
                    fwd_s = time.monotonic() - t0
                    elapsed_min = (time.monotonic() - loop_start) / 60.0
                    peak = (
                        torch.cuda.max_memory_allocated() / (1 << 30)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    mprint(
                        f"[activation/automodel] iter {step + 1}/{total} "
                        f"({fwd_s:.1f}s/iter, peak {peak:.1f} GiB, elapsed {elapsed_min:.1f} min)"
                    )
                next_step = step + 1
                checkpoint_due = next_step % checkpoint_interval == 0
                if checkpoint_due:
                    self._save_resume_checkpoint(next_step, total)
                    if self.dist_env.is_main:
                        mprint(
                            f"[activation/automodel] checkpointed exact scorer state at "
                            f"{next_step}/{total}"
                        )
                # Decide the cap only at a checkpoint boundary.  Every rank is already
                # synchronized by _save_resume_checkpoint, so one broadcast per 50 steps
                # avoids both rank-local deadline races and a costly per-step collective.
                should_stop = checkpoint_due and (
                    max_run_minutes > 0
                    and (time.monotonic() - loop_start) / 60.0 >= max_run_minutes
                    and next_step < total
                )
                if checkpoint_due and max_run_minutes > 0 and torch.distributed.is_initialized():
                    stop_flag = torch.tensor(
                        int(should_stop if self.dist_env.is_main else False),
                        device=torch.device("cuda", torch.cuda.current_device()),
                        dtype=torch.int32,
                    )
                    torch.distributed.broadcast(stop_flag, src=0)
                    should_stop = bool(stop_flag.item())
                if should_stop:
                    raise RuntimeError(
                        f"Activation scoring reached the per-allocation runtime cap at "
                        f"{next_step}/{total}; rerun the identical command to resume exactly"
                    )

        if self._scorer_groups is None:
            # Backward-compat: old-style single-group (no scorer_groups set).
            assert self._activations_log_dir is not None
            results = write_scores(self._scorers, self._activations_log_dir, self._groups)
            if self.dist_env.is_main:
                mprint(f"[activation/automodel] wrote {len(results)} module scores")
            self._clear_resume_checkpoint()
            return results

        all_group_results = []
        for group_scorers, group_dir in self._scorer_groups:
            if group_dir is not None and group_scorers:
                group_result = write_scores(group_scorers, group_dir, self._groups)
                all_group_results.append((group_dir, group_result))
        results = all_group_results
        if self.dist_env.is_main:
            mprint(
                f"[activation/automodel] wrote {sum(len(r) for _, r in all_group_results)} module scores"
            )
        self._clear_resume_checkpoint()
        return results

    def run_train_validation_loop(self):
        """Alias so the reference ``run.py`` entry point drives scoring."""
        return self.run_scoring()
