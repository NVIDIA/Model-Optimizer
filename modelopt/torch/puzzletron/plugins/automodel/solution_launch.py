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

"""Entry point for AutoModel-backed replace-1-block (solution) scoring.

Sorted teacher + dynamic prune (no realized intermediate checkpoints):

* Build the sorted teacher once (D2) if absent, then load it ONCE (sharded).
* Phase 1: forward the calibration set unpruned -> cache per-batch final hidden states + the
  LM-head weight (``TeacherTargetCache``). The sorted teacher is functionally identical to the
  teacher, so these are the true teacher targets.
* Phase 2 (per solution): dynamically prune the candidate's ONE block on the already-loaded model
  via ``recipe.prune_block_context`` (D3/D4 — mask down_proj/o_proj input for removal, swap merged
  K/V for merge), score against the cached targets, write ``solution_<i>.json``. No reload, no temp
  checkpoint, one model resident.

Solutions are already partitioned across *nodes* by the caller; within a node every rank
participates in each solution's sharded forward.
"""

import json
import logging
import math
import os
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import torch
import torch.distributed as torch_dist

import modelopt.torch.utils.distributed as dist

from ...replacement_library.replacement_utils import parse_layer_replacement
from ...tools.logger import mprint
from ...tools.validate_puzzle_with_multi_replacements import load_puzzle_solutions
from ...tools.validation_utils import write_results
from .config import build_solution_recipe_config, solution_scoring_params
from .launch import _free_scoring_memory
from .module_trace import synchronized_module_trace
from .patch import _precache_trust_remote_code_distributed, apply_patch
from .solution_metrics import aggregate_solution_scores, retain_teacher_channels, score_batch
from .teacher_cache import TeacherTargetCache

logger = logging.getLogger(__name__)

__all__ = [
    "launch_score_solution_parents_automodel",
    "launch_score_solutions_automodel",
]


def _resolve_output_dir(scoring):
    if scoring.get("output_dir", None) is not None:
        return Path(scoring.output_dir)
    solutions_path = Path(scoring.solutions_path)
    return solutions_path.with_name(f"{solutions_path.stem}--validation")


def _solution_output_location(scoring, output_dir: Path, solution_id: int) -> tuple[Path, str]:
    """Resolve an optional per-solution output while preserving the default layout."""

    routed_outputs = scoring.get("solution_output_dirs", None)
    if routed_outputs is not None:
        routed_output = routed_outputs.get(str(solution_id), None)
        if routed_output is not None:
            return Path(str(routed_output)), "solution_0"
    return output_dir, f"solution_{solution_id}"


def _solution_result_path(scoring, output_dir: Path, solution_id: int) -> Path:
    solution_output, solution_name = _solution_output_location(
        scoring, output_dir, solution_id
    )
    return solution_output / f"{solution_name}.json"


def _load_model_config_distributed(checkpoint_dir, descriptor, *, loader):
    """Pre-cache dynamic modules once before every rank imports a config."""

    trust_remote_code = descriptor.requires_trust_remote_code()
    _precache_trust_remote_code_distributed(
        checkpoint_dir,
        trust_remote_code=trust_remote_code,
    )
    return loader(checkpoint_dir, trust_remote_code=trust_remote_code)


def _load_solution_work(scoring, output_dir: Path) -> tuple[list[dict], list[int]]:
    """Load pending candidates, or no candidates for a source-baseline-only evaluation."""

    if bool(scoring.get("baseline_only", False)):
        return [], []
    solutions = load_puzzle_solutions(
        Path(scoring.solutions_path),
        scoring.get("sort_solutions_by", None),
        scoring.get("bigger_is_better", False),
    )
    ids = scoring.get("solutions_to_validate", None) or list(range(len(solutions)))
    if bool(scoring.get("skip_existing_solutions", True)):
        ids = [
            i
            for i in ids
            if not _solution_result_path(scoring, output_dir, i).exists()
        ]
    return solutions, ids


def _trace_batch(phase: str, batch_idx: int, **fields) -> None:
    """Emit opt-in per-rank batch boundaries for distributed hang diagnosis."""
    if os.environ.get("PUZZLETRON_TRACE_BATCHES") != "1":
        return
    rank = torch_dist.get_rank() if torch_dist.is_available() and torch_dist.is_initialized() else 0
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[solution/automodel/trace] rank={rank} phase={phase} batch={batch_idx} {details}",
        flush=True,
    )


def _is_output_writer(recipe) -> bool:
    """Return True on one rank that actually owns final hidden states/LM head.

    Pipeline layouts do not guarantee that the global last rank is the rank with
    Puzzletron's captured outputs.  All ranks participate in this discovery so PP/TP/CP
    variants can choose a deterministic writer from the output-owning ranks.
    """
    import torch.distributed as torch_dist

    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return bool(recipe.has_outputs)
    rank = torch_dist.get_rank()
    gathered: list[tuple[int, bool]] = [None] * torch_dist.get_world_size()  # type: ignore[list-item]
    torch_dist.all_gather_object(gathered, (rank, bool(recipe.has_outputs)))
    output_ranks = [rank for rank, has_outputs in gathered if has_outputs]
    return bool(recipe.has_outputs) and bool(output_ranks) and rank == min(output_ranks)


def _quarantine_failed_realization(checkpoint_dir: Path) -> Path | None:
    """Remove a realized parent checkpoint that failed during first load."""

    from ...checkpoint_transactions import invalidate_realization, remove_realization_temp_dir

    if not dist.is_master():
        dist.barrier()
        return None
    remove_realization_temp_dir(checkpoint_dir)
    quarantined = invalidate_realization(checkpoint_dir)
    dist.barrier()
    return quarantined


def _run_recipe(
    recipe_dict,
    pruning_cfg,
    eval_iters,
    use_puzzletron_dataloader,
    data_cfg=None,
):
    """Wrap a recipe dict in a ConfigNode and build the ReplaceBlockScoringRecipe."""
    from nemo_automodel.components.config.loader import ConfigNode

    from .solution_recipe import ReplaceBlockScoringRecipe

    recipe = ReplaceBlockScoringRecipe(
        ConfigNode(recipe_dict),
        pruning_cfg=pruning_cfg,
        eval_iters=eval_iters,
        use_puzzletron_dataloader=use_puzzletron_dataloader,
        data_cfg=data_cfg,
    )
    recipe.setup()
    return recipe


def _extract_teacher_targets(recipe, cache: TeacherTargetCache, params: dict | None = None) -> dict | None:
    """Phase 1: fill the cache and optionally compute the teacher baseline metrics."""
    if recipe.has_outputs:
        cache.set_lm_head_weight(recipe.lm_head_weight())
    per_batch = []
    tp_group = recipe.tensor_parallel_group()
    trace_writer = _is_output_writer(recipe)
    # Writer discovery is a world-wide collective.  Cache its result while all
    # ranks are in the same control-flow position; calling it again after the
    # output PP stage performs token-group reductions would mismatch the global
    # barrier reached by non-output PP stages.
    recipe._puzzletron_output_writer = trace_writer
    batch_started = time.perf_counter()
    for batch_idx, (hidden, targets) in enumerate(recipe.iterate_captures()):
        _trace_batch("teacher_capture_yield", batch_idx, has_hidden=hidden is not None)
        if hidden is not None:
            torch.cuda.synchronize(hidden.device)
        forward_seconds = time.perf_counter() - batch_started
        if hidden is not None:
            cache.append_hidden(hidden)
            if params is not None and targets is not None and recipe.has_outputs:
                lm_head_w = recipe.lm_head_weight()
                metric_started = time.perf_counter()
                per_batch.append(
                    score_batch(
                        hidden,
                        lm_head_w,
                        hidden,
                        lm_head_w,
                        targets,
                        temperature=params["temperature"],
                        chunk_size=params["chunk_size"],
                        lm_head_backend=params["lm_head_backend"],
                        tp_group=tp_group,
                        flash_kld_token_chunk_size=params["flash_kld_token_chunk_size"],
                        flash_kld_reduction_backend=params["flash_kld_reduction_backend"],
                        **recipe.current_metric_masks(),
                    )
                )
                _trace_batch("teacher_score_complete", batch_idx)
                torch.cuda.synchronize(hidden.device)
                if trace_writer:
                    print(
                        "[solution/automodel] teacher batch timing "
                        f"batch={batch_idx} forward_s={forward_seconds:.3f} "
                        f"lm_head_s={time.perf_counter() - metric_started:.3f} "
                        f"backend={params['lm_head_backend']} "
                        f"reduction={params['flash_kld_reduction_backend']}",
                        flush=True,
                    )
        batch_started = time.perf_counter()
    # Only the output stage(s) populate (and later read) the cache; non-output ranks never call
    # cache.hidden(), so sealing them — which would assert on the missing LM-head weight — is wrong.
    if recipe.has_outputs:
        cache.seal()
    if recipe.has_outputs and per_batch:
        token_group = recipe._groups.token_group if recipe._groups is not None else None
        return aggregate_solution_scores(
            per_batch,
            token_group=token_group,
            cp_group=recipe.context_parallel_group(),
        )
    return None


def _source_hidden_channel_indices(
    source_dir: Path,
    source_width: int,
    teacher_width: int,
    *,
    retained_width: int | None = None,
    source_and_teacher_share_basis: bool = False,
) -> tuple[int, ...]:
    """Map a source checkpoint's local hidden basis to the original teacher basis."""
    source_width = int(source_width)
    teacher_width = int(teacher_width)
    retained_width = source_width if retained_width is None else int(retained_width)
    if retained_width > teacher_width:
        raise ValueError(
            f"retained hidden width {retained_width} exceeds teacher width {teacher_width}"
        )
    if source_and_teacher_share_basis:
        if retained_width > source_width:
            raise ValueError(
                f"retained hidden width {retained_width} exceeds source width {source_width}"
            )
        return tuple(range(retained_width))
    permutation_path = Path(source_dir) / "sorted_permutations.json"
    if not permutation_path.is_file():
        return tuple(range(retained_width))
    payload = json.loads(permutation_path.read_text())
    order = payload.get("embedding.hidden_order")
    if order is None:
        return tuple(range(retained_width))
    order = tuple(int(index) for index in order)
    if len(order) != teacher_width or sorted(order) != list(range(teacher_width)):
        raise ValueError(
            f"invalid embedding.hidden_order in {permutation_path}: "
            f"length={len(order)} teacher_width={teacher_width}"
        )
    if source_width > teacher_width:
        raise ValueError(
            f"source hidden width {source_width} exceeds teacher width {teacher_width}"
        )
    return order[:retained_width]


def _score_candidate(
    recipe,
    cache,
    params,
    output_dir,
    scoring,
    name,
    payload,
    prune_target,
    *,
    bypass_checkpoint_dir=None,
    retained_hidden_indices: tuple[int, ...] | None = None,
    hidden_width: int | None = None,
) -> dict | None:
    """Phase 2: dynamically prune one block on the loaded sorted teacher, score, write its json.

    ``prune_target`` (layer_idx + orig/target dims) is applied for the whole calibration sweep via
    ``recipe.prune_block_context`` — no model reload, no temp checkpoint.
    """
    per_batch = []
    tp_group = recipe.tensor_parallel_group()
    is_writer = _is_output_writer(recipe)
    lm_head_w = recipe.lm_head_weight() if recipe.has_outputs else None
    batch_started = time.perf_counter()
    with (
        _candidate_execution_context(
            recipe,
            prune_target,
            bypass_checkpoint_dir=bypass_checkpoint_dir,
            hidden_width=hidden_width,
        ),
        synchronized_module_trace(recipe),
    ):
        for batch_idx, (hidden, targets) in enumerate(recipe.iterate_captures()):
            _trace_batch("candidate_capture_yield", batch_idx, has_hidden=hidden is not None, name=name)
            if hidden is not None:
                torch.cuda.synchronize(hidden.device)
            forward_seconds = time.perf_counter() - batch_started
            if hidden is None:
                batch_started = time.perf_counter()
                continue
            teacher_hidden = cache.hidden(batch_idx, device=hidden.device, dtype=hidden.dtype)
            teacher_w = cache.lm_head(device=hidden.device, dtype=hidden.dtype)
            candidate_hidden = hidden
            candidate_w = lm_head_w
            if bool(scoring.get("zero_pad_hidden_to_teacher_width", False)):
                envelope_width = int(teacher_hidden.shape[-1])
                candidate_width = int(candidate_hidden.shape[-1])
                if candidate_width > envelope_width:
                    raise ValueError(
                        "candidate hidden width exceeds its teacher envelope: "
                        f"candidate={candidate_width} teacher={envelope_width}"
                    )
                padding = envelope_width - candidate_width
                if padding:
                    candidate_hidden = torch.nn.functional.pad(candidate_hidden, (0, padding))
                    candidate_w = torch.nn.functional.pad(candidate_w, (0, padding))
            hidden_metric_teacher, _ = retain_teacher_channels(
                candidate_hidden,
                candidate_w,
                teacher_hidden,
                teacher_w,
                channel_indices=retained_hidden_indices,
            )
            _trace_batch("candidate_retained", batch_idx, name=name)
            metric_started = time.perf_counter()
            per_batch.append(
                score_batch(
                    candidate_hidden, candidate_w, teacher_hidden, teacher_w, targets,
                    temperature=params["temperature"], chunk_size=params["chunk_size"],
                    lm_head_backend=params["lm_head_backend"],
                    tp_group=tp_group,
                    flash_kld_token_chunk_size=params["flash_kld_token_chunk_size"],
                    flash_kld_reduction_backend=params["flash_kld_reduction_backend"],
                    hidden_metric_teacher=hidden_metric_teacher,
                    **recipe.current_metric_masks(),
                )
            )
            _trace_batch("candidate_score_complete", batch_idx, name=name)
            torch.cuda.synchronize(hidden.device)
            if is_writer:
                print(
                    "[solution/automodel] candidate batch timing "
                    f"name={name} batch={batch_idx} forward_s={forward_seconds:.3f} "
                    f"lm_head_s={time.perf_counter() - metric_started:.3f} "
                    f"backend={params['lm_head_backend']} "
                    f"reduction={params['flash_kld_reduction_backend']}",
                    flush=True,
                )
            batch_started = time.perf_counter()

    # Reduce per-sample lists across the data (dp_cp) group, then the single writer emits json.
    token_group = recipe._groups.token_group if recipe._groups is not None else None
    observability = recipe.observability_metadata()
    if recipe.has_outputs and per_batch:
        losses = aggregate_solution_scores(
            per_batch,
            token_group=token_group,
            cp_group=recipe.context_parallel_group(),
        )
        if is_writer:
            write_results(
                output_dir,
                name,
                scoring,
                {**losses, **payload, "observability": observability},
            )
            return losses
    return None


def _metric_average(raw: dict, metric: str) -> float:
    value = raw.get(metric)
    if isinstance(value, dict):
        value = value.get("avg")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite parent equivalence metric {metric}={value}")
    return value


def _validate_parent_equivalence(
    *,
    teacher_result_path: Path,
    parent_result_path: Path,
    tolerances: dict,
    hidden_basis_permuted: bool = False,
) -> dict:
    teacher = json.loads(teacher_result_path.read_text())
    parent = json.loads(parent_result_path.read_text())
    checks: dict[str, dict[str, float | bool]] = {}

    teacher_lm = _metric_average(teacher, "lm_loss")
    parent_lm = _metric_average(parent, "lm_loss")
    lm_delta = abs(parent_lm - teacher_lm)
    lm_limit = float(tolerances.get("max_abs_lm_loss_delta", 1.0e-3))
    checks["abs_lm_loss_delta"] = {
        "value": lm_delta,
        "limit": lm_limit,
        "passed": lm_delta <= lm_limit,
    }

    metric_limits = {
        "kl_div": float(tolerances.get("max_kl_div", 1.0e-2)),
        "cosine_embedding_loss_hidden_states": float(
            tolerances.get("max_cosine_embedding_loss_hidden_states", 1.0e-2)
        ),
        "normalized_mse_loss_hidden_states": float(
            tolerances.get("max_normalized_mse_loss_hidden_states", 2.0e-2)
        ),
        "mse_loss_hidden_states": float(
            tolerances.get("max_mse_loss_hidden_states", 1.0e-1)
        ),
        "mae_loss_hidden_states": float(
            tolerances.get("max_mae_loss_hidden_states", 5.0e-1)
        ),
    }
    for metric, limit in metric_limits.items():
        value = _metric_average(parent, metric)
        if hidden_basis_permuted:
            checks[metric] = {
                "value": value,
                "limit": limit,
                "passed": True,
                "gated": False,
                "reason": "basis_permuted",
            }
        else:
            checks[metric] = {
                "value": value,
                "limit": limit,
                "passed": value <= limit,
                "gated": True,
            }

    top_1 = _metric_average(parent, "top_1_logit_agreement")
    top_1_limit = float(tolerances.get("min_top_1_logit_agreement", 0.9))
    checks["top_1_logit_agreement"] = {
        "value": top_1,
        "limit": top_1_limit,
        "passed": top_1 >= top_1_limit,
    }

    failed = [name for name, check in checks.items() if not check["passed"]]
    findings = [
        {
            "stage": "width_sanity",
            "message": (
                f"parent equivalence check {name} failed: "
                f"value={check['value']:.6g} limit={check['limit']:.6g}"
            ),
            "evidence": {
                "kind": "parent_equivalence",
                "check": name,
                "value": check["value"],
                "limit": check["limit"],
            },
            "severity": "warning",
        }
        for name, check in checks.items()
        if not check["passed"]
    ]
    summary = {
        "teacher_result": str(teacher_result_path),
        "parent_result": str(parent_result_path),
        "hidden_basis_permuted": bool(hidden_basis_permuted),
        "checks": checks,
        "passed": not failed,
        "findings": findings,
    }
    return summary


def _extract_single_sequence_replacement(solution: dict) -> dict:
    """Return the one replacement that this replace-one-block score is validating.

    ``chosen_replacements`` contains the candidate plus teacher-fill replacements for
    every other layer, sorted by layer index.  Using its first element often scores
    the unchanged layer-0 teacher entry instead of the candidate.  The library writes
    the intended one-block candidate explicitly as ``single_sequence_replacement``.
    """
    solution = solution.get("puzzle_solution", solution)
    if "single_sequence_replacement" not in solution:
        raise KeyError("replace-one-block scoring requires single_sequence_replacement")
    return parse_layer_replacement(solution["single_sequence_replacement"])


def _solution_hidden_width(solution: dict) -> int | None:
    solution = solution.get("puzzle_solution", solution)
    value = solution.get("hidden_width")
    return None if value is None else int(value)


def _solution_prune_target(layer_replacements, teacher_block_configs, num_q_heads, head_dim) -> dict | None:
    """Resolve a single-block replacement into prune_block_context kwargs (orig + target dims).

    Attention targets are interpreted as sorted-prefix removal: reducing KV groups also removes
    their corresponding query heads unless ``num_query_heads`` is explicitly set.
    """
    rep = layer_replacements[0] if isinstance(layer_replacements, list) else layer_replacements
    diagnostic = rep.get("diagnostic") or {}
    if int(diagnostic.get("num_changed_layers", 1)) == 0:
        # Parent-equivalence scoring is intentionally unsliced.  Entering an identity
        # dynamic-block context is both unnecessary and unsafe for modules that lazily
        # materialize runtime-only parameters (for example Mamba ``_fp32_params``).
        return None
    layer_idx = int(rep["parent_layer_indices"][0])
    child = rep["child_block_configs"][0]
    teacher = teacher_block_configs[layer_idx]
    if child == teacher:
        return None

    child_ffn = child.get_subblock("ffn")
    child_attn = child.get_subblock("attention")
    teacher_ffn = teacher.get_subblock("ffn")
    teacher_attn = teacher.get_subblock("attention")

    t_ffn = (
        child_ffn.intermediate_size
        if child_ffn is not None and not child_ffn.no_op
        else None
    )
    t_kv = (
        child_attn.num_kv_heads
        if child_attn is not None and not child_attn.no_op
        else None
    )
    t_q = (
        child_attn.num_query_heads
        if child_attn is not None and not child_attn.no_op
        else None
    )
    orig_kv = (
        teacher_attn.num_kv_heads
        if teacher_attn is not None and not teacher_attn.no_op
        else None
    )
    target_num_q = None
    if t_kv is not None:
        if t_q is not None:
            target_num_q = t_q
        elif orig_kv is not None and t_kv < orig_kv:
            target_num_q = t_kv * (num_q_heads // orig_kv)
        else:
            target_num_q = num_q_heads
    return {
        "layer_idx": layer_idx,
        "teacher_block_config": teacher,
        "child_block_config": child,
        "orig_intermediate": (
            teacher_ffn.intermediate_size
            if teacher_ffn is not None and not teacher_ffn.no_op
            else None
        ),
        "target_intermediate": t_ffn,
        "orig_num_q": num_q_heads,
        "orig_num_kv": orig_kv,
        "target_num_q": target_num_q,
        "target_num_kv": t_kv,
        "head_dim": head_dim,
        # Diagnostic expert ids refer to the original checkpoint.  Every
        # scoring parent is already permuted into that method's ranked order,
        # so execution must retain the compact prefix rather than applying the
        # original ids a second time.  Keep the ids in diagnostic metadata for
        # reporting only.
        "expert_keep_ids": None,
    }


def _solution_prune_targets(
    solution: dict,
    teacher_block_configs,
    num_q_heads: int,
    head_dim: int,
) -> dict | list[dict] | None:
    """Resolve a replace-one or complete-architecture solution into runtime targets."""

    payload = solution.get("puzzle_solution", solution)
    if "single_sequence_replacement" in payload:
        replacement = parse_layer_replacement(payload["single_sequence_replacement"])
        return _solution_prune_target(
            replacement,
            teacher_block_configs,
            num_q_heads,
            head_dim,
        )
    replacements = [
        parse_layer_replacement(replacement)
        for replacement in payload.get("chosen_replacements", ())
    ]
    if not replacements:
        raise KeyError(
            "solution scoring requires single_sequence_replacement or chosen_replacements"
        )
    targets = [
        target
        for replacement in replacements
        if (
            target := _solution_prune_target(
                replacement,
                teacher_block_configs,
                num_q_heads,
                head_dim,
            )
        )
        is not None
    ]
    return targets


@contextmanager
def _candidate_execution_context(
    recipe,
    prune_target: dict | list[dict] | tuple[dict, ...] | None,
    *,
    bypass_checkpoint_dir=None,
    hidden_width: int | None = None,
):
    """Overlay every changed layer, apply its architecture, and restore deterministically."""

    is_architecture = isinstance(prune_target, (list, tuple))
    targets = list(prune_target) if is_architecture else [prune_target]
    targets = [target for target in targets if target is not None]
    with ExitStack() as stack:
        if bypass_checkpoint_dir is not None:
            for target in targets:
                stack.enter_context(
                    recipe.block_checkpoint_overlay_context(
                        bypass_checkpoint_dir,
                        int(target["layer_idx"]),
                        offload_restore=is_architecture and len(targets) > 1,
                    )
                )
        if is_architecture:
            stack.enter_context(recipe.architecture_context(targets))
        elif targets:
            stack.enter_context(recipe.prune_block_context(**targets[0]))
        stack.enter_context(recipe.hidden_width_context(hidden_width))
        yield


def launch_score_solutions_automodel(hydra_cfg, num_nodes: int = 1, node_index: int = 0) -> None:
    """Run replace-1-block scoring on a single, once-loaded **sorted teacher** (no temp checkpoints).

    Builds the sorted teacher (D2) if absent, loads it once, caches teacher targets, then scores
    each candidate by dynamically pruning its one block (D3/D4) — no per-candidate reload.
    """
    from ...anymodel.model_descriptor import ModelDescriptorFactory
    from ...block_config import maybe_cast_block_configs
    from ...pruning.sorted_teacher import build_sorted_teacher
    from ...tools.checkpoint_utils import load_model_config

    scoring = hydra_cfg.scoring
    params = solution_scoring_params(hydra_cfg)
    mprint(f"[solution/automodel] scoring params: {params}")
    output_dir = _resolve_output_dir(scoring)
    apply_patch()

    descriptor = ModelDescriptorFactory.get(hydra_cfg.get("descriptor", None))
    teacher_dir = scoring.get("teacher_dir", None) or f"{hydra_cfg.puzzle_dir}/ckpts/teacher"
    activations_log_dir = scoring.get("activations_log_dir", None) or hydra_cfg.pruning.get(
        "activations_log_dir", None
    )

    # ---- Build/resolve source and target checkpoints ----
    # Normal replace-1-block scoring loads the sorted teacher, caches its full-model
    # targets, then dynamically slices that same model.  Bypass diagnostics need a
    # slightly different shape: cache targets from the original sorted teacher, but
    # slice a bypass-trained checkpoint.  ``source_checkpoint_dir`` and
    # ``target_teacher_dir`` keep that explicit while preserving the default path.
    default_sorted_dir = Path(hydra_cfg.puzzle_dir) / "ckpts" / "sorted_teacher"
    source_dir = Path(scoring.get("source_checkpoint_dir", None) or default_sorted_dir)
    target_dir = Path(scoring.get("target_teacher_dir", None) or source_dir)
    if (
        scoring.get("source_checkpoint_dir", None) is None
        and not (default_sorted_dir / "config.json").exists()
    ):
        if dist.is_master():
            mprint(f"[solution/automodel] building sorted teacher -> {default_sorted_dir}")
            sort_cfg = hydra_cfg.get("sort", {})
            embedding_widths = tuple(
                hydra_cfg.get("embedding_pruning", {}).get("widths", ()) or ()
            )
            build_sorted_teacher(
                teacher_dir,
                activations_log_dir,
                default_sorted_dir,
                descriptor,
                deferred_axes=tuple(sort_cfg.get("deferred_axes", ()) or ()),
                mamba_state_score_key=str(
                    sort_cfg.get("mamba_state_score_key", "ssm_channel_contrib")
                ),
                embedding_widths=embedding_widths,
            )
        dist.barrier()
    if not (source_dir / "config.json").exists():
        raise FileNotFoundError(f"source checkpoint missing config.json: {source_dir}")
    if not (target_dir / "config.json").exists():
        raise FileNotFoundError(f"target checkpoint missing config.json: {target_dir}")
    bypass_checkpoint_dir = scoring.get("bypass_checkpoint_dir", None)
    if bypass_checkpoint_dir is not None:
        bypass_checkpoint_dir = Path(bypass_checkpoint_dir)
        if not (bypass_checkpoint_dir / "config.json").exists():
            raise FileNotFoundError(
                f"bypass overlay checkpoint missing config.json: {bypass_checkpoint_dir}"
            )

    # Teacher dims for resolving per-candidate prune targets.
    config = _load_model_config_distributed(
        teacher_dir,
        descriptor,
        loader=load_model_config,
    )
    teacher_block_configs = maybe_cast_block_configs(config.block_configs)
    lm = descriptor.get_language_model_config(config)
    num_q = lm.num_attention_heads
    head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // num_q)

    baseline_only = bool(scoring.get("baseline_only", False))
    solutions, ids = _load_solution_work(scoring, output_dir)
    solution_widths = {
        width for solution in solutions if (width := _solution_hidden_width(solution)) is not None
    }
    if len(solution_widths) > 1:
        raise ValueError(
            "one replacement-scoring sweep may contain only one hidden width; "
            f"found {sorted(solution_widths)}"
        )
    source_config = _load_model_config_distributed(
        source_dir,
        descriptor,
        loader=load_model_config,
    )
    source_hidden_width = int(descriptor.get_language_model_config(source_config).hidden_size)
    same_checkpoint = source_dir.resolve() == target_dir.resolve()
    retained_hidden_indices = _source_hidden_channel_indices(
        source_dir,
        source_hidden_width,
        int(lm.hidden_size),
        retained_width=(
            int(lm.hidden_size)
            if bool(scoring.get("zero_pad_hidden_to_teacher_width", False))
            else source_hidden_width
        ),
        source_and_teacher_share_basis=same_checkpoint,
    )
    if solution_widths and max(solution_widths) > source_hidden_width:
        raise ValueError(
            "replacement solution exceeds source hidden width: "
            f"solutions={sorted(solution_widths)} source={source_hidden_width}"
        )
    if not ids and not baseline_only:
        mprint("[solution/automodel] no pending solutions to validate; skipping model load")
        return

    # ---- Cache teacher targets, then score candidates from the requested source. ----
    mprint(
        "[solution/automodel] checkpoint roles | "
        f"target={target_dir} source={source_dir}"
    )
    def score_pending(recipe, cache) -> None:
        sliced_teacher_baseline = None
        if baseline_only or bool(scoring.get("score_source_baseline", True)):
            mprint("[solution/automodel] Phase 1b: scoring source-checkpoint baseline")
            sliced_teacher_baseline = _score_candidate(
                recipe,
                cache,
                params,
                output_dir,
                scoring,
                name="sliced_teacher",
                payload={
                    "role": "source_checkpoint_baseline",
                    "source_checkpoint_dir": str(source_dir),
                    **dict(scoring.get("baseline_payload", {}) or {}),
                },
                prune_target=None,
                retained_hidden_indices=retained_hidden_indices,
            )
            dist.barrier()
        for i_solution in ids:
            solution = solutions[i_solution]
            prune_target = _solution_prune_targets(
                solution,
                teacher_block_configs,
                num_q,
                head_dim,
            )
            solution_output, solution_name = _solution_output_location(
                scoring, output_dir, i_solution
            )
            mprint(
                f"[solution/automodel] Phase 2: scoring solution_{i_solution} "
                f"as {solution_output / solution_name} {prune_target}"
            )
            _score_candidate(
                recipe, cache, params, solution_output, scoring,
                name=solution_name,
                payload={
                    "i_solution": 0 if solution_name == "solution_0" else i_solution,
                    "puzzle_solution": solution,
                    "hidden_width": solution.get("hidden_width"),
                    "sliced_teacher_baseline": sliced_teacher_baseline,
                },
                prune_target=prune_target,
                bypass_checkpoint_dir=bypass_checkpoint_dir,
                retained_hidden_indices=retained_hidden_indices,
                hidden_width=_solution_hidden_width(solution),
            )
            dist.barrier()

    recipe = _run_recipe(
        build_solution_recipe_config(hydra_cfg, target_dir),
        scoring, params["eval_iters"], params["use_puzzletron_dataloader"], params["data_cfg"],
    )
    try:
        mprint("[solution/automodel] Phase 1: caching teacher targets")
        cache = TeacherTargetCache(device=params["teacher_cache_device"])
        teacher_scores = _extract_teacher_targets(recipe, cache, params)
        is_writer = bool(getattr(recipe, "_puzzletron_output_writer", False))
        if teacher_scores is not None and is_writer:
            write_results(output_dir, "teacher", scoring, teacher_scores)
        dist.barrier()
        mprint(f"[solution/automodel] cached targets for {len(cache)} batch(es)")
        if same_checkpoint:
            score_pending(recipe, cache)
    finally:
        recipe.teardown_capture()
        _free_scoring_memory(recipe)

    if same_checkpoint:
        return

    recipe = _run_recipe(
        build_solution_recipe_config(hydra_cfg, source_dir),
        scoring, params["eval_iters"], params["use_puzzletron_dataloader"], params["data_cfg"],
    )
    try:
        score_pending(recipe, cache)
    finally:
        recipe.teardown_capture()
        _free_scoring_memory(recipe)


def launch_score_solution_parents_automodel(hydra_cfg) -> None:
    """Score complete candidate sweeps while loading each parent exactly once.

    The first parent must be the original teacher. Its hidden states and LM-head
    weight remain in the bounded CPU cache while the resident model is replaced
    by each sorted parent. Every non-original parent is evaluated unsliced
    against that common cache and must pass the configured equivalence gate
    before any dynamic candidate is scored.
    """

    from ...anymodel.model_descriptor import ModelDescriptorFactory
    from ...block_config import maybe_cast_block_configs
    from ...tools.checkpoint_utils import load_model_config

    apply_patch()
    scoring = hydra_cfg.scoring
    params = solution_scoring_params(hydra_cfg)
    parents = [dict(parent) for parent in scoring.get("parent_sweeps", ())]
    if not parents:
        raise ValueError("parent_sweeps must contain original, activation, and reverse parents")
    roles = [str(parent.get("role")) for parent in parents]
    if roles[0] != "original" or len(roles) != len(set(roles)):
        raise ValueError(f"parent_sweeps must start with unique role='original'; got {roles}")

    descriptor = ModelDescriptorFactory.get(hydra_cfg.get("descriptor", None))
    teacher_dir = Path(scoring.get("teacher_dir", None) or f"{hydra_cfg.puzzle_dir}/ckpts/teacher")
    teacher_config = _load_model_config_distributed(
        teacher_dir,
        descriptor,
        loader=load_model_config,
    )
    teacher_block_configs = maybe_cast_block_configs(teacher_config.block_configs)
    lm = descriptor.get_language_model_config(teacher_config)
    num_q = lm.num_attention_heads
    head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // num_q)
    force_rescore = bool(scoring.get("force_rescore", False))
    tolerances = dict(scoring.get("parent_equivalence_tolerances", {}))
    manifest_path = Path(scoring.parent_sweep_manifest)
    manifest = {
        "version": 1,
        "status": "running",
        "parent_order": roles,
        "checkpoint_loads": {role: 0 for role in roles},
        "parents": {},
    }

    def write_manifest() -> None:
        if dist.is_master():
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        dist.barrier()

    write_manifest()
    cache = TeacherTargetCache(device=params["teacher_cache_device"])
    original_result_path: Path | None = None

    try:
        for parent in parents:
            role = str(parent["role"])
            evaluation_mode = str(parent.get("evaluation_mode", "runtime_slice"))
            skip_parent_equivalence = bool(parent.get("skip_parent_equivalence", False))
            if evaluation_mode not in {"runtime_slice", "realized_baseline"}:
                raise ValueError(
                    f"unsupported parent evaluation_mode={evaluation_mode!r} for role={role}"
                )
            checkpoint_dir = Path(parent["checkpoint_dir"])
            solutions_path = Path(parent["solutions_path"])
            output_dir = Path(parent["output_dir"])
            if not solutions_path.is_file():
                raise FileNotFoundError(f"{role} parent missing solutions: {solutions_path}")
            output_dir.mkdir(parents=True, exist_ok=True)

            solutions = load_puzzle_solutions(solutions_path, None, False)
            pending_ids = [
                idx
                for idx in range(len(solutions))
                if force_rescore or not (output_dir / f"solution_{idx}.json").is_file()
            ]
            parent_result_path = output_dir / "parent.json"
            needs_parent_evaluation = (
                evaluation_mode != "realized_baseline"
                and not skip_parent_equivalence
                and (force_rescore or not parent_result_path.is_file())
            )
            if not pending_ids and not needs_parent_evaluation:
                manifest["parents"][role] = {
                    "checkpoint_dir": str(checkpoint_dir),
                    "solutions": len(solutions),
                    "pending": 0,
                    "skipped_model_load": True,
                }
                write_manifest()
                continue

            mprint(
                "[solution/automodel] parent sweep load | "
                f"role={role} checkpoint={checkpoint_dir} "
                f"solutions={len(solutions)} pending={len(pending_ids)}"
            )
            try:
                if not (checkpoint_dir / "config.json").is_file():
                    raise FileNotFoundError(
                        f"{role} parent missing config.json: {checkpoint_dir}"
                    )
                parent_config = _load_model_config_distributed(
                    checkpoint_dir,
                    descriptor,
                    loader=load_model_config,
                )
                parent_width = int(
                    descriptor.get_language_model_config(parent_config).hidden_size
                )
                retained_hidden_indices = _source_hidden_channel_indices(
                    checkpoint_dir,
                    parent_width,
                    int(lm.hidden_size),
                )
                recipe = _run_recipe(
                    build_solution_recipe_config(hydra_cfg, checkpoint_dir),
                    scoring,
                    params["eval_iters"],
                    params["use_puzzletron_dataloader"],
                    params["data_cfg"],
                )
            except BaseException as exc:
                if evaluation_mode == "realized_baseline":
                    quarantined = _quarantine_failed_realization(checkpoint_dir)
                    manifest["parents"][role] = {
                        "checkpoint_dir": str(checkpoint_dir),
                        "quarantined": str(quarantined) if quarantined else None,
                        "load_error": f"{type(exc).__name__}: {exc}",
                    }
                    write_manifest()
                raise
            manifest["checkpoint_loads"][role] += 1
            if manifest["checkpoint_loads"][role] != 1:
                raise RuntimeError(f"parent {role} loaded more than once: {manifest['checkpoint_loads']}")
            write_manifest()

            parent_summary = None
            try:
                if role == "original":
                    mprint("[solution/automodel] parent sweep: caching original teacher targets")
                    teacher_scores = _extract_teacher_targets(recipe, cache, params)
                    if teacher_scores is not None and bool(
                        getattr(recipe, "_puzzletron_output_writer", False)
                    ):
                        write_results(
                            output_dir,
                            "parent",
                            scoring,
                            {
                                **teacher_scores,
                                "parent_role": role,
                                "checkpoint_dir": str(checkpoint_dir),
                            },
                        )
                    dist.barrier()
                    original_result_path = parent_result_path
                    if not original_result_path.is_file():
                        raise RuntimeError(
                            f"original teacher result was not written: {original_result_path}"
                        )
                    parent_summary = {"passed": True, "reference": True}
                elif evaluation_mode == "runtime_slice" and not skip_parent_equivalence:
                    if original_result_path is None or len(cache) == 0 and recipe.has_outputs:
                        raise RuntimeError("original teacher cache is unavailable for sorted parent")
                    if needs_parent_evaluation:
                        mprint(
                            "[solution/automodel] parent sweep equivalence | "
                            f"role={role} checkpoint={checkpoint_dir}"
                        )
                        _score_candidate(
                            recipe,
                            cache,
                            params,
                            output_dir,
                            scoring,
                            name="parent",
                            payload={
                                "parent_role": role,
                                "checkpoint_dir": str(checkpoint_dir),
                            },
                            prune_target=None,
                            retained_hidden_indices=retained_hidden_indices,
                        )
                    dist.barrier()
                    parent_summary = _validate_parent_equivalence(
                        teacher_result_path=original_result_path,
                        parent_result_path=parent_result_path,
                        tolerances=tolerances,
                        hidden_basis_permuted=bool(
                            parent.get("hidden_basis_permuted", False)
                        ),
                    )
                    mprint(
                        "[solution/automodel] parent equivalence | "
                        f"role={role} passed={parent_summary['passed']} "
                        f"checks={parent_summary['checks']}"
                    )
                elif evaluation_mode == "runtime_slice":
                    parent_summary = {
                        "passed": True,
                        "reference": False,
                        "gated": False,
                        "reason": "full-width equivalence is owned by sort_equivalence",
                    }
                else:
                    parent_summary = {
                        "passed": True,
                        "reference": False,
                        "gated": False,
                        "reason": "physically realized candidate is intentionally pruned",
                    }

                for solution_idx in pending_ids:
                    solution = solutions[solution_idx]
                    if evaluation_mode == "realized_baseline":
                        prune_target = None
                    else:
                        layer_replacement = _extract_single_sequence_replacement(solution)
                        prune_target = _solution_prune_target(
                            layer_replacement,
                            teacher_block_configs,
                            num_q,
                            head_dim,
                        )
                    mprint(
                        "[solution/automodel] parent sweep candidate | "
                        f"role={role} solution={solution_idx} target={prune_target}"
                    )
                    _score_candidate(
                        recipe,
                        cache,
                        params,
                        output_dir,
                        scoring,
                        name=f"solution_{solution_idx}",
                        payload={
                            "i_solution": solution_idx,
                            "puzzle_solution": solution,
                            "parent_role": role,
                            "checkpoint_dir": str(checkpoint_dir),
                        },
                        prune_target=prune_target,
                        retained_hidden_indices=retained_hidden_indices,
                        hidden_width=(
                            None
                            if evaluation_mode == "realized_baseline"
                            else _solution_hidden_width(solution)
                        ),
                    )
                    dist.barrier()
            finally:
                recipe.teardown_capture()
                _free_scoring_memory(recipe)

            manifest["parents"][role] = {
                "checkpoint_dir": str(checkpoint_dir),
                "solutions": len(solutions),
                "scored_this_run": len(pending_ids),
                "output_dir": str(output_dir),
                "equivalence": parent_summary,
            }
            write_manifest()

        manifest["status"] = "complete"
        write_manifest()
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest()
        raise
