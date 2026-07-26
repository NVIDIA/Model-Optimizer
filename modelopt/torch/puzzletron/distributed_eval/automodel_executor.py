"""Long-lived AutoModel executor for distributed replace-block evaluation."""

from __future__ import annotations

import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..anymodel.registry import resolve_descriptor_from_pretrained
from .schema import EvaluationRequest, EvaluationResult


def _resolve_executor_descriptor(cfg, source_dir: Path):
    descriptor_name = cfg.get("descriptor", None)
    model_cfg = cfg.get("model", {}) or {}
    descriptor_name = descriptor_name or model_cfg.get("descriptor_override", None)
    if descriptor_name:
        return ModelDescriptorFactory.get(descriptor_name)
    resolution = resolve_descriptor_from_pretrained(
        str(source_dir),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    runtime_cfg = cfg.get("_runtime", None)
    if runtime_cfg is None:
        cfg["_runtime"] = {}
        runtime_cfg = cfg["_runtime"]
    runtime_cfg["descriptor"] = resolution.name
    return resolution.descriptor


class AutoModelReplaceBlockExecutor:
    """Load a sorted teacher once and evaluate dynamic one-block candidates."""

    def __init__(self, hydra_cfg):
        self.cfg = hydra_cfg
        self.recipe = None
        self.cache = None
        self.params = None
        self.teacher_block_configs = None
        self.num_q = None
        self.head_dim = None
        self.bypass_checkpoint_dir = None
        self.is_output_writer = False
        self.source_hidden_width = None
        self.sliced_teacher_baseline = None
        self.latest_observability = None
        self._setup_complete = False

    def capabilities(self) -> dict:
        force_hf = None
        if self.params is not None:
            force_hf = self.params["force_hf"]
        return {
            "handlers": ["replace_block", "depth_candidate"],
            "force_hf": force_hf,
            "teacher_target_cache": True,
            "batch_cache_identity": (self.cfg.get("data", {}) or {}).get("fingerprint"),
            "prefix_activation_cache": False,
        }

    def setup(self) -> None:
        if self._setup_complete:
            return
        import modelopt.torch.utils.distributed as dist

        from ..block_config import maybe_cast_block_configs
        from ..plugins.automodel.config import build_solution_recipe_config, solution_scoring_params
        from ..plugins.automodel.launch import _free_scoring_memory
        from ..plugins.automodel.load import validate_force_hf_ep
        from ..plugins.automodel.patch import apply_patch
        from ..plugins.automodel.solution_launch import _extract_teacher_targets, _run_recipe
        from ..plugins.automodel.teacher_cache import TeacherTargetCache
        from ..tools.checkpoint_utils import load_model_config

        scoring = self.cfg.scoring
        self.params = solution_scoring_params(self.cfg)
        apply_patch()
        teacher_dir = Path(
            scoring.get("teacher_dir", None) or f"{self.cfg.puzzle_dir}/ckpts/teacher"
        )
        sorted_dir = Path(self.cfg.puzzle_dir) / "ckpts" / "sorted_teacher"
        source_dir = Path(scoring.get("source_checkpoint_dir", None) or sorted_dir)
        target_dir = Path(scoring.get("target_teacher_dir", None) or source_dir)
        descriptor = _resolve_executor_descriptor(self.cfg, source_dir)
        for role, directory in (("source", source_dir), ("target", target_dir)):
            if not (directory / "config.json").is_file():
                raise FileNotFoundError(
                    f"Distributed evaluation requires a prebuilt sorted teacher; "
                    f"{role} checkpoint is missing config.json: {directory}"
                )
        self.bypass_checkpoint_dir = scoring.get("bypass_checkpoint_dir", None)
        if self.bypass_checkpoint_dir is not None:
            self.bypass_checkpoint_dir = Path(self.bypass_checkpoint_dir)

        model_config = load_model_config(
            teacher_dir,
            trust_remote_code=descriptor.requires_trust_remote_code(),
        )
        self.teacher_block_configs = maybe_cast_block_configs(model_config.block_configs)
        language_config = descriptor.get_language_model_config(model_config)
        self.num_q = int(language_config.num_attention_heads)
        self.head_dim = int(
            getattr(language_config, "head_dim", None)
            or (language_config.hidden_size // self.num_q)
        )
        source_config = load_model_config(
            source_dir,
            trust_remote_code=descriptor.requires_trust_remote_code(),
        )
        self.source_hidden_width = int(
            descriptor.get_language_model_config(source_config).hidden_size
        )

        recipe_dict = build_solution_recipe_config(self.cfg, target_dir)
        distributed = recipe_dict.get("distributed", {})
        validate_force_hf_ep(
            self.params["force_hf"],
            int(distributed.get("ep_size", 1) or 1),
        )
        target_recipe = _run_recipe(
            recipe_dict,
            scoring,
            self.params["eval_iters"],
            self.params["use_puzzletron_dataloader"],
            self.params["data_cfg"],
        )
        self.cache = TeacherTargetCache(device=self.params["teacher_cache_device"])
        _extract_teacher_targets(target_recipe, self.cache, self.params)
        dist.barrier()

        if source_dir.resolve() == target_dir.resolve():
            self.recipe = target_recipe
        else:
            target_recipe.teardown_capture()
            _free_scoring_memory(target_recipe)
            self.recipe = _run_recipe(
                build_solution_recipe_config(self.cfg, source_dir),
                scoring,
                self.params["eval_iters"],
                self.params["use_puzzletron_dataloader"],
                self.params["data_cfg"],
            )
        # AutoModel PP containers can retain a final norm/LM head on a rank
        # where the pipeline stage does not actually execute them.  Elect from
        # ranks that observed teacher captures instead of static module
        # ownership.  This collective runs once while setup is synchronized;
        # candidate loops only consult the cached rank-local boolean.
        import torch.distributed as torch_dist

        rank = torch_dist.get_rank() if torch_dist.is_initialized() else 0
        observed = bool(len(self.cache))
        observed_by_rank = [(rank, observed)]
        if torch_dist.is_initialized():
            observed_by_rank = [None] * torch_dist.get_world_size()
            torch_dist.all_gather_object(observed_by_rank, (rank, observed))
        output_ranks = [item_rank for item_rank, has_capture in observed_by_rank if has_capture]
        if not output_ranks:
            raise RuntimeError("No AutoModel rank captured teacher final hidden states")
        self.is_output_writer = observed and rank == min(output_ranks)
        if rank == 0:
            print(
                "[distributed-eval/automodel] "
                f"observed_output_ranks={output_ranks} writer={min(output_ranks)}",
                flush=True,
            )
        self.sliced_teacher_baseline = self._score(None)
        self._setup_complete = True

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult | None:
        if request.handler not in {"replace_block", "depth_candidate"}:
            raise NotImplementedError(f"Unsupported evaluation handler {request.handler!r}")
        if not self._setup_complete:
            raise RuntimeError("AutoModelReplaceBlockExecutor.setup() was not called")
        from ..plugins.automodel.solution_launch import _solution_prune_target
        from ..replacement_library.replacement_utils import parse_layer_replacement

        request_width = request.payload.get("hidden_width")
        if request_width is not None and int(request_width) != self.source_hidden_width:
            raise ValueError(
                "RPC candidate/source hidden-width mismatch: "
                f"request={request_width} source={self.source_hidden_width}"
            )

        raw_replacements = (
            request.payload.get("layer_replacements", [])
            if request.handler == "depth_candidate"
            else [request.payload["layer_replacement"]]
        )
        prune_targets = [
            target
            for raw in raw_replacements
            if (
                target := _solution_prune_target(
                    parse_layer_replacement(raw),
                    self.teacher_block_configs,
                    self.num_q,
                    self.head_dim,
                )
            )
            is not None
        ]
        layer_indices = [int(target["layer_idx"]) for target in prune_targets]
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError(
                "depth_candidate must contain at most one cumulative replacement per layer"
            )
        started = time.perf_counter()
        import torch.distributed as torch_dist

        if not torch_dist.is_initialized() or torch_dist.get_rank() == 0:
            print(
                "[distributed-eval/automodel] "
                f"request_start={request.request_id} layers={layer_indices}",
                flush=True,
            )
        score_target = (
            prune_targets
            if request.handler == "depth_candidate"
            else (prune_targets[0] if prune_targets else None)
        )
        metrics = self._score(score_target)
        if metrics is None:
            return None
        if not torch_dist.is_initialized() or torch_dist.get_rank() == 0:
            print(
                "[distributed-eval/automodel] "
                f"request_done={request.request_id} "
                f"seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
        counts = {
            name: len(value.get("per_sample", []))
            for name, value in metrics.items()
            if isinstance(value, dict)
        }
        return EvaluationResult(
            request_id=request.request_id,
            campaign_id=request.campaign_id,
            metrics=metrics,
            counts=counts,
            timing={"evaluation_seconds": time.perf_counter() - started},
            provenance={
                "handler": request.handler,
                "evaluator_revision": request.evaluator_revision,
                "micro_batch_size": self.params.get("micro_batch_size"),
                "hidden_width": self.source_hidden_width,
                "sliced_teacher_baseline": self.sliced_teacher_baseline,
                "observability": self.latest_observability,
            },
        )

    def _score(self, prune_target: dict | list[dict] | None) -> dict | None:
        import torch.distributed as torch_dist

        import modelopt.torch.utils.distributed as dist

        from ..plugins.automodel.solution_metrics import (
            aggregate_solution_scores,
            retain_teacher_channels,
            score_batch,
        )

        recipe = self.recipe
        cache = self.cache
        params = self.params
        per_batch = []
        tp_group = recipe.tensor_parallel_group()
        candidate_lm_head = recipe.lm_head_weight() if recipe.has_outputs else None
        raw_targets = prune_target if isinstance(prune_target, list) else [prune_target]
        prune_targets = [dict(target) for target in raw_targets if target is not None]
        layer_indices = [int(target["layer_idx"]) for target in prune_targets]
        owned_layers = [
            layer_idx
            for layer_idx in layer_indices
            if recipe._find_decoder_layer(layer_idx) is not None
        ]
        overlay_started = time.perf_counter()
        with ExitStack() as stack:
            for target in prune_targets:
                layer_idx = int(target["layer_idx"])
                bypass_dir = target.pop(
                    "bypass_checkpoint_dir", self.bypass_checkpoint_dir
                )
                stack.enter_context(
                    recipe.block_checkpoint_overlay_context(bypass_dir, layer_idx)
                    if bypass_dir is not None
                    else nullcontext()
                )
                stack.enter_context(recipe.prune_block_context(**target))
            if owned_layers:
                print(
                    "[distributed-eval/automodel] "
                    f"rank={torch_dist.get_rank()} layers={owned_layers} "
                    f"overlay_and_prune_ready_seconds={time.perf_counter() - overlay_started:.3f}",
                    flush=True,
                )
            for batch_idx, (hidden, targets) in enumerate(recipe.iterate_captures()):
                if hidden is None:
                    continue
                teacher_hidden = cache.hidden(
                    batch_idx,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                teacher_lm_head = cache.lm_head(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                teacher_hidden, teacher_lm_head = retain_teacher_channels(
                    hidden,
                    candidate_lm_head,
                    teacher_hidden,
                    teacher_lm_head,
                )
                batch_started = time.perf_counter()
                per_batch.append(
                    score_batch(
                        hidden,
                        candidate_lm_head,
                        teacher_hidden,
                        teacher_lm_head,
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
                if self.is_output_writer:
                    print(
                        "[distributed-eval/automodel] "
                        f"layers={layer_indices} batch={batch_idx} "
                        f"score_seconds={time.perf_counter() - batch_started:.3f}",
                        flush=True,
                    )
        token_group = recipe._groups.token_group if recipe._groups is not None else None
        cp_group = recipe.context_parallel_group()
        self.latest_observability = recipe.observability_metadata()
        local_metrics = None
        if recipe.has_outputs and per_batch:
            reduced = aggregate_solution_scores(
                per_batch,
                token_group=token_group,
                cp_group=cp_group,
            )
            if self.is_output_writer:
                local_metrics = reduced
        dist.barrier()
        if not torch_dist.is_available() or not torch_dist.is_initialized():
            return local_metrics
        gathered: list[dict | None] = [None] * torch_dist.get_world_size()
        torch_dist.all_gather_object(gathered, local_metrics)
        return next((value for value in gathered if value is not None), None)

    def close(self) -> None:
        if self.recipe is None:
            return
        from ..plugins.automodel.launch import _free_scoring_memory

        self.recipe.teardown_capture()
        _free_scoring_memory(self.recipe)
        self.recipe = None
        self.cache = None
        self._setup_complete = False

    def prepare_prefix_cache(self, *args, **kwargs):
        raise NotImplementedError(
            "Prefix activation caching is intentionally disabled until topology-specific "
            "AutoModel correctness experiments pass"
        )
