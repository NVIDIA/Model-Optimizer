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

"""Entry point for AutoModel-backed activation scoring.

Dispatched from ``activation_scoring.launch_score_activations`` when
``pruning.backend == "automodel"``.
"""

import logging
import os

from ...tools.logger import mprint
from .config import build_recipe_config, scoring_params
from .load import validate_force_hf_ep
from .patch import apply_patch

logger = logging.getLogger(__name__)

__all__ = ["launch_score_activations_automodel", "launch_score_activations_automodel_multipass"]


def _observed_num_nodes(requested: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))))
    return max(int(requested), (world_size + local_world_size - 1) // local_world_size)


def launch_score_activations_automodel(hydra_cfg, num_nodes: int = 1, node_index: int = 0):
    """Run pruning activation scoring on a NeMo-AutoModel-parallelized teacher.

    ``num_nodes``/``node_index`` are accepted for interface parity with the other
    multi-node stages but are informational here: data parallelism for the AutoModel
    backend comes from the global (torchrun) world size + a ``DistributedSampler``,
    and the per-target SUM-reduce/GATHER make the result independent of node count —
    so no modulo work-splitting is needed.
    """
    params = scoring_params(hydra_cfg)
    if params["method"] is None:
        raise ValueError(
            "AutoModel scoring requires pruning.activation_hooks_kwargs.method to be set."
        )
    validate_force_hf_ep(params["force_hf"], params["ep_size"])

    # Patch nemo_automodel so from_pretrained loads the AnyModel teacher, then build
    # the recipe config and run the forward-only scoring recipe.
    apply_patch()
    recipe_dict = build_recipe_config(hydra_cfg)

    dist_cfg = recipe_dict.get("distributed", {})
    model_cfg = recipe_dict.get("model", {})
    mprint(
        "[activation/automodel] backend ACTIVE | "
        f"descriptor={model_cfg.get('anymodel_descriptor')} force_hf={model_cfg.get('force_hf')} "
        f"method={params['method']} eval_iters={params['eval_iters']} "
        f"validation_full_iters={params['hook_kwargs'].get('validation_full_iters')}"
    )
    mprint(
        "[activation/automodel] requested parallel sizes (from recipe): "
        f"tp={dist_cfg.get('tp_size')} pp={dist_cfg.get('pp_size')} cp={dist_cfg.get('cp_size')} "
        f"ep={dist_cfg.get('ep_size')} dp={dist_cfg.get('dp_size')}"
    )

    # NeMo recipes require a ConfigNode (not a plain dict): the base recipe stores cfg
    # as-is and setup_distributed only extracts the `distributed` sub-section when cfg is
    # NOT a dict, otherwise it forwards the whole config as the strategy kwargs. Wrap after
    # apply_patch() so the model `_target_` (from_pretrained) resolves to the patched fn.
    from nemo_automodel.components.config.loader import ConfigNode

    recipe_config = ConfigNode(recipe_dict)

    # Imported lazily: this pulls in nemo_automodel (subclasses a NeMo recipe).
    from .scoring_recipe import ActivationScoringRecipe

    logger.info(
        "Launching AutoModel activation scoring (method=%s, nodes=%d, idx=%d)",
        params["method"],
        num_nodes,
        node_index,
    )
    recipe = ActivationScoringRecipe(
        recipe_config,
        pruning_mixin=hydra_cfg.pruning.pruning_mixin,
        hook_kwargs=params["hook_kwargs"],
        pruning_cfg=hydra_cfg.pruning,
        activations_log_dir=params["activations_log_dir"],
        eval_iters=params["eval_iters"],
        use_puzzletron_dataloader=params["use_puzzletron_dataloader"],
        data_cfg=params["data_cfg"],
        embedding_pruning_cfg=params["embedding_pruning_cfg"],
    )
    recipe.setup()
    try:
        recipe.run_scoring()
        _write_completion_marker(
            params["activations_log_dir"],
            params,
            num_nodes,
            observability=recipe.observability_metadata(),
        )
    finally:
        _free_scoring_memory(recipe)


def launch_score_activations_automodel_multipass(
    hydra_cfg, passes, pass_names, parent, num_nodes: int = 1, node_index: int = 0
):
    """Run ALL activation passes in ONE forward sweep by combining all hook specs.

    Each pass attaches hooks to different module targets (e.g. ``mlp.down_proj`` for FFN,
    ``self_attn.o_proj`` for attention).  The hooks are fully independent, so registering
    all of them before a single calibration loop is exact and O(1) in data cost.

    ``passes`` and ``pass_names`` come from ``_run_activation_passes``; ``parent`` is the
    shared parent ``activations_log_dir``.
    """
    from pathlib import Path as _Path

    # Build a scorer spec for every pass: {pruning_mixin, hook_kwargs, activations_log_dir}.
    specs = []
    attention_scored_axes = hydra_cfg.pruning.get("attention_scored_axes", None)
    attention_token_chunk_size = hydra_cfg.pruning.get("attention_token_chunk_size", None)
    for name, pass_cfg in zip(pass_names, passes):
        mixin = pass_cfg.get("pruning_mixin", None)
        raw_hk = pass_cfg.get("activation_hooks_kwargs", None)
        hook_kwargs = dict(raw_hk) if raw_hk else {}
        if hook_kwargs.get("method") == "grouped_attention_contribution":
            if attention_scored_axes is not None and "scored_axes" not in hook_kwargs:
                hook_kwargs["scored_axes"] = attention_scored_axes
            if attention_token_chunk_size is not None and "token_chunk_size" not in hook_kwargs:
                hook_kwargs["token_chunk_size"] = attention_token_chunk_size
        specs.append(
            {
                "pruning_mixin": mixin,
                "hook_kwargs": hook_kwargs,
                "activations_log_dir": str(_Path(parent) / name),
            }
        )

    # Temporarily apply first pass's overrides onto cfg so scoring_params / build_recipe_config
    # read the right pruning_mixin, hook_kwargs, activations_log_dir for the primary spec.
    _override_keys = ("pruning_mixin", "activation_hooks_kwargs", "hook_class", "mlp_init_mode",
                      "activations_log_dir")
    _saved = {k: hydra_cfg.pruning.get(k, None) for k in _override_keys}
    try:
        first_pass = passes[0]
        for key in _override_keys[:-1]:  # all except activations_log_dir
            val = first_pass.get(key, None)
            if val is not None:
                hydra_cfg.pruning[key] = val
        hydra_cfg.pruning.activations_log_dir = specs[0]["activations_log_dir"]

        params = scoring_params(hydra_cfg)
        if params["method"] is None:
            raise ValueError(
                "AutoModel multi-pass scoring: first pass must set "
                "activation_hooks_kwargs.method."
            )
        validate_force_hf_ep(params["force_hf"], params["ep_size"])
        apply_patch()
        recipe_dict = build_recipe_config(hydra_cfg)
    finally:
        for key, val in _saved.items():
            hydra_cfg.pruning[key] = val

    # scoring_params computes validation_full_iters = eval_samples // micro_batch_size and
    # injects it into hook_kwargs.  The per-pass specs were built from raw YAML before this
    # call, so none of them have it yet.  Inject it now into every spec; setdefault means an
    # explicit per-pass override in activation_hooks_kwargs is respected.
    vfi = params["hook_kwargs"].get("validation_full_iters")
    for spec in specs:
        spec["hook_kwargs"].setdefault("validation_full_iters", vfi)

    dist_cfg = recipe_dict.get("distributed", {})
    mprint(
        "[activation/automodel] multi-pass (single forward) ACTIVE | "
        f"passes={pass_names} | "
        f"tp={dist_cfg.get('tp_size')} pp={dist_cfg.get('pp_size')} "
        f"dp={dist_cfg.get('dp_size')} eval_iters={params['eval_iters']}"
    )

    from nemo_automodel.components.config.loader import ConfigNode

    recipe_config = ConfigNode(recipe_dict)

    from .scoring_recipe import ActivationScoringRecipe

    recipe = ActivationScoringRecipe(
        recipe_config,
        pruning_mixin=specs[0]["pruning_mixin"],
        hook_kwargs=specs[0]["hook_kwargs"],
        pruning_cfg=hydra_cfg.pruning,
        activations_log_dir=specs[0]["activations_log_dir"],
        eval_iters=params["eval_iters"],
        use_puzzletron_dataloader=params["use_puzzletron_dataloader"],
        data_cfg=params["data_cfg"],
        embedding_pruning_cfg=params["embedding_pruning_cfg"],
        extra_scorer_specs=specs[1:],
    )
    recipe.setup()
    try:
        recipe.run_scoring()
        # Write completion markers for each pass dir so the resume short-circuit works.
        for spec in specs:
            _write_completion_marker(
                spec["activations_log_dir"],
                {**params, "method": spec["hook_kwargs"].get("method")},
                num_nodes,
                observability=recipe.observability_metadata(),
            )
    finally:
        _free_scoring_memory(recipe)


def _write_completion_marker(
    activations_log_dir,
    params,
    num_nodes: int,
    *,
    observability: dict | None = None,
) -> None:
    """Write ``args.json`` once scoring finishes so ``check_scoring_completion`` recognizes a
    completed AutoModel run and the resume short-circuit can skip it (it requires both the
    ``rank_*.pth`` shards *and* ``args.json``, mirroring the legacy writer).
    """
    import json

    import modelopt.torch.utils.distributed as dist

    dist.barrier()  # all stage writers have flushed their rank_*.pth
    if dist.is_master():
        from pathlib import Path

        out_dir = Path(activations_log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(
            json.dumps(
                {
                    "backend": "automodel",
                    "method": params.get("method"),
                    "eval_samples": params.get("eval_samples"),
                    "micro_batch_size": params.get("micro_batch_size"),
                    "eval_iters": params.get("eval_iters"),
                    "num_nodes": _observed_num_nodes(num_nodes),
                    "observability": observability or {},
                },
                indent=2,
            )
        )
    dist.barrier()


def _free_scoring_memory(recipe) -> None:
    """Release the scoring model + scorers from GPU before later in-process stages.

    The scoring stage holds substantial GPU memory: the parallelized NeMo model and — for the
    iterative method — every scorer caches the full gathered ``down_proj`` weight. The pruning
    and bypass-distillation stages run in the *same* torchrun process and build their own models
    on all ranks, so without freeing this first they OOM (legacy scoring did not leave a resident
    model). The ``torch.distributed`` process group is intentionally left intact — the post-stage
    barrier and the later stages still need it.
    """
    import gc

    import torch

    close_observability = getattr(recipe, "close_observability", None)
    if close_observability is not None:
        close_observability()

    for scorer in getattr(recipe, "_scorers", None) or []:
        try:
            scorer.remove()  # detach the forward hook; drops the scorer's cached GPU tensors
        except Exception:  # noqa: BLE001
            pass
    # Drop large references the recipe holds so they can be collected. Keep the recipe's
    # distributed/env state; only the memory-heavy model/optimizer/scorer state is released.
    for attr in ("_scorers", "model_parts", "model", "optimizer", "pp", "dataloader"):
        if hasattr(recipe, attr):
            setattr(recipe, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
