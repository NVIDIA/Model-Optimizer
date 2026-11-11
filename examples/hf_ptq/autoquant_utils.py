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

"""Recipe translation and search helpers for the Hugging Face AutoQuantize example."""

import argparse
import copy
import os
import warnings
from fnmatch import fnmatch
from typing import Any

import torch
from torch.utils.data import DataLoader

import modelopt.torch.quantization as mtq
from modelopt.recipe import ModelOptAutoQuantizeRecipe, load_recipe
from modelopt.recipe.presets import KV_CACHE_NONE, KV_QUANT_CFG_CHOICES, QUANT_CFG_CHOICES
from modelopt.torch.opt.searcher import is_directory_checkpoint
from modelopt.torch.quantization.algorithms import AUTO_QUANTIZE_SEARCHERS
from modelopt.torch.quantization.kv_cache_auto_quant import _forward_kl
from modelopt.torch.quantization.model_quant import _process_search_space
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import safe_load
from modelopt.torch.utils.dataset_utils import create_forward_loop, model_type_is_enc_dec
from modelopt.torch.utils.distributed import size as world_size

__all__ = ["auto_quantize"]

_FSDP2_AUTOQUANT_WARNING = (
    "AutoQuantize with --use_fsdp2 is experimental. "
    "Resume with the same world size, model, recipe, and calibration data."
)


def _get_autoquant_memory_probe(recipe: ModelOptAutoQuantizeRecipe):
    """Include retained reference logits and KL intermediates in batch-size probing."""
    stages = (recipe.auto_quantize, recipe.kv_auto_quantize)
    if not any(stage is not None and stage.auto_quantize_method == "kl_div" for stage in stages):
        return None
    has_kv_kl = any(
        stage is not None
        and stage.auto_quantize_method == "kl_div"
        and stage.constraints.cost_model == "kv_cache"
        for stage in stages
    )

    def probe(model, inputs):
        # Encoder-decoder forwards need decoder inputs; retain their generation-based probe.
        if model_type_is_enc_dec(model):
            model.generate(inputs)
            return
        # Composed searches must still budget for the gradient stage's activations.
        if torch.is_grad_enabled():
            model(inputs)

        def get_logits():
            logits = model(inputs).logits
            if has_kv_kl:
                return _select_unpadded_logits(logits, {"attention_mask": torch.ones_like(inputs)})
            return logits

        with torch.no_grad():
            logits_ref = get_logits()
            log_prob_ref = torch.log_softmax(logits_ref.float(), dim=-1)
            # Keep the previous candidate alive during the next forward, as the scorer does.
            for _ in range(2):
                logits_quant = get_logits()
                _forward_kl(logits_quant, log_prob_ref)

    return probe


# Presets safe to mix into an AutoQuantize search *and* write via the unified HF checkpoint
# exporter. Export-compatibility is a property of the export path, not of a preset's validity for
# plain PTQ, so this is a curated set rather than something derived from QUANT_CFG_CHOICES.
# TODO: drop the partial-model presets (e.g. nvfp4_mlp_only, nvfp4_experts_only) from this set as future work.
_AUTO_QUANTIZE_QFORMATS: frozenset[str] = frozenset(
    {
        "fp8",
        "int8_smoothquant",
        "int8_weight_only",
        "int4_awq",
        "nvfp4",
        "nvfp4_awq_lite",
        "nvfp4_w4a4_weight_mse_fp8_sweep",
        "w4a8_awq_beta",
        "w4a16_nvfp4",
        "fp8_2d_blockwise_weight_only",
        "w4a8_mxfp4_fp8",
        "nvfp4_mlp_only",
        "nvfp4_experts_only",
        "nvfp4_omlp_only",
        "nvfp4_w4a4_weight_local_hessian",
        "mxfp8",
    }
)


def auto_quantize(
    args: argparse.Namespace,
    language_model: torch.nn.Module,
    calib_dataloader: DataLoader,
    aq_config,
    full_model: torch.nn.Module | None = None,
    fixed_quantize_config=None,
    allow_uniform_kv: bool = True,
    checkpoint: str | None = None,
):
    """Recipe-driven auto_quantize, organized around an AutoQuantizeConfig.

    The sole AutoQuantize entry point: it is driven by the recipe's AutoQuantizeConfig and optional
    fixed PTQ config, then wraps ``mtq.auto_quantize``.
    """
    if args.calib_with_images:
        raise NotImplementedError(
            "AutoQuantize with image-text calibration is not supported yet. "
            "Please run plain PTQ (e.g., --qformat nvfp4) with --calib_with_images."
        )
    assert args.inference_pipeline_parallel <= 1, (
        "Auto Quantization is not supported for pipeline parallel size > 1"
    )

    inputs = _mtq_inputs_from_auto_quantize_config(
        aq_config,
        args,
        fixed_quantize_config=fixed_quantize_config,
        allow_uniform_kv=allow_uniform_kv,
    )
    if args.use_fsdp2:
        warnings.warn(_FSDP2_AUTOQUANT_WARNING)
    # base-model lm_head handling (mirrors the CLI helper)
    is_base_model = (
        full_model is not None
        and language_model is not full_model
        and not hasattr(language_model, "lm_head")
        and hasattr(full_model, "lm_head")
    )
    if is_base_model:
        assert full_model is not None
        lm_head = full_model.lm_head

        def loss_func(output, data):
            logits = lm_head(output.last_hidden_state)
            labels = data["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            return torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
    else:

        def loss_func(output, data):
            return output.loss

    method_name = (
        inputs["method"]["method"] if isinstance(inputs["method"], dict) else inputs["method"]
    )
    if method_name == "gradient":

        def forward_step(model, batch):
            inputs_ = {k: v for k, v in batch.items() if k != "labels"} if is_base_model else batch
            return model(**inputs_)

    elif method_name in ("kl_div", "aumann_shapley"):

        def forward_step(model, batch):
            inputs_ = {k: v for k, v in batch.items() if k != "labels"} if is_base_model else batch
            output = model(**inputs_)
            if output is None:
                # Layerwise calibration of a candidate stops the forward once it has the inputs.
                return None
            if is_base_model:
                assert full_model is not None
                logits = full_model.lm_head(output.last_hidden_state)
            else:
                logits = output.logits
            if inputs["search_domain"] == "kv_cache":
                return _select_unpadded_logits(logits, batch)
            return logits

        if method_name == "aumann_shapley":
            loss_func = None

    else:
        raise ValueError(
            f"Invalid auto_quantize method: {method_name}. Must be 'gradient', 'kl_div', "
            "or 'aumann_shapley'"
        )

    score_batch_size = args.batch_size * (world_size() if args.use_fsdp2 else 1)
    auto_quantize_kwargs: dict[str, Any] = {
        "constraints": inputs["constraints"],
        "data_loader": calib_dataloader,
        "forward_step": forward_step,
        "quantization_formats": inputs["quantization_formats"],
        "num_calib_steps": len(calib_dataloader),
        "num_score_steps": min(
            len(calib_dataloader), max(inputs["score_size"] // score_batch_size, 1)
        ),
        "verbose": True,
        "disabled_layers": inputs["disabled_layers"],
        "method": inputs["method"],
        "checkpoint": checkpoint,
    }
    if inputs["search_domain"] == "weight":
        auto_quantize_kwargs.update(
            {
                "loss_func": loss_func,
                "fixed_quantization_config": inputs["fixed_quantization_config"],
                "module_search_spaces": inputs["module_search_spaces"],
            }
        )

    format_shard = getattr(args, "auto_quantize_format_shard", None)
    if format_shard is not None:
        auto_quantize_kwargs["format_shard"] = format_shard
    language_model, _ = mtq.auto_quantize(
        language_model,
        **auto_quantize_kwargs,
    )
    # A format shard, and a search whose finalizing run calibrates the KV cache, only save scores.
    if (
        inputs["search_domain"] == "kv_cache"
        or format_shard is not None
        or getattr(aq_config, "finalize_algorithm", None) is not None
    ):
        return language_model

    # KV cache quantization is uniform; applied after the LP search.
    kv_cache_quant_cfg = inputs["kv_cache_quant_cfg"]
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    print(f"{'Enable' if kv_cache_quant_cfg is not None else 'Disable'} KV cache quantization")
    if kv_cache_quant_cfg is not None:
        kv_entries = _kv_cache_entries(kv_cache_quant_cfg)
        mtq.set_quantizer_by_cfg(language_model, quant_cfg=kv_entries)
        if not _kv_cfg_uses_constant_amax(kv_entries):
            with mtq.set_quantizer_by_cfg_context(
                language_model,
                [{"quantizer_name": "*", "enable": False}, *kv_entries],
            ):
                mtq.calibrate(language_model, algorithm="max", forward_loop=calibrate_loop)
    return language_model


def _mtq_inputs_from_auto_quantize_config(
    aq_config,
    args: argparse.Namespace,
    fixed_quantize_config=None,
    allow_uniform_kv: bool = True,
) -> dict:
    """Map a resolved AutoQuantizeConfig to mtq.auto_quantize inputs.

    Single, testable place where a recipe maps to mtq inputs. ``fixed_quantize_config`` is the
    optional normal PTQ baseline for modules outside explicit search spaces. ``disabled_layers``
    and candidate cost come entirely from the recipe (no model introspection). KV cache falls back
    to ``--kv_cache_qformat`` when the recipe omits it.
    """
    constraints = aq_config.constraints.model_dump(exclude_none=True)
    method_options = aq_config.method_options or {}
    if aq_config.uses_predicted_damage_target:
        # The recipe validator rejects an explicit bit budget; remove only the schema default.
        constraints.pop("effective_bits", None)

    is_kv_search = aq_config.constraints.cost_model == "kv_cache"
    if is_kv_search:
        return {
            "search_domain": "kv_cache",
            "constraints": constraints,
            "quantization_formats": [
                fmt.model_dump(exclude_none=True) for fmt in aq_config.candidate_formats
            ],
            "disabled_layers": aq_config.disabled_layers,
            "method": aq_config.auto_quantize_method,
            "score_size": aq_config.score_size,
        }

    method = aq_config.auto_quantize_method
    if method_options:
        method = {"method": method, **method_options}
    # cost_excluded_layers (sibling of disabled_layers) maps to the mtq cost key: these layers are
    # kept out of the bit-budget denominator (cost_weight 0) — e.g. VL vision towers — distinct from
    # disabled_layers, which removes them from the search.
    if aq_config.cost_excluded_layers:
        constraints.setdefault("cost", {})["excluded_module_name_patterns"] = (
            aq_config.cost_excluded_layers
        )
    if not allow_uniform_kv:
        kv_cache_quant_cfg = None
    elif aq_config.kv_cache is not None:
        kv_cache_quant_cfg = aq_config.kv_cache.model_dump()
    elif args.kv_cache_qformat == KV_CACHE_NONE:
        kv_cache_quant_cfg = None
    else:
        kv_cache_quant_cfg = copy.deepcopy(KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])
    # Translate each candidate to its mtq preset dict and, in the same pass, guard export
    # compatibility (fails fast, before the expensive search). Custom configs matching no shipped
    # preset can't be verified, so warn rather than block.
    quantization_formats = _mtq_candidate_formats(aq_config.candidate_formats)
    fixed_quantization_config = (
        _mtq_candidate_formats([fixed_quantize_config])[0]
        if fixed_quantize_config is not None
        else None
    )
    module_search_spaces = [
        {
            "module_name_patterns": search_space.module_name_patterns,
            "quantization_formats": _mtq_candidate_formats(search_space.candidate_formats),
            "allow_no_quant": search_space.allow_no_quant,
        }
        for search_space in aq_config.module_search_spaces
    ]
    return {
        "search_domain": "weight",
        "constraints": constraints,
        "quantization_formats": quantization_formats,
        "fixed_quantization_config": fixed_quantization_config,
        "module_search_spaces": module_search_spaces,
        "disabled_layers": aq_config.disabled_layers,
        "kv_cache_quant_cfg": kv_cache_quant_cfg,
        "method": method,
        "score_size": aq_config.score_size,
    }


def _mtq_candidate_formats(formats) -> list[dict]:
    """Translate recipe candidate formats to export-compatible mtq configs."""
    quantization_formats = []
    for fmt in formats:
        preset_name, quant_cfg = _match_candidate_to_preset(fmt)
        if preset_name is not None and preset_name not in _AUTO_QUANTIZE_QFORMATS:
            raise ValueError(
                f"AutoQuantize candidate_formats entry '{preset_name}' is not supported for "
                "unified checkpoint export. Use an export-compatible format."
            )
        if preset_name is None:
            warnings.warn(
                "An AutoQuantize candidate_formats entry matches no shipped preset; its export "
                "compatibility cannot be verified. Ensure it is safe for HF checkpoint export."
            )
        quantization_formats.append(quant_cfg)
    return quantization_formats


def _match_candidate_to_preset(fmt) -> tuple[str | None, dict]:
    """Match a recipe candidate against the shipped QUANT_CFG_CHOICES presets by value.

    Returns ``(preset_name, quant_cfg)``: ``preset_name`` is the matched preset (or None for a
    custom config matching none), and ``quant_cfg`` is the dict passed to mtq.auto_quantize.
    Passing the matched preset dict (rather than the candidate's own dump) keeps the search naming
    the candidate after the preset (e.g. FP8_DEFAULT_CFG), consistent with CLI-produced checkpoints.

    ``effective_bits`` is cost-only metadata (it does not affect export), so it is excluded when
    identifying the preset — otherwise a per-candidate override would make a shipped preset look
    "custom" and slip past the export-compat whitelist. Any override is preserved in the return.
    """
    stripped = fmt.model_dump(exclude_unset=True)
    match_key = {k: v for k, v in stripped.items() if k != "effective_bits"}
    for name, preset in QUANT_CFG_CHOICES.items():
        if preset == match_key:
            if "effective_bits" in stripped:
                return name, {**preset, "effective_bits": stripped["effective_bits"]}
            return name, preset
    return None, fmt.model_dump()


def _quantize_config_explicitly_enables_kv(quant_cfg: dict[str, Any]) -> bool:
    """Detect explicit K/V rules while preserving their ordered override semantics."""
    names = ("k_bmm_quantizer", "v_bmm_quantizer")
    enabled_by_parent = {None: dict.fromkeys(names, False)}
    for entry in quant_cfg["quant_cfg"]:
        pattern = entry["quantizer_name"]
        if pattern != "*" and not any(marker in pattern for marker in ("bmm", "attn", "attention")):
            continue
        basename_pattern = pattern.rsplit(".", 1)[-1]
        matched_names = [
            name for name in names if fnmatch(name, basename_pattern) or pattern.endswith(name)
        ]
        if not matched_names:
            continue

        parent_class = entry.get("parent_class")
        if parent_class is None:
            scopes = enabled_by_parent.values()
        else:
            scopes = [enabled_by_parent.setdefault(parent_class, enabled_by_parent[None].copy())]
        for enabled in scopes:
            for name in matched_names:
                enabled[name] = entry["enable"]
    return any(any(enabled.values()) for enabled in enabled_by_parent.values())


def _recipe_is_auto_quantize(recipe: str | None) -> bool:
    """True if ``recipe`` resolves to an AutoQuantize recipe (peeked before model load)."""
    return recipe is not None and isinstance(load_recipe(recipe), ModelOptAutoQuantizeRecipe)


def _recipe_is_kv_auto_quantize(recipe: str | None) -> bool:
    """True if ``recipe`` resolves to a KV AutoQuantize recipe (peeked before model load)."""
    if recipe is None:
        return False
    loaded_recipe = load_recipe(recipe)
    return isinstance(loaded_recipe, ModelOptAutoQuantizeRecipe) and any(
        stage is not None and stage.constraints.cost_model == "kv_cache"
        for stage in (loaded_recipe.auto_quantize, loaded_recipe.kv_auto_quantize)
    )


def _check_auto_quantize_shard_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Parse ``--auto_quantize_format_shard`` and refuse shard flags a run cannot use."""
    if args.auto_quantize_format_shard is not None:
        try:
            index, count = (int(v) for v in args.auto_quantize_format_shard.split("/"))
        except ValueError:
            parser.error("--auto_quantize_format_shard must be INDEX/COUNT, e.g. 0/5.")
        if not 0 <= index < count:
            parser.error("--auto_quantize_format_shard needs 0 <= INDEX < COUNT.")
        args.auto_quantize_format_shard = (index, count)
    if (args.auto_quantize_format_shard is not None or args.auto_quantize_merge_shards) and not (
        args.auto_quantize_checkpoint and _recipe_is_auto_quantize(args.recipe)
    ):
        parser.error(
            "--auto_quantize_format_shard and --auto_quantize_merge_shards need a weight "
            "AutoQuantize --recipe and --auto_quantize_checkpoint."
        )
    if args.auto_quantize_format_shard is not None and args.auto_quantize_merge_shards:
        parser.error("A format shard cannot also merge shards.")
    if args.auto_quantize_format_shard is not None and args.batch_size == 0:
        # A probed batch size follows each shard's free memory; the merge needs equal step counts.
        parser.error(
            "--auto_quantize_format_shard needs an explicit --batch_size, the same in every shard."
        )


def _auto_quantize_finalize_config(
    recipe: ModelOptAutoQuantizeRecipe, args: argparse.Namespace
) -> dict | None:
    """The frozen quantize config to calibrate the source weights with, once the scores are saved.

    Refuses ``finalize_algorithm`` and format shards where the scores cannot pass between runs,
    and merges ``--auto_quantize_merge_shards`` into ``--auto_quantize_checkpoint`` first. Returns
    None while the search still has to run, and always for a format shard.
    """
    primary = recipe.auto_quantize
    finalize = primary.finalize_algorithm is not None
    # Off unless given, like the shard flag auto_quantize reads, for args built without them.
    format_shard = getattr(args, "auto_quantize_format_shard", None)
    merge_shards = getattr(args, "auto_quantize_merge_shards", None)
    if not (finalize or format_shard is not None or merge_shards):
        return None
    if (
        primary.constraints.cost_model == "kv_cache"
        or recipe.kv_auto_quantize is not None
        or not args.auto_quantize_checkpoint
    ):
        raise ValueError(
            "auto_quantize.finalize_algorithm, --auto_quantize_format_shard and "
            "--auto_quantize_merge_shards need a weight search without a KV-cache search, "
            "and --auto_quantize_checkpoint to hand the scores between runs."
        )
    if dist.is_initialized():
        # Each rank would search into its own file, which the merge and the scores lookup below
        # do not read.
        raise ValueError(
            "auto_quantize.finalize_algorithm, --auto_quantize_format_shard and "
            "--auto_quantize_merge_shards run in one process; launch hf_ptq without torchrun "
            "and without --use_fsdp2."
        )
    checkpoint = _search_checkpoint_file(args.auto_quantize_checkpoint)
    if merge_shards and not os.path.exists(checkpoint):
        mtq.merge_auto_quantize_checkpoints(
            [_search_checkpoint_file(path) for path in merge_shards.split(",")], checkpoint
        )
    if not finalize or format_shard is not None:
        return None
    search_state = _scored_auto_quantize_state(checkpoint)
    if search_state is None:
        return None
    if search_state.get("method") not in (None, primary.auto_quantize_method):
        # A resumed search refuses another method's scores; finalizing reads them as saved.
        raise ValueError(
            f"{checkpoint} holds {search_state['method']} scores, not the recipe's "
            f"{primary.auto_quantize_method}. Finalize them with the recipe that scored them, "
            "or search again into another --auto_quantize_checkpoint."
        )
    # So are scores taken for other candidate formats, processed as the search processes them.
    inputs = _mtq_inputs_from_auto_quantize_config(
        primary, args, fixed_quantize_config=recipe.quantize
    )
    with warnings.catch_warnings():  # the search already warned about its custom formats
        warnings.simplefilter("ignore")
        formats, search_spaces, fixed = _process_search_space(
            inputs["quantization_formats"],
            inputs["module_search_spaces"],
            inputs["fixed_quantization_config"],
        )
    searcher = AUTO_QUANTIZE_SEARCHERS[primary.auto_quantize_method]()
    searcher.config = {
        **searcher.default_search_config,
        "quantization_formats": formats,
        "module_search_spaces": search_spaces,
        "fixed_quantization_config": fixed,
    }
    mismatch = searcher.candidate_mismatch(search_state)
    if mismatch:
        raise ValueError(
            f"{checkpoint} does not hold the recipe's scores: {mismatch} Finalize them with the "
            "recipe that scored them, or search again into another --auto_quantize_checkpoint."
        )
    # Freeze the per-layer formats for this budget, with the recipe's KV-cache config.
    quant_cfg = mtq.get_auto_quantize_config(
        search_state,
        {"effective_bits": primary.constraints.effective_bits},
        algorithm=primary.finalize_algorithm,
    )
    kv_cfg = inputs["kv_cache_quant_cfg"]
    if kv_cfg is not None:
        quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(quant_cfg, _kv_cache_entries(kv_cfg))
    return quant_cfg


def _auto_quantize_finalizes(recipe: ModelOptAutoQuantizeRecipe, args: argparse.Namespace) -> bool:
    """Whether this run finalizes saved scores, or the shards it merges, instead of searching."""
    if (
        recipe.auto_quantize.finalize_algorithm is None
        or args.auto_quantize_format_shard is not None
        or not args.auto_quantize_checkpoint
    ):
        return False
    if args.auto_quantize_merge_shards:
        return True
    checkpoint = _search_checkpoint_file(args.auto_quantize_checkpoint)
    return _scored_auto_quantize_state(checkpoint) is not None


def _kv_cache_entries(kv_cache_quant_cfg: dict) -> list[dict]:
    """A uniform KV-cache config's entries to add after the selected weight formats.

    Its ``*`` entries are left out: there they would reset every weight quantizer.
    """
    return [e for e in copy.deepcopy(kv_cache_quant_cfg["quant_cfg"]) if e["quantizer_name"] != "*"]


def _search_checkpoint_file(checkpoint: str) -> str:
    """The file a one-process search keeps at ``checkpoint``: a directory's ``rank0.pth``.

    An existing file is read as is, even without an extension, as the searcher's load does.
    """
    if os.path.isfile(checkpoint) or not is_directory_checkpoint(checkpoint):
        return checkpoint
    return os.path.join(checkpoint, "rank0.pth")


def _scored_auto_quantize_state(checkpoint: str) -> dict | None:
    """The search state in ``checkpoint`` (a file) if it holds every format's scores."""
    if not os.path.isfile(checkpoint):
        return None
    # Memory-mapped on the CPU: finalizing reads the scores, not every format's calibration.
    state = safe_load(checkpoint, map_location="cpu", mmap=True)
    stats = state.get("candidate_stats") or {}
    if state.get("format_shard") is not None or not stats:
        return None
    return state


def _select_unpadded_logits(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Return logits only for token positions selected by ``attention_mask``."""
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return logits
    if logits.shape[:-1] != attention_mask.shape:
        raise ValueError(
            "AutoQuantize KL logits and attention_mask must have matching token dimensions; "
            f"got {tuple(logits.shape[:-1])} and {tuple(attention_mask.shape)}."
        )
    return logits[attention_mask.bool()]


def _kv_cfg_uses_constant_amax(kv_quant_cfg: list[dict[str, Any]]) -> bool:
    """Return True if this KV cfg pins ``use_constant_amax`` on the bmm quantizer.

    Cast-style KV presets (e.g. ``fp8_cast`` / ``nvfp4_cast``) set
    ``use_constant_amax: true`` on the ``*[kv]_bmm_quantizer`` entry; that flag
    means there is no data-driven calibration to run, so callers should skip
    the KV-only calibration pass. Detect the property from the YAML contents
    rather than from the preset name so new cast-style presets work
    automatically.
    """
    for entry in kv_quant_cfg:
        if entry.get("quantizer_name") != "*[kv]_bmm_quantizer":
            continue
        cfg = entry.get("cfg") or {}
        return bool(cfg.get("use_constant_amax"))
    return False
