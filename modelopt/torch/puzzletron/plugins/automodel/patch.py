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

"""Runtime patch so NeMo AutoModel loads Puzzletron AnyModel checkpoints.

After :func:`apply_patch`,
``NeMoAutoModelForCausalLM.from_pretrained(path, anymodel_descriptor=..., force_hf=...)``
applies ModelOpt's heterogeneous-layer support:

* ``force_hf=True`` (the path implemented here) loads the plain HuggingFace model
  and wraps construction in ``deci_x_patcher`` so each decoder layer picks up its
  per-layer ``BlockConfig``. Works for every registered AnyModel descriptor; no
  expert parallel.
* ``force_hf=False`` (NeMo custom models, e.g. MoE/EP) additionally uses
  ``automodel_patcher`` + an ``AutoModelDescriptor``. Those live in
  ``modelopt.torch.puzzletron.anymodel`` and are added in a later milestone; this
  module degrades gracefully (the custom path is a no-op) until they exist.

The patch also teaches NeMo's HuggingFace storage reader to follow
``model.safetensors.index.json`` for standard sharded checkpoints.

This is a port/hardening of the reference ``automodel_distillation/patch_automodel.py``.
NeMo and ModelOpt-anymodel imports are lazy (inside :func:`apply_patch`) so importing
this module does not require ``nemo_automodel``. Call :func:`apply_patch` before
loading any model; :func:`remove_patch` restores the original state.
"""

import json
import logging
import os
import threading
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["apply_patch", "auto_detect_block_configs", "load_block_configs", "remove_patch"]

_anymodel_ctx = threading.local()


def _get_ctx_stack() -> list:
    if not hasattr(_anymodel_ctx, "stack"):
        _anymodel_ctx.stack = []
    return _anymodel_ctx.stack


def load_block_configs(block_configs_path: str | Path) -> list[dict]:
    """Load a list of per-layer block configs from JSON.

    Accepts either a bare list or a ``{"block_configs": [...]}`` wrapper.
    """
    from ...block_config import maybe_cast_block_configs

    path = Path(block_configs_path)
    if not path.exists():
        raise FileNotFoundError(f"Block configs not found: {path}")
    with open(path) as f:
        out = json.load(f)
    if isinstance(out, dict) and "block_configs" in out:
        out = out["block_configs"]
    out = maybe_cast_block_configs(out)
    logger.info("Loaded %d block configs from %s", len(out), path)
    return out


def auto_detect_block_configs(checkpoint_dir: str | Path) -> list[dict] | None:
    """Return the per-layer block configs from an AnyModel checkpoint, or ``None``.

    Prefers ``<checkpoint_dir>/block_configs.json`` (the reference layout); falls back to
    the top-level ``block_configs`` key in ``config.json`` (where the puzzletron converter
    writes them). Reading these is what makes heterogeneous teachers/students load and score
    correctly — ``deci_x_patcher`` applies them per layer, and the scorer resolver reads the
    per-layer dims from them.
    """
    checkpoint_dir = Path(checkpoint_dir)
    block_configs_path = checkpoint_dir / "block_configs.json"
    if block_configs_path.exists():
        return load_block_configs(block_configs_path)

    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        block_configs = config.get("block_configs")
        if block_configs:
            from ...block_config import maybe_cast_block_configs

            block_configs = maybe_cast_block_configs(block_configs)
            logger.info(
                "Loaded %d block configs from %s (config.json)", len(block_configs), config_path
            )
            return block_configs
    return None


def _native_checkpoint_requires_heterogeneous_adapter(
    checkpoint_dir: str | Path,
    *,
    block_configs_override,
) -> bool:
    """Whether native construction must consume Puzzletron shape overrides.

    Converted teachers retain their original native architecture and their
    block configs only describe the already-serialized shapes. Native
    AutoModel can load those checkpoints unchanged. Realized ``AnyModel``
    checkpoints and explicit runtime block overrides are genuinely
    heterogeneous and still require a registered native adapter.
    """
    if block_configs_override is not None:
        return True
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        return True
    try:
        with open(config_path) as stream:
            architectures = json.load(stream).get("architectures") or []
    except (json.JSONDecodeError, OSError):
        return True
    return "AnyModel" in architectures


def _restore_native_anymodel_architecture(
    checkpoint_dir: str | Path,
    config_override,
):
    """Return an in-memory config override that selects the native base model.

    Realized Puzzletron checkpoints intentionally advertise ``AnyModel`` on
    disk so HuggingFace and inference engines route them through the generic
    heterogeneous loader.  NeMo AutoModel's native-class resolver, however,
    uses ``config.architectures`` before Puzzletron patches decoder-layer
    construction.  Without this transient override, ``force_hf=False`` falls
    through to the stock HF model even when a native descriptor is registered.

    ``base_architecture`` is part of the AnyModel wire format.  Restore it only
    in the config object used for native construction; never rewrite the saved
    checkpoint.  A dict override is preferred because AutoModel deep-merges it
    while loading the complete config from disk.
    """
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.is_file():
        return config_override
    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return config_override

    architectures = config_data.get("architectures") or []
    if "AnyModel" not in architectures:
        return config_override
    base_architecture = config_data.get("base_architecture")
    if not isinstance(base_architecture, str) or not base_architecture:
        raise ValueError(
            f"Native AnyModel loading requires a non-empty base_architecture in {config_path}"
        )

    if config_override is None:
        restored = {"architectures": [base_architecture]}
    elif isinstance(config_override, dict):
        restored = dict(config_override)
        restored["architectures"] = [base_architecture]
    else:
        restored = deepcopy(config_override)
        restored.architectures = [base_architecture]
    logger.info(
        "AnyModel: transiently restoring native base architecture %s from %s",
        base_architecture,
        config_path,
    )
    return restored


def _precache_trust_remote_code_distributed(
    pretrained_model_name_or_path: str | Path,
    *,
    trust_remote_code: bool,
) -> None:
    """Materialize HF dynamic modules once before all ranks import them.

    Hugging Face's dynamic-module cache lock is process-local.  With PP/EP/TP
    torchrun launches, all ranks can race to create/import the same local
    ``transformers_modules/<basename>/<hash>`` package; a rank may import
    ``modeling_*.py`` before another rank has finished copying its relative
    imports such as ``configuration_*.py``.  Rank 0 pre-caches the ``auto_map``
    modules, then every rank imports from the completed cache.
    """
    checkpoint_dir = Path(pretrained_model_name_or_path)
    if not trust_remote_code or not checkpoint_dir.is_dir():
        return

    try:
        import torch.distributed as dist
    except Exception:  # noqa: BLE001
        dist = None

    distributed = bool(dist is not None and dist.is_available() and dist.is_initialized())
    rank = dist.get_rank() if distributed else 0
    error: str | None = None

    if rank == 0:
        try:
            from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config

            load_model_config(checkpoint_dir, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"

    if distributed:
        payload = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]
        if error is None:
            dist.barrier()

    if error is not None:
        raise RuntimeError(
            f"Failed to pre-cache trust_remote_code modules for {checkpoint_dir}: {error}"
        )


def _patch_get_init_context(model_init) -> None:
    """Re-wrap ``PreTrainedModel.get_init_context`` to tolerate newer transformers.

    NeMo's ``_patched_get_init_context`` does not forward extra kwargs (e.g.
    ``allow_all_kernels`` added in transformers 5.3). Forward ``**kwargs`` to the
    real original and preserve NeMo's meta-device filtering. Optional/best-effort.
    """
    from transformers import PreTrainedModel

    def _fixed_get_init_context(cls, *args, **kwargs):
        contexts = model_init._patched_get_init_context.__wrapped__(cls, *args, **kwargs)
        if model_init._get_hf_meta_device_disabled():
            return model_init._filter_meta_device_from_init_context(contexts)
        return contexts

    PreTrainedModel.get_init_context = classmethod(_fixed_get_init_context)


def _patch_nemotron_h_backbone_alias() -> None:
    """Let NeMo's NemotronH parallelizer handle HF remote-code checkpoints.

    The current HF NemotronH trust-remote-code class exposes the decoder as
    ``model`` while NeMo's optimized NemotronH parallelization strategy expects
    ``backbone``.  The model's own source declares conversion mappings between
    the two names, so expose ``backbone`` as a transient, non-registered alias
    before NeMo parallelizes.  ``object.__setattr__`` avoids adding a duplicate
    child module to ``nn.Module._modules``.
    """
    import torch.nn as nn
    from nemo_automodel.components.distributed import parallelizer

    strategy_cls = getattr(parallelizer, "NemotronHParallelizationStrategy", None)
    if strategy_cls is None or getattr(strategy_cls, "_puzzletron_backbone_alias_patch", False):
        return

    orig_parallelize = strategy_cls.parallelize

    class _ModuleDictLayerList(nn.ModuleList):
        def __init__(self, module_dict):
            object.__setattr__(self, "_puzzletron_module_dict", module_dict)
            object.__setattr__(self, "_puzzletron_keys", list(module_dict.keys()))
            super().__init__(list(module_dict.values()))

        def __setitem__(self, idx, module):
            super().__setitem__(idx, module)
            key = self._puzzletron_keys[idx]
            self._puzzletron_module_dict[key] = module

    class _NemotronHBackboneAlias:
        def __init__(self, inner):
            self._inner = inner

        @property
        def layers(self):
            layers = getattr(self._inner, "layers")
            if isinstance(layers, nn.ModuleDict):
                return _ModuleDictLayerList(layers)
            if isinstance(layers, dict):
                return nn.ModuleList(list(layers.values()))
            return layers

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _parallelize_with_backbone_alias(self, model, *args, **kwargs):
        if (
            model.__class__.__name__ == "NemotronHForCausalLM"
            and "backbone" not in getattr(model, "_modules", {})
            and hasattr(model, "model")
        ):
            existing = getattr(model, "backbone", None)
            if existing is None or existing is getattr(model, "model"):
                alias = _NemotronHBackboneAlias(model.model)
                object.__setattr__(model, "backbone", alias)
                layers = alias.layers
                try:
                    first_layer = next(iter(layers))
                except StopIteration:
                    first_layer = None
                logger.info(
                    "AnyModel: added transient NemotronH backbone -> model alias "
                    "(layers=%s, len=%s, first=%s)",
                    type(layers).__name__,
                    len(layers) if hasattr(layers, "__len__") else "unknown",
                    type(first_layer).__name__ if first_layer is not None else None,
                )
        return orig_parallelize(self, model, *args, **kwargs)

    strategy_cls.parallelize = _parallelize_with_backbone_alias
    strategy_cls._puzzletron_backbone_alias_patch = True


def _patch_native_nemotron_stage_local_initialization() -> None:
    """Make native Nemotron3 weight init tolerate pipeline-pruned stages.

    NeMo initializes model parameters after the PP splitter has replaced modules outside the
    local stage with ``None``.  The native Nemotron3 initializer assumes a full model and
    unconditionally touches ``embed_tokens``, ``norm``, and ``lm_head``.  For Puzzletron's
    force_hf=False/EP path we only need initialization to run for modules present on this PP
    stage before checkpoint loading fills real weights, so absent modules are skipped.
    """
    import torch
    import torch.nn as nn
    from nemo_automodel.components.models.nemotron_v3 import model as nemotron_v3_model

    model_cls = getattr(nemotron_v3_model, "NemotronV3Model", None)
    lm_cls = getattr(nemotron_v3_model, "NemotronHForCausalLM", None)
    if model_cls is None or lm_cls is None:
        return
    if getattr(model_cls, "_puzzletron_stage_local_init_patch", False):
        return

    orig_model_init = model_cls.initialize_weights
    orig_lm_init = lm_cls.initialize_weights

    @torch.no_grad()
    def _stage_local_model_initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        device_ctx = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with device_ctx:
            embed_tokens = getattr(self, "embed_tokens", None)
            if embed_tokens is not None:
                nn.init.normal_(
                    embed_tokens.weight,
                    mean=0.0,
                    std=self.config.initializer_range,
                )
            norm = getattr(self, "norm", None)
            if norm is not None:
                norm.reset_parameters()

        layers = getattr(self, "layers", None)
        values = layers.values() if hasattr(layers, "values") else []
        for block in values:
            if block is not None and hasattr(block, "init_weights"):
                block.init_weights(buffer_device=device_ctx)

    @torch.no_grad()
    def _stage_local_lm_initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        device_ctx = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "initialize_weights"):
            model.initialize_weights(buffer_device=device_ctx)
        lm_head = getattr(self, "lm_head", None)
        if lm_head is not None:
            with device_ctx:
                nn.init.normal_(lm_head.weight, mean=0.0, std=self.config.initializer_range)
        _cast_stage_local_model_to_dtype(self, dtype)

    model_cls._puzzletron_orig_initialize_weights = orig_model_init
    lm_cls._puzzletron_orig_initialize_weights = orig_lm_init
    model_cls.initialize_weights = _stage_local_model_initialize_weights
    lm_cls.initialize_weights = _stage_local_lm_initialize_weights
    model_cls._puzzletron_stage_local_init_patch = True
    lm_cls._puzzletron_stage_local_init_patch = True


def _cast_stage_local_model_to_dtype(model, dtype) -> None:
    """Cast a stage-local Nemotron model while preserving its intrinsic FP32 parameters."""
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype

    cast_model_to_dtype(model, dtype, skip_modules=("_fp32_params",))


def _patch_hf_storage_reader(auto_model) -> None:
    """Teach NeMo's HF storage reader to find shards via ``model.safetensors.index.json``.

    The original ``read_metadata`` does a flat ``fs.ls()`` on the checkpoint root.
    When an index file is present we feed the reader its authoritative shard list.
    """
    from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader

    orig_read_metadata = _HuggingFaceStorageReader.read_metadata

    def _patched_read_metadata(self):
        index_path = os.path.join(self.path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            return orig_read_metadata(self)
        with open(index_path) as f:
            index = json.load(f)
        shard_files = [os.path.join(self.path, rel) for rel in set(index["weight_map"].values())]
        logger.debug(
            "AnyModel patch: using model.safetensors.index.json to discover %d shard(s) in %s",
            len(shard_files),
            self.path,
        )
        orig_fs_ls = self.fs.ls

        def _ls_override(path, **kwargs):
            if os.path.normpath(path) == os.path.normpath(self.path):
                return shard_files
            return orig_fs_ls(path, **kwargs)

        self.fs.ls = _ls_override
        try:
            return orig_read_metadata(self)
        finally:
            self.fs.ls = orig_fs_ls

    _HuggingFaceStorageReader.read_metadata = _patched_read_metadata
    auto_model._anymodel_orig_read_metadata = orig_read_metadata
    auto_model._anymodel_hf_storage_reader_cls = _HuggingFaceStorageReader


def _patch_stage_local_hf_load(auto_model) -> None:
    """Use range-based HF safetensor reads for PP-local stages.

    NeMo's dtype-preserving init shortcut loads a full HF state dict whenever
    ``len(model_state.model) == 1``.  After pipeline splitting, each rank also
    owns exactly one model object, but that object is only a stage-local slice
    (``model.layers`` is a ``ModuleDict`` subset).  Loading the full checkpoint
    there defeats meta-device initialization and can mmap every 120B shard on
    every PP rank.  For stage-local models, keep the bf16 meta skeleton and let
    DCP/HF storage read only the tensor ranges requested by that stage.
    """
    import nemo_automodel.components.checkpoint.checkpointing as checkpointing
    import torch.nn as nn

    if getattr(checkpointing.Checkpointer, "_puzzletron_stage_local_load_patch", False):
        return

    orig_load_model = checkpointing.Checkpointer.load_model

    def _layer_container(model):
        for owner_name in ("model", "backbone"):
            owner = getattr(model, owner_name, None)
            layers = getattr(owner, "layers", None) if owner is not None else None
            if layers is not None:
                return layers
        return getattr(model, "layers", None)

    def _is_pp_stage_local_model(model) -> bool:
        config = getattr(model, "config", None)
        expected_layers = getattr(config, "num_hidden_layers", None)
        if expected_layers is None:
            return False
        layers = _layer_container(model)
        if layers is None or not hasattr(layers, "__len__"):
            return False
        if len(layers) >= int(expected_layers):
            return False
        # NeMo's PP splitter represents kept layer subsets as ModuleDicts.  We
        # also accept any shorter layer container to cover future splitters.
        return isinstance(layers, (nn.ModuleDict, nn.ModuleList, list, tuple))

    def _patched_load_model(
        self,
        model,
        model_path,
        is_init_step=False,
        use_checkpoint_id=True,
        key_mapping=None,
    ):
        primary = model[0] if isinstance(model, (list, tuple)) and model else model
        is_hf_checkpoint = checkpointing._is_safetensors_checkpoint(
            model_path
        ) or checkpointing._is_bin_checkpoint(model_path)
        if is_init_step and is_hf_checkpoint and _is_pp_stage_local_model(primary):
            if key_mapping is None:
                model_type = getattr(getattr(primary, "config", None), "model_type", None)
                model_key_mapping = getattr(primary, "_checkpoint_conversion_mapping", None)
                get_mapping = getattr(checkpointing, "get_combined_key_mapping", None)
                if callable(get_mapping):
                    key_mapping = get_mapping(model_type, model_key_mapping)
                if key_mapping is None and model_type == "nemotron_h":
                    key_mapping = {r"^backbone": "model"}
            orig_is_safetensors = checkpointing._is_safetensors_checkpoint
            orig_is_bin = checkpointing._is_bin_checkpoint
            checkpointing._is_safetensors_checkpoint = lambda path: False
            checkpointing._is_bin_checkpoint = lambda path: False
            logger.info(
                "AnyModel: using stage-local HF checkpoint load for PP split "
                "(layers=%s/%s, key_mapping=%s, path=%s)",
                len(_layer_container(primary)),
                getattr(primary.config, "num_hidden_layers", "unknown"),
                key_mapping,
                model_path,
            )
            try:
                return orig_load_model(
                    self,
                    model,
                    model_path,
                    is_init_step=is_init_step,
                    use_checkpoint_id=use_checkpoint_id,
                    key_mapping=key_mapping,
                )
            finally:
                checkpointing._is_safetensors_checkpoint = orig_is_safetensors
                checkpointing._is_bin_checkpoint = orig_is_bin

        return orig_load_model(
            self,
            model,
            model_path,
            is_init_step=is_init_step,
            use_checkpoint_id=use_checkpoint_id,
            key_mapping=key_mapping,
        )

    checkpointing.Checkpointer.load_model = _patched_load_model
    checkpointing.Checkpointer._puzzletron_stage_local_load_patch = True
    auto_model._anymodel_checkpointing_mod = checkpointing
    auto_model._anymodel_orig_checkpointer_load_model = orig_load_model


def _patch_moe_apply_bias(auto_model) -> None:
    """Replace NeMo MoE ``_apply_bias`` with a compile-safe vectorized version.

    The original calls ``tokens_per_expert.tolist()``, which forces per-expert
    ``.item()`` calls under ``torch.compile`` and can blow past the NCCL timeout.
    ``@torch.compiler.disable`` makes it an opaque eager op. Optional/best-effort.
    """
    import nemo_automodel.components.moe.experts as moe_experts
    import torch

    @torch.compiler.disable
    def _apply_bias_vectorized(value, bias, tokens_per_expert, permuted_probs=None):
        if bias is None:
            return value
        shape = value.shape
        view = value.view(-1, shape[-1])
        n_experts = bias.shape[0]
        expert_ids = torch.repeat_interleave(
            torch.arange(n_experts, device=view.device, dtype=torch.long),
            tokens_per_expert.to(device=view.device),
        )
        bias_per_token = bias[expert_ids]
        if permuted_probs is not None:
            result = view + bias_per_token * permuted_probs
        else:
            result = view + bias_per_token
        return result.view(shape).to(value.dtype)

    auto_model._anymodel_orig_apply_bias = moe_experts._apply_bias
    auto_model._anymodel_moe_experts_mod = moe_experts
    moe_experts._apply_bias = _apply_bias_vectorized


def _patch_causal_mask_kwarg(auto_model) -> None:
    """Reconcile the mask-builder kwargs NeMo passes with this transformers' signature.

    NeMo's pipeline-parallel forward has an on-the-fly causal-mask fallback
    (``pipelining/hf_utils.py``) that calls ``create_causal_mask(**mask_kwargs)`` with a kwarg
    set frozen for an older transformers: it passes ``input_embeds`` (renamed to
    ``inputs_embeds``) and ``cache_position`` (now "deprecated and unused" and dropped from the
    signature). The fallback fires whenever the dataloader does not precompute
    ``causal_mask_mapping`` (ours does not), so the PP path dies with ``TypeError: ...
    unexpected keyword argument``. NeMo imports the builder *inside* the forward, so wrapping
    the module attribute is picked up at call time.

    The wrapper renames ``input_embeds`` -> ``inputs_embeds`` then drops any kwarg the real
    function does not accept (unless it takes ``**kwargs``). Correct callers (transformers' own
    modeling code) pass only supported kwargs, so they go through untouched. Optional/best-effort.
    """
    import inspect

    import transformers.masking_utils as masking_utils

    def _wrap(orig):
        params = inspect.signature(orig).parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        allowed = set(params)

        def _compat(*args, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            if not accepts_var_kw:
                dropped = [k for k in kwargs if k not in allowed]
                for k in dropped:
                    kwargs.pop(k)
                if dropped:
                    logger.debug("causal-mask shim dropped unsupported kwarg(s): %s", dropped)
            return orig(*args, **kwargs)

        _compat.__wrapped__ = orig
        return _compat

    saved = {}
    for fn_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        orig = getattr(masking_utils, fn_name, None)
        if orig is None or getattr(orig, "__wrapped__", None) is not None:
            continue  # missing or already wrapped (idempotent)
        saved[fn_name] = orig
        setattr(masking_utils, fn_name, _wrap(orig))

    auto_model._anymodel_masking_utils_mod = masking_utils
    auto_model._anymodel_orig_mask_fns = saved


def apply_patch() -> None:
    """Patch ``nemo_automodel`` to load Puzzletron AnyModel checkpoints. Idempotent."""
    import nemo_automodel._transformers.auto_model as auto_model
    import nemo_automodel._transformers.model_init as model_init

    if getattr(auto_model, "_anymodel_patch_applied", False):
        logger.debug("AnyModel patch already applied")
        return

    # Optional, version-fragile transformers compat shim — never fatal.
    try:
        _patch_get_init_context(model_init)
    except Exception as e:  # noqa: BLE001
        logger.warning("AnyModel patch: skipping get_init_context shim (%s)", e)

    try:
        _patch_nemotron_h_backbone_alias()
    except Exception as e:  # noqa: BLE001
        logger.warning("AnyModel patch: skipping NemotronH backbone alias shim (%s)", e)

    try:
        _patch_native_nemotron_stage_local_initialization()
    except Exception as e:  # noqa: BLE001
        logger.warning("AnyModel patch: skipping native Nemotron stage-local init shim (%s)", e)

    # deci_x_patcher + ModelDescriptorFactory are required (force_hf=True path).
    from modelopt.torch.puzzletron.anymodel import ModelDescriptorFactory, deci_x_patcher

    # AutoModelDescriptorFactory + automodel_patcher power the force_hf=False
    # (NeMo custom model) path; they are added in a later milestone. Optional here.
    try:
        from modelopt.torch.puzzletron.anymodel import AutoModelDescriptorFactory, automodel_patcher
    except ImportError:
        AutoModelDescriptorFactory, automodel_patcher = None, None

    orig_init_model = auto_model._init_model
    orig_from_pretrained = auto_model._BaseNeMoAutoModelClass.from_pretrained.__func__

    def _patched_init_model(cls, *model_args, **kwargs):
        stack = _get_ctx_stack()
        block_configs, anymodel_descriptor = stack[-1] if stack else (None, None)
        active_descriptor = None

        with ExitStack() as es:
            if block_configs is not None and anymodel_descriptor is not None:
                hf_descriptor = ModelDescriptorFactory.get(anymodel_descriptor)
                if hf_descriptor is not None:
                    es.enter_context(
                        deci_x_patcher(model_descriptor=hf_descriptor, block_configs=block_configs)
                    )
                    logger.info(
                        "AnyModel: deci_x_patcher with %d block configs (descriptor=%s)",
                        len(block_configs),
                        anymodel_descriptor,
                    )

                automodel_descriptor_cls = (
                    AutoModelDescriptorFactory.get(anymodel_descriptor)
                    if AutoModelDescriptorFactory is not None
                    else None
                )
                if automodel_descriptor_cls is not None:
                    active_descriptor = automodel_descriptor_cls()
                    es.enter_context(
                        automodel_patcher(descriptor=active_descriptor, block_configs=block_configs)
                    )
                    logger.info(
                        "AnyModel: automodel_patcher with %d block configs (descriptor=%s)",
                        len(block_configs),
                        anymodel_descriptor,
                    )

                if hf_descriptor is None and automodel_descriptor_cls is None:
                    logger.warning(
                        "anymodel_descriptor=%r not found in ModelDescriptorFactory or "
                        "AutoModelDescriptorFactory; no AnyModel patching applied",
                        anymodel_descriptor,
                    )

            result = orig_init_model(cls, *model_args, **kwargs)

        # Custom-model (force_hf=False) post-init hooks; no-op for the HF path.
        if active_descriptor is not None and isinstance(result, (tuple, list)) and len(result) == 2:
            is_custom_model, model = result
            if is_custom_model:
                active_descriptor.patch_state_dict_adapter(model)
            else:
                active_descriptor.patch_hf_model_checkpoint_mapping(model)

        return result

    def _patched_from_pretrained_impl(cls, *args, **kwargs):
        kwargs = dict(kwargs)
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)
        anymodel_descriptor = kwargs.pop("anymodel_descriptor", None)
        block_configs_path = kwargs.pop("block_configs_path", None)
        block_configs_override = kwargs.pop("block_configs", None)
        if args:
            pretrained_model_name_or_path = pretrained_model_name_or_path or args[0]
            model_args = args[1:]
        else:
            model_args = ()
        if pretrained_model_name_or_path is None:
            raise TypeError(
                "from_pretrained() missing 1 required argument: 'pretrained_model_name_or_path'"
            )
        if isinstance(pretrained_model_name_or_path, type):
            raise TypeError(
                "pretrained_model_name_or_path must be a path (str or PathLike), got a type. "
                "Ensure model.pretrained_model_name_or_path is the checkpoint path."
            )

        block_configs = None
        if anymodel_descriptor is not None:
            if block_configs_override is not None:
                block_configs = block_configs_override
            elif block_configs_path is not None:
                block_configs = load_block_configs(block_configs_path)
            elif Path(pretrained_model_name_or_path).is_dir():
                block_configs = auto_detect_block_configs(pretrained_model_name_or_path)
                if block_configs:
                    logger.info(
                        "Auto-detected %d block configs from %s/block_configs.json",
                        len(block_configs),
                        pretrained_model_name_or_path,
                    )

        if block_configs is not None:
            from ...block_config import maybe_cast_block_configs

            block_configs = maybe_cast_block_configs(block_configs)
            # ``force_hf=False`` means "prefer native" in AutoModel. A native
            # architecture can only consume heterogeneous Puzzletron block
            # configs when a matching AutoModelDescriptor exists; otherwise
            # select AutoModel's HF path, where the regular ModelDescriptor
            # patch is available. Homogeneous models (no block configs) keep
            # the caller's native preference unchanged.
            native_descriptor = (
                AutoModelDescriptorFactory.get(anymodel_descriptor)
                if AutoModelDescriptorFactory is not None
                else None
            )
            if kwargs.get("force_hf", False) is False and native_descriptor is not None:
                native_config_override = _restore_native_anymodel_architecture(
                    pretrained_model_name_or_path,
                    kwargs.get("config"),
                )
                if native_config_override is not None:
                    kwargs["config"] = native_config_override
            if kwargs.get("force_hf", False) is False and native_descriptor is None:
                requires_adapter = _native_checkpoint_requires_heterogeneous_adapter(
                    pretrained_model_name_or_path,
                    block_configs_override=block_configs_override,
                )
                if requires_adapter:
                    kwargs["force_hf"] = True
                    logger.info(
                        "AnyModel: falling back to HF for descriptor=%s because the "
                        "heterogeneous checkpoint has no native adapter",
                        anymodel_descriptor,
                    )
                else:
                    logger.info(
                        "AnyModel: retaining native AutoModel for canonical teacher "
                        "descriptor=%s; block configs match serialized base shapes",
                        anymodel_descriptor,
                    )

        _precache_trust_remote_code_distributed(
            pretrained_model_name_or_path,
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
        )

        stack = _get_ctx_stack()
        stack.append((block_configs, anymodel_descriptor))
        adapter_context = nullcontext()
        if (
            block_configs is not None
            and anymodel_descriptor is not None
            and AutoModelDescriptorFactory is not None
        ):
            adapter_descriptor_cls = AutoModelDescriptorFactory.get(anymodel_descriptor)
            if adapter_descriptor_cls is not None:
                adapter_context = adapter_descriptor_cls.native_state_dict_adapter_context(
                    block_configs
                )
        try:
            # Adapter conversion happens after bare model construction, while
            # AutoModel applies distributed infrastructure and loads shards.
            # Keep the heterogeneous bridge active for the entire load rather
            # than only for decoder-layer __init__.
            with adapter_context:
                return orig_from_pretrained(
                    cls,
                    pretrained_model_name_or_path,
                    *model_args,
                    **kwargs,
                )
        finally:
            stack.pop()

    # Required: discover indexed safetensor shards.
    _patch_hf_storage_reader(auto_model)
    _patch_stage_local_hf_load(auto_model)

    auto_model._init_model = _patched_init_model
    # Install as a classmethod (NOT a descriptor returning a fresh functools.partial):
    # build_model gates the efficient meta-init + mesh-sharded load on
    # ``cfg.model._target_ in (NeMoAutoModelForCausalLM.from_pretrained, ...)``. A fresh
    # partial per attribute access never compares equal there, so the model would be built
    # dense on the real device (no sharding) and OOM. A classmethod yields a stable bound
    # method that compares equal across accesses, so the check passes.
    auto_model._BaseNeMoAutoModelClass.from_pretrained = classmethod(_patched_from_pretrained_impl)
    auto_model._anymodel_orig_init_model = orig_init_model
    auto_model._anymodel_orig_from_pretrained = orig_from_pretrained

    # Optional MoE compile-safety patch — never fatal.
    try:
        _patch_moe_apply_bias(auto_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("AnyModel patch: skipping MoE _apply_bias patch (%s)", e)

    # Optional PP causal-mask kwarg compat shim — never fatal.
    try:
        _patch_causal_mask_kwarg(auto_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("AnyModel patch: skipping causal-mask kwarg shim (%s)", e)

    auto_model._anymodel_patch_applied = True
    logger.info("Applied AnyModel patch to nemo_automodel._transformers.auto_model")


def remove_patch() -> None:
    """Restore ``nemo_automodel`` to its original state. Safe if not applied."""
    import nemo_automodel._transformers.auto_model as auto_model

    if not getattr(auto_model, "_anymodel_patch_applied", False):
        logger.debug("AnyModel patch was not applied")
        return

    auto_model._init_model = auto_model._anymodel_orig_init_model
    # Restore as a classmethod so ``cls`` is passed automatically.
    auto_model._BaseNeMoAutoModelClass.from_pretrained = classmethod(
        auto_model._anymodel_orig_from_pretrained
    )
    if hasattr(auto_model, "_anymodel_hf_storage_reader_cls"):
        auto_model._anymodel_hf_storage_reader_cls.read_metadata = (
            auto_model._anymodel_orig_read_metadata
        )
        del auto_model._anymodel_hf_storage_reader_cls
        del auto_model._anymodel_orig_read_metadata
    if hasattr(auto_model, "_anymodel_checkpointing_mod"):
        auto_model._anymodel_checkpointing_mod.Checkpointer.load_model = (
            auto_model._anymodel_orig_checkpointer_load_model
        )
        auto_model._anymodel_checkpointing_mod.Checkpointer._puzzletron_stage_local_load_patch = False
        del auto_model._anymodel_checkpointing_mod
        del auto_model._anymodel_orig_checkpointer_load_model
    if hasattr(auto_model, "_anymodel_moe_experts_mod"):
        auto_model._anymodel_moe_experts_mod._apply_bias = auto_model._anymodel_orig_apply_bias
        del auto_model._anymodel_moe_experts_mod
        del auto_model._anymodel_orig_apply_bias
    if hasattr(auto_model, "_anymodel_masking_utils_mod"):
        for fn_name, orig in auto_model._anymodel_orig_mask_fns.items():
            setattr(auto_model._anymodel_masking_utils_mod, fn_name, orig)
        del auto_model._anymodel_masking_utils_mod
        del auto_model._anymodel_orig_mask_fns
    del auto_model._anymodel_orig_init_model
    del auto_model._anymodel_orig_from_pretrained
    auto_model._anymodel_patch_applied = False
    logger.info("Removed AnyModel patch from nemo_automodel._transformers.auto_model")
