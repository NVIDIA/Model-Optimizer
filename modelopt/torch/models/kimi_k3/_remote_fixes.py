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

"""What Kimi-K3's own remote modeling needs to run on current transformers and fla-core.

Each patch applies in place and is idempotent: the KDA (linear-attention) layers' ``A_log``
allocation and their ``sigmoid(beta)``, which fla-core 0.5 skips; the causal-mask and cache
signatures transformers 5.13 and 5.15 changed; the ``tie_weights`` keyword arguments and the
``OutputRecorder`` import newer transformers dropped. Most take the module the checkpoint's remote
code was imported as. An adapter that loads the checkpoint's own code applies them before it builds
the model.
"""

import sys

import torch
import torch.nn as nn


def _install_output_recorder_shim() -> None:
    """Restore the generic-utils symbol the remote modeling imports.

    The remote ``modeling_kimi`` does ``from transformers.utils.generic import OutputRecorder`` (used
    once, as output-recording metadata), but that symbol moved to ``transformers.modeling_utils`` in
    recent transformers. Re-export it into ``utils.generic`` (or inject a benign stub) BEFORE the remote
    modeling imports so the import succeeds. Idempotent.
    """
    import transformers.utils.generic as _g

    if hasattr(_g, "OutputRecorder"):
        return
    found = None
    for _p in ("transformers.modeling_utils", "transformers.utils"):
        try:
            _m = __import__(_p, fromlist=["OutputRecorder"])
            if hasattr(_m, "OutputRecorder"):
                found = _m.OutputRecorder
                break
        except Exception:
            pass
    if found is None:

        class OutputRecorder:  # minimal stub: satisfies the import + the metadata use only
            def __init__(self, *a, **k):
                self.args, self.kwargs = a, k

        found = OutputRecorder
    _g.OutputRecorder = found


def _find_module_defining(class_name: str, package: str):
    """Return the loaded module of ``package`` that DEFINES a class named ``class_name``, or None.

    ``package`` is where one checkpoint's remote code was imported (``transformers_modules.<repo>...``);
    another revision loaded in the same process has a package of its own, whose classes are not this
    checkpoint's to patch.

    Identify the remote module by the CLASS it defines, NOT by its file name -- the plugin depends only on
    the class names it patches, never on the remote code's file layout, so this survives remote-file
    renames (e.g. ``modeling_kimi`` vs ``modeling_kimi_linear``). transformers
    owns parsing + importing the remote files (via ``trust_remote_code`` + ``auto_map``); we only locate
    the already-imported module. ``__module__ == module.__name__`` requires the class to be DEFINED here
    (not merely imported), which uniquely picks the defining module (e.g. the backbone that defines
    ``KimiSparseMoeBlock``, not the VLM wrapper that may import it). Guarded with try/except because
    iterating ``sys.modules`` can touch lazy/magic modules -- notably ``torch.classes``, whose
    ``__getattr__`` returns a namespace proxy for ANY name and raises on use (the proxy is not a ``type``,
    so it is rejected regardless).
    """
    for name, mod in list(sys.modules.items()):
        if mod is None or name.rpartition(".")[0] != package:
            continue
        try:
            obj = getattr(mod, class_name, None)
        except Exception:
            continue
        if isinstance(obj, type) and getattr(obj, "__module__", None) == getattr(
            mod, "__name__", None
        ):
            return mod
    return None


def _patch_kda_a_log(kmod) -> None:
    """Fix a stale-HF-port bug in the KDA (linear-attention) layers.

    ``KimiDeltaAttention.__init__`` allocates ``A_log`` as ``empty(num_heads)`` (96), but the checkpoint stores it as
    ``[head_dim]`` (128 = ``linear_attn_config['head_dim']``). ``from_pretrained`` therefore raises a size mismatch on
    ``layers.*.self_attn.A_log`` (and ``ignore_mismatched_sizes=True`` would REINIT the trained decay -> garbage). The
    forward passes ``self.A_log`` straight to FLA's ``chunk_kda`` / ``fused_recurrent_kda`` (co-designed with this
    checkpoint), so the ONLY bug is the allocation shape. Re-allocate ``A_log`` as ``[head_dim]`` right after the
    original __init__ so the trained values load into the right shape; ``torch.empty`` is fine (overwritten by the
    checkpoint on load). Idempotent.
    """
    from modelopt.torch.utils import print_rank_0

    KDA = getattr(kmod, "KimiDeltaAttention", None)  # noqa: N806 (a class, keeping its own name)
    if KDA is None or getattr(KDA, "_kimi_k3_a_log_patched", False):
        return
    _orig_init = KDA.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        a_log = getattr(self, "A_log", None)
        # Only the stale layout: a 1-D A_log of the wrong length. Any other is left for the load to check.
        if a_log is not None and a_log.ndim == 1 and a_log.shape[0] != self.head_dim:
            self.A_log = nn.Parameter(
                torch.empty(self.head_dim, dtype=a_log.dtype, device=a_log.device)
            )

    KDA.__init__ = _init
    KDA._kimi_k3_a_log_patched = True
    print_rank_0(
        "[kimi_k3] patched KimiDeltaAttention.A_log alloc: [num_heads] -> [head_dim] (checkpoint shape)"
    )


def _patch_kda_beta_sigmoid(kmod) -> None:
    """Apply the KDA gate's sigmoid in torch, and clear the kernel flag that asks for it.

    fla-core 0.5.x silently ignores ``use_beta_sigmoid_in_kernel``, so its ``chunk_kda`` /
    ``fused_recurrent_kda`` use the raw gate, and on trained weights the delta-rule recurrence
    (``|1 - beta*||k||^2| > 1``) diverges to inf, a NaN amax in calibration. Random weights keep
    ``beta`` small enough to hide it. Idempotent.
    """
    import functools
    import inspect

    from modelopt.torch.utils import print_rank_0

    if getattr(kmod, "_kimi_k3_beta_sigmoid_patched", False):
        return
    kernels = [
        f for f in ("chunk_kda", "fused_recurrent_kda") if getattr(kmod, f, None) is not None
    ]
    if not kernels:
        if getattr(kmod, "KimiDeltaAttention", None) is None:
            return  # no KDA layers to patch
        # Unpatched, the divergence would surface only as a NaN amax after the whole calibration.
        raise RuntimeError(
            "kimi_k3: the remote module defining KimiDeltaAttention has no chunk_kda or "
            "fused_recurrent_kda to apply sigmoid(beta) around, so its KDA recurrence would "
            "diverge to inf on trained weights; the checkpoint's remote code is not the one this "
            "adapter patches."
        )
    for fname in kernels:
        orig = getattr(kmod, fname)
        if getattr(orig, "_kimi_k3_beta_wrapped", False):
            continue

        def _make(orig_fn, name):
            # Where beta sits when passed by position. The flag only ever arrives by name: fla-core
            # takes it through **kwargs -- which is exactly why it ignores it -- so it is read off the
            # call's keywords, never off the kernel's signature.
            try:
                params = list(inspect.signature(orig_fn).parameters)
            except (TypeError, ValueError):
                params = []
            beta_pos = params.index("beta") if "beta" in params else None

            # Keeps the kernel's name and signature, which code calling it may inspect.
            @functools.wraps(orig_fn)
            def _wrapped(*args, **kwargs):
                if kwargs.get("use_beta_sigmoid_in_kernel"):
                    pos = None if "beta" in kwargs else beta_pos
                    if pos is not None and pos >= len(args):
                        pos = None
                    beta = args[pos] if pos is not None else kwargs.get("beta")
                    if not isinstance(beta, torch.Tensor):
                        # Passed on, the flag fla-core ignores leaves the raw gate in.
                        raise RuntimeError(
                            f"kimi_k3: {name} was asked to apply sigmoid(beta) in the kernel, which "
                            "fla-core ignores, but this adapter cannot find the beta gate to apply it "
                            f"itself (signature: {params or 'unavailable'}); on the raw gate the KDA "
                            "recurrence diverges to inf on trained weights."
                        )
                    squashed = torch.sigmoid(beta.float()).to(beta.dtype)
                    if pos is None:
                        kwargs["beta"] = squashed
                    else:
                        args = (*args[:pos], squashed, *args[pos + 1 :])
                    kwargs["use_beta_sigmoid_in_kernel"] = False
                return orig_fn(*args, **kwargs)

            setattr(_wrapped, "_kimi_k3_beta_wrapped", True)
            return _wrapped

        setattr(kmod, fname, _make(orig, fname))
    kmod._kimi_k3_beta_sigmoid_patched = True
    print_rank_0(
        "[kimi_k3] patched chunk_kda/fused_recurrent_kda: apply sigmoid(beta) in torch "
        "(fla-core use_beta_sigmoid_in_kernel is a no-op -> KDA state explodes on real weights)"
    )


def _shim_create_causal_mask(kmod) -> None:
    """tf-version compat for the backbone forward's ``create_causal_mask`` call.

    The remote code calls it as ``create_causal_mask(config=, input_embeds=, attention_mask=, cache_position=,
    past_key_values=, position_ids=)``, in older spellings: every transformers this supports (5.5 on) names the second
    ``inputs_embeds``, and tf 5.13's no longer accepts ``cache_position`` -> ``TypeError`` at the first calibration
    forward. The remote module binds ``create_causal_mask`` at import (``from transformers.masking_utils import ...``),
    so wrap that module-level reference: translate ``input_embeds`` -> ``inputs_embeds`` and drop ``cache_position``
    where the installed signature does not take it. Idempotent.
    """
    import functools
    import inspect

    from modelopt.torch.utils import print_rank_0

    orig = getattr(kmod, "create_causal_mask", None)
    if orig is None or getattr(orig, "_kimi_k3_ccm_shim", False):
        return
    try:
        params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        params = {}
    # None: every keyword passes, as for a signature that takes **kwargs.
    accepted = (
        set(params)
        if params and all(p.kind is not inspect.Parameter.VAR_KEYWORD for p in params.values())
        else None
    )

    @functools.wraps(orig)
    def _wrapped(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if accepted is not None and "cache_position" not in accepted:
            # Only this one: any other keyword the installed signature rejects shapes the mask, so
            # dropping it would build a wrong mask silently instead of raising.
            kwargs.pop("cache_position", None)
        return orig(*args, **kwargs)

    setattr(_wrapped, "_kimi_k3_ccm_shim", True)
    kmod.create_causal_mask = _wrapped
    print_rank_0(
        "[kimi_k3] shimmed create_causal_mask (input_embeds->inputs_embeds; drop cache_position; tf>=5.13)"
    )


def _patch_kimi_cache_get_mask_sizes(kmod) -> None:
    """tf-version compat for the KDA hybrid cache.

    tf 5.13's mask preprocessing calls ``cache.get_mask_sizes(q_length: int, layer_idx)`` (q_length =
    inputs_embeds.shape[1], an int), but the remote ``KimiDynamicCache.get_mask_sizes(cache_position, layer_idx)`` does
    ``cache_position.shape[0]`` -- it expects a TENSOR (older tf) -> ``AttributeError: 'int' object has no attribute
    'shape'`` in the calibration forward. Re-implement to accept either an int query_length or a tensor cache_position
    (the body is otherwise trivial: kv_length = query_length + past_seen_tokens, kv_offset = 0).

    tf 5.15's mask preprocessing also asks the cache for ``get_query_offset(layer_idx)``, which the remote
    cache predates (``AttributeError`` once ``generate`` decodes with a cache). Add it when the class has
    none, as transformers' own caches answer it: the tokens already cached. Idempotent.
    """
    from modelopt.torch.utils import print_rank_0

    Cache = getattr(kmod, "KimiDynamicCache", None)  # noqa: N806 (a class, keeping its own name)
    if Cache is None or getattr(Cache, "_kimi_k3_gms_patched", False):
        return

    def _get_mask_sizes(self, query_length, layer_idx):
        ql = query_length.shape[0] if torch.is_tensor(query_length) else int(query_length)
        past_seen_tokens = self.get_seq_length(layer_idx)
        return ql + past_seen_tokens, 0

    def _get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx)

    Cache.get_mask_sizes = _get_mask_sizes
    # The cache's own method is kept; ours replaces an inherited one, as the cache's own answer.
    has_own = "get_query_offset" in vars(Cache)
    if not has_own:
        Cache.get_query_offset = _get_query_offset
    Cache._kimi_k3_gms_patched = True
    added = "" if has_own else " and added get_query_offset (tf>=5.15)"
    print_rank_0(
        f"[kimi_k3] patched KimiDynamicCache.get_mask_sizes (accept int query_length; tf>=5.13){added}"
    )


def _shim_tie_weights_kwargs(package: str) -> None:
    """Absorb the ``recompute_mapping`` kwarg newer transformers passes to ``tie_weights``.

    Transformers 5.13's ``from_pretrained`` calls ``model.tie_weights(recompute_mapping=...)``, but
    Kimi-K3's remote ``KimiK3ForConditionalGeneration.tie_weights(self)`` has the older no-kwarg signature
    (it just delegates to ``self.language_model.tie_weights()``) -> ``TypeError``. Wrap it to swallow the
    extra kwargs (they only concern lm_head/embeddings tying, which the delegate handles). Idempotent.
    (Same class of fix as kimi_vl.)

    Locate the VLM wrapper by the class it DEFINES (``KimiK3ForConditionalGeneration``) via
    ``_find_module_defining`` -- name-independent and safe against the ``torch.classes`` magic module
    (see that helper). The LM ``KimiLinearForCausalLM.tie_weights`` is the native ``PreTrainedModel`` one
    (accepts the kwargs) so it needs no wrap.
    """

    def _make_tie(orig):
        def _tie(self, *args, **kwargs):
            return orig(self)

        setattr(_tie, "_kimi_k3_kwargs_shim", True)
        return _tie

    mod = _find_module_defining("KimiK3ForConditionalGeneration", package)
    if mod is None:
        return
    cls = getattr(mod, "KimiK3ForConditionalGeneration", None)
    if cls is None:
        return
    tw = cls.__dict__.get("tie_weights")  # only if defined ON the class
    if tw is not None and not getattr(tw, "_kimi_k3_kwargs_shim", False):
        cls.tie_weights = _make_tie(tw)
