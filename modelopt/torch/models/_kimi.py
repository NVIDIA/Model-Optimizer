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

"""What the Kimi-K2.5 and Kimi-VL adapters change in the modeling code their checkpoints ship.

Both checkpoints' own VLM class builds its language model as ``DeepseekV3ForCausalLM(
config.text_config)``, a DeepSeek-V3 whose MoE is an ``nn.ModuleList`` of per-expert Linears that
expert parallelism cannot shard. The adapters point that name at transformers' native class, and
keep the vision tower and projector from the checkpoint's code.
"""

# modelopt.torch.utils and transformers are imported inside the functions: the model registry imports
# this module, and it must import without transformers, the optional [hf] extra.
from ._remote_code import _apply, keep_patched, stage_full_remote_closure


def checkpoint_to_patch(
    label: str, hf_config, ckpt_path: str | None, trust_remote_code: bool
) -> str:
    """Return the checkpoint whose own modeling code the adapter ``label`` patches.

    Refuses first, before its ``prepare`` changes anything, a config the adapter cannot patch.
    """
    if type(hf_config).__module__.startswith("transformers."):
        # transformers' own port (Kimi-K2.5's, from 5.13), which the patch never reaches: a load
        # without trust_remote_code builds it, and so does one of a checkpoint naming no config class.
        raise ValueError(
            f"{label}: expert parallelism runs on the checkpoint's own modeling code, but this "
            f"config is transformers' own port ({type(hf_config).__module__}). Load it with "
            "trust_remote_code=True, from a checkpoint whose auto_map names its own AutoConfig."
        )
    if not hasattr(hf_config, "text_config"):
        raise ValueError(f"{label}: hf_config has no .text_config (expected a VLM config)")
    if ckpt_path is None:
        raise ValueError(
            f"{label}: expert parallelism patches the checkpoint's own modeling code, so it needs "
            "the checkpoint's path (apply_distributed_support's ckpt_path)."
        )
    if not trust_remote_code:
        # Patching imports the checkpoint's own modeling code, and transformers'
        # get_class_from_dynamic_module runs it without checking trust_remote_code itself.
        raise ValueError(
            f"{label}: expert parallelism imports the checkpoint's own modeling code; "
            "load it with trust_remote_code=True."
        )
    return ckpt_path


def refuse_quantization_only_on_text_config(label: str, hf_config) -> None:
    """Refuse a checkpoint whose ``quantization_config`` only its ``text_config`` declares.

    The adapter replaces ``text_config`` with a native config without it, and the expert-parallel
    load would then read the quantized experts as stored. It loads unquantized checkpoints only, so
    the way in is an unquantized copy of the model.
    """
    if (
        getattr(hf_config.text_config, "quantization_config", None) is not None
        and getattr(hf_config, "quantization_config", None) is None
    ):
        raise ValueError(
            f"{label}: the checkpoint declares quantization_config on text_config only, which the "
            "native deepseek_v3 config replacing it drops, so the expert-parallel load would read "
            "its experts as stored. Start from an unquantized copy of the model."
        )


def install_compat_import_shims() -> None:
    """Add stubs for symbols the Kimi modeling imports that newer transformers removed, so it imports.

    Each guards a torch.fx / legacy-attn-mask code path the native LM + MoonViT forward never hit, so a benign stub is
    safe. Idempotent (only sets missing attrs).
    """
    from modelopt.torch.utils import print_rank_0

    try:
        import transformers.utils.import_utils as iu

        if not hasattr(iu, "is_torch_fx_available"):
            iu.is_torch_fx_available = lambda: False
    except Exception as e:
        print_rank_0(f"[kimi] compat-shim warning (import_utils): {type(e).__name__}: {e}")
    try:
        import transformers.pytorch_utils as pu

        if not hasattr(pu, "is_torch_greater_or_equal_than_1_13"):
            pu.is_torch_greater_or_equal_than_1_13 = True
    except Exception as e:
        print_rank_0(f"[kimi] compat-shim warning (pytorch_utils): {type(e).__name__}: {e}")
    try:
        import transformers.modeling_attn_mask_utils as amu

        if not hasattr(amu, "_prepare_4d_causal_attention_mask"):

            def _prepare_4d_causal_attention_mask(*args, **kwargs):  # pragma: no cover
                raise RuntimeError(
                    "kimi: remote _prepare_4d_causal_attention_mask stub called -- the native LM "
                    "builds its own mask, so this path should be unreachable."
                )

            amu._prepare_4d_causal_attention_mask = _prepare_4d_causal_attention_mask
    except Exception as e:
        print_rank_0(f"[kimi] compat-shim warning (attn_mask_utils): {type(e).__name__}: {e}")


def make_importable() -> None:
    """The names newer transformers dropped, and the code's two-deep relative imports staged."""
    install_compat_import_shims()
    # transformers stages only 1-hop relative imports; Kimi's remote code is 2 hops deep.
    stage_full_remote_closure()


def make_buildable(
    modeling: str, vlm_class: str, hf_config, ckpt_path: str | None, trust_remote_code: bool
) -> None:
    """What both checkpoints' own code needs to import and build on this transformers, and no more.

    The names newer transformers dropped, the code's two-deep relative imports staged, and the VLM
    class's ``tie_weights`` taking the keyword arguments transformers' ``post_init`` passes it. None
    of it changes what the model computes or loads, so the plan decision runs it before ``prepare``,
    whatever plan comes out. A config of transformers' own port, or code this load may not run, has
    nothing of the checkpoint's to build. Idempotent.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if type(hf_config).__module__.startswith("transformers.") or ckpt_path is None:
        return
    if not trust_remote_code:
        return
    make_importable()
    get_class_from_dynamic_module(
        f"{modeling}.{vlm_class}", ckpt_path, trust_remote_code=trust_remote_code
    )
    # Remote code only: transformers' own port of a model can share the module's name.
    _apply(f".{modeling}", lambda module: _shim_tie_weights(module, vlm_class))


def _shim_tie_weights(module, vlm_class: str) -> bool:
    """Have ``module``'s VLM class take the keyword arguments transformers passes ``tie_weights``.

    The remote ``tie_weights()`` takes none, but transformers' ``post_init`` and load call
    ``tie_weights(recompute_mapping=False)``: a TypeError at construction. Returns whether it changed
    the class.
    """
    kimi_cls = getattr(module, vlm_class, None)
    if kimi_cls is None:
        return False
    own_tie = vars(kimi_cls).get("tie_weights")
    if own_tie is None or getattr(own_tie, "_kimi_tie_weights_shim", False):
        return False
    kimi_cls.tie_weights = _drop_tie_weights_kwargs(own_tie)
    return True


def _drop_tie_weights_kwargs(orig_tie):
    """Wrap a remote ``tie_weights()`` that predates the kwargs newer transformers pass to it.

    Dropping ``missing_keys`` as well as ``recompute_mapping`` is safe because the remote
    ``tie_weights`` only calls its language model's, and the Kimi checkpoints set
    ``tie_word_embeddings=False``: nothing is tied, so no tied key is left to report to the load. A
    tied one would still load its embedding into both, with ``lm_head.weight`` reported missing.
    """

    def _tie_weights(self, *args, **kwargs):  # drop recompute_mapping etc.
        return orig_tie(self)

    setattr(_tie_weights, "_kimi_tie_weights_shim", True)
    return _tie_weights


def _native_for_the_prepared_config(own_class, native_class):
    """The ``DeepseekV3ForCausalLM`` the patched module builds its language model with.

    transformers' native class for the native ``DeepseekV3Config`` ``prepare`` puts on
    ``text_config``, and the checkpoint's own class for its own config. The patch outlives the load
    it was made for: a later load in the process that its plan keeps off expert parallelism builds
    from the config as read, and so gets the checkpoint's own language model, as it would with no
    patch.
    """
    from transformers import DeepseekV3Config

    def build(config, *args, **kwargs):
        built = native_class if isinstance(config, DeepseekV3Config) else own_class
        return built(config, *args, **kwargs)

    setattr(build, "_kimi_native_swap", True)
    return build


def patch_language_model_to_native(
    label: str, modeling: str, vlm_class: str, ckpt_path: str, trust_remote_code: bool
) -> None:
    """Point the checkpoint's ``modeling`` module's ``DeepseekV3ForCausalLM`` at the native class.

    Loads the checkpoint's remote ``modeling`` module and replaces its ``DeepseekV3ForCausalLM``, so
    ``vlm_class.__init__`` builds the EP-capable native deepseek_v3 for ``self.language_model``
    instead of the per-expert Linears version, from the config ``prepare`` made
    (:func:`_native_for_the_prepared_config`).
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM as NativeDeepseekV3ForCausalLM

    from modelopt.torch.utils import print_rank_0

    # Force the dynamic-module loader to stage + import the module into sys.modules.
    get_class_from_dynamic_module(
        f"{modeling}.{vlm_class}", ckpt_path, trust_remote_code=trust_remote_code
    )

    def to_native(module):
        current = getattr(module, "DeepseekV3ForCausalLM", None)
        if current is None:
            # Not "already native": the VLM class then builds its language model from a class this
            # swap does not reach (renamed, or imported inside __init__), with per-expert Linears.
            raise RuntimeError(
                f"{label}: {module.__name__} defines no module-level DeepseekV3ForCausalLM to swap "
                "for the transformers-native class; the checkpoint's modeling code changed."
            )
        if getattr(current, "_kimi_native_swap", False):
            return False
        module.DeepseekV3ForCausalLM = _native_for_the_prepared_config(
            current, NativeDeepseekV3ForCausalLM
        )
        _shim_tie_weights(module, vlm_class)
        return True

    # The dynamic-module loader names the module like
    # ``transformers_modules.<short-repo>.<hash>.<modeling>``. It stays patched when from_pretrained
    # resolves the class again, even from a re-executed module.
    patched = keep_patched(f".{modeling}", to_native)
    if not patched:
        raise RuntimeError(
            f"{label}: could not find a loaded {modeling} in sys.modules to monkey-patch"
        )
    print_rank_0(
        f"[{label}] monkey-patched DeepseekV3ForCausalLM -> transformers-native (+tie_weights shim) "
        f"in: {patched}"
    )
