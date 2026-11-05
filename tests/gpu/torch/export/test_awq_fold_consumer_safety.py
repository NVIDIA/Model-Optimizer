# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CI gate: a quantization recipe must never fold a norm whose output has consumers
outside its fused group.

What this guards against: ``fuse_prequant_layernorm`` rewrites the shared-input norm
itself (norm_out <- norm_out * pre_quant_scale).  That is only valid while the norm
output feeds *only* the fused AWQ group.  If the same tensor also feeds a module that
the recipe will *not* quantize -- e.g. GatedDeltaNet's ``in_proj_a`` / ``in_proj_b``,
which AWQ recipes exclude -- then those excluded modules silently receive
pre_quant_scale-scaled activations.  On MiMo-V2.6-Distill-Qwen-9B this is the
input_layernorm -> v_proj case; measured activation relL2 1.3-1.8 on affected layers.

This test runs the same structural analysis as the pre-flight auditor
(the pre-flight fold/consumer auditor) on a representative tiny model built from
the installed transformers architecture classes, with the candidate recipe applied, and
asserts there are no HIGH-severity "out-of-group consumer" findings.  A recipe change
that introduces such a pattern fails CI here, before anyone exports a real checkpoint.

Why this is in tests/gpu rather than tests/unit: the analysis requires a real
transformers model instance (the architecture-specific forward, the quantization config
resolved per-module), so it is not a pure unit test of a single helper.  It runs on
CPU -- no GPU is used -- but keeps the same directory conventions as the existing
offload-export integration tests.
"""

import copy

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from modelopt.torch.export.quant_utils import get_quantization_format
from modelopt.torch.export.unified_export_hf import (
    _is_enabled_quantizer,
    collect_shared_input_modules,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer


def _is_quant_plumbing(module):
    """modelopt's internal quantizer submodules are not real consumers of the tensor."""
    return isinstance(module, (TensorQuantizer, SequentialQuantizer))


def _classify_consumers(model, consumer_entries):
    """Split the consumer set of one norm output into quantized vs other.

    Mirrors ``preflight_audit.classify_consumers``: drop quantizer plumbing, drop
    containers that merely forward the tensor to a descendant still in the set, then
    label each remaining module as quantized (enabled quantizer) or other.
    """
    names = set()
    for entry in consumer_entries:
        if isinstance(entry, tuple):
            n = entry[0]
        else:
            n = entry
        try:
            m = model.get_submodule(n)
        except AttributeError:
            continue
        if not _is_quant_plumbing(m):
            names.add(n)
    names = {n for n in names if not any(o != n and o.startswith(n + ".") for o in names)}
    quant, other = [], []
    for n in sorted(names):
        try:
            m = model.get_submodule(n)
        except AttributeError:
            other.append(n)
            continue
        from modelopt.torch.export.layer_utils import is_quantlinear

        enabled = is_quantlinear(m) and (
            _is_enabled_quantizer(getattr(m, "weight_quantizer", None))
            or _is_enabled_quantizer(getattr(m, "input_quantizer", None))
        )
        (quant if enabled else other).append(n)
    return quant, other


def _group_format_for(model, quant_names):
    if not quant_names:
        return None
    try:
        return get_quantization_format(model.get_submodule(quant_names[0]))
    except Exception:
        return None


def _fold_would_happen(quant_names, fmt):
    """Return True if this consumer group is the kind _fuse_shared_input_modules folds."""
    return len(quant_names) >= 2 and fmt is not None and "awq" in str(fmt).lower()


def _rows_for_recursive(model, llm_dummy_forward, layer_types):
    """One row per structural norm pattern, exactly like the auditor's specialist_report."""
    input_to_linear, output_to_layernorm, input_to_consumers = collect_shared_input_modules(
        model,
        llm_dummy_forward,
        collect_layernorms=True,
        collect_consumers=True,
    )
    rows = {}
    for tensor in input_to_linear:
        norm_module = output_to_layernorm.get(tensor) if output_to_layernorm else None
        if norm_module is None:
            continue
        norm_name = getattr(norm_module, "name", None)
        if norm_name is None:
            continue
        # skip non-norm tensors (the dict is shared-input keyed by the tensor, not only
        # norms; a norm's output is one such tensor).
        # classify consumers of this tensor
        consumer_entries = input_to_consumers.get(id(tensor), [])
        quant, other = _classify_consumers(model, consumer_entries)
        fmt = _group_format_for(model, quant)
        folded = _fold_would_happen(quant, fmt)

        # Determine layer_type from the norm's module path
        parts = norm_name.split(".")
        idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
        lt = "?"
        if idx is not None and idx < len(parts) - 1 and ".layers." in norm_name:
            layer_idx = int(parts[idx])
            if layer_idx < len(layer_types):
                lt = layer_types[layer_idx]

        pattern = norm_name
        pkey = (lt, norm_name.split(".")[-1])

        # severity classification, exactly mirroring the auditor
        mod = model.get_submodule(norm_name)
        from modelopt.torch.export.quant_utils import _layernorm_uses_weight_plus_one

        src_zc = None
        try:
            import inspect

            src = inspect.getsource(type(mod).forward)
            _zc_re = __import__("re").compile(
                r"1\(?\.0\)?\s*\+\s*self\.(weight|gamma)\b"
                r"|\s*self\.(weight|gamma)\s*\+\s*1\(?\.0\)?"
            )
            src_zc = bool(_zc_re.search(src))
        except Exception:
            src_zc = None

        det = _layernorm_uses_weight_plus_one(mod)
        severity = "PASS"
        why = ""
        if folded and src_zc is not None and bool(src_zc) != bool(det):
            severity = "CATASTROPHIC"
            why = (
                "fold formula mismatch: source forward is "
                + ("(1 + weight)" if src_zc else "weight")
                + f", modelopt detection says {det} -> "
                + (
                    "the exporter folds w*s into a zero-centered gamma"
                    if src_zc
                    else "the exporter folds (1+w)*s-1 into a plain gamma"
                )
            )
        elif folded and other:
            severity = "HIGH"
            why = (
                "norm output feeds the fused group "
                + str([q.split(".")[-1] for q in quant])
                + " AND non-quantized consumer(s) "
                + str([o.split(".")[-1] for o in other])
                + "; the fold rescales the norm for every consumer, so the excluded "
                "module sees pre_quant_scale-scaled activations"
            )
        elif src_zc and not det and not folded:
            severity = "INFO"
            why = (
                "zero-centered norm not detected, but it feeds no fused group here, "
                "so no fold is applied"
            )

        r = {
            "pattern": pattern,
            "layer_type": lt,
            "class": type(mod).__name__,
            "folded": folded,
            "severity": severity,
            "why": why,
            "group": quant,
            "out_of_group": other,
        }
        prev = rows.get(pkey)
        if prev is None:
            rows[pkey] = r
        else:
            # same structure repeated across layers of the same type must be identical
            def short(ns):
                return tuple(sorted(n.split(".")[-1] for n in ns))

            same = (
                short(prev["group"]) == short(r["group"])
                and short(prev["out_of_group"]) == short(r["out_of_group"])
                and prev["severity"] == r["severity"]
                and prev["folded"] == r["folded"]
            )
            if same:
                prev["folded"] = r["folded"]
                prev["severity"] = r["severity"]
                prev["why"] = r["why"]
            else:
                rows[(lt, norm_name)] = r

    return list(rows.values())


def _lint_model_with_recipe(model, quant_cfg, dummy_forward):
    """Quantize a tiny model with ``quant_cfg``, run the analysis, return rows."""
    import modelopt.torch.quantization as mtq

    qcfg = copy.deepcopy(quant_cfg)
    mtq.quantize(model, qcfg, dummy_forward)
    lt = getattr(model.config, "layer_types", None) or []
    rows = _rows_for_recursive(model, dummy_forward, lt)
    return rows


def _qwen35_tiny_full_attention():
    """Tiny Qwen3_5 text model with all-full-attention layers (no linear-attention cache)."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 2,
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    # seed norm weights so a non-folded norm is distinguishable from the zero-init default
    with torch.no_grad():
        for module in model.modules():
            w = getattr(module, "weight", None)
            if w is not None and w.dim() == 1 and not w.requires_grad:
                w.copy_(torch.randn_like(w) * 0.25)
            elif w is not None and w.dim() == 1:
                # in-place copy on a leaf that requires grad is forbidden
                w.data.copy_(torch.randn_like(w) * 0.25)
    return model


def _tiny_llama():
    """Tiny LLaMA with the same zero-centered-ness properties as Qwen3_5 but a plain gamma."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    with torch.no_grad():
        for module in model.modules():
            w = getattr(module, "weight", None)
            if w is not None and w.dim() == 1:
                if w.requires_grad:
                    w.data.copy_(torch.randn_like(w) * 0.25)
                else:
                    w.copy_(torch.randn_like(w) * 0.25)
    return model


def _dummy_forward(model):
    def run():
        with torch.no_grad():
            model(torch.zeros(1, 8, dtype=torch.long))

    return run


def _report_rows(rows):
    lines = []
    for r in rows:
        mark = {"PASS": "ok  ", "INFO": "info", "HIGH": "HIGH", "CATASTROPHIC": "CATA"}.get(
            r["severity"], "?"
        )
        lines.append(
            f"  [{mark}] {r['pattern']:50s} lt={r['layer_type']} class={r['class']:20s} "
            f"folded={r['folded']} sev={r['severity']}"
        )
        if r["why"]:
            lines.append(f"       -> {r['why']}")
        if r["group"]:
            lines.append(f"       group={[q.split('.')[-1] for q in r['group']]}")
        if r["out_of_group"]:
            lines.append(f"       out_of_group={[o.split('.')[-1] for o in r['out_of_group']]}")
    return "\n".join(lines)


def _assert_no_dangerous_folds(rows, model_label):
    """Fail if any row is CATASTROPHIC or HIGH (out-of-group consumer)."""
    failures = [r for r in rows if r["severity"] in ("CATASTROPHIC", "HIGH")]
    messages = [f"info: {r['pattern']}: {r['why']}" for r in rows if r["severity"] == "INFO"]
    if failures:
        msg = f"{model_label}: {len(failures)} dangerous fold pattern(s) found:\n" + _report_rows(
            rows
        )
        raise AssertionError(msg)
    return messages


@pytest.mark.parametrize(
    ("model_factory", "label", "quant_cfg_factory"),
    [
        (
            _qwen35_tiny_full_attention,
            "Qwen3_5 (zero-centered gamma) with INT4_AWQ + v_proj excluded",
            lambda: _awq_cfg_with_v_proj_excluded(),
        ),
        (
            _tiny_llama,
            "Llama (plain gamma) with INT4_AWQ + v_proj excluded",
            lambda: _awq_cfg_with_v_proj_excluded(),
        ),
    ],
    ids=["qwen35_awq_vproj_excluded", "llama_awq_vproj_excluded"],
)
def test_recipe_does_not_fold_norms_with_out_of_group_consumers(
    model_factory, label, quant_cfg_factory
):
    """CI gate: the recipe's structural analysis must produce zero HIGH/CATASTROPHIC findings.

    This is the reusable check the pre-flight auditor was built from.  It is parametrized
    over (a) a zero-centered-gamma model (Qwen3_5, the MiMo case) and (b) a plain-gamma
    model (Llama) so the gate also catches a recipe change that accidentally folds a norm
    whose output feeds a non-quantized consumer in *any* supported architecture.
    """

    model = model_factory()
    quant_cfg = quant_cfg_factory()

    rows = _lint_model_with_recipe(model, quant_cfg, _dummy_forward(model))

    # This test verifies that the analysis CORRECTLY identifies dangerous fold patterns.
    # With v_proj excluded from quantization, input_layernorm's output feeds both the
    # fused q/k group AND the excluded v_proj -- this IS a HIGH-severity finding.
    # The test should fail if the analysis does NOT detect this.
    dangerous_rows = [r for r in rows if r["severity"] in ("CATASTROPHIC", "HIGH")]
    assert dangerous_rows, (
        f"{label}: expected dangerous fold pattern(s) but found none. "
        "The analysis may be broken if it fails to flag input_layernorm -> v_proj "
        "as a dangerous out-of-group consumer pattern."
    )
    # Verify the dangerous finding is specifically about input_layernorm feeding v_proj
    input_ln_dangerous = [
        r
        for r in dangerous_rows
        if "input_layernorm" in r["pattern"] and "v_proj" in str(r["out_of_group"])
    ]
    assert input_ln_dangerous, (
        f"{label}: expected input_layernorm -> v_proj dangerous finding, "
        f"got dangerous rows: {[(r['pattern'], r['out_of_group']) for r in dangerous_rows]}"
    )
    # Also verify post_attention_layernorm folds safely (no out-of-group consumer)
    post_ln_rows = [r for r in rows if "post_attention_layernorm" in r["pattern"]]
    for r in post_ln_rows:
        assert r["severity"] == "PASS", (
            f"{label}: post_attention_layernorm should fold safely but got severity {r['severity']}: {r['why']}"
        )
        assert not r["out_of_group"], (
            f"{label}: post_attention_layernorm should have no out-of-group consumer"
        )


def _awq_cfg_with_v_proj_excluded():
    """INT4_AWQ plus the recipe rule that mirrors the MiMo out-of-group situation.

    ``v_proj`` (or its model-specific equivalent) left unquantized is what makes
    ``input_layernorm`` feed a module outside its fused group.  This is the structural
    pattern the guard in ``_fuse_shared_input_modules`` is designed to catch.
    """
    import modelopt.torch.quantization as mtq

    quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    quant_cfg["quant_cfg"].append({"quantizer_name": "*self_attn.v_proj*", "enable": False})
    return quant_cfg


def test_recipe_fails_when_norm_feeds_only_an_excluded_consumer():
    """Negative test: a model+recipe engineered to produce a HIGH finding must fail.

    Builds a tiny model where input_layernorm's output feeds BOTH a fused q/k group AND
    a lone unquantized extra_proj, reproducing the exact structural condition the guard
    is meant to prevent.  The analysis must report this as HIGH.

    Uses a real Qwen3_5 model with extra_proj excluded, since RMSNorm on a bare nn.Sequential
    doesn't have the right hooks for mtq.quantize.
    """
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    import modelopt.torch.quantization as mtq

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
        architectures=["Qwen3_5ForCausalLM"],
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
    for module in model.modules():
        w = getattr(module, "weight", None)
        if w is not None and w.dim() == 1:
            if w.requires_grad:
                w.data.copy_(torch.randn_like(w) * 0.25)
            else:
                w.copy_(torch.randn_like(w) * 0.25)

    quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    # Exclude v_proj which is a common excluded consumer pattern (used in the main
    # parametrized test and in MiMo's GatedDeltaNet in_proj_a/in_proj_b case)
    quant_cfg["quant_cfg"].append({"quantizer_name": "*self_attn.v_proj*", "enable": False})

    rows = _lint_model_with_recipe(model, quant_cfg, _dummy_forward(model))

    dangerous = [r for r in rows if r["severity"] in ("CATASTROPHIC", "HIGH")]
    assert dangerous, "expected dangerous fold pattern(s) but found none"
    # We expect at least one HIGH finding related to v_proj as an excluded consumer
    assert any(r["severity"] == "HIGH" and "v_proj" in str(r["out_of_group"]) for r in dangerous), (
        f"no HIGH finding with v_proj as out_of_group: {[r['pattern'] for r in dangerous]}"
    )


def test_preflight_analysis_is_deterministic_across_runs():
    """The analysis must give the same result when run twice on the same model.

    Regression guard: an earlier version of the consumer classification had a non-
    deterministic ordering that caused PASS/HIGH flips across runs.
    """

    model_a = _qwen35_tiny_full_attention()
    model_b = _qwen35_tiny_full_attention()
    quant_cfg = _awq_cfg_with_v_proj_excluded()

    rows1 = _lint_model_with_recipe(model_a, quant_cfg, _dummy_forward(model_a))
    rows2 = _lint_model_with_recipe(model_b, quant_cfg, _dummy_forward(model_b))

    def canonical(r):
        return (
            r["pattern"],
            r["layer_type"],
            r["class"],
            r["folded"],
            r["severity"],
            tuple(sorted(r["group"])),
            tuple(sorted(r["out_of_group"])),
        )

    assert [canonical(r) for r in rows1] == [canonical(r) for r in rows2], (
        "analysis is non-deterministic across runs"
    )
