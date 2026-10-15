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

"""Step-family PTQ modeling (HF model types ``step3p5``, ``step3p7``).

The package is named for the first Step release; later revisions share its expert-indexed
``MoELinear`` layout and are matched by :func:`_is_step_family_model`.
"""

import inspect
import re

import torch
import torch.nn as nn
from torch import Tensor

from modelopt.torch.models import hf_model_type
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
from modelopt.torch.quantization.plugins.custom import CUSTOM_MODEL_PLUGINS

__all__: list[str] = []


class _QuantMoELinear(QuantModule):
    """Quantization wrapper for expert-indexed MoELinear modules (fused expert weights).

    MoELinear has weight shape [num_experts, out_features, in_features] with
    forward(x, expert_id). We expand it into per-expert nn.Linear modules so
    each expert gets its own weight_quantizer and input_quantizer, calibrated
    only on tokens actually routed to that expert.

    On export, _reconstruct_fused_moe_linear() stacks the per-expert quantized
    weights and scales back into the original 3D format.

    Note: we use expansion-then-reconstruction rather than the add_module() approach
    because vLLM requires stacked 3D scaling factors; per-expert expanded keys are
    not accepted by the downstream serving engine.
    """

    def _setup(self):
        # Deferred: accelerate is a separate optional dependency, and the HF plugin (which
        # imports this module) must stay importable without it.
        from accelerate import init_empty_weights

        # Accelerate's CPU/disk offload (`device_map="auto"`, `--offload_folder`) leaves
        # `weight` as a meta tensor and keeps the real value in the module's offload hook,
        # keyed on the original `weight` name. Expanding that would copy meta storage into
        # every expert and then delete the key the hook restores into, silently producing a
        # checkpoint of zeros. Refuse instead of corrupting.
        if self.weight.is_meta or getattr(getattr(self, "_hf_hook", None), "offload", False):
            raise NotImplementedError(
                f"{type(self).__name__}: expert-indexed MoELinear weights cannot be quantized "
                "while offloaded by Accelerate (the weight is a meta tensor whose value lives "
                "in the offload hook). Load the model without CPU/disk offload — more GPUs, or "
                "a device_map that keeps the MoE layers resident — and re-run."
            )

        dtype, device = self.weight.dtype, self.weight.device

        with init_empty_weights():
            experts = nn.ModuleList(
                [
                    nn.Linear(self.in_features, self.out_features, bias=False)
                    for _ in range(self.num_experts)
                ]
            )

        for i in range(self.num_experts):
            experts[i].to_empty(device=device)
            with torch.no_grad():
                experts[i].weight.data = self.weight[i].detach().to(dtype=dtype, device=device)

        delattr(self, "weight")
        self.experts = experts

    def forward(self, x, expert_id):
        # experts[expert_id] is a _QuantLinear after quantization wrapping, providing
        # per-expert input_quantizer and weight_quantizer.
        #
        # MoELinear.forward always promotes to fp32 for the matmul regardless of storage
        # dtype (`F.linear(x.float(), self.weight[expert_id].float())`), so leaving the
        # expert's weight at its native storage dtype (e.g. bf16) and downcasting x to
        # match before the matmul would compute in bf16 and change the model's output even
        # with every quantizer disabled -- the bf16 rounding this class exists to quantize
        # *past*, not to reintroduce as a side effect of conversion.
        #
        # A prior version of this fix instead expanded every expert's weight in fp32
        # permanently in `_setup`. That reproduces Step's fp32 compute but turns a per-call
        # transient promotion into persistent model state: on Step-3.7's full routed-expert
        # set (42 layers x 3 projections x 288 experts x 4096 x 1280), doubling from bf16 to
        # fp32 adds roughly 354 GiB held throughout calibration, on top of device placement
        # already sized for bf16 -- a model that loaded successfully can then OOM. It also
        # left disabled/unquantized experts reconstructed at fp32 in the exported checkpoint.
        #
        # Instead, only the one expert actually being called is promoted, transiently, for
        # the duration of this one call -- matching Step's own per-call `.float()` memory
        # profile instead of Step-3.7's full expert set. `expert.weight` is read here
        # outside any `quantize_weight()` context, so `_get_quantized_weight` passes it
        # through unchanged and this is the real underlying nn.Parameter (the same pattern
        # `_setup` above uses), not a value computed by the quantizer -- so reassigning its
        # `.data` genuinely mutates the persisted storage, not a transient wrapper.
        #
        # This must keep calling `expert(x)` (`__call__`, not `.forward()`) rather than
        # reimplementing the input/weight-quantize/output-quantize sequence inline: some
        # calibration algorithms (e.g. `local_hessian_calibrate`) register a
        # `forward_pre_hook` directly on the quantized Linear module, which only fires
        # through standard `nn.Module.__call__` dispatch.
        expert = self.experts[expert_id]
        original_weight = expert.weight.data
        with torch.no_grad():
            expert.weight.data = original_weight.float()
        try:
            out = expert(x.float())
        finally:
            with torch.no_grad():
                expert.weight.data = original_weight
        return out.float()


def _is_expert_indexed_moe_linear(module: nn.Module) -> bool:
    """Whether ``module`` packs one projection's experts into an expert-indexed 3-D weight.

    The Step family (``stepfun-ai/Step-3.5-Flash``, ``stepfun-ai/Step-3.7-Flash``) ships a
    custom ``MoELinear`` via ``trust_remote_code``: a plain ``nn.Module`` holding a single
    ``weight`` of shape ``[num_experts, out_features, in_features]``, whose
    ``forward(x, expert_id)`` runs ``F.linear`` against the selected expert's slice. The
    weights therefore live on the projection submodule rather than on the expert container,
    which is what :func:`_fused_experts_wrapper_class` looks for, and the module is not an
    ``nn.Linear``, so neither the fused-experts path nor the plain linear path claims it.

    Detection is structural rather than keyed on class names so new Step revisions are picked
    up without another hardcoded name, but the shape alone is not a sufficient contract: the
    replacement forward indexes ``self.experts[expert_id]``, so it only works for callers that
    pass a **scalar expert index**. Grouped-GEMM MoE layers share the exact same 3-D weight and
    attribute set while passing a per-expert token-count *tensor* instead (e.g. Moondream3's
    ``MoeFusedLinear.forward(input, m_sizes)``, which would raise ``TypeError: only integer
    tensors of a single element can be converted to an index`` on the first calibration
    forward). The second parameter must therefore be named ``expert_id``, which is the
    scalar-index contract both Step revisions declare.
    """
    weight = getattr(module, "weight", None)
    if not isinstance(weight, (nn.Parameter, Tensor)) or weight.dim() != 3:
        return False
    if not all(hasattr(module, attr) for attr in ("num_experts", "in_features", "out_features")):
        return False
    # The wrapper rebuilds the weight as `num_experts` Linears of (out_features, in_features),
    # so a 3-D weight laid out any other way would silently copy the wrong slices.
    if tuple(weight.shape) != (module.num_experts, module.out_features, module.in_features):
        return False
    try:
        params = list(inspect.signature(type(module).forward).parameters.values())[1:]
    except (TypeError, ValueError):
        return False
    # The replacement forward is exactly `(x, expert_id)`, so anything the caller could pass
    # beyond those two — a keyword-only `router_state`, *args, **kwargs — would raise once
    # converted. Require the signature to match what the wrapper can honour.
    return (
        len(params) == 2
        and all(p.kind is p.POSITIONAL_OR_KEYWORD and p.default is p.empty for p in params)
        and params[1].name == "expert_id"
    )


_STEP_FAMILY_RE = re.compile(r"(?i)^step\d")


def _is_step_family_model(model: nn.Module) -> bool:
    """Whether ``model`` is a Step-family root model (Step-3.5, Step-3.7, or a future revision).

    Matched against the ``step<digit>`` convention shared by ``model_type`` (``"step3p5"``,
    ``"step3p7"``) and the remote-code class name (``Step3p5ForCausalLM``,
    ``Step3p7ForConditionalGeneration``), not an exact revision, so a new Step release is
    still picked up without another hardcoded name. This is deliberately narrower than the
    shape/signature check in :func:`_is_expert_indexed_moe_linear` alone: that check accepts
    any module with a matching 3-D weight and an ``(x, expert_id)`` forward, which is a
    coincidence risk on its own -- an unrelated architecture happening to reuse the parameter
    name ``expert_id`` with different semantics (a per-expert bias or post-scale, say) would
    be claimed and have that behavior silently dropped by the replacement wrapper. Gating on
    the model family keeps the shape check doing what it is actually good at: telling
    Step revisions apart without a class-name allowlist, rather than distinguishing Step
    from arbitrary third-party MoE code.
    """
    model_type = hf_model_type(model) or ""
    return bool(_STEP_FAMILY_RE.match(model_type) or _STEP_FAMILY_RE.match(type(model).__name__))


def register_moe_linear_on_the_fly(model):
    """Register expert-indexed ``MoELinear`` modules (Step-3.5 / Step-3.7) for quantization.

    Without this the routed experts carry no quantizer at all: an experts-only recipe matches
    nothing and the export writes a checkpoint with ``quant_algo: null``.
    """
    if not _is_step_family_model(model):
        return
    visited_types = set()
    for name, module in model.named_modules():
        mod_type = type(module)
        if mod_type in visited_types or QuantModuleRegistry.get(mod_type) is not None:
            continue
        visited_types.add(mod_type)

        if _is_expert_indexed_moe_linear(module):
            print(
                f"\033[1mDetected expert-indexed MoE linear '{name}' of type "
                f"{mod_type.__name__}, registering with _QuantMoELinear.\033[0m"
            )
            QuantModuleRegistry.register({mod_type: f"hf.{mod_type.__name__}"})(_QuantMoELinear)


def _reconstruct_fused_moe_linear(model: nn.Module) -> None:
    """Reconstruct :class:`_QuantMoELinear` per-expert weights back to the 3-D MoELinear format.

    After _process_quantized_modules, each expert's nn.Linear inside the wrapper has:
      - weight: fp4-quantized tensor [out_features, in_features]
      - weight_scale, weight_scale_2: per-block / global scales
      - input_scale: activation scale (if calibrated)

    This stacks them back into the original MoELinear layout so the exported state_dict
    uses the original key names (e.g. moe.up_proj.weight with shape [N, out, in]).

    Matched by wrapper type rather than by the dynamically generated class name (``Quant`` +
    the model's own class name): a model whose class is not spelled ``MoELinear`` would
    otherwise quantize normally but export unusable per-expert keys.
    """
    for _name, module in model.named_modules():
        if not isinstance(module, _QuantMoELinear):
            continue

        n = module.num_experts
        experts = module.experts

        # Reconstruct 3D weight: [num_experts, out_features, in_features]
        module.weight = nn.Parameter(
            torch.stack([experts[i].weight.data for i in range(n)]),
            requires_grad=False,
        )

        # Stack per-expert scales back under the original attribute names.
        # Check all experts: some may lack input_scale if they were never routed
        # during calibration, so only stack when every expert has the attribute.
        for attr in ("weight_scale", "weight_scale_2", "input_scale"):
            if all(hasattr(experts[i], attr) for i in range(n)):
                module.register_buffer(
                    attr,
                    torch.stack([getattr(experts[i], attr) for i in range(n)]),
                )

        # Remove expanded experts — the reconstructed 3D tensors replace them
        del module.experts


CUSTOM_MODEL_PLUGINS.add(register_moe_linear_on_the_fly)
