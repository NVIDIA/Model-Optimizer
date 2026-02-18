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

"""Utilities for Mixture-of-Experts (MoE) model export."""

from pathlib import Path

import torch.nn as nn


def sync_expert_amax_low_tokens(model: nn.Module, threshold: float = 0.05):
    """Sync expert amax values across experts if the number of tokens routed to an expert is less than the threshold.

    For each MoE layer, this function collects the maximum amax value across all experts for
    each input quantizer, then overwrites the amax of experts whose token count falls below
    ``threshold * mean_token_count`` with that maximum.
    """
    for module_name, module in model.named_modules():
        if not (hasattr(module, "expert_token_count") and module.expert_token_count.numel() > 0):
            continue

        experts = list(module.experts.children())
        num_experts = module.expert_token_count.shape[0]

        expert_amax_max = {}
        for expert in experts:
            for quantizer_name, quantizer in expert.named_modules():
                # We do not sync amax for AWQ.
                if hasattr(quantizer, "pre_quant_scale") and quantizer.pre_quant_scale is not None:
                    return

                if (
                    "input_quantizer" in quantizer_name
                    and hasattr(quantizer, "_amax")
                    and quantizer._amax is not None
                    and quantizer._amax.numel() == 1
                ):
                    prev = expert_amax_max.get(quantizer_name)
                    cur = quantizer._amax.detach().clone()
                    if prev is None or cur > prev:
                        expert_amax_max[quantizer_name] = cur

        if not expert_amax_max:
            continue

        avg_token_count = module.expert_token_count.float().mean().item()
        token_threshold = avg_token_count * threshold

        print(f"[sync_expert_amax] {module_name}")
        print(f"  token counts : {module.expert_token_count.tolist()}")
        print(f"  avg={avg_token_count:.1f}  threshold(<)={token_threshold:.1f}")
        print(f"  tracked quantizers: {list(expert_amax_max.keys())}")

        for i in range(num_experts):
            token_count_i = module.expert_token_count[i].item()
            if token_count_i < token_threshold:
                expert_i = experts[i]
                print(f"  expert {i}: token_count={token_count_i} — syncing amax")
                for quantizer_name, quantizer in expert_i.named_modules():
                    if quantizer_name in expert_amax_max:
                        old_val = quantizer._amax.item() if quantizer._amax is not None else None
                        quantizer._amax = expert_amax_max[quantizer_name].clone()
                        print(f"    {quantizer_name}: {old_val} -> {quantizer._amax.item()}")


def save_expert_token_count_table(
    model: nn.Module, output_dir: str | Path | None = None, threshold: float = 0.05
):
    """Collect expert_token_count from all quantized MoE layers and save as an HTML table.

    The table has rows for each MoE layer and columns for each expert, with cell values
    showing the number of tokens routed to that expert during calibration.

    Args:
        model: The model containing quantized MoE layers with ``expert_token_count`` attributes.
        output_dir: Directory to save the HTML file. Defaults to current directory.
        threshold: Threshold for low token count to sync amax. Defaults to 0.05.
    """
    rows = []
    for name, module in model.named_modules():
        if hasattr(module, "expert_token_count") and module.expert_token_count.numel() > 0:
            rows.append((name, module.expert_token_count))

    if not rows:
        return

    num_experts = rows[0][1].shape[0]
    html_parts = [
        "<html><head><style>",
        "table { border-collapse: collapse; font-family: monospace; }",
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }",
        "th { background: #f0f0f0; }",
        "</style></head><body>",
        "<h2>Expert Token Counts (per MoE layer)</h2>",
        "<table><tr><th>Layer/Expert</th>",
    ]
    html_parts.extend(f"<th>{i}</th>" for i in range(num_experts))
    html_parts.append("</tr>")

    for name, counts in rows:
        avg = counts.float().mean().item()
        html_parts.append(f"<tr><td>{name}</td>")
        for c in counts.tolist():
            if avg > 0 and c < avg * threshold:
                style = ' style="background: #ffcccc;"'
            else:
                style = ""
            html_parts.append(f"<td{style}>{c}</td>")
        html_parts.append("</tr>")

    html_parts.append("</table></body></html>")
    html_content = "\n".join(html_parts)

    if output_dir is None:
        output_dir = Path(".")
    output_path = Path(output_dir) / "moe.html"
    output_path.write_text(html_content)
    print(f"Expert token count table saved to {output_path}")
