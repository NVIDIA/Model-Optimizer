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

"""Summarize run directories written by ``bypass_baseline`` and ``refit_dense`` into one table.

Usage: python -m examples.local_quantization_distillation.hessian_refit.summarize <run_dir>...
Each run directory holds <config>/metrics.json files; one row is printed per config.
"""

import argparse
import glob
import json
import os
from pathlib import Path


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def row_from(path):
    data = json.loads(Path(path).read_text())
    args = data.get("args", {})
    before = data.get("before") or {}
    after = data.get("after") or {}
    repair = data.get("repair") or {}
    units = repair.get("units") or {}
    row = {
        "run": Path(path).parent.parent.name,
        "config": Path(path).parent.name,
        "format": args.get("format"),
        "updates": repair.get("updates", args.get("updates")),
        "kld_before": before.get("kld_teacher_candidate"),
        "kld_after": after.get("kld_teacher_candidate"),
        "ppl_teacher": after.get("teacher_ppl") or before.get("teacher_ppl"),
        "ppl_before": before.get("ppl"),
        "ppl_after": after.get("ppl"),
    }
    if units:
        base = [u.get("baseline_validation_mse") for u in units.values()]
        final = [u.get("validation_mse") for u in units.values()]
        if all(v is not None for v in base):
            row["unit_mse_base"] = sum(base) / len(base)
        if all(v is not None for v in final):
            row["unit_mse_final"] = sum(final) / len(final)
        row["gpu_hours"] = repair.get("process_gpu_hours")
        row["peak_gib"] = (repair.get("peak_allocated_bytes_max") or 0) / 2**30
    summary = data.get("summary") or {}
    if "val_unit_mse_ptq" in summary:
        row["interior"] = summary.get("interior")
        row["gram_tokens"] = summary.get("gram_tokens")
        row["unit_mse_base"] = summary["val_unit_mse_ptq"]
        row["unit_mse_interior"] = summary.get("val_unit_mse_after_interior")
        row["unit_mse_final"] = summary["val_unit_mse_refit"]
        if "kld_after_bypass_first" in summary:
            row["updates"] = summary.get("bypass_first_updates")
            row["kld_bypass_first"] = summary["kld_after_bypass_first"]
            row["unit_mse_bypass_first"] = summary.get("val_unit_mse_after_bypass_first")
            row["gpu_hours"] = summary.get("bypass_gpu_hours")
        if "kld_after_bypass" in summary:
            row["updates"] = summary.get("bypass_updates")
            row["kld_after_bypass"] = summary["kld_after_bypass"]
            row["unit_mse_bypass"] = summary.get("val_unit_mse_after_bypass")
            row["gpu_hours"] = summary.get("bypass_gpu_hours")
        row["gram_s"] = summary.get("gram_seconds")
        row["refit_s"] = summary.get("refit_seconds")
    elif data.get("validation"):
        validation = data["validation"]
        row["unit_mse_base"] = sum(v["ptq_last_linear_mse"] for v in validation.values()) / len(
            validation
        )
        row["unit_mse_final"] = sum(v["refit_last_linear_mse"] for v in validation.values()) / len(
            validation
        )
        row["gram_s"] = data.get("gram_seconds")
        row["refit_s"] = data.get("refit_seconds")
    if "kld_before" in row and row["kld_before"] and row["kld_after"] is not None:
        row["kld_rel_gain"] = 1 - row["kld_after"] / row["kld_before"]
    row["done"] = "finished_unix" in data
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--compact", action="store_true", help="only the decisive columns")
    args = parser.parse_args()
    files = []
    for pattern in args.paths:
        for root in glob.glob(pattern):
            files.extend(glob.glob(os.path.join(root, "*", "metrics.json")))
            if os.path.basename(root) == "metrics.json":
                files.append(root)
    rows = [row_from(f) for f in sorted(set(files))]
    if not rows:
        print("no metrics.json found")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if args.compact:
        keep = [
            "config",
            "format",
            "updates",
            "interior",
            "gram_tokens",
            "kld_before",
            "kld_after",
            "kld_bypass_first",
            "kld_after_bypass",
            "unit_mse_base",
            "unit_mse_bypass_first",
            "unit_mse_interior",
            "unit_mse_final",
            "unit_mse_bypass",
            "gpu_hours",
            "gram_s",
            "refit_s",
            "done",
        ]
        columns = [c for c in keep if c in columns]
        for row in rows:
            row["config"] = row["config"][:28]
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in columns}
    print(" | ".join(c.ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" | ".join(fmt(row.get(c)).ljust(widths[c]) for c in columns))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
