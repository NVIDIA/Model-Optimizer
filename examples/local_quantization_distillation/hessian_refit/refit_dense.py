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

"""Dense-unit driver: GPTQ PTQ -> KLD -> closed-form scale refit -> KLD [-> Bypass -> KLD].

For every selected unit, the inputs of the chosen linears are captured on the teacher
(``Z_t``) and on the student replay (``Z_s``) and accumulated as Grams over valid tokens.
The native IQ scale fields of ``down_proj`` are then refit exactly (``block_ls.py``), since
the unit output is linear in them. With ``--interior layerwise`` the gate and up
projections are first refit against their own teacher outputs on the shared unit input,
and the down Grams are re-collected with the updated student. ``--updates N`` runs an
ordinary Bypass repair of all scale fields from the refit state afterwards.

Single process on one GPU; run as a module from the repository root. Experimental.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

import modelopt.torch.distill as mtd
import modelopt.torch.quantization as mtq
from modelopt.torch.distill.plugins.transformers import (
    TransformersLocalDistillationAdapter,
    transformers_family,
)
from modelopt.torch.quantization.repair import QuantizationRepairPolicy

from ..evaluate import evaluate_pair
from ..prepare_data import PackedBatchSequence
from . import block_gn, block_ls
from .bypass_baseline import DEFAULT_DAMP, Head, gpu_stats, load_model

INTERIOR = ("gate_proj", "up_proj")
LAST = "down_proj"


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-class", default="auto", choices=["auto", "causal", "image-text"])
    p.add_argument("--unit-types", nargs="+", default=["mlp"])
    p.add_argument("--interior", default="none", choices=["none", "layerwise", "gn"])
    p.add_argument("--gn-iters", type=int, default=2)
    p.add_argument("--gn-damp", type=float, default=0.3, help="LM damping relative to the diagonal")
    p.add_argument("--gn-chunk", type=int, default=256)
    p.add_argument("--gn-ls-packs", type=int, default=4, help="packs for the exact line search")
    p.add_argument("--gn-packs", type=int, default=8, help="packs for the Gauss-Newton statistics")
    p.add_argument(
        "--gn-unit-group", type=int, default=0, help="units per statistics pass (0: all at once)"
    )
    p.add_argument(
        "--gn-rotate", type=int, default=1, help="1: fresh statistics packs every iteration"
    )
    p.add_argument(
        "--gn-granularity",
        default="group",
        choices=["group", "block"],
        help="block: only the FP16 scale of each 256-weight block moves (relative step)",
    )
    p.add_argument(
        "--gn-step-clip",
        type=float,
        default=0.5,
        help="block granularity: bound on the relative scale change per iteration",
    )
    p.add_argument(
        "--gn-down-packs",
        type=int,
        default=0,
        help="packs for the per-iteration down Grams (0: gram-packs)",
    )
    p.add_argument(
        "--gn-monitor",
        type=int,
        default=1,
        help="1: validation Grams every iteration (else final only)",
    )
    p.add_argument(
        "--bypass-first",
        action="store_true",
        help="run the Bypass updates before the refit instead of after",
    )
    p.add_argument("--gn-eta-max", type=float, default=1.0)
    p.add_argument("--sequence-length", type=int, default=32768)
    p.add_argument("--gram-split", default="train")
    p.add_argument("--gram-packs", type=int, default=64, help="32K packs for the Grams")
    p.add_argument("--gram-batch-size", type=int, default=1)
    p.add_argument("--validation-packs", type=int, default=None)
    p.add_argument("--diagnostic-packs", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--calib-batch-size", type=int, default=8)
    p.add_argument("--perc-damp", type=float, default=None, help="GPTQ damping")
    p.add_argument("--gptq-block-size", type=int, default=256)
    p.add_argument("--no-importance-weighted", action="store_true")
    p.add_argument("--ls-damp", type=float, default=0.01, help="ridge for the scale solve")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--updates", type=int, default=0, help="Bypass updates after the refit (0: none)"
    )
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--scale-lr", type=float, default=0.01)
    p.add_argument("--validation-interval", type=int, default=64)
    p.add_argument("--skip-before-eval", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class GramAccumulator:
    """Student/teacher Grams of one linear's input over valid tokens."""

    def __init__(self, dim, device):
        self.h_ss = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        self.h_st = torch.zeros_like(self.h_ss)
        self.h_tt = torch.zeros_like(self.h_ss)
        self.tokens = 0

    @torch.no_grad()
    def add(self, z_s, z_t, mask):
        if z_s.shape != z_t.shape:
            raise ValueError("Student and teacher linear inputs differ in shape.")
        flat_s = z_s.reshape(-1, z_s.shape[-1])
        flat_t = z_t.reshape(-1, z_t.shape[-1])
        if mask is not None:
            flat_mask = mask.reshape(-1).to(flat_s.device)
            flat_s, flat_t = flat_s[flat_mask], flat_t[flat_mask]
        flat_s, flat_t = flat_s.float(), flat_t.float()
        self.h_ss.addmm_(flat_s.T, flat_s)
        self.h_st.addmm_(flat_s.T, flat_t)
        self.h_tt.addmm_(flat_t.T, flat_t)
        self.tokens += flat_s.shape[0]

    @torch.no_grad()
    def mse(self, weight_s, weight_t):
        """Exact ``|| Z_s W_s^T - Z_t W_t^T ||^2 / (tokens * rows)``."""
        w_s = weight_s.to(device=self.h_ss.device, dtype=torch.float64)
        w_t = weight_t.to(device=self.h_ss.device, dtype=torch.float64)
        value = torch.einsum("rp,pq,rq->", w_s, self.h_ss.double(), w_s)
        value -= 2 * torch.einsum("rp,pq,rq->", w_s, self.h_st.double(), w_t)
        value += torch.einsum("rp,pq,rq->", w_t, self.h_tt.double(), w_t)
        return float(value) / max(1, self.tokens * w_t.shape[0])


@torch.no_grad()
def accumulate(adapter, teacher, student, units, batches, linears, device):
    """One teacher pass with student replay per unit; Grams keyed by (unit, linear)."""
    grams, stash, handles = {}, {}, []
    for unit in units:
        for name in linears:
            linear_t = teacher.get_submodule(unit.teacher_path).get_submodule(name)
            linear_s = student.get_submodule(unit.student_path).get_submodule(name)
            grams[unit.name, name] = GramAccumulator(linear_t.in_features, device)
            handles.append(
                linear_t.register_forward_pre_hook(
                    lambda module, args, key=(unit.name, name, "t"): stash.__setitem__(key, args[0])
                )
            )
            handles.append(
                linear_s.register_forward_pre_hook(
                    lambda module, args, key=(unit.name, name, "s"): stash.__setitem__(key, args[0])
                )
            )

    def consume(unit, record):
        module = student.get_submodule(unit.student_path)
        adapter.replay(module, record)
        for name in linears:
            grams[unit.name, name].add(
                stash.pop((unit.name, name, "s")),
                stash.pop((unit.name, name, "t")),
                record.valid_mask,
            )

    try:
        for batch in batches:
            adapter.teacher_microbatch(teacher, units, batch, consume)
    finally:
        for handle in handles:
            handle.remove()
    return grams


def apply_fields(state, g, local, rows):
    """Encode (g, local) into the adapter's log-scale masters."""
    g0 = state.global_base.reshape(rows, -1).double()
    ratio = torch.where(g0 > 0, g / g0.clamp_min(1e-30), torch.ones_like(g0)).clamp_min(1e-8)
    state.global_delta.copy_(
        ratio.log().reshape_as(state.global_delta).to(state.global_delta.dtype)
    )
    local_ratio = (2 * local.double() + 1) / state.local_base.reshape(rows, -1).double()
    state.local_delta.copy_(
        local_ratio.log().reshape_as(state.local_delta).to(state.local_delta.dtype)
    )


def state_levels(state):
    """Signed integer levels (rows, n) and current non-negative steps (rows, K) of a state."""
    fmt = state.format
    rows, n = state.weight_shape
    levels, steps = fmt.decode_int8(
        state.packed, torch.tensor(state.weight_shape, dtype=torch.int64)
    )
    blocks = n // block_ls.GGML_BLOCK
    sign = state.global_sign.reshape(rows, blocks)
    signed = (levels.reshape(rows, blocks, -1).float() * sign.unsqueeze(-1)).reshape(rows, n)
    return signed, steps.reshape(rows, -1).abs().float()


@torch.no_grad()
def prepare_interior(student, units, repair):
    """Prepare repair policies and read levels/steps of gate/up and the down weight."""
    policies, levels = {}, {}
    for unit in units:
        module = student.get_submodule(unit.student_path)
        policy = QuantizationRepairPolicy()
        policy.prepare(module, repair)
        policies[unit.name] = policy
        gate, up, down = (module.get_submodule(n) for n in (*INTERIOR, LAST))
        lev_g, s_g = state_levels(gate.weight_quantizer._local_repair_state)
        lev_u, s_u = state_levels(up.weight_quantizer._local_repair_state)
        w_down = down.weight_quantizer(down.weight).detach().float()  # (d_model, rows)
        levels[unit.name] = {
            "lev_g": lev_g,
            "lev_u": lev_u,
            "s_g": s_g,
            "s_u": s_u,
            "w_down": w_down,
        }
    return policies, levels


@torch.no_grad()
def interior_pass(adapter, teacher, student, units, batches, callback):
    """One teacher pass; ``callback(unit_name, x, g, u, r)`` gets valid-token tensors."""
    stash, handles = {}, []
    for unit in units:
        module = student.get_submodule(unit.student_path)
        gate, up = (module.get_submodule(n) for n in INTERIOR)
        handles.append(
            gate.register_forward_pre_hook(
                lambda m, a, key=(unit.name, "x"): stash.__setitem__(key, a[0])
            )
        )
        handles.append(
            gate.register_forward_hook(
                lambda m, a, o, key=(unit.name, "g"): stash.__setitem__(key, o)
            )
        )
        handles.append(
            up.register_forward_hook(
                lambda m, a, o, key=(unit.name, "u"): stash.__setitem__(key, o)
            )
        )

    def consume(unit, record):
        module = student.get_submodule(unit.student_path)
        prediction = adapter.replay(module, record)
        x, g, u = (stash.pop((unit.name, key)) for key in ("x", "g", "u"))
        r = prediction - record.target
        mask = record.valid_mask
        if mask is not None:
            m = mask.reshape(-1).to(x.device)
            x, g, u, r = (t.reshape(-1, t.shape[-1])[m] for t in (x, g, u, r))
        else:
            x, g, u, r = (t.reshape(-1, t.shape[-1]) for t in (x, g, u, r))
        callback(unit.name, x, g, u, r)

    try:
        for batch in batches:
            adapter.teacher_microbatch(teacher, units, batch, consume)
    finally:
        for handle in handles:
            handle.remove()


class GaussNewtonStatsHolder:
    def __init__(self, rows, n, device, module):
        state = module.get_submodule(INTERIOR[0]).weight_quantizer._local_repair_state
        self.group, self.groups_per_block, self.local_max = block_ls.format_geometry(state.format)
        self.stats = block_gn.GaussNewtonStats(rows, n // self.group, device)


@torch.no_grad()
def gn_update_unit(student, unit, holder, policy, levels, target_g, target_u, eta_info, args):
    """Project the scaled Gauss-Newton targets to legal fields and finalize the unit."""
    module = student.get_submodule(unit.student_path)
    stats = holder.stats
    lv = levels[unit.name]
    try:
        report = {
            "tokens": stats.tokens,
            "unit_mse_before": stats.sq_err / max(1, stats.tokens * lv["w_down"].shape[0]),
            "line_search": eta_info,
        }
        for name, target, which, current in (
            (INTERIOR[0], target_g, "gate", lv["s_g"]),
            (INTERIOR[1], target_u, "up", lv["s_u"]),
        ):
            metric = stats.metric(which)
            state = module.get_submodule(name).weight_quantizer._local_repair_state
            rows = state.weight_shape[0]
            g0 = state.global_base.reshape(rows, -1).double()
            local0 = state.local_base.reshape(rows, -1).sub(1).div(2).round().long()
            g, local, s, history = block_gn.project(
                metric,
                target,
                g0,
                local0,
                holder.groups_per_block,
                holder.local_max,
                damp=args.ls_damp,
            )
            report[name] = {
                "projection_history": history,
                "local_changed_fraction": float((local != local0).double().mean()),
                "global_rel_change_mean": float(((g - g0).abs() / g0.clamp_min(1e-12)).mean()),
                "step_rel_norm": float(
                    (target - current.double()).norm() / current.double().norm()
                ),
            }
            apply_fields(state, g, local, rows)
            del metric
        policy.finalize_()
    except BaseException:
        policy.close()
        raise
    return report


@torch.no_grad()
def gn_update_unit_block(student, unit, policy, delta_g, delta_u, eta, eta_info, args):
    """Scale each block's FP16 scale by ``1 + eta * delta``; local scales stay as they are."""
    module = student.get_submodule(unit.student_path)
    try:
        report = {"line_search": eta_info}
        for name, delta in ((INTERIOR[0], delta_g), (INTERIOR[1], delta_u)):
            state = module.get_submodule(name).weight_quantizer._local_repair_state
            rows = state.weight_shape[0]
            g0 = state.global_base.reshape(rows, -1).double()
            local0 = state.local_base.reshape(rows, -1).sub(1).div(2).round().long()
            factor = (1 + eta * delta.reshape(rows, -1)).clamp(
                1 - args.gn_step_clip, 1 + args.gn_step_clip
            )
            report[name] = {"global_rel_change_mean": float((factor - 1).abs().mean())}
            apply_fields(state, g0 * factor, local0, rows)
        policy.finalize_()
    except BaseException:
        policy.close()
        raise
    return report


def dequantized(linear):
    with torch.no_grad():
        return linear.weight_quantizer(linear.weight).detach().float()


@torch.no_grad()
def refit_linears(student, unit, names, grams, teacher_weights, repair, args, device):
    """Refit the native IQ scale fields of ``names`` inside one unit; returns reports."""
    module = student.get_submodule(unit.student_path)
    policy = QuantizationRepairPolicy()
    policy.prepare(module, repair)
    reports = {}
    try:
        for name in names:
            linear = module.get_submodule(name)
            state = linear.weight_quantizer._local_repair_state
            fmt = state.format
            group, groups_per_block, local_max = block_ls.format_geometry(fmt)
            rows, n = state.weight_shape
            gram = grams[unit.name, name]
            teacher_weight = teacher_weights[unit.name, name].to(device=device, dtype=torch.float32)
            if tuple(teacher_weight.shape) != (rows, n):
                raise ValueError(f"{name}: teacher weight shape differs from the student payload.")
            levels, _ = fmt.decode_int8(
                state.packed, torch.tensor(state.weight_shape, dtype=torch.int64)
            )
            levels = levels.reshape(rows, n).float()
            blocks = n // block_ls.GGML_BLOCK
            sign = state.global_sign.reshape(rows, blocks)
            signed = (levels.reshape(rows, blocks, -1) * sign.unsqueeze(-1)).reshape(rows, n)
            g0 = state.global_base.reshape(rows, blocks).double()
            local0 = state.local_base.reshape(rows, -1).sub(1).div(2).round().long()
            u = teacher_weight @ gram.h_st.T
            a, c = block_ls.group_quadratics(signed, gram.h_ss, u, group)
            const = float(
                torch.einsum(
                    "rp,pq,rq->",
                    teacher_weight.double(),
                    gram.h_tt.double(),
                    teacher_weight.double(),
                )
            )
            g, local, s, history = block_ls.refit_hierarchy(
                a, c, g0, local0, groups_per_block, local_max, damp=args.ls_damp, rounds=args.rounds
            )
            denominator = max(1, gram.tokens) * rows
            reports[name] = {
                "rows": rows,
                "in_features": n,
                "group": group,
                "tokens": gram.tokens,
                "mse": {key: (value + const) / denominator for key, value in history.items()},
                "local_changed_fraction": float((local != local0).double().mean()),
                "global_rel_change_mean": float(((g - g0).abs() / g0.clamp_min(1e-12)).mean()),
            }
            ratio = torch.where(g0 > 0, g / g0.clamp_min(1e-30), torch.ones_like(g0)).clamp_min(
                1e-8
            )
            state.global_delta.copy_(
                ratio.log().reshape_as(state.global_delta).to(state.global_delta.dtype)
            )
            local_ratio = (2 * local.double() + 1) / state.local_base.reshape(rows, -1).double()
            state.local_delta.copy_(
                local_ratio.log().reshape_as(state.local_delta).to(state.local_delta.dtype)
            )
        policy.finalize_()
    except BaseException:
        policy.close()
        raise
    return reports


def main():
    args = parse()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log = {"args": vars(args), "started_unix": time.time()}

    def dump():
        (out / "metrics.json").write_text(json.dumps(log, indent=2, default=str) + "\n")

    def run_bypass_stage(log, key):
        bypass = mtd.LocalDistillationConfig(
            unit_types=tuple(args.unit_types),
            mode="scale_only",
            updates=args.updates,
            sequence_length=args.sequence_length,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            scale_lr=args.scale_lr,
            validation_interval=args.validation_interval,
            seed=args.seed,
        )
        train = PackedBatchSequence(
            args.data, "train", global_batch_size=args.global_batch_size, device=device
        )
        torch.cuda.reset_peak_memory_stats(device)
        result = mtd.local_distill(
            student, teacher, train, bypass, validation_dataloader=validation, adapter=adapter
        )
        log["bypass"] = result.metrics
        log[key] = evaluate_pair(teacher, result.student, diagnostic, adapter=adapter)
        print(json.dumps({key: log[key]["kld_teacher_candidate"]}), flush=True)

    repair = mtd.LocalDistillationConfig(
        unit_types=tuple(args.unit_types),
        mode="scale_only",
        updates=0,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    teacher = load_model(args.model, device, args.model_class)
    student = copy.deepcopy(teacher)
    adapter = TransformersLocalDistillationAdapter()
    adapter.prepare(student, teacher, [], repair)  # single process: records the student

    grams_data = Head(
        PackedBatchSequence(
            args.data, args.gram_split, global_batch_size=args.gram_batch_size, device=device
        ),
        args.gram_packs // args.gram_batch_size,
    )
    calibration = PackedBatchSequence(
        args.data, "calibration", global_batch_size=args.calib_batch_size, device=device
    )
    validation = PackedBatchSequence(
        args.data, "validation", global_batch_size=args.eval_batch_size, device=device
    )
    diagnostic = PackedBatchSequence(
        args.data, "diagnostic", global_batch_size=args.eval_batch_size, device=device
    )
    if args.validation_packs:
        validation = Head(validation, args.validation_packs // args.eval_batch_size)
    if args.diagnostic_packs:
        diagnostic = Head(diagnostic, args.diagnostic_packs // args.eval_batch_size)
    log["data"] = {
        "gram_batches": len(grams_data),
        "gram_tokens_nominal": len(grams_data)
        * args.gram_batch_size
        * grams_data.inner.sequence_length,
        "validation_batches": len(validation),
        "diagnostic_batches": len(diagnostic),
    }

    # ---- PTQ (identical recipe to bypass_baseline) ----------------------------------------
    t0 = time.perf_counter()
    patterns = transformers_family(teacher).weight_patterns(teacher, repair)
    damp = args.perc_damp if args.perc_damp is not None else DEFAULT_DAMP.get(args.format, 0.01)
    recipe = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            *[
                {
                    "quantizer_name": p,
                    "enable": True,
                    "cfg": {"num_bits": args.format, "backend": "ggml"},
                }
                for p in patterns
            ],
        ],
        "algorithm": {
            "method": "gptq",
            "block_size": args.gptq_block_size,
            "importance_weighted": not args.no_importance_weighted,
            "perc_damp": damp,
            "fused": False,
        },
    }

    @torch.no_grad()
    def calibrate(model):
        for batch in calibration:
            adapter.teacher_microbatch(model, [], batch, lambda *_: None)

    mtq.quantize(student, recipe, forward_loop=calibrate)
    log["ptq_seconds"] = time.perf_counter() - t0
    dump()

    units = adapter.resolve_units(student, teacher, repair)
    log["units"] = [unit.name for unit in units]
    linears = (*INTERIOR, LAST) if args.interior == "layerwise" else (LAST,)
    teacher_weights = {
        (unit.name, name): teacher.get_submodule(unit.teacher_path)
        .get_submodule(name)
        .weight.detach()
        for unit in units
        for name in linears
    }
    ptq_down = {
        unit.name: dequantized(student.get_submodule(unit.student_path).get_submodule(LAST)).cpu()
        for unit in units
    }

    if not args.skip_before_eval:
        log["before"] = evaluate_pair(teacher, student, diagnostic, adapter=adapter)
        print(json.dumps({"before": log["before"]}), flush=True)
        dump()

    if args.updates > 0 and args.bypass_first:
        run_bypass_stage(log, "after_bypass_first")
        ptq_down = {
            unit.name: dequantized(
                student.get_submodule(unit.student_path).get_submodule(LAST)
            ).cpu()
            for unit in units
        }
        dump()

    # ---- Pass 1: Grams of every selected linear at the current state -----------------------
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    grams = accumulate(adapter, teacher, student, units, grams_data, linears, device)
    val_grams = accumulate(adapter, teacher, student, units, validation, (LAST,), device)
    log["gram_seconds"] = time.perf_counter() - t0
    log["gram_gpu"] = gpu_stats(device)
    log["gram_tokens"] = grams[units[0].name, LAST].tokens
    log["validation_tokens"] = val_grams[units[0].name, LAST].tokens
    unit_mse = {
        unit.name: {
            "ptq": val_grams[unit.name, LAST].mse(
                ptq_down[unit.name], teacher_weights[unit.name, LAST]
            )
        }
        for unit in units
    }
    dump()

    # ---- Interior refit (layer-level LS on the shared unit input), then re-collect down ----
    t0 = time.perf_counter()
    reports = {}
    if args.interior == "layerwise":
        for unit in units:
            reports[unit.name] = refit_linears(
                student, unit, INTERIOR, grams, teacher_weights, repair, args, device
            )
        grams = accumulate(adapter, teacher, student, units, grams_data, (LAST,), device)
        val_grams = accumulate(adapter, teacher, student, units, validation, (LAST,), device)
        for unit in units:
            unit_mse[unit.name]["after_interior"] = val_grams[unit.name, LAST].mse(
                ptq_down[unit.name], teacher_weights[unit.name, LAST]
            )
    if args.interior == "gn":
        log["gn"] = []
        for iteration in range(args.gn_iters):
            t_it = time.perf_counter()
            step_reports = {}
            group_size = args.gn_unit_group or len(units)
            for group_start in range(0, len(units), group_size):
                group = units[group_start : group_start + group_size]
                policies, levels_cache = prepare_interior(student, group, repair)
                holders = {
                    unit.name: GaussNewtonStatsHolder(
                        levels_cache[unit.name]["lev_g"].shape[0],
                        levels_cache[unit.name]["lev_g"].shape[1],
                        device,
                        student.get_submodule(unit.student_path),
                    )
                    for unit in group
                }
                if args.gn_granularity == "block":
                    # Relative coordinates: the Jacobian factor is the dequantized weight pooled per block.
                    for unit in group:
                        lv, holder = levels_cache[unit.name], holders[unit.name]
                        rows, n = lv["lev_g"].shape
                        lv["lev_g"] = block_ls.expand_steps(lv["lev_g"], lv["s_g"], holder.group)
                        lv["lev_u"] = block_ls.expand_steps(lv["lev_u"], lv["s_u"], holder.group)
                        lv["s_g"] = torch.zeros(rows, n // block_ls.GGML_BLOCK, device=device)
                        lv["s_u"] = torch.zeros(
                            lv["lev_u"].shape[0],
                            lv["lev_u"].shape[1] // block_ls.GGML_BLOCK,
                            device=device,
                        )
                        holder.group = block_ls.GGML_BLOCK
                        holder.stats = block_gn.GaussNewtonStats(
                            rows, n // block_ls.GGML_BLOCK, device
                        )

                def accumulate_stats(name, x, g, u, r, holders=holders, levels_cache=levels_cache):
                    lv = levels_cache[name]
                    holders[name].stats.add(
                        x,
                        g,
                        u,
                        r,
                        lv["w_down"],
                        lv["lev_g"],
                        lv["lev_u"],
                        holders[name].group,
                        chunk=args.gn_chunk,
                    )

                stats_batches = Head(
                    grams_data.inner,
                    args.gn_packs // args.gram_batch_size,
                    start=(iteration * args.gn_packs // args.gram_batch_size)
                    if args.gn_rotate
                    else 0,
                )
                interior_pass(adapter, teacher, student, group, stats_batches, accumulate_stats)
                steps = {
                    unit.name: holders[unit.name].stats.step(
                        levels_cache[unit.name]["s_g"], levels_cache[unit.name]["s_u"], args.gn_damp
                    )
                    for unit in group
                }
                searches = {
                    unit.name: block_gn.LineSearch(
                        steps[unit.name][0],
                        steps[unit.name][1],
                        levels_cache[unit.name]["lev_g"],
                        levels_cache[unit.name]["lev_u"],
                        levels_cache[unit.name]["w_down"],
                        holders[unit.name].group,
                    )
                    for unit in group
                }
                search_batches = Head(
                    grams_data.inner,
                    args.gn_ls_packs // args.gram_batch_size,
                    start=stats_batches.start,
                )
                interior_pass(
                    adapter,
                    teacher,
                    student,
                    group,
                    search_batches,
                    lambda name, x, g, u, r, searches=searches: searches[name].add(
                        x, g, u, r, chunk=args.gn_chunk
                    ),
                )
                for unit in group:
                    delta_g, delta_u = steps[unit.name]
                    eta, eta_info = searches[unit.name].eta(args.gn_eta_max)
                    lv = levels_cache[unit.name]
                    if args.gn_granularity == "block":
                        step_reports[unit.name] = gn_update_unit_block(
                            student,
                            unit,
                            policies[unit.name],
                            delta_g,
                            delta_u,
                            eta,
                            {"eta": eta, **eta_info},
                            args,
                        )
                    else:
                        target_g = lv["s_g"].double() + eta * delta_g
                        target_u = lv["s_u"].double() + eta * delta_u
                        step_reports[unit.name] = gn_update_unit(
                            student,
                            unit,
                            holders[unit.name],
                            policies[unit.name],
                            levels_cache,
                            target_g,
                            target_u,
                            {"eta": eta, **eta_info},
                            args,
                        )
                    del holders[unit.name], steps[unit.name], searches[unit.name]
                    torch.cuda.empty_cache()
                del holders, steps, searches, levels_cache, policies
                torch.cuda.empty_cache()
            etas = [r["line_search"]["eta"] for r in step_reports.values()]
            down_batches = Head(
                grams_data.inner, (args.gn_down_packs or args.gram_packs) // args.gram_batch_size
            )
            grams = accumulate(adapter, teacher, student, units, down_batches, (LAST,), device)
            last = iteration == args.gn_iters - 1
            monitor = args.gn_monitor or last
            if monitor:
                val_grams = accumulate(
                    adapter, teacher, student, units, validation, (LAST,), device
                )
            for unit in units:
                if monitor:
                    unit_mse[unit.name][f"gn{iteration}_before_down"] = val_grams[
                        unit.name, LAST
                    ].mse(
                        dequantized(student.get_submodule(unit.student_path).get_submodule(LAST)),
                        teacher_weights[unit.name, LAST],
                    )
                reports.setdefault(unit.name, {})[f"gn{iteration}"] = step_reports[unit.name]
                reports[unit.name][f"down_gn{iteration}"] = refit_linears(
                    student, unit, (LAST,), grams, teacher_weights, repair, args, device
                )[LAST]
                if monitor:
                    unit_mse[unit.name][f"gn{iteration}"] = val_grams[unit.name, LAST].mse(
                        dequantized(student.get_submodule(unit.student_path).get_submodule(LAST)),
                        teacher_weights[unit.name, LAST],
                    )
            entry = {
                "iteration": iteration,
                "seconds": time.perf_counter() - t_it,
                "eta_mean": sum(etas) / len(etas),
                "eta_min": min(etas),
                "eta_max": max(etas),
            }
            if monitor:
                entry["val_unit_mse_before_down"] = sum(
                    v[f"gn{iteration}_before_down"] for v in unit_mse.values()
                ) / len(unit_mse)
                entry["val_unit_mse"] = sum(v[f"gn{iteration}"] for v in unit_mse.values()) / len(
                    unit_mse
                )
            log["gn"].append(entry)
            print(json.dumps(log["gn"][-1]), flush=True)
            dump()
        for unit in units:
            unit_mse[unit.name]["refit"] = unit_mse[unit.name][f"gn{args.gn_iters - 1}"]
    else:
        for unit in units:
            reports.setdefault(unit.name, {}).update(
                refit_linears(student, unit, (LAST,), grams, teacher_weights, repair, args, device)
            )
            refit_down = dequantized(student.get_submodule(unit.student_path).get_submodule(LAST))
            unit_mse[unit.name]["refit"] = val_grams[unit.name, LAST].mse(
                refit_down, teacher_weights[unit.name, LAST]
            )
            print(
                json.dumps(
                    {
                        "unit": unit.name,
                        "validation_unit_mse": unit_mse[unit.name],
                        "gram_mse": {n: r["mse"] for n, r in reports[unit.name].items()},
                    }
                ),
                flush=True,
            )
    log["refit"] = reports
    log["validation_unit_mse"] = unit_mse
    log["refit_seconds"] = time.perf_counter() - t0
    dump()

    log["after"] = evaluate_pair(teacher, student, diagnostic, adapter=adapter)
    dump()

    # ---- Optional Bypass from the refit state ----------------------------------------------
    if args.updates > 0 and not args.bypass_first:
        run_bypass_stage(log, "after_bypass")
        dump()

    log["finished_unix"] = time.time()
    n = max(1, len(units))
    summary = {
        "format": args.format,
        "interior": args.interior,
        "gram_tokens": log["gram_tokens"],
        "kld_before": log.get("before", {}).get("kld_teacher_candidate"),
        "kld_after": log["after"]["kld_teacher_candidate"],
        "ppl_before": log.get("before", {}).get("ppl"),
        "ppl_after": log["after"].get("ppl"),
        "val_unit_mse_ptq": sum(v["ptq"] for v in unit_mse.values()) / n,
        "val_unit_mse_after_interior": (sum(v["after_interior"] for v in unit_mse.values()) / n)
        if args.interior == "layerwise"
        else None,
        "gn": log.get("gn"),
        "val_unit_mse_refit": sum(v["refit"] for v in unit_mse.values()) / n,
        "gram_seconds": log["gram_seconds"],
        "refit_seconds": log["refit_seconds"],
    }
    if "after_bypass_first" in log:
        units_b = log["bypass"]["units"]
        summary.update(
            bypass_first_updates=args.updates,
            kld_after_bypass_first=log["after_bypass_first"]["kld_teacher_candidate"],
            val_unit_mse_after_bypass_first=sum(
                u.get("validation_mse", 0) for u in units_b.values()
            )
            / max(1, len(units_b)),
            bypass_gpu_hours=log["bypass"].get("process_gpu_hours"),
        )
    if "after_bypass" in log:
        units_b = log["bypass"]["units"]
        summary.update(
            bypass_updates=args.updates,
            kld_after_bypass=log["after_bypass"]["kld_teacher_candidate"],
            ppl_after_bypass=log["after_bypass"].get("ppl"),
            val_unit_mse_after_bypass=sum(u.get("validation_mse", 0) for u in units_b.values())
            / max(1, len(units_b)),
            bypass_gpu_hours=log["bypass"].get("process_gpu_hours"),
        )
    log["summary"] = summary
    dump()
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
