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

"""Local-distillation entry point for existing ModelOpt quantized students."""

import json
import math
import os
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

import modelopt.torch.opt as mto
from modelopt.torch.utils import print_rank_0

from .config import LocalDistillationConfig
from .local_checkpoint import LocalRepairCheckpoint, _batch_hash
from .local_units import ModuleLocalDistillationAdapter, mse_statistics

__all__ = ["LocalDistillationResult", "local_distill"]


@dataclass
class LocalDistillationResult:
    """A finalized student and lightweight measurements, without its teacher."""

    student: nn.Module
    metrics: dict[str, Any] = field(default_factory=dict)
    report_paths: dict[str, str] = field(default_factory=dict)


def local_distill(
    student,
    teacher,
    train_dataloader,
    config=None,
    *,
    validation_dataloader=None,
    adapter=None,
) -> LocalDistillationResult:
    """Repair independently selected units of an existing PTQ model.

    Preflight is transactional. A zero-update call checks ownership and returns
    the original PTQ model, without adding a distillation wrapper or teacher.
    """
    from modelopt.torch.quantization.repair import preflight_quantizers
    from modelopt.torch.quantization.utils import is_quantized

    config = (
        config
        if isinstance(config, LocalDistillationConfig)
        else LocalDistillationConfig(**(config or {}))
    )
    adapter = adapter or ModuleLocalDistillationAdapter()
    if not is_quantized(student) and not getattr(adapter, "allows_empty_stage", False):
        raise ValueError("Local quantization distillation requires a ModelOpt PTQ student.")
    units = adapter.resolve_units(student, teacher, config)
    if config.updates == 0:
        return LocalDistillationResult(
            student, {"updates": 0, "units": [unit.name for unit in units]}
        )
    report = preflight_quantizers(
        [student.get_submodule(unit.student_path) for unit in units], config
    )
    if hasattr(adapter, "validate_preflight"):
        adapter.validate_preflight(report)
    else:
        report.require_supported()
    return _LocalRunner(student, teacher, units, config, adapter).run(
        train_dataloader, validation_dataloader
    )


def _learning_rate_factor(config, step):
    warmup = math.ceil(config.updates * config.warmup_fraction)
    if step <= warmup:
        return step / max(1, warmup)
    fraction = (step - warmup) / max(1, config.updates - warmup)
    if config.scheduler == "cosine":
        return (1 + math.cos(math.pi * fraction)) / 2
    return 1 - fraction if config.scheduler == "linear" else 1.0


class _LocalRunner:
    def __init__(self, student, teacher, units, config, adapter):
        self.student, self.teacher, self.units = student, teacher, units
        self.config, self.adapter = config, adapter
        self.policies, self.optimizers = {}, {}
        self.metrics: dict[str, Any] = {
            "updates": 0,
            "units": {unit.name: {} for unit in units},
            "history": [],
        }
        self.best = {}
        self.checkpoint = None
        self.consumed_batches = 0
        self.consumed_hashes = []
        self.runtime = None

    def _parameters(self, unit):
        return {
            name: parameter
            for name, parameter in self.student.get_submodule(unit.student_path).named_parameters()
            if parameter.requires_grad
        }

    def _parameter_state(self, unit):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._parameters(unit).items()
        }

    def _load_parameters(self, unit, values):
        parameters = self._parameters(unit)
        if parameters.keys() != values.keys():
            raise ValueError("Repair checkpoint trainable field ownership differs.")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(values[name])

    def _guard(self, operation):
        # Only wrap rank-local work. Model/gradient collectives cannot be made
        # transactional by catching an exception after peers have entered them.
        value, error = None, None
        try:
            value = operation()
        except Exception as exc:
            error = exc
        if hasattr(self.adapter, "synchronize_error"):
            self.adapter.synchronize_error(
                None if error is None else f"{type(error).__name__}: {error}"
            )
        if error is not None:
            raise error
        return value

    def _validate_output(self):
        if self.config.output_dir:
            output = Path(self.config.output_dir)
            if output.is_symlink() or (
                output.exists() and (not output.is_dir() or any(output.iterdir()))
            ):
                raise ValueError("Repair output_dir must be a new or empty directory.")

    def _next_first(self, batches):
        try:
            return next(batches)
        except StopIteration as exc:
            raise ValueError("Repair training data is empty.") from exc

    def _window(self, batches, dataloader):
        window, samples = [], 0
        while samples < self.update_batch_size:
            try:
                batch = next(batches)
            except StopIteration as exc:
                raise ValueError(
                    "Repair data exhausted; supply unique samples for every configured update."
                ) from exc
            samples += self.adapter.batch_size(batch)
            if samples > self.update_batch_size:
                raise ValueError(
                    "Dataloader batches must exactly fill the configured global batch."
                )
            window.append(batch)
            if self.checkpoint is not None and not hasattr(dataloader, "state_dict"):
                self.consumed_hashes.append(_batch_hash(batch))
            self.consumed_batches += 1
        return window

    def _prepare(self, first_batch):
        from modelopt.torch.quantization.repair import QuantizationRepairPolicy

        self.runtime = (
            self.adapter.prepare(self.student, self.teacher, self.units, self.config)
            if hasattr(self.adapter, "prepare")
            else None
        )
        if (
            self.runtime is None
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise ValueError("Distributed local repair requires the AutoModel runtime adapter.")
        self.update_batch_size = (
            self.runtime.local_batch_size if self.runtime else self.config.global_batch_size
        )
        if self.config.resume_from or self.config.checkpoint_dir:
            self.checkpoint = LocalRepairCheckpoint(
                self.student, self.teacher, self.config, self.runtime
            )
        if self.runtime:
            self.runtime.initialize(
                self.student, self.teacher, self.units, self.adapter, first_batch
            )
        for unit in self.units:
            policy = QuantizationRepairPolicy()
            self.policies[unit.name] = policy
            module = self.student.get_submodule(unit.student_path)
            groups = (
                self.runtime.prepare_policy(module, policy, self.config)
                if self.runtime
                else policy.prepare(module, self.config)
            )
            if self.runtime:
                self.runtime.register(self.student.get_submodule(unit.student_path), policy)
            optimizer = (
                torch.optim.Adagrad if self.config.optimizer == "adagrad" else torch.optim.AdamW
            )
            self.metrics["units"][unit.name]["trainable_parameters_local"] = sum(
                parameter.numel() for group in groups for parameter in group.params
            )
            self.optimizers[unit.name] = optimizer(
                [
                    {"params": group.params, "lr": group.lr, "initial_lr": group.lr}
                    for group in groups
                ],
                weight_decay=0,
            )

    def _microbatches(self, batches):
        def local_microbatches():
            for batch in batches:
                size = (
                    self.config.micro_batch_size
                    or getattr(self.runtime, "micro_batch_size", None)
                    or self.adapter.batch_size(batch)
                )
                yield from self.adapter.microbatches(batch, size)

        iterator = iter(local_microbatches())
        while True:
            available, batch, error = False, None, None
            try:
                batch = next(iterator)
                available = True
            except StopIteration:
                pass
            except Exception as exc:
                error = exc
            if hasattr(self.adapter, "synchronize_microbatch"):
                self.adapter.synchronize_microbatch(
                    available, None if error is None else f"{type(error).__name__}: {error}"
                )
            if error is not None:
                raise error
            if not available:
                return
            yield batch

    def _record_coverage(self, unit, coverage):
        if not coverage:
            return
        report = self.metrics["units"][unit.name].setdefault("expert_coverage", {})
        for name, counts in coverage.items():
            previous = report.get(name, {}).get("routed_assignments", [0] * counts.numel())
            accumulated = [old + new for old, new in zip(previous, counts.tolist(), strict=True)]
            report[name] = {
                "routed_assignments": accumulated,
                "first_observed_step": report.get(name, {}).get(
                    "first_observed_step", self.metrics["updates"] + 1
                ),
                "last_observed_step": self.metrics["updates"] + 1,
                "unobserved_experts": [i for i, count in enumerate(accumulated) if count == 0],
            }

    def _measure(self, batches, training):
        totals = {}

        def consume(unit, record):
            with torch.set_grad_enabled(training):
                module = self.student.get_submodule(unit.student_path)
                with self.adapter.observe_replay(module, training) as coverage:
                    prediction = self.adapter.replay(module, record)
                if training:
                    self._record_coverage(unit, coverage)
                error, count, energy = mse_statistics(prediction, record.target, record.valid_mask)
                if training:
                    error.backward()
            values = torch.stack(
                (
                    error.detach().double(),
                    count.double(),
                    energy.detach().double(),
                    count.double()
                    / (
                        math.prod(record.target.shape[record.valid_mask.ndim :])
                        if record.valid_mask is not None
                        else record.target.shape[-1]
                    ),
                )
            )
            if unit.name not in totals:
                totals[unit.name] = values
            else:
                totals[unit.name].add_(values)

        for microbatch in self._microbatches(batches):
            self.adapter.teacher_microbatch(self.teacher, self.units, microbatch, consume)
        return self.runtime.reduce_statistics(totals, training) if self.runtime else totals

    def _validate(self, dataloader, baseline=False):
        values = self._measure(dataloader, False)
        for unit in self.units:
            if unit.name not in values:
                raise ValueError(f"Validation did not execute repair boundary {unit.name!r}.")
            error, count, _ = values[unit.name][:3].tolist()
            if count == 0:
                raise ValueError("Validation requires valid hidden-state tokens.")
            key = "baseline_validation_mse" if baseline else "validation_mse"
            self.metrics["units"][unit.name][key] = error / count
            if self.config.selection == "best_validation_per_unit":
                previous = self.best.get(unit.name)
                if previous is None or error / count < previous["mse"]:
                    self.best[unit.name] = {
                        "mse": error / count,
                        "step": self.metrics["updates"],
                        "parameters": self._parameter_state(unit),
                    }

    def _save(self, dataloader, step):
        assert self.checkpoint is not None
        data = (
            {"state": dataloader.state_dict()}
            if hasattr(dataloader, "state_dict")
            else {"consumed_batches": self.consumed_batches, "prefix_hashes": self.consumed_hashes}
        )
        return self.checkpoint.save(
            self.config.checkpoint_dir,
            step,
            {
                "parameters": {unit.name: self._parameter_state(unit) for unit in self.units},
                "optimizers": {
                    name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
                },
                "metrics": self.metrics,
                "best": self.best,
                "data": data,
            },
        )

    def _restore(self, dataloader):
        assert self.checkpoint is not None
        step, state = self.checkpoint.load(self.config.resume_from)

        def restore_cursor():
            if "state" in state["data"]:
                dataloader.load_state_dict(state["data"]["state"])
                return iter(dataloader)
            self.consumed_batches = state["data"]["consumed_batches"]
            hashes = state["data"]["prefix_hashes"]
            if len(hashes) != self.consumed_batches:
                raise ValueError("Repair checkpoint data cursor is incomplete.")
            batches = iter(dataloader)
            for expected in hashes:
                if _batch_hash(next(batches)) != expected:
                    raise ValueError("Consumed sample content/order changed.")
            self.consumed_hashes = list(hashes)
            return batches

        batches = self.checkpoint.restore_data_cursor(restore_cursor)
        for unit in self.units:
            self._load_parameters(unit, state["parameters"][unit.name])
            self.optimizers[unit.name].load_state_dict(state["optimizers"][unit.name])
        self.metrics, self.best = state["metrics"], state["best"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"])
        return step, batches

    def _publish(self):
        if self.runtime:
            return self.adapter.publish(self.student, self.config, self.metrics)
        destination = Path(self.config.output_dir).resolve()
        pending = destination.parent / ("." + destination.name + "-" + uuid.uuid4().hex)
        pending.mkdir(parents=True)
        mto.save(self.student, pending / "modelopt.pt")
        (pending / "metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
        (pending / "config.json").write_text(self.config.model_dump_json(indent=2) + "\n")
        os.replace(pending, destination)
        return {
            "checkpoint": str(destination / "modelopt.pt"),
            "metrics": str(destination / "metrics.json"),
        }

    def _record_runtime(self, started, devices):
        for device in devices:
            torch.cuda.synchronize(device)
        local = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": sum(
                torch.cuda.max_memory_allocated(device) for device in devices
            ),
            "cuda_devices": len(devices),
        }
        values = [local]
        if self.runtime:
            values = self.runtime.gather_values(local)
        self.metrics.update(
            elapsed_seconds=max(value["elapsed_seconds"] for value in values),
            elapsed_seconds_local=local["elapsed_seconds"],
            peak_allocated_bytes_local=local["peak_allocated_bytes"],
            peak_allocated_bytes_max=max(value["peak_allocated_bytes"] for value in values),
            # Process lifetime in this call, including validation/save waits.
            # Scheduler allocation hours include startup and must be read from Slurm.
            process_gpu_hours=sum(
                value["elapsed_seconds"] * value["cuda_devices"] for value in values
            )
            / 3600,
        )

    def run(self, dataloader, validation):
        if validation is not None and iter(validation) is validation:
            raise ValueError("Validation must be re-iterable for baseline and final measurements.")
        if self.config.selection != "final" and validation is None:
            raise ValueError("Best-state selection requires validation data.")
        if (self.config.resume_from or self.config.checkpoint_dir) and not (
            isinstance(dataloader, Sequence)
            or all(hasattr(dataloader, name) for name in ("state_dict", "load_state_dict"))
        ):
            raise ValueError(
                "Checkpointed repair requires a stateful dataloader or deterministic batch Sequence."
            )
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        batches = iter(dataloader)
        flags = {
            module: module.training
            for model in (self.student, self.teacher)
            for module in model.modules()
        }
        devices = {
            value.device
            for model in (self.student, self.teacher)
            for value in chain(model.parameters(), model.buffers())
            if value.is_cuda
        }
        mesh = getattr(getattr(self.adapter, "mesh_context", None), "device_mesh", None)
        if mesh is not None and mesh.device_type == "cuda":
            devices.add(torch.device("cuda", torch.cuda.current_device()))
        for device in devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            self.student.eval()
            self.teacher.eval()
            self._guard(self._validate_output)
            first_batch = self._guard(lambda: self._next_first(batches))
            self._prepare(first_batch)
            batches = chain((first_batch,), batches)
            del first_batch
            start = 0
            if self.config.resume_from:
                start, batches = self._restore(dataloader)
            elif validation is not None:
                self._validate(validation, baseline=True)
            for step in range(start + 1, self.config.updates + 1):
                window = self._guard(lambda: self._window(batches, dataloader))
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = group["initial_lr"] * _learning_rate_factor(self.config, step)
                totals = self._measure(window, True)
                row: dict[str, Any] = {"step": step, "units": {}}
                for unit in self.units:
                    if unit.name not in totals:
                        raise ValueError(f"Training did not execute repair boundary {unit.name!r}.")
                    error, count, energy, tokens = totals[unit.name].tolist()
                    optimizer = self.optimizers[unit.name]
                    denominator = energy if self.config.loss == "normalized_mse" else count
                    if denominator <= 0:
                        raise ValueError("Repair requires a nonempty objective denominator.")
                    parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    if self.runtime:
                        self.runtime.reduce_gradients(parameters)
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(denominator)
                    norm = (
                        self.runtime.clip_gradients(parameters, self.config.gradient_clip_norm)
                        if self.runtime
                        else nn.utils.clip_grad_norm_(
                            parameters, self.config.gradient_clip_norm, error_if_nonfinite=True
                        )
                    )
                    optimizer.step()
                    self.policies[unit.name].project_()
                    row["units"][unit.name] = {
                        "mse": error / count,
                        "elements": count,
                        "gradient_norm": float(norm),
                        "valid_tokens": int(tokens),
                        "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    }
                self.metrics["history"].append(row)
                self.metrics["updates"] = step
                if step == start + 1 or step % 16 == 0:
                    values = list(row["units"].values())
                    print_rank_0(
                        json.dumps(
                            {
                                "repair_step": step,
                                "total_steps": self.config.updates,
                                "mean_unit_mse": sum(value["mse"] for value in values)
                                / max(1, len(values)),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )
                    )
                if (
                    validation is not None
                    and self.config.validation_interval
                    and step % self.config.validation_interval == 0
                ):
                    self._validate(validation)
                if self.config.checkpoint_dir and (
                    step % self.config.checkpoint_interval == 0 or step == self.config.updates
                ):
                    self._save(dataloader, step)
            if validation is not None:
                self._validate(validation)
            if self.config.selection == "best_validation_per_unit":
                for unit in self.units:
                    self._load_parameters(unit, self.best[unit.name]["parameters"])
                    self.metrics["units"][unit.name]["selected_step"] = self.best[unit.name]["step"]
                    self.metrics["units"][unit.name]["selected_validation_mse"] = self.best[
                        unit.name
                    ]["mse"]
            for unit in self.units:
                policy = self.policies[unit.name]
                if self.runtime:
                    self.runtime.finalize_policy(
                        self.student.get_submodule(unit.student_path), policy
                    )
                else:
                    policy.finalize_()
        finally:
            for policy in self.policies.values():
                policy.close()
            if self.runtime:
                self.runtime.close()
            for module, training in flags.items():
                module.training = training
        self._record_runtime(started, devices)
        paths = self._publish() if self.config.output_dir else {}
        self._record_runtime(started, devices)
        if paths:
            Path(paths["metrics"]).write_text(json.dumps(self.metrics, indent=2) + "\n")
        return LocalDistillationResult(self.student, self.metrics, paths)
