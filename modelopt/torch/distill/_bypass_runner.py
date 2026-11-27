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

"""Full-global-batch IQ Bypass updates over streamed independent boundaries."""

import logging
import random
from functools import partial

import numpy as np
import torch

from modelopt.torch.quantization.repair import (
    IQRepairPolicy,
    _validate_ownership,
    _validate_quantizer,
)
from modelopt.torch.quantization.utils.core_utils import (
    enable_weight_access_and_writeback,
    module_name_maps,
)

from ._bypass_capture import mse_statistics
from ._bypass_optimizer import Lion, learning_rate

_LOG = logging.getLogger(__name__)
_OPTIMIZERS = {"lion": Lion}


class _BypassRunner:
    def __init__(self, student, teacher, units, plan, config, adapter):
        self.student, self.teacher, self.units = student, teacher, units
        self.plan, self.config, self.adapter = plan, config, adapter
        self.policies = {}
        self.optimizers = {}
        self.fields = {}
        self.modules = {u.name: student.get_submodule(u.student_path) for u in units}
        self.names = module_name_maps(student)

    def _weight_access(self, name, *, writeback=True, rollback=False):
        return enable_weight_access_and_writeback(
            self.modules[name], self.student, self.names, writeback=writeback, rollback=rollback
        )

    def _window(self, batches):
        window, samples = [], 0
        while samples < self.config.global_batch_size:
            try:
                batch = next(batches)
            except StopIteration as exc:
                raise ValueError(
                    "Bypass data exhausted; provide unique samples for every update"
                ) from exc
            self.adapter.validate_batch(batch, self.config)
            samples += self.adapter.batch_size(batch)
            if samples > self.config.global_batch_size:
                raise ValueError("Dataloader batches must exactly fill the configured global batch")
            window.append(batch)
        return window

    def _measure(self, batches, training):
        statistics = {}

        def consume(unit, record):
            with torch.enable_grad() if training else torch.no_grad():
                prediction = self.adapter.replay(self.modules[unit.name], record)
                error, count = mse_statistics(prediction, record.target, record.valid_mask)
                if training:
                    error.backward()
                if unit.name not in statistics:
                    statistics[unit.name] = [error.detach(), count.detach()]
                else:
                    statistics[unit.name][0].add_(error.detach())
                    statistics[unit.name][1].add_(count)

        for batch in batches:
            for microbatch in self.adapter.microbatches(batch, self.config.micro_batch_size):
                self.adapter.teacher_microbatch(self.teacher, self.units, microbatch, consume)
        if set(statistics) != set(self.modules):
            raise ValueError("Every selected Bypass boundary must execute in the teacher forward")
        return statistics

    def _update(self, statistics, step):
        # Normalize and clip every unit before any update, so invalid gradients cannot
        # partially advance the model. Counts cover the entire global batch.
        for unit in self.units:
            count = statistics[unit.name][1]
            if count.item() == 0:
                raise ValueError("A Bypass update has no valid target elements")
            fields = self.fields[unit.name]
            for parameter in fields:
                if parameter.grad is not None:
                    parameter.grad.div_(count.to(parameter.device))
            torch.nn.utils.clip_grad_norm_(
                fields, self.config.max_grad_norm, error_if_nonfinite=True
            )
        lr = learning_rate(self.config, step)
        for unit in self.units:
            optimizer = self.optimizers[unit.name]
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            self.policies[unit.name].project_()
        if step == 1 or step % 16 == 0 or step == self.config.num_steps:
            _LOG.info(
                "Bypass step=%d/%d lr=%g mse=%s",
                step,
                self.config.num_steps,
                lr,
                {name: (error / count).item() for name, (error, count) in statistics.items()},
            )

    def run(self, train_dataloader, validation_dataloader=None):
        """Finalize the supplied student and restore all caller modes/gradient flags."""
        devices = sorted(
            {
                p.device.index
                for model in (self.student, self.teacher)
                for p in (*model.parameters(), *model.buffers())
                if p.is_cuda
            }
        )
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(self.config.seed)
                np.random.seed(self.config.seed % 2**32)
                torch.random.default_generator.manual_seed(self.config.seed)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(self.config.seed)
                return self._run(train_dataloader, validation_dataloader, devices)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _run(self, train_dataloader, validation_dataloader, devices):
        batches = iter(train_dataloader)
        first = self._window(batches)
        modes = {
            module: module.training
            for root in (self.student, self.teacher)
            for module in root.modules()
        }
        teacher_flags = [
            (module, name, parameter.requires_grad)
            for module in self.teacher.modules()
            for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False)
        ]
        error = None
        try:
            self.student.eval()
            self.teacher.eval().requires_grad_(False)
            for index, unit in enumerate(self.units):
                policy = IQRepairPolicy()
                self.policies[unit.name] = policy
                with self._weight_access(unit.name, writeback=False):
                    fields = policy.prepare_plan(self.plan, index)
                self.fields[unit.name] = fields
                self.optimizers[unit.name] = _OPTIMIZERS[self.config.optimizer](
                    fields, lr=self.config.max_lr
                )
            for step in range(1, self.config.num_steps + 1):
                window = first if step == 1 else self._window(batches)
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                with torch.profiler.record_function("modelopt.bypass.update"):
                    self._update(self._measure(window, training=True), step)
                window.clear()
            states = {
                binding: state
                for policy in self.policies.values()
                for binding, state, _, _ in policy.adapters
            }
            for binding, state in states.items():
                _validate_quantizer(binding.quantizer)
                if binding.quantizer.num_bits != state.name:
                    raise ValueError("The native IQ format changed during repair")
            _validate_ownership((self.student,), (states,))
            del states
            for name, policy in self.policies.items():
                policy.finalize_(
                    weight_access=partial(self._weight_access, name),
                )
            if validation_dataloader is not None:
                statistics = self._measure(validation_dataloader, training=False)
                _LOG.info(
                    "Bypass final validation mse=%s",
                    {n: (e / c).item() for n, (e, c) in statistics.items()},
                )
            return self.student
        except BaseException as exc:
            error = exc
            raise
        finally:
            cleanup_error = None
            for name, policy in self.policies.items():
                try:
                    if policy._pending_rollback is not None:
                        policy._restore_publication(partial(self._weight_access, name), None)
                    policy.close()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        cleanup_error.add_note(f"Another Bypass policy cleanup failed: {exc!r}")
            for module, name, flag in teacher_flags:
                module.get_parameter(name).requires_grad_(flag)
            for module, mode in modes.items():
                module.training = mode
            if cleanup_error is not None:
                if error is not None:
                    error.add_note(f"Bypass cleanup also failed: {cleanup_error!r}")
                else:
                    raise cleanup_error
