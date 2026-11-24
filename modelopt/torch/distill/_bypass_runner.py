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

from modelopt.torch.quantization.repair import IQRepairPolicy
from modelopt.torch.quantization.utils.core_utils import (
    enable_weight_access_and_writeback,
    module_name_maps,
)

from ._bypass_capture import mse_statistics
from ._bypass_optimizer import Lion, learning_rate

_LOG = logging.getLogger(__name__)
_OPTIMIZERS = {"lion": Lion}


class _BypassRunner:
    def __init__(self, student, teacher, units, plan, config, adapter, runtime=None):
        self.student, self.teacher, self.units = student, teacher, units
        self.plan, self.config, self.adapter = plan, config, adapter
        self.runtime = runtime
        self.policies = {}
        self.optimizers = {}
        self.fields = {}
        self.modules = {u.name: student.get_submodule(u.student_path) for u in units}
        self.names = module_name_maps(student)

    def _weight_access(self, name, *, writeback=True, rollback=False):
        return enable_weight_access_and_writeback(
            self.modules[name], self.student, self.names, writeback=writeback, rollback=rollback
        )

    def _iterator(self, loader):
        iterator, error = None, None
        try:
            iterator = iter(loader)
        except Exception as exc:
            if self.runtime is None:
                raise
            error = f"{type(exc).__name__}: {exc}"
        if self.runtime is not None:
            self.runtime.synchronize_error(error)
        assert iterator is not None
        return iterator

    def _validation_batches(self, loader):
        iterator = self._iterator(loader)
        while True:
            batch, present, error, count = None, False, None, 0
            try:
                batch = next(iterator)
                present = True
                self.adapter.validate_batch(batch, self.config)
                count = (
                    self.adapter.batch_size(batch) + self.config.micro_batch_size - 1
                ) // self.config.micro_batch_size
            except StopIteration:
                pass
            except Exception as exc:
                if self.runtime is None:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            if self.runtime is not None:
                high_present, low_present, high_count, low_count, failed = self.runtime.max_values(
                    [int(present), -int(present), count, -count, int(error is not None)]
                )
                if failed:
                    errors = [message for message in self.runtime.gather_values(error) if message]
                    raise ValueError(f"Bypass validation reader failed: {errors}")
                if high_present != -low_present or high_count != -low_count:
                    raise ValueError("Bypass ranks have unequal validation schedules")
            if not present:
                return
            yield batch

    def _window(self, batches):
        window, samples, error = [], 0, None
        batch_size = (
            self.config.global_batch_size if self.runtime is None else self.runtime.local_batch_size
        )
        try:
            while samples < batch_size:
                batch = next(batches)
                self.adapter.validate_batch(batch, self.config)
                samples += self.adapter.batch_size(batch)
                if samples > batch_size:
                    raise ValueError(
                        "Dataloader batches must exactly fill each data lane's global-batch share"
                    )
                window.append(batch)
        except StopIteration:
            error = "Bypass data exhausted; provide unique samples for every update"
        except Exception as exc:
            if self.runtime is None:
                raise
            error = f"{type(exc).__name__}: {exc}"
        if self.runtime is not None:
            self.runtime.synchronize_error(error)
            count = sum(
                (self.adapter.batch_size(b) + self.config.micro_batch_size - 1)
                // self.config.micro_batch_size
                for b in window
            )
            high, low = self.runtime.max_values([count, -count])
            if high != -low:
                raise ValueError("Bypass ranks have unequal microbatch schedules")
        elif error:
            raise ValueError(error)
        return window

    def _measure(self, batches, training):
        statistics = {}

        def consume(unit, record):
            with torch.enable_grad() if training else torch.no_grad():
                if self.runtime is None:
                    prediction = self.adapter.replay(self.modules[unit.name], record)
                else:
                    # PR8's Transformers _document_teacher sets this only inside
                    # CPSequenceLayout.attention(); this slice uses replicated CP replay.
                    record = self.runtime.record(
                        unit, record, native_context=getattr(self.adapter, "_native_context", False)
                    )
                    prediction = self.runtime.replay(
                        self.modules[unit.name], record, self.adapter, training
                    )
                error, count = mse_statistics(prediction, record.target, record.valid_mask)
                if training:
                    error.backward()
                if unit.name not in statistics:
                    statistics[unit.name] = [error.detach(), count.detach()]
                else:
                    statistics[unit.name][0].add_(error.detach())
                    statistics[unit.name][1].add_(count)

        for batch in batches if training else self._validation_batches(batches):
            for microbatch in self.adapter.microbatches(batch, self.config.micro_batch_size):
                self.adapter.teacher_microbatch(self.teacher, self.units, microbatch, consume)
        if self.runtime is not None:
            return self.runtime.reduce_statistics(statistics, self.modules)
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
            if self.runtime is not None:
                self.runtime.reduce_gradients(fields)
            for parameter in fields:
                if parameter.grad is not None:
                    parameter.grad.div_(count.to(parameter.device))
            if self.runtime is None:
                torch.nn.utils.clip_grad_norm_(
                    fields, self.config.max_grad_norm, error_if_nonfinite=True
                )
            else:
                self.runtime.clip_gradients(fields, self.config.max_grad_norm)
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
        batches = self._iterator(train_dataloader)
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
        try:
            self.student.eval()
            self.teacher.eval().requires_grad_(False)
            if self.runtime is not None:
                self.runtime.initialize(
                    self.student, self.teacher, self.units, self.adapter, first[0]
                )
            for index, unit in enumerate(self.units):
                policy = IQRepairPolicy()
                self.policies[unit.name] = policy
                if self.runtime is None:
                    with self._weight_access(unit.name, writeback=False):
                        fields = policy.prepare_plan(self.plan, index)
                else:
                    fields = self.runtime.prepare_policy(self.plan, index, policy)
                self.fields[unit.name] = fields
                self.optimizers[unit.name] = _OPTIMIZERS[self.config.optimizer](
                    fields, lr=self.config.max_lr
                )
            if self.runtime is not None:
                self.runtime.validate_gradient_schedules(self.fields)
            for step in range(1, self.config.num_steps + 1):
                window = first if step == 1 else self._window(batches)
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                with torch.profiler.record_function("modelopt.bypass.update"):
                    self._update(self._measure(window, training=True), step)
                window.clear()
            for name, policy in self.policies.items():
                if self.runtime is None:
                    policy.finalize_(weight_access=partial(self._weight_access, name))
                else:
                    self.runtime.finalize_policy(self.modules[name], policy)
            if validation_dataloader is not None:
                statistics = self._measure(validation_dataloader, training=False)
                _LOG.info(
                    "Bypass final validation mse=%s",
                    {n: (e / c).item() for n, (e, c) in statistics.items()},
                )
            if self.runtime is not None:
                self.student._modelopt_bypass_execution = {
                    "context_parallel_modes": sorted(self.runtime.context_modes),
                    "data_lanes": self.runtime.context.dp_size,
                    "local_batch_size": self.runtime.local_batch_size,
                }
                _LOG.info("Bypass execution: %s", self.student._modelopt_bypass_execution)
            return self.student
        finally:
            for policy in self.policies.values():
                policy.close()
            if self.runtime is not None:
                self.runtime.close()
            for module, name, flag in teacher_flags:
                module.get_parameter(name).requires_grad_(flag)
            for module, mode in modes.items():
                module.training = mode
