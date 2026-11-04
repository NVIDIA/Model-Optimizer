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

"""Forward-only teacher stages followed by stage-local student replay.

Adapted from feature/puzzletron_v2's AutoModel local-KD/scoring recipes.
Calling eval on a training pipeline schedule still issues backward P2P operations.
"""

import torch

from ...layerwise_distillation_model import local_replay_boundaries
from ...local_units import teacher_forward_kwargs

__all__ = []


def _validate_pipeline(pipeline, teacher):
    if pipeline is None:
        raise ValueError("PP repair requires the existing teacher AutoPipeline.")
    info = pipeline.info
    if (
        not info.enabled
        or not info.model_parts
        or len(info.model_parts) != 1
        or not info.stages
        or len(info.stages) != 1
        or getattr(info.schedule, "_stage", None) is None
    ):
        raise ValueError(
            "Local repair requires non-interleaved PP with one physical stage per rank."
        )
    if info.model_parts[0] is not teacher:
        raise ValueError("The teacher must be the current AutoPipeline's local model part.")


class _TeacherPipeline:
    def __init__(self, pipeline, teacher):
        _validate_pipeline(pipeline, teacher)
        self.pipeline = pipeline
        self.original_schedule = pipeline.info.schedule
        self.stage = pipeline.info.stages[0]
        self.original_backward = self.stage.has_backward
        self.active = False
        self.sequence_length = None
        self.teacher = teacher
        self.original_meta_hook = getattr(teacher, "get_pipeline_stage_metas", None)
        self.had_meta_override = "get_pipeline_stage_metas" in teacher.__dict__
        self.original_meta_override = teacher.__dict__.get("get_pipeline_stage_metas")
        self.trim_logits = "logits_to_keep" in teacher_forward_kwargs(teacher, {})

    def _activate(self, sequence_length):
        if not self.active:
            from torch.distributed.pipelining.schedules import _ScheduleForwardOnly

            schedule = self.original_schedule
            self.pipeline.info.schedule = _ScheduleForwardOnly(
                self.stage,
                n_microbatches=1,
                loss_fn=None,
                args_chunk_spec=getattr(schedule, "_args_chunk_spec", None),
                kwargs_chunk_spec=getattr(schedule, "_kwargs_chunk_spec", None),
                output_merge_spec=getattr(schedule, "_output_merge_spec", None),
                scale_grads=False,
            )
            # Native static metadata must describe the forward-only stage too.
            self.pipeline._pp_current_seq_len = None
            if self.trim_logits and self.stage.is_last:
                self.teacher.get_pipeline_stage_metas = self._limited_output_metas
            self.active = True
        self.pipeline.update_seq_len(sequence_length)
        self.sequence_length = sequence_length

    def _limited_output_metas(self, *, is_first, microbatch_size, seq_len, dtype):
        if self.original_meta_hook is not None:
            inputs, outputs = self.original_meta_hook(
                is_first=is_first, microbatch_size=microbatch_size, seq_len=seq_len, dtype=dtype
            )
        else:
            from nemo_automodel.components.distributed.pipelining.functional import (
                _get_hidden_and_vocab_size,
            )

            hidden, vocab = _get_hidden_and_vocab_size(self.teacher.config)
            inputs = (
                torch.empty(
                    (microbatch_size, seq_len) if is_first else (microbatch_size, seq_len, hidden),
                    device="meta",
                    dtype=torch.long if is_first else dtype,
                ),
            )
            outputs = (torch.empty(microbatch_size, seq_len, vocab, device="meta", dtype=dtype),)
        if getattr(self.teacher, "lm_head", None) is None or getattr(
            self.teacher, "_pp_return_hidden_states", False
        ):
            return inputs, outputs
        # Only terminal logits shrink. Inter-stage activations retain every token.
        return inputs, (outputs[0][..., -1:, :], *outputs[1:])

    def run(self, teacher, units, batch, adapter, consume):
        inputs = batch["input_ids"]
        if inputs.shape[0] != self.pipeline.pp_microbatch_size:
            raise ValueError("Repair microbatch must match AutoPipeline.pp_microbatch_size.")
        self._activate(inputs.shape[1])
        kwargs = {
            key: value
            for key, value in batch.items()
            if key
            not in ("input_ids", "labels", "valid_mask", "loss_mask", "sample_ids", "document_ids")
        }
        kwargs = teacher_forward_kwargs(teacher, kwargs)
        records = []
        try:
            with (
                local_replay_boundaries(
                    teacher,
                    units,
                    adapter,
                    batch,
                    lambda unit, record: records.append((unit, record)),
                ),
                torch.no_grad(),
            ):
                # Finish the current teacher schedule before any student
                # backward. Records are bounded to this microbatch and stage.
                self.pipeline.step(inputs if self.pipeline.info.has_first_stage else None, **kwargs)
            while records:
                unit, record = records.pop(0)
                consume(unit, record)
                del record
        finally:
            records.clear()

    def prime(self, student, teacher, batch, adapter):
        def prime_student(module, args, kwargs):
            args, kwargs = adapter.snapshot_inputs(args, kwargs)
            with torch.no_grad():
                student(*args, **kwargs)

        handle = teacher.register_forward_pre_hook(prime_student, with_kwargs=True)
        try:
            microbatch = next(adapter.microbatches(batch, self.pipeline.pp_microbatch_size))
            adapter.teacher_microbatch(teacher, [], microbatch, lambda unit, record: None)
        finally:
            handle.remove()

    def close(self):
        if self.active:
            self.pipeline.info.schedule = self.original_schedule
            self.stage.has_backward = self.original_backward
            if self.trim_logits and self.stage.is_last:
                if self.had_meta_override:
                    self.teacher.get_pipeline_stage_metas = self.original_meta_override
                else:
                    del self.teacher.get_pipeline_stage_metas
            self.pipeline._pp_current_seq_len = None
            if self.sequence_length is not None:
                self.pipeline.update_seq_len(self.sequence_length)
            self.active = False
