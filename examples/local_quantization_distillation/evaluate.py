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

"""Paired held-out PPL and full-vocabulary KL(teacher || candidate)."""

import inspect
import json
import math
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from modelopt.torch.distill.local_units import _valid_token_mask
from modelopt.torch.distill.plugins.transformers.output_head import (
    OutputHeadProjection,
    capture_head_inputs,
)
from modelopt.torch.utils import print_rank_0

__all__ = ["diagnostic_statistics", "evaluate_pair"]


@torch.no_grad()
def diagnostic_statistics(
    teacher_logits, candidate_logits, labels, valid_mask, *, token_chunk=128, vocabulary_chunk=8192
):
    """Return candidate NLL/count, KL/count and teacher NLL without a full FP32 copy.

    Labels are *already shifted*, within-document next-token targets. KL includes
    every valid input token, including prompts and document-final tokens. Vocabulary
    chunks use exact full-vocabulary normalizers; this is not a top-k approximation.
    """
    if teacher_logits.shape != candidate_logits.shape or teacher_logits.ndim < 2:
        raise ValueError("Paired logits must have identical token/vocabulary dimensions.")
    if labels.shape != teacher_logits.shape[:-1] or valid_mask.shape != labels.shape:
        raise ValueError("Diagnostic targets and mask must match the local token layout.")
    if min(token_chunk, vocabulary_chunk) < 1:
        raise ValueError("Diagnostic chunk sizes must be positive.")
    vocabulary = teacher_logits.shape[-1]
    teacher_logits = teacher_logits.reshape(-1, vocabulary)
    candidate_logits = candidate_logits.reshape(-1, vocabulary)
    labels, valid_mask = labels.flatten(), valid_mask.flatten().bool()
    totals = teacher_logits.new_zeros(5, dtype=torch.float64)
    for start in range(0, labels.numel(), token_chunk):
        stop = start + token_chunk
        target, valid = labels[start:stop], valid_mask[start:stop]
        predicted = valid & target.ne(-100)
        left, right = teacher_logits[start:stop], candidate_logits[start:stop]
        normalizers = []
        for logits in (left, right):
            norm = torch.full((logits.shape[0],), -torch.inf, device=logits.device)
            for column in range(0, vocabulary, vocabulary_chunk):
                block = logits[:, column : column + vocabulary_chunk].float()
                norm = torch.logaddexp(norm, block.logsumexp(-1))
            normalizers.append(norm)
        teacher_norm, candidate_norm = normalizers
        safe_target = target.masked_fill(~predicted, 0).unsqueeze(-1)
        totals[0] += (
            (candidate_norm - right.gather(-1, safe_target).squeeze(-1).float())[predicted]
            .double()
            .sum()
        )
        totals[1] += predicted.sum()
        totals[4] += (
            (teacher_norm - left.gather(-1, safe_target).squeeze(-1).float())[predicted]
            .double()
            .sum()
        )
        for column in range(0, vocabulary, vocabulary_chunk):
            teacher_logp = (
                left[:, column : column + vocabulary_chunk].float() - teacher_norm[:, None]
            )
            candidate_logp = (
                right[:, column : column + vocabulary_chunk].float() - candidate_norm[:, None]
            )
            kl = (teacher_logp.exp() * (teacher_logp - candidate_logp)).sum(-1)
            totals[2] += kl[valid].double().sum()
        totals[3] += valid.sum()
    return totals


def _model_logits(model, family, inputs, projection=None):
    target = family.teacher_forward_target(model) if family is not None else model
    parameters = inspect.signature(target.forward).parameters
    inputs = dict(inputs)
    for name, value in (("use_cache", False), ("logits_to_keep", 0), ("num_logits_to_keep", 0)):
        if name in parameters:
            inputs[name] = value
    had_override, previous = "forward" in model.__dict__, model.__dict__.get("forward")
    if target is not model:
        model.forward = lambda *args, **kwargs: target(*args, **kwargs)
    try:
        if projection is None:
            output = model(**inputs)
        else:
            with capture_head_inputs(projection.head) as captured:
                model(**inputs)
            if len(captured) != 1:
                raise ValueError("Chunked diagnostics require one pointwise output-head call.")
            output = captured[0]
    finally:
        if target is not model:
            if had_override:
                model.forward = previous
            else:
                delattr(model, "forward")
    logits = (
        output
        if isinstance(output, torch.Tensor)
        else output.logits
        if hasattr(output, "logits")
        else output[0]
    )
    return logits.full_tensor() if isinstance(logits, DTensor) else logits


@torch.no_grad()
def _projected_statistics(
    left, right, teacher, candidate, labels, valid, token_chunk, vocabulary_chunk, slots, sliceable
):
    left, right = left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])
    labels, valid = labels.flatten(), valid.flatten().bool()
    totals = left.new_zeros(5, dtype=torch.float64)
    vocabulary = teacher.head.weight.shape[0]
    for start in range(0, slots, token_chunk):
        end = min(start + token_chunk, labels.numel())
        target, mask = labels[start:end], valid[start:end]
        if start >= labels.numel():
            # Every model/FSDP rank must still execute the same projection schedule.
            a, b = left.new_zeros((1, left.shape[-1])), right.new_zeros((1, right.shape[-1]))
            target = labels.new_full((1,), -100)
            mask = valid.new_zeros((1,))
        else:
            a, b = left[start:end], right[start:end]
        if not sliceable:
            t_logits = teacher.project(a)
            s_logits = candidate.project(b)
            totals += diagnostic_statistics(
                t_logits,
                s_logits,
                target,
                mask,
                token_chunk=token_chunk,
                vocabulary_chunk=vocabulary_chunk,
            )
            del t_logits, s_logits
            continue
        predicted = mask & target.ne(-100)
        t_max = torch.full((a.shape[0],), -torch.inf, device=a.device)
        s_max = torch.full_like(t_max, -torch.inf)
        t_mass = torch.zeros_like(t_max, dtype=torch.float64)
        s_mass, difference = torch.zeros_like(t_mass), torch.zeros_like(t_mass)
        t_target, s_target = torch.zeros_like(t_max), torch.zeros_like(t_max)
        for column in range(0, vocabulary, vocabulary_chunk):
            columns = slice(column, min(column + vocabulary_chunk, vocabulary))
            t_logits, s_logits = (
                teacher.project(a, columns).float(),
                candidate.project(b, columns).float(),
            )
            next_t = torch.maximum(t_max, t_logits.amax(-1))
            next_s = torch.maximum(s_max, s_logits.amax(-1))
            t_scale = (t_max.double() - next_t.double()).exp()
            s_scale = (s_max.double() - next_s.double()).exp()
            t_exp = (t_logits - next_t[:, None]).exp()
            s_exp = (s_logits - next_s[:, None]).exp()
            t_mass = t_mass * t_scale + t_exp.sum(-1, dtype=torch.float64)
            s_mass = s_mass * s_scale + s_exp.sum(-1, dtype=torch.float64)
            difference = difference * t_scale + (t_exp * (t_logits - s_logits)).sum(
                -1, dtype=torch.float64
            )
            t_max, s_max = next_t, next_s
            owned = predicted & (target >= column) & (target < columns.stop)
            index = (target - column).clamp(0, columns.stop - column - 1).unsqueeze(-1)
            t_target += torch.where(owned, t_logits.gather(-1, index).squeeze(-1), 0)
            s_target += torch.where(owned, s_logits.gather(-1, index).squeeze(-1), 0)
        t_norm = t_max.double() + t_mass.log()
        s_norm = s_max.double() + s_mass.log()
        totals[0] += (s_norm - s_target)[predicted].sum()
        totals[1] += predicted.sum()
        totals[4] += (t_norm - t_target)[predicted].sum()
        # KL = E_teacher[teacher_logits - candidate_logits] + log Z_candidate - log Z_teacher.
        totals[2] += (difference / t_mass + s_norm - t_norm)[mask].sum()
        totals[3] += mask.sum()
    return totals, "chunked_linear" if sliceable else "chunked_head"


def _perplexity(mean_cross_entropy):
    # PPL is secondary: an unrepresentable exponential must not hide finite KL.
    try:
        value = math.exp(mean_cross_entropy)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


@torch.no_grad()
def evaluate_pair(
    teacher, candidate, dataloader, *, adapter=None, token_chunk=128, vocabulary_chunk=8192
):
    """Compare matching full HF tasks on document-isolated, full-context inputs.

    Labels are already shifted next-token targets. Supply a Transformers adapter
    for packed or distributed models. Diagnostics replicate context across CP;
    only unique data lanes contribute to the reported token counts.
    """
    if min(token_chunk, vocabulary_chunk) < 1:
        raise ValueError("Diagnostic chunk sizes must be positive.")
    family, context = None, None
    if adapter is not None:
        from modelopt.torch.distill.local_parallel import _all_rank_values
        from modelopt.torch.distill.plugins.transformers import transformers_family
        from modelopt.torch.distill.plugins.transformers.packing import (
            document_batching,
            empty_document_batch,
        )

        family = adapter.family_adapter or transformers_family(teacher)
        context = adapter.context
        if adapter.allows_empty_stage:
            raise ValueError("Whole-model diagnostics require PP=1.")
    if family is None:
        # Optional HF family resolution; ordinary Torch/logits-only tasks still work.
        from modelopt.torch.distill.plugins.transformers import transformers_family

        family = transformers_family(teacher)
    heads, error = None, None
    try:
        heads = (OutputHeadProjection(teacher, family), OutputHeadProjection(candidate, family))
    except (ValueError, TypeError, AttributeError) as exc:
        if adapter is None:
            raise
        error = f"{type(exc).__name__}: {exc}"
    if adapter is not None:
        adapter.synchronize_error(error)
    assert heads is not None
    use_head = all(head.head is not None for head in heads)
    if adapter is not None:
        adapter.synchronize_error(
            "Every model rank must expose matching output-head capabilities."
            if len(set(_all_rank_values(context, use_head))) != 1
            else None
        )
    modes = set()
    teacher_flags = {
        module: module.training for model in (teacher, candidate) for module in model.modules()
    }
    device = next(candidate.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    try:
        teacher.eval()
        candidate.eval()
        iterator, exhausted, index = iter(dataloader), object(), 0
        batch: Any
        while True:
            batch, error = exhausted, None
            try:
                batch = next(iterator)
            except StopIteration:
                pass
            except Exception as exc:
                if adapter is None:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            if adapter is not None:
                adapter.synchronize_microbatch(batch is not exhausted, error)
            if batch is exhausted:
                break
            index += 1
            if adapter is None:
                if batch.get("_packed_seq_ids") is not None:
                    raise ValueError("Packed diagnostics require the model's Transformers adapter.")
                documents, slots = [batch], 1
            else:
                documents, error = [], None
                try:
                    with document_batching(family.document_batching_supported(teacher)):
                        documents = family.document_batches(candidate, batch)
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                adapter.synchronize_error(error)
                slots = max(_all_rank_values(context, len(documents)))
            for slot in range(slots):
                document = documents[slot] if slot < len(documents) else empty_document_batch(batch)
                error = None
                try:
                    labels = document.get("labels", torch.full_like(document["input_ids"], -100))
                    valid = _valid_token_mask(document)
                    if valid is None:
                        valid = torch.ones_like(labels, dtype=torch.bool)
                    if labels.shape != document["input_ids"].shape or valid.shape != labels.shape:
                        raise ValueError("Diagnostic labels and validity must match token IDs.")
                    if labels.dtype != torch.long:
                        raise ValueError("Diagnostic labels must use torch.long token IDs.")
                    if use_head:
                        vocabulary = heads[0].head.weight.shape[0]
                        if vocabulary != heads[1].head.weight.shape[0]:
                            raise ValueError("Paired output vocabularies differ.")
                        if bool(
                            (
                                valid & labels.ne(-100) & ((labels < 0) | (labels >= vocabulary))
                            ).any()
                        ):
                            raise ValueError(
                                "Diagnostic targets are outside the output vocabulary."
                            )
                        for projection in heads:
                            projection.family.transform_logits(
                                projection.model, torch.zeros((1, 1), device=labels.device)
                            )
                    kwargs = (
                        adapter.prepare_batch(teacher, document)
                        if adapter is not None
                        else {
                            key: value
                            for key, value in document.items()
                            if key
                            not in (
                                "labels",
                                "valid_mask",
                                "sample_ids",
                                "document_ids",
                                "loss_mask",
                                "padding_mask",
                            )
                        }
                    )
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                    if adapter is None:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                if adapter is not None:
                    adapter.synchronize_error(error)
                target = _model_logits(teacher, family, kwargs, heads[0] if use_head else None)
                prediction = _model_logits(
                    candidate, family, kwargs, heads[1] if use_head else None
                )
                if use_head:
                    error = (
                        "Captured output-head inputs must cover every diagnostic token."
                        if target.shape[:-1] != labels.shape
                        or prediction.shape[:-1] != labels.shape
                        else None
                    )
                    if adapter is not None:
                        adapter.synchronize_error(error)
                    elif error:
                        raise ValueError(error)
                    token_slots = (
                        max(_all_rank_values(context, labels.numel()))
                        if adapter is not None
                        else labels.numel()
                    )
                    with heads[0].weight_window(), heads[1].weight_window():
                        sliceable = heads[0].sliceable and heads[1].sliceable
                        if adapter is not None:
                            sliceable = all(_all_rank_values(context, sliceable))
                        values, mode = _projected_statistics(
                            target,
                            prediction,
                            *heads,
                            labels,
                            valid,
                            token_chunk,
                            vocabulary_chunk,
                            token_slots,
                            sliceable,
                        )
                    totals += values
                    modes.add(mode)
                else:
                    error = None
                    if target.shape != prediction.shape or target.shape[:-1] != labels.shape:
                        error = "Paired logits and diagnostic targets must have matching shapes."
                    elif bool(
                        (
                            valid & labels.ne(-100) & ((labels < 0) | (labels >= target.shape[-1]))
                        ).any()
                    ):
                        error = "Diagnostic targets are outside the output vocabulary."
                    if adapter is not None:
                        adapter.synchronize_error(error)
                    elif error:
                        raise ValueError(error)
                    totals += diagnostic_statistics(
                        target,
                        prediction,
                        labels,
                        valid.bool(),
                        token_chunk=token_chunk,
                        vocabulary_chunk=vocabulary_chunk,
                    )
                    modes.add("full_logits")
                del target, prediction
            if index == 1 or index % 4 == 0:
                print_rank_0(
                    json.dumps(
                        {
                            "diagnostic_batches_local": index,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                )
        data_mesh = getattr(context, "data_mesh", None)
        if data_mesh is not None and data_mesh.size() > 1:
            dist.all_reduce(totals, group=data_mesh.get_group())
        nll, target_count, kl, token_count, teacher_nll = totals.tolist()
        if token_count <= 0:
            raise ValueError("Held-out diagnostics require valid tokens.")
        cross_entropy = nll / target_count if target_count else None
        teacher_cross_entropy = teacher_nll / target_count if target_count else None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "projection_modes": sorted(modes),
            "ppl": _perplexity(cross_entropy) if cross_entropy is not None else None,
            "mean_cross_entropy": cross_entropy,
            "teacher_ppl": _perplexity(teacher_cross_entropy)
            if teacher_cross_entropy is not None
            else None,
            "teacher_mean_cross_entropy": teacher_cross_entropy,
            "kld_teacher_candidate": kl / token_count,
            "next_token_targets": int(target_count),
            "kl_tokens": int(token_count),
            "elapsed_seconds_local": time.perf_counter() - started,
            "peak_allocated_bytes_local": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
    finally:
        for module, training in teacher_flags.items():
            module.training = training
