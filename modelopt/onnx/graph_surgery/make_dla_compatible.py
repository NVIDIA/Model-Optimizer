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

"""Load, orchestrate, validate and save the documented DLA transformation phases."""

import copy
import gc
from collections.abc import Iterable
from pathlib import Path

import onnx
from onnxscript import ir

from ..logging_config import logger
from .dla_transforms import stages
from .dla_transforms.schemas import upgrade_quantizers
from .dla_transforms.validation import quantized_types, validate

__all__ = ["dla_make_dla_compatible"]


def _transform_make_dla_compatible(
    model: onnx.ModelProto,
    *,
    qdq_quantized_dtypes: Iterable[int | str] | None = None,
    remove_all_qdq: bool = False,
    force_fp16_activations: bool = False,
    verbose: bool = True,
) -> onnx.ModelProto:
    """Apply the IR pipeline to an independent model, refusing unsupported input forms."""
    validate(model)
    if remove_all_qdq and qdq_quantized_dtypes is not None:
        raise ValueError("Specify remove_all_qdq or qdq_quantized_dtypes, not both")
    dtypes = quantized_types(qdq_quantized_dtypes)
    if any(item.domain == "ai.onnx" for item in (*model.opset_import, *model.graph.node)):
        # ORT shape inference requires the canonical standard domain; retain caller ownership.
        model = copy.deepcopy(model)
        for node in model.graph.node:
            if node.domain == "ai.onnx":
                node.domain = ""
        imports = {
            "" if item.domain == "ai.onnx" else item.domain: item.version
            for item in model.opset_import
        }
        del model.opset_import[:]
        model.opset_import.extend(
            onnx.helper.make_opsetid(domain, v) for domain, v in imports.items()
        )
    if force_fp16_activations:
        # Standard Q/DQ supports FP16 floating-point operands/results only from opset 19.
        # Upgrade before inference and rewrites so later FP16 conversion remains schema-valid.
        model = upgrade_quantizers(model)
    # ORT's optional, heavy inference import is needed for contrib QDQ typing.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    typed = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    working = ir.serde.deserialize_model(typed)
    counts = stages.run(
        working,
        quantized_dtypes=dtypes,
        remove_all_qdq=remove_all_qdq,
        force_fp16_activations=force_fp16_activations,
    )
    result = ir.serde.serialize_model(working)
    # Release the cyclic IR and its tensor owners before inference/checking copies data.
    del typed, working
    gc.collect()
    result = SymbolicShapeInference.infer_shapes(result, auto_merge=True)
    onnx.checker.check_model(result, full_check=True)
    if verbose:
        logger.info("DLA transformations applied: %s", counts)
    return result


def dla_make_dla_compatible(
    model_path: str,
    output_path: str,
    *,
    use_external_data: bool = True,
    external_data_name: str | None = None,
    qdq_quantized_dtypes: Iterable[int | str] | None = None,
    remove_all_qdq: bool = False,
    force_fp16_activations: bool = False,
    verbose: bool = True,
) -> onnx.ModelProto:
    """Apply DLA graph surgeries and save an ONNX model with native public shapes.

    Supports top-level graphs with standard opset 17, 18, 20 or 21. Always runs
    preprocessing, connected-pattern rewrites, individual-operator rewrites and
    cleanup in that order, followed by optional FP16 preparation.

    Zero-branch Where lowering assumes finite FP32 values and may change signed
    zero. Optional FP16 L2 decomposition uses a denominator floor of 2**-14,
    with possible overflow and underflow.
    FP16 QDQ may upgrade accepted opset 17/18 inputs to opset 19 before any phase.
    See dla_transforms/README.md for supported forms and how to add a transform.

    The returned model retains initializer data in memory, including when the
    saved model uses external data.
    """
    model = _transform_make_dla_compatible(
        onnx.load(model_path, load_external_data=True),
        qdq_quantized_dtypes=qdq_quantized_dtypes,
        remove_all_qdq=remove_all_qdq,
        force_fp16_activations=force_fp16_activations,
        verbose=verbose,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # External-data saving clears raw_data; preserve the returned model's weights.
    onnx.save(
        copy.deepcopy(model) if use_external_data else model,
        output_path,
        save_as_external_data=use_external_data,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
    )
    return model
