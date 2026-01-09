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

"""Utility functions related to Onnx."""

import base64
import inspect
import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from typing import Any

import onnx
import torch
import torch.nn as nn
from onnx import ModelProto
from onnxconverter_common import convert_float_to_float16
from torch.nn.parallel import DataParallel, DistributedDataParallel

from modelopt.onnx.autocast.convert import convert_to_f16
from modelopt.onnx.export import (
    FP8QuantExporter,
    INT4QuantExporter,
    INT8QuantExporter,
    MXFP8QuantExporter,
    NVFP4QuantExporter,
    ONNXQuantExporter,
)
from modelopt.onnx.quantization.qdq_utils import qdq_to_dq, replace_zero_scale_with_smallest_nonzero
from modelopt.onnx.utils import (
    get_input_names,
    get_input_shapes,
    get_node_names,
    get_output_names,
    get_output_shapes,
    infer_shapes,
    remove_node_training_mode,
)
from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers
from modelopt.torch.utils import flatten_tree, standardize_named_model_args
from modelopt.torch.utils._pytree import TreeSpec

from ..utils.onnx_optimizer import Optimizer
from .onnx_utils import check_model_uses_external_data

ModelMetadata = dict[str, Any]
ModelType = Any
ValueInfoType = Any

# a few constants...
DEFAULT_ONNX_OPSET = 20
ONNX_EXPORT_OUT_PREFIX = "out"
TWO_GB = 2 * 1024 * 1024 * 1024


class OnnxBytes:
    """A class to save and load onnx models as bytes."""

    def __init__(self, onnx_load_path: str) -> None:
        """Loads the model from the specified path.

        If the model is loaded without external data format, then it is saved as a dictionary where
        the key is the model name and the value is the model bytes.
        If the model is loaded with external data format, then the model is saved as a dictionary
        where the keys include all the file names in the model directory and the value are the corresponding file bytes.
        For external data format, we assume that the external data for the model is saved in the same directory
        as the model file.

        Args:
            onnx_load_path: The path to load the .onnx model file.
        """
        self.onnx_load_path = os.path.abspath(onnx_load_path)
        self.onnx_model = {}
        self.model_name = ""

        if has_external_data(onnx_load_path):
            onnx_model_dir = os.path.dirname(self.onnx_load_path)
            for onnx_model_file in os.listdir(onnx_model_dir):
                with open(os.path.join(onnx_model_dir, onnx_model_file), "rb") as f:
                    self.onnx_model[onnx_model_file] = f.read()
                if onnx_model_file.endswith(".onnx"):
                    if self.model_name != "":
                        raise ValueError("Multiple onnx files found in the directory")
                    self.model_name = onnx_model_file.replace(".onnx", "")
        else:
            onnx_model_file = os.path.basename(self.onnx_load_path)
            if not onnx_model_file.endswith(".onnx"):
                raise ValueError("The file should be a .onnx file")
            with open(self.onnx_load_path, "rb") as f:
                self.onnx_model[onnx_model_file] = f.read()
            self.model_name = onnx_model_file.replace(".onnx", "")

    def write_to_disk(self, onnx_save_dir: str = "", clean_dir: bool = True) -> None:
        """Write ONNX model(s) to the specified directory.

        Args:
            onnx_save_dir: Directory path for saving. Defaults to current directory if empty.
            clean_dir: Whether to remove existing directory first.
        """
        # Determine save directory
        save_dir = os.path.abspath(onnx_save_dir) if onnx_save_dir else os.getcwd()

        # Clean existing directory if requested
        if clean_dir and os.path.exists(save_dir) and onnx_save_dir:
            print(f"Removing existing directory: {save_dir}")
            shutil.rmtree(save_dir)

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Write model files
        print(f"Writing ONNX model to directory: {save_dir}")
        for filename, file_bytes in self.onnx_model.items():
            with open(os.path.join(save_dir, filename), "wb") as f:
                f.write(file_bytes)

    def to_bytes(self) -> bytes:
        """Returns the bytes of the object that can be restored using the OnnxBytes.from_bytes method."""
        serialized_model = {}
        for file_name, file_bytes in self.onnx_model.items():
            serialized_model[file_name] = base64.b64encode(file_bytes).decode("utf-8")

        # Create a dictionary with all necessary attributes
        data = {
            "onnx_load_path": self.onnx_load_path,
            "model_name": self.model_name,
            "onnx_model": serialized_model,
        }

        return json.dumps(data).encode("utf-8")

    def get_onnx_model_file_bytes(self) -> bytes:
        """Returns the bytes of the onnx model file.

        Note: Even if the model has external data, this function will return the bytes of the main onnx model file.
        To get the bytes of the external data, use the get_external_data_bytes() method.
        """
        return self.onnx_model[self.model_name + ".onnx"]

    @classmethod
    def from_bytes(cls, onnx_bytes: bytes) -> "OnnxBytes":
        """Returns the OnnxBytes object from the bytes."""
        data = json.loads(onnx_bytes.decode("utf-8"))

        # Create a new instance without calling __init__ and set the attributes
        instance = cls.__new__(cls)
        instance.onnx_load_path = data["onnx_load_path"]
        instance.model_name = data["model_name"]
        instance.onnx_model = {}
        for file_name, encoded_bytes in data["onnx_model"].items():
            instance.onnx_model[file_name] = base64.b64decode(encoded_bytes)

        return instance


def _to_expected_onnx_type(val: Any) -> Any:
    """Convert the given value to the expected onnx type.

    During the onnx export process, plain numeric types (floats and ints) are converted to torch
    tensors. This function pre-converts the given val to a tensor in case val is a int or float for
    easier handling of such input values during the onnx export process.
    """
    if isinstance(val, (int, float)):
        return torch.tensor(val).to(type(val))
    return val


def generate_onnx_input(
    model_metadata: ModelMetadata, input: Any | tuple, ignore_nesting: bool = False
) -> dict[str, Any]:
    """Generate input for onnx model from model's forward signature and provided input.

    Args:
        model_metadata: The model's metadata.
        input: A tuple of args/kwargs or torch.Tensor feed into the model's ``forward()`` method,
            see :meth:`standardize_model_args() <modelopt.torch.utils.network.standardize_model_args>`
            for more info on the convention.
        ignore_nesting: If True, only the last part of the nested input name will be considered.
            eg. if the input name is x.y.z, only z will be considered.

    Returns:
        Args flattened into one dictionary with serialized keys compatible with provided onnx.

    .. note::

        This function performs a sanity check on the provided input data to filter out args that
        are constants (instead of input nodes) in the onnx graph.


    Some more relevant background of why we want to flatten the input pytree here:

        * In the onnx export process, nested python data structures (like nested lists, tuples,
            dictionaries) are being recursed into until leaf objects corresponding to tensors are
            encountered.

        * This is used to flatten the input in an onnx model to a list of tensors.

        * However, this is a fairly complex process for the user to understand in case their models
            takes a nested data structure. They have to understand how to manually flatten the data
            structure in the *correct* order in order for them to run inference on a device_model or
            onnx model.

        * With this function this additional complexity can be abstracted away from the user.

        * Example: if the original model took ``[x, {"y":y, "z" : [z1,z2]}]`` they can still provide
            this nested data structure instead of the expected onnx input list of ``[x, y, z1, z2]``
            --> flattening and unflattering is handled internally.
    """
    # get named args and set of params where we added default values
    named_args, args_with_default = standardize_named_model_args(model_metadata["signature"], input)

    # retrieve onnx input names
    onnx_input_names = model_metadata["input_onnx_names"]
    input_none_names = model_metadata["input_none_names"]

    # capture flattened names of args from default values
    named_default_args = {k: v for k, v in named_args.items() if k in args_with_default}
    _, tree_spec_default_args = flatten_tree(named_default_args)

    # capture flattened args without default args that do not appear in onnx graph
    values, tree_spec = flatten_tree(named_args)
    if not ignore_nesting:
        flat_kv = dict(zip(tree_spec.names, values))
    else:
        flat_kv = {k.split(".")[-1]: v for k, v in zip(tree_spec.names, values)}

    # We wanna consider four types of flattened args:
    # 1. Args that appear in the onnx graph
    # 2. Args that are not their default value
    # 3. Args that were provided as None during conversion but are not None right now
    # 4. Args that were provided as None during conversion and are None right now

    args_in_onnx = {k for k in flat_kv if k in onnx_input_names}
    args_not_default = {k for k in flat_kv if k not in tree_spec_default_args.names}
    args_not_none = {k for k, v in flat_kv.items() if k in input_none_names and v is not None}
    args_none = {k for k, v in flat_kv.items() if k in input_none_names and v is None}

    # identify unexpected args from these 4 types
    unexpected_args = ((args_not_default - args_none) | args_not_none) - args_in_onnx
    if unexpected_args:
        print(
            "The following args were provided that do not appear in the onnx graph of your model "
            "since they are treated as constants in the onnx graph:"
            + "\t\n".join(unexpected_args)
            + "\nConsider removing these args from your input that are constants in the onnx model "
            "or recompiling your onnx model with new constant values!"
        )

    # return the args that are relevant for the onnx graph in the right type
    return {k: _to_expected_onnx_type(v) for k, v in flat_kv.items() if k in args_in_onnx}


def optimize(name, onnx_graph, verbose=False):
    """Optimizes onnx graph."""
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    # TODO: fold constants is not working for some models from deploy_models(NestedOutModel, ArgsKwargsModel1)
    # opt.fold_constants()
    # opt.info(name + ": fold_constants")
    onnx_graph = opt.infer_shapes(return_onnx=True)
    opt.info(name + ": shape inference")
    return onnx_graph


def split_args_kwargs(args_tuple):
    """Splits args_tuple into positional arguments and keyword arguments."""
    split_index = len(args_tuple)

    for i, item in enumerate(reversed(args_tuple)):
        if not isinstance(item, dict):
            split_index = len(args_tuple) - i
            break

    pos_args = args_tuple[:split_index]
    kw_args = {}
    for d in args_tuple[split_index:]:
        kw_args.update(d)

    kw_args = None if kw_args == {} else kw_args

    # remove empty dict if it is the last element
    if pos_args[-1] == {}:
        pos_args = pos_args[:-1]

    return pos_args, kw_args


def is_int4_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in INT4 mode.
    This method does not check if the model has been quantized in mixed precision format."""
    for _, module in model.named_modules():
        if (
            hasattr(module, "input_quantizer")
            and hasattr(module, "weight_quantizer")
            and module.weight_quantizer._num_bits == 4
            and module.input_quantizer._disabled
        ):
            return True
    return False


def is_fp4_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in NVFP4 mode."""
    for _, module in model.named_modules():
        if (
            hasattr(module, "input_quantizer")
            and module.input_quantizer.block_sizes
            and module.input_quantizer.block_sizes.get("scale_bits", None) == (4, 3)
        ):
            return True
    return False


def is_mxfp8_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in MXFP8 mode."""
    for _, module in model.named_modules():
        if (
            hasattr(module, "input_quantizer")
            and module.input_quantizer.block_sizes
            and module.input_quantizer.block_sizes.get("scale_bits", None) == (8, 0)
        ):
            return True
    return False


def is_int8_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in INT8 mode."""
    for _, module in model.named_modules():
        if (
            hasattr(module, "weight_quantizer")
            and hasattr(module, "input_quantizer")
            and module.weight_quantizer.is_enabled
            and module.input_quantizer.is_enabled
            and module.weight_quantizer._num_bits == 8
            and module.input_quantizer._num_bits == 8
        ):
            return True
    return False


def is_fp8_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in FP8 mode."""
    for _, module in model.named_modules():
        if (
            hasattr(module, "weight_quantizer")
            and hasattr(module, "input_quantizer")
            and module.weight_quantizer.is_enabled
            and module.input_quantizer.is_enabled
            and module.weight_quantizer._num_bits == (4, 3)
            and module.input_quantizer._num_bits == (4, 3)
            # Exclude MXFP8 which also uses (4,3) but has block_sizes with scale_bits
            and not (
                module.input_quantizer.block_sizes
                and module.input_quantizer.block_sizes.get("scale_bits", None) == (8, 0)
            )
        ):
            return True
    return False


def quantize_weights(model: nn.Module, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Real quantizes the weights in the onnx model.

    Applies weight quantization to an ONNX model based on the quantization scheme detected
    in the PyTorch model. Supports INT4, FP4, and MXFP8 quantization formats.

    The function performs a four-stage process for each detected quantization type:
    1. Pre-process - Restructure the graph for quantization
    2. Compute scales - Calculate quantization scaling factors
    3. Compress weights - Convert weights to the target quantized format
    4. Post-process - Apply any final transformations or cleanup

    Args:
        model (nn.Module): The original PyTorch model used to detect quantization schemes.
            This model should have been quantized using modelopt's quantization APIs.
        onnx_model (onnx.ModelProto): The ONNX model whose weights will be quantized.

    Returns:
        onnx.ModelProto: The ONNX model with quantized weights applied. The returned model
            contains compressed weight tensors in the appropriate quantization format.

    Notes:
        - Multiple quantization formats can be applied sequentially if the model contains
          different quantization schemes for different layers
        - The function checks for INT4, FP4, and MXFP8 quantization in the PyTorch model
        - Each quantization exporter modifies the ONNX graph in-place before returning
    """

    onnx_exporters: list[type[ONNXQuantExporter]] = []
    if is_int4_quantized(model):
        onnx_exporters.append(INT4QuantExporter)
    if is_fp4_quantized(model):
        onnx_exporters.append(NVFP4QuantExporter)
    if is_mxfp8_quantized(model):
        onnx_exporters.append(MXFP8QuantExporter)
    if is_fp8_quantized(model):
        onnx_exporters.append(FP8QuantExporter)
    if is_int8_quantized(model):
        onnx_exporters.append(INT8QuantExporter)

    for onnx_exporter in onnx_exporters:
        onnx_model = onnx_exporter.process_model(onnx_model)

    return onnx_model


def get_onnx_bytes_and_metadata(
    model: nn.Module,
    dummy_input: Any | tuple,
    model_name: str = "",
    onnx_load_path: str = "",
    dynamic_axes: dict = {},
    remove_exported_model: bool = True,
    dynamo_export: bool = False,
    onnx_opset: int = DEFAULT_ONNX_OPSET,
    dq_only: bool = False,
    weights_dtype: str = "fp32",
) -> tuple[bytes, ModelMetadata]:
    """Get onnx model in bytes from input pytorch model together with the input/output of model.

    Arguments:
        model: PyTorch model to export to onnx.
        dummy_input: A tuple of args/kwargs or torch.Tensor, see
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info on the convention.
        model_name: The name of the model. If not provided, the model name will be inferred from the model class name.
        onnx_load_path: The path to load the onnx model.
        dynamic_axes: A dictionary of dynamic shapes used for exporting the torch model to onnx.
        remove_exported_model: If True, the onnx model will be cleared from the disk after the
            export process.
        dynamo_export: If True, the model is exported using `dynamo=True` in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
        onnx_opset: The onnx opset version to use for exporting the model.
        dq_only: If True, the exported onnx model is converted to a dq_only model.
        weights_dtype: The dtype of the weights in the onnx model.

    Returns:
        bytes: Onnx model in bytes.
        ModelMetadata: The model's meta data.

    Raises:
        ValueError: If nn.Module is not passed as model.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("Only PyTorch model compilation is supported.")

    assert weights_dtype in ["fp32", "fp16", "bf16"], (
        "weights_dtype must be one of fp32, fp16, or bf16"
    )

    # unwrap DDP and DP models
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    # Standardize model args and also tensorize them so they also appear in the onnx graph!
    # Floats/ints are tensorized when they are provided, but not tensorized when they are not
    # provided which is somewhat inconsistent (we always tensorize them!)
    named_args, _ = standardize_named_model_args(model, dummy_input)
    named_args = {k: _to_expected_onnx_type(v) for k, v in named_args.items()}

    # Also standardize dummy_input again so we can use it
    dummy_input = tuple(named_args.values())
    if dummy_input and isinstance(dummy_input[-1], dict):
        dummy_input = (*dummy_input, {})  # we need to add an extra dict for the fake kwargs!

    # Get input tree spec, see generate_onnx_input for more info as well on this
    flat_input, tree_spec_input = flatten_tree(named_args)

    # input names are the names of the flattened input tree spec but without None values
    input_names = [k for k, v in zip(tree_spec_input.names, flat_input) if v is not None]

    # we also want to record the input names that are None so we can remove them from the input
    # during inference.
    input_none_names = list(set(tree_spec_input.names) - set(input_names))

    use_torch_autocast = not (
        is_fp4_quantized(model) or is_mxfp8_quantized(model) or weights_dtype == "fp32"
    )
    autocast = torch.autocast("cuda") if use_torch_autocast else nullcontext()

    # Get output once (we export in inference mode - so also using inference mode here!)
    with torch.inference_mode(), autocast:
        output = model(*named_args.values())

    # Get output tree spec
    flat_output, tree_spec_output = flatten_tree(output, prefix=ONNX_EXPORT_OUT_PREFIX)

    # output names are the names of the flattened input tree spec but without None values
    output_names = [k for k, v in zip(tree_spec_output.names, flat_output) if v is not None]

    if onnx_load_path != "":
        onnx_model = OnnxBytes(onnx_load_path)
        onnx_model_graph = onnx.load(onnx_load_path)
        model_metadata = create_model_metadata(
            tree_spec_input, tree_spec_output, input_none_names, onnx_model_graph, model
        )
        return onnx_model.to_bytes(), model_metadata

    # Export onnx model from pytorch model
    # As the maximum size of protobuf is 2GB, we cannot use io.BytesIO() buffer during export.
    model_name = model_name or model.__class__.__name__
    onnx_path = tempfile.mkdtemp(prefix=f"modelopt_{model_name}_")
    onnx_save_path = os.path.join(onnx_path, f"{model_name}.onnx")

    # Configure quantizers if the model is quantized in NVFP4 or MXFP8 mode
    quantizer_context = (
        configure_linear_module_onnx_quantizers(model)
        if is_fp4_quantized(model) or is_mxfp8_quantized(model)
        else nullcontext()
    )
    with torch.inference_mode(), autocast, quantizer_context:
        additional_kwargs = {}
        if not dynamo_export:
            additional_kwargs["dynamic_axes"] = dynamic_axes
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=onnx_opset,
            dynamo=dynamo_export,
            **additional_kwargs,
        )

    # Check that export worked
    assert len(os.listdir(onnx_path)) > 0, "Torch to onnx export failed."

    # Load the onnx graph for optimizaiton
    onnx_graph = onnx.load(onnx_save_path, load_external_data=True)

    try:
        onnx_graph = infer_shapes(onnx_graph)
    except Exception as e:
        print(f"Shape inference failed: {e}")

    # Optimize the onnx graph
    onnx_opt_graph = optimize(model.__class__.__name__, onnx_graph)

    # Remove training_mode attribute from BatchNormalization nodes
    onnx_opt_graph = remove_node_training_mode(onnx_opt_graph, "BatchNormalization")

    model_metadata = create_model_metadata(
        tree_spec_input, tree_spec_output, input_none_names, onnx_opt_graph, model
    )

    # TODO: Remove manual ir_version change once ORT supports ir_version 11
    onnx_opt_graph.ir_version = 10

    # Convert dummy TRT_FP4QDQ nodes to 2DQ format if the model is quantized in FP4 mode
    # Or convert weights to MXFP8 format if the model is quantized in MXFP8 mode
    if is_int4_quantized(model) or is_fp4_quantized(model) or is_mxfp8_quantized(model):
        onnx_opt_graph = quantize_weights(model, onnx_opt_graph)

    if dq_only:
        onnx_opt_graph = qdq_to_dq(onnx_opt_graph)

    try:
        # TODO: Single-precision torch model assumed
        param_dtype = next(model.parameters()).dtype
    except StopIteration:
        param_dtype = torch.float32
    if weights_dtype in ["fp16", "bf16"] and param_dtype == torch.float32:
        if is_int4_quantized(model) or is_mxfp8_quantized(model):
            assert weights_dtype == "fp16", "BF16 + MXFP8/INT4 mixed precision is not supported yet"
            onnx_opt_graph = convert_float_to_float16(
                onnx_opt_graph,
                keep_io_types=False,
                disable_shape_infer=True,
                check_fp16_ready=False,
            )
        else:
            onnx_opt_graph = convert_to_f16(
                onnx_opt_graph, low_precision_type=weights_dtype, keep_io_types=False
            )

        # TensorRT expects all scales to be postive
        onnx_opt_graph = replace_zero_scale_with_smallest_nonzero(onnx_opt_graph)

    # If the onnx model contains external data store the external tensors in one file and save the onnx model
    if has_external_data(onnx_save_path):
        tensor_paths = get_external_tensor_paths(onnx_path)
        onnx.save_model(
            onnx_opt_graph,
            onnx_save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{model_name}.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )
        for path in tensor_paths:
            os.remove(path)
    else:
        onnx.save_model(onnx_opt_graph, onnx_save_path)

    onnx_bytes = OnnxBytes(onnx_save_path)

    if remove_exported_model:
        shutil.rmtree(onnx_path)
    return onnx_bytes.to_bytes(), model_metadata


def get_external_tensor_paths(model_dir: str) -> list[str]:
    """Get the paths of the external data tensors in the model."""
    return [
        os.path.join(model_dir, file)
        for file in os.listdir(model_dir)
        if not file.endswith(".onnx")
    ]


def has_external_data(onnx_model_path: str):
    """Check if the onnx model has external data."""
    onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    return check_model_uses_external_data(onnx_model)


def create_model_metadata(
    tree_spec_input: TreeSpec,
    tree_spec_output: TreeSpec,
    input_none_names: list[str],
    onnx_graph: ModelProto,
    model: nn.Module,
) -> ModelMetadata:
    """Create model metadata from the given input.

    Args:
        tree_spec_input: pytree spec describing the structure of the pytree for the model input.
        tree_spec_output: pytree spec describing the structure of the pytree for the model output.
        input_none_names: List of input names with values that are None.
        onnx_opt_graph: Graph of the onnx model.
        model: Pytorch model.

    Returns:
        ModelMetadata: The DeviceModel metadata.
    """
    return {
        "input_tree_spec": tree_spec_input,
        "input_shapes": get_input_shapes(onnx_graph),
        "input_onnx_names": get_input_names(onnx_graph),
        "input_none_names": input_none_names,
        "output_tree_spec": tree_spec_output,
        "output_shapes": get_output_shapes(onnx_graph),
        "output_onnx_names": get_output_names(onnx_graph),
        "signature": inspect.signature(model.forward),
        "onnx_node_names": get_node_names(onnx_graph),
        "is_bytes_pickled": onnx_graph.ByteSize() > TWO_GB,
        "config": model.config if hasattr(model, "config") else None,
    }
