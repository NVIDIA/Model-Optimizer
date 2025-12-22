AutoCast (ONNX)
###############

AutoCast is a tool for converting FP32 ONNX models to mixed precision FP32-FP16 or FP32-BF16 models.
While casting FP32 to FP16/BF16, some nodes might be more sensitive to affecting accuracy.
AutoCast intelligently selects nodes to keep in FP32 precision to maintain model accuracy while benefiting from
reduced precision on the rest of the nodes. AutoCast automatically injects cast operations around the selected
nodes.

Basic Commandline Usage
-----------------------

.. argparse::
   :module: modelopt.onnx.autocast.__main__
   :func: get_parser
   :prog: python -m modelopt.onnx.autocast

Python API Usage
----------------

AutoCast can also be used programmatically through its Python API:

.. code-block:: python

   import onnx
   from modelopt.onnx.autocast import convert_to_mixed_precision

   # Convert model to mixed precision
   converted_model = convert_to_mixed_precision(
      onnx_path="model.onnx",
      low_precision_type="fp16",            # or "bf16"
      nodes_to_exclude=None,                # optional list of node name patterns to keep in FP32
      op_types_to_exclude=None,             # optional list of op types to keep in FP32
      nodes_to_include=None,                # optional list of node name patterns to force-include in low precision
      op_types_to_include=None,             # optional list of op types to force-include in low precision
      data_max=512,                         # threshold for node outputs
      init_max=65504,                       # threshold for initializers
      keep_io_types=False,                  # whether to preserve input/output types
      calibration_data=None,                # optional path to input data file
      init_conversion_max_bytes=None,       # maximum size in bytes for initializer conversion
      providers=["cpu"],                    # list of Execution Providers for ONNX-Runtime backend
      trt_plugins=[],                       # list of TensorRT plugin library paths in .so format
      max_depth_of_reduction=None,          # maximum depth of reduction allowed in low precision
      opset=None,                           # optional target ONNX opset version (default: 13 for fp16, 22 for bf16)
   )

   # Save the converted model
   onnx.save(converted_model, "converted_model.onnx")

How It Works
------------

AutoCast follows these steps to convert a model:

#. **Model Loading and Sanitization**:

   - Loads the ONNX model
   - Performs graph sanitization and optimizations
   - Ensures minimum opset version requirements (22 for BF16, 13 for FP16 by default, or user-specified via ``--opset``)

#. **Node Classification**:

   - Analyzes each node in the graph
   - Determines which nodes should remain in FP32 based on input and output tensors magnitudes, operation types and node name patterns
   - If a calibration dataset is provided, it will be used to generate intermediate tensor magnitudes for more accurate node classification, otherwise random data will be used.
   - Use ``nodes_to_include`` and ``op_types_to_include`` to force-include nodes in low precision, even if they would otherwise be excluded.
   
   - Default classification rules. Nodes that meet any of these rules will be kept in high precision:
     - Node I/O magnitudes are higher than ``data_max`` (default: 512). Due to precision limitations, compute of high magnitude tensors in low precision might not be accurate. The unit in last place (ULP) for 512 is 0.5, for 1024 it is 1.0, etc.
     - Initializers magnitudes are higher than ``init_max`` (default: 65504). Initializers are often used for non-compute intensive operations and are more likely to be controlled by the user. However, values above ``init_max`` will cause overflow, therefore they are kept in high precision.
   
   Additional classification rules (disabled by default):
     - ``max_depth_of_reduction``: Require nodes with a high depth of reduction (e.g., large matrix multiplications, convolutions with large kernels) to be kept in high precision.
     - ``nodes_to_exclude``: List of regex patterns for node names to keep in high precision.
     - ``op_types_to_exclude``: List of operation types to keep in high precision.
     - ``nodes_to_include``: List of regex patterns for node names to force-include in low precision.
     - ``op_types_to_include``: List of operation types to force-include in low precision.
     - ``custom_rule``: Optional custom rule for node classification (inherits from NodeRuleBase).

#. **Precision Conversion**:

   - Converts eligible nodes to lower precision
   - Automatically inserts necessary cast operations
   - Automatically replaces initializers with lower precision values

#. **Validation and Export**:

   - Verifying that the model is a valid ONNX model (using onnx.checker)
   - Checking that the output tensors are not disconnected
   - Verifying that the original and current network inputs/outputs names match
   - Ensuring that the input and output types are handled according to keep_io_types
   - Saves the converted model

Best Practices
--------------

#. **Start with Default Settings**:

   - Begin with default thresholds and gradually adjust based on accuracy requirements.

#. **Monitor Node Conversion**:

   - Use INFO level logging to see what percentage of nodes were converted to lower precision.
   - Use DEBUG level logging to see more detailed information about the node classification process.

#. **Preserve Critical Operations**:

   - Use ``op_types_to_exclude`` for operations known to be sensitive to precision reduction.

#. **Validate with Real Data**:

   - Provide representative input data using the ``calibration_data`` option for more accurate node classification.
   - The input names and shapes in ``calibration_data`` should match the ones in the given ONNX model.

#. **Control Reduction Depth**:
   - Use ``max_depth_of_reduction`` to limit the depth of reduction operations that can be converted to low precision.
   Operations with higher reduction depths (e.g., large matrix multiplications, convolutions with large kernels) may be more sensitive to precision loss.

#. **BF16 Conversion**:

   - BF16 conversion is not supported for all operations.
   - AutoCast will automatically convert the model to opset 22 to enable more BF16 operations.
   - Use ``--op_types_to_exclude`` to exclude operations that are not supported in BF16.
   - BF16 accuracy may require additional tuning of the ``data_max`` and ``init_max`` thresholds.
   - TensorRT might not be able to support all BF16 converted models.

#. **Large Initializers**

   - Attempting to convert very large initializers, might cause host memory issues.
   - Use ``--init_conversion_max_bytes`` to limit the size of initializers that will be converted at compile time.
   - Initializers larger than ``--init_conversion_max_bytes`` will be converted at runtime (using a cast operation).

#. **TensorRT custom op support**

   - Refer to :ref:`TensorRT Execution Provider requirements <ort_ep_requirements>`.
   - When a custom op is detected, the TensorRT Execution Provider is automatically enabled.
   - To also enable the CUDA execution provider, use ``--providers cpu cuda:x``, where ``x`` is your device ID (``x=0`` if your system only has 1 GPU).
   - Use ``--trt_plugins`` to provide the paths to the necessary TensorRT plugin libraries (in ``.so`` format).

#. **Opset Version Control**

   - Use ``--opset`` to specify a target ONNX opset version for the converted model.
   - If not specified, AutoCast keeps the existing model's opset, subject to a minimum opset based on precision type (13 for FP16, 22 for BF16).
   - A warning will be issued if you specify an opset lower than recommended minimum.
   - A warning will be issued if you specify an opset lower than the original model's opset, as downgrading opset versions may cause compatibility issues.
   - The opset may be automatically increased beyond your specified value if certain operations require it (e.g., quantization nodes require opset >= 19).

Limitations and Restrictions
----------------------------
- AutoCast does not yet support quantized models.
- BF16 conversion is not supported for all operations
- Large models (e.g. over 2GB) might cause memory issues.

Example Usage
-------------

Basic conversion to FP16:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx

Basic conversion with verbose logging and custom output path:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx --output_path custom_path.onnx --log_level DEBUG

Convert to BF16 with custom data magnitude threshold and custom disabled op types:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx \
        --low_precision_type bf16 \
        --data_max 256 \
        --op_types_to_exclude Resize

Bypass data magnitude check and keep specific node names in FP32:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx --data_max inf --nodes_to_exclude ".*attn.*"

Limit depth of reduction for precision-sensitive operations:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx --max_depth_of_reduction 1024

Specify a target opset version:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx --opset 19

Convert to BF16 with a specific opset:

.. code-block:: bash

   python -m modelopt.onnx.autocast --onnx_path model.onnx --low_precision_type bf16 --opset 22
