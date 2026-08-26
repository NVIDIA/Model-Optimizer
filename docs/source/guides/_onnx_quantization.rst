================================
ONNX Quantization - Linux (Beta)
================================

ModelOpt provides ONNX quantization that works together with `TensorRT Explicit Quantization (EQ) <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-quantization>`_. The key advantages offered by ModelOpt's ONNX quantization:

#. Easy to use for non-expert users.
#. White-box design allowing expert users to customize the quantization process.
#. Better support for vision transformers.

Currently ONNX quantization supports FP8, INT4 and INT8 quantization.

.. note::

    ModelOpt ONNX quantization generates new ONNX models with QDQ nodes following TensorRT rules.
    For real speedup, the generated ONNX should be compiled into TensorRT engine.

.. _ort_ep_requirements:

Requirements
============

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Execution Provider
     - Requirements
   * - CPU
     - * Default
   * - CUDA
     - * Add ``libcudnn*.so*`` path to ``LD_LIBRARY_PATH`` (check required cuDNN version `here <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>`_)
   * - TensorRT
     - * Same requirement as CUDA EP
       * TensorRT >= 10.0 (add TensorRT ``lib/`` path to ``LD_LIBRARY_PATH`` and  install python wheel). Please refer to `TensorRT 10.0 download link <https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-12.4.tar.gz>`_.

Apply Post Training Quantization (PTQ)
======================================

PTQ should be done with a calibration dataset. If calibration dataset is not provided, ModelOpt will use random scales for the QDQ nodes.

Prepare calibration dataset
---------------------------
ModelOpt accepts an NPY array for a single-input model or an NPZ archive whose keys match the model input names.

.. code-block:: python

    # Example numpy file for single-input ONNX
    calib_data = np.random.randn(batch_size, channels, h, w)
    np.save("calib_data.npy", calib_data)

    # Example numpy file for single/multi-input ONNX
    # Dict key should match the input names of ONNX
    calib_data = {
        "input_name": np.random.randn(*shape),
        "input_name2": np.random.randn(*shape2),
    }
    np.savez("calib_data.npz", **calib_data)

Large calibration datasets can instead be streamed from a directory. Store one batch per ``.npy`` file for a single-input model or one batch per ``.npz`` file for a single- or multi-input model. A directory cannot mix both formats. Each batch is checked against the ONNX input names, shapes, and dtypes before it is loaded.

Call PTQ function
-----------------
.. code-block:: python

    from pathlib import Path

    import modelopt.onnx.quantization as moq
    import numpy as np

    calibration_path = Path("calib_data.npz")
    if calibration_path.suffix == ".npz":
        with np.load(calibration_path, allow_pickle=False) as archive:
            calibration_data = {name: archive[name] for name in archive.files}
    else:
        calibration_data = np.load(calibration_path, allow_pickle=False)

    moq.quantize(
        onnx_path=onnx_path,
        calibration_data=calibration_data,
        output_path="quant.onnx",
        quantize_mode="int8",
    )

To stream directory-backed batches, create a calibration data reader:

.. code-block:: python

    from modelopt.onnx.quantization.calib_utils import create_directory_calibration_reader

    calibration_reader = create_directory_calibration_reader(
        "calibration_batches",
        onnx_path,
        max_batches=512,
    )
    moq.quantize(
        onnx_path=onnx_path,
        calibration_data_reader=calibration_reader,
        output_path="quant.onnx",
        quantize_mode="int8",
    )

Optionally enable Autotune for more optimized Q/DQ placement. Note that this will likely increase the time required to calibrate the model.

.. code-block:: python

    moq.quantize(
        ...
        # Default Autotune settings, can be tuned with the autotune_* arguments below.
        autotune=True,
    )

Alternatively, you can call PTQ function in command line:

.. code-block:: bash

    python -m modelopt.onnx.quantization \
        --onnx_path=model.onnx \
        --calibration_data_path=calibration_batches \
        --max_calibration_batches=512 \
        --output_path=quant.onnx

.. argparse::
   :module: modelopt.onnx.quantization.__main__
   :func: get_parser
   :prog: python -m modelopt.onnx.quantization


If the model contains custom ops, enable calibration with the TensorRT Execution Provider backend with CUDA and CPU fallback (``--calibration_eps trt cuda:0 cpu``) and, if relevant, provide the location of the TensorRT plugin in ``.so`` format via the ``--trt_plugins`` flag.

By default, after running the calibration, the quantization tool will insert the QDQ nodes by following TensorRT friendly QDQ insertion algorithm. Users can change the default quantization behavior by tweaking the API params like op_types_to_quantize, op_types_to_exclude etc. See the :meth:`modelopt.onnx.quantization.quantize() <modelopt.onnx.quantization.quantize>` for details.


Deploy Quantized ONNX Model
===========================


``trtexec`` is a command-line tool provided by TensorRT. Typically, it's within the ``/usr/src/tensorrt/bin/`` directory. Below is a simple command to compile the quantized onnx model generated by the previous step into a TensorRT engine file.

.. code-block:: bash

    trtexec --onnx=quant.onnx --saveEngine=quant.engine --best

Compare the performance
=======================

The following command will build the engine using fp16 precision. After building, check the reported "Latency" and "Throughput" fields and compare.


.. code-block:: bash

    trtexec --onnx=original.onnx --saveEngine=fp16.engine --fp16


.. note::

    If you replace ``--fp16`` flag with ``--best`` flag, this command will create an int8 engine with TensorRT's implicit quantization.
