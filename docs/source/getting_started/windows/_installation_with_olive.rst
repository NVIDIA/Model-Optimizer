.. _Install-Page-Olive-Windows:

===================================
Install ModelOpt-Windows with Olive
===================================

ModelOpt-Windows can be installed and used through Olive to perform model optimization using quantization technique. Follow the steps below to configure Olive for use with ModelOpt-Windows.

Setup Steps for Olive with ModelOpt-Windows
-------------------------------------------

**1. Installation**

   - **Install Olive and the Model Optimizer:** Run the following command to install Olive with NVIDIA Model Optimizer - Windows:

     .. code-block:: bash

         pip install olive-ai[nvmo]

   - **Install Prerequisites:** Ensure all required dependencies are installed. For example, to use DirectML Execution-Provider (EP) based onnxruntime and onnxruntime-genai packages, run the following commands:

     .. code-block:: shell

            $ pip install onnxruntime-genai-directml>=0.4.0
            $ pip install onnxruntime-directml==1.20.0

   - Above onnxruntime and onnxruntime-genai packages enable Olive workflow with DirectML Execution-Provider (EP). To use other EPs, install corresponding packages.

   - Additionally, ensure that dependencies for Model Optimizer - Windows are met as mentioned in the :ref:`Install-Page-Standalone-Windows`.

**2. Configure Olive for Model Optimizer – Windows**

   - **New Olive Pass:** Olive introduces a new pass, ``NVModelOptQuantization`` (or “nvmo”), specifically designed for model quantization using Model Optimizer – Windows.
   - **Add to Configuration:** To apply quantization to your target model, include this pass in the Olive configuration file. [Refer `this <https://github.com/microsoft/Olive/blob/main/docs/source/features/quantization.md#nvidia-tensorrt-model-optimizer-windows>`_ guide for details about this pass.].

**3. Setup Other Passes in Olive Configuration**

   - **Add Other Passes:** Add additional passes to the Olive configuration file as needed for the desired Olive workflow of your input model.

**4. Install other dependencies**

   - Install other requirements as needed by the Olive scripts and config.

**5. Run the Optimization**

   - **Execute Optimization:** To start the optimization process, run the following commands:

     .. code-block:: shell

            $ olive run --config <config json> --setup
            $ olive run --config <config json>

     Alternatively, you can execute the optimization using the following Python code:

     .. code-block:: python

            from olive.workflows import run as olive_run

            olive_run("config.json")


**Note**:

#. Currently, the Model Optimizer - Windows only supports Onnx Runtime GenAI based LLM models in the Olive workflow.
#. To get started with Olive, refer to the official `Olive documentation <https://microsoft.github.io/Olive/>`_.
