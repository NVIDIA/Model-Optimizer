=================================================================
Unified HuggingFace Checkpoint
=================================================================

We support exporting modelopt-optimized Hugging Face models (transformers and diffusers pipelines/components) and Megatron Core models to a unified checkpoint format that can be deployed in various inference frameworks such as TensorRT-LLM, vLLM, and SGLang.

The workflow is as follows:

#. Load the Huggingface models or Megatron Core models, `quantize with modelopt <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/hf_ptq#ptq-post-training-quantization>`_ , and export to the unified checkpoint format, where the layer structures and tensor names are aligned with the original checkpoint.
#. Load the unified checkpoint in the supported inference framework for accelerated inference.


Export Quantized Model
======================

The modelopt quantized model can be exported to the unified checkpoint format stored as

#. A group of safetensors files, containing quantized model weights and scaling factors.
#. A ``hf_quant_config.json`` file containing quantization configurations.
#. Other json files that store the model structure information, tokenizer information, and metadata.


The export API (:meth:`export_hf_checkpoint <modelopt.torch.export.unified_export_hf.export_hf_checkpoint>`) can be used as follows:

.. code-block:: python

    from modelopt.torch.export import export_hf_checkpoint

    with torch.inference_mode():
        export_hf_checkpoint(
            model,  # The quantized model.
            export_dir,  # The directory where the exported files will be stored.
        )

.. note::
   ``export_hf_checkpoint`` also supports diffusers pipelines and components (e.g., UNet/transformer). See the
   diffusers quantization examples for end-to-end workflows and CLI usage.

Deployment Support Matrix
==============================================

Supported Quantization Formats
------------------------------

The unified HF export API supports the following quantization formats:

1. FP8 - 8-bit floating point
2. FP8_PB - 8-bit floating point with per-block scaling
3. NVFP4 - NVIDIA 4-bit floating point
4. NVFP4_AWQ - NVIDIA 4-bit floating point with AWQ optimization
5. INT4_AWQ - 4-bit integer with AWQ optimization
6. W4A8_AWQ - 4-bit weights and 8-bit activations with AWQ optimization

Minimum Framework Versions
--------------------------

===============  =================
Framework        Minimum version
===============  =================
TensorRT-LLM     v1.2.0
vLLM             v0.10.1
SGLang           v0.4.10
===============  =================

These are the oldest versions expected to load a unified HF checkpoint. The deployment suite itself
targets newer ones — TensorRT-LLM containers in ``.github/workflows/`` are on the 1.3.x line. Older
TensorRT-LLM releases may still serve FP8 checkpoints; that is simply not exercised, so v1.2.0 is
the oldest version stated here rather than the oldest that works.

.. _unified-hf-support-matrix:

Model Support Matrix
--------------------

What this matrix is based on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Entries are drawn from the release deployment suite,
`tests/examples/hf_ptq/test_deploy.py <https://github.com/NVIDIA/Model-Optimizer/blob/main/tests/examples/hf_ptq/test_deploy.py>`_.
For each entry it loads the exported checkpoint in the framework and generates from four short text
prompts, asserting that each returns non-empty output.

Two limits are worth stating plainly, because they bound what any ✅ below can mean:

* **These are declared cases, not PR-gated coverage.** The suite is marked ``release`` and collects
  only when pytest is given ``--run-release``, which no workflow in ``.github/workflows/`` currently
  passes. A green check on a pull request does not mean these cases ran.
* **Each case is a load-and-generate smoke check on the text path.** It does not verify accuracy,
  image or audio inputs, diffusion output, or that speculative decoding actually engages.

Legend:

* ✅ — declared in the release deployment suite, subject to the two limits above.
* ⚠ — expected to work, but not a suite entry: either carried over from earlier documentation, or
  present as a case that does not exercise the feature the row names.
* ``-`` — not in the suite. It may still work; see `Models not listed here`_.

Language models
~~~~~~~~~~~~~~~

============================================  ==============  ============  ======  ========
Model                                         Quant format    TensorRT-LLM  vLLM    SGLang
============================================  ==============  ============  ======  ========
Llama 3.1, 3.3                                FP8, NVFP4      ✅             ✅       ✅
Llama 4 Scout, Maverick                       FP8             ✅             ✅       ✅
Llama 4 Scout                                 NVFP4           ✅             ✅       ✅
Llama 4 Maverick                              NVFP4           ⚠             \-      \-
Llama Nemotron Super 49B v1, v1.5             FP8             ✅             ✅       ✅
Llama Nemotron Ultra 253B v1                  FP8             ✅             ✅       ✅
Nemotron 3 Nano 30B-A3B                       FP8, NVFP4      ✅             ✅       ✅
Nemotron 3 Super 120B-A12B                    FP8, NVFP4      ✅             ✅       ✅
Nemotron 3 Ultra 550B-A55B                    NVFP4           ✅             ✅       ✅
DeepSeek R1, R1-0528                          NVFP4           ✅             ✅       ✅
DeepSeek R1, V3                               FP8             ⚠             ⚠       ⚠
DeepSeek V3, V3.1, V3.2                       NVFP4           ✅             ✅       ✅
DeepSeek V4 Flash                             NVFP4           ✅             ✅       ✅
DeepSeek V4 Pro                               NVFP4           \-            ✅       ✅
Qwen 3 8B, 14B                                FP8, NVFP4      ✅             ✅       ✅
Qwen 3 32B                                    NVFP4           ✅             ✅       ✅
Qwen 3 MoE 235B-A22B                          FP8, NVFP4      ✅             ✅       ✅
Qwen 3 MoE 30B-A3B                            NVFP4           ✅             ✅       ✅
Qwen 3 Coder 480B-A35B                        NVFP4           ✅             ✅       ✅
Qwen 3-Next 80B-A3B                           NVFP4           ✅             ✅       ✅
Qwen 3.5 397B-A17B                            NVFP4           ✅             ✅       ✅
Qwen 3.5 122B-A10B, Qwen 3.6 35B-A3B          NVFP4           \-            ✅       \-
Qwen 2.5                                      FP8             ⚠             ⚠       ⚠
Qwen 2.5                                      NVFP4           ⚠             ⚠       \-
QwQ-32B                                       FP8             ⚠             ⚠       ⚠
QwQ-32B                                       NVFP4           ⚠             ⚠       \-
Gemma 4 31B                                   NVFP4           ✅             ✅       ✅
Gemma 4 26B-A4B                               NVFP4           \-            ✅       \-
GLM-4.7, GLM-5, GLM-5.2                       NVFP4           ✅             ✅       ✅
GLM-5.1                                       NVFP4           \-            ✅       ✅
Kimi K2-Thinking, K2.5                        NVFP4           ✅             ✅       ✅
Kimi K2.6                                     NVFP4           \-            ✅       \-
MiniMax M2.5, M3                              NVFP4           ✅             ✅       ✅
Mixtral 8x7B                                  FP8             ⚠             ⚠       ⚠
Mixtral 8x7B                                  NVFP4           ⚠             \-      \-
============================================  ==============  ============  ======  ========

Vision-language and multimodal models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For VLMs, modelopt quantizes the language model only; the vision encoder is kept in high precision.
The exported checkpoint therefore relies on the serving framework's own multimodal support for that
architecture — see the
`TensorRT-LLM multimodal support matrix <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md#multimodal-feature-support-matrix-pytorch-backend>`_.

.. important::
   ✅ in this table is **text-only smoke coverage**. The suite sends the same plain-text prompts it
   uses for language models, so no image or audio input reaches the processor or vision encoder.
   These entries show that the quantized checkpoint loads and that its language path generates —
   they do not demonstrate multimodal serving.

============================================  ==============  ============  ======  ========
Model                                         Quant format    TensorRT-LLM  vLLM    SGLang
============================================  ==============  ============  ======  ========
Qwen 2.5-VL 7B                                FP8, NVFP4      ✅             ✅       ✅
Qwen 3-VL 235B-A22B                           NVFP4           ✅             ✅       ✅
Nemotron 3 Nano Omni 30B-A3B                  FP8, NVFP4      ✅             ✅       ✅
============================================  ==============  ============  ======  ========

Speculative decoding drafters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drafters are deployed on top of their base checkpoint.

Two caveats specific to this table:

* **Most entries are doubly conditional.** Beyond the ``--run-release`` gate, the drafter cases in
  ``test_eagle`` also require ``MODELOPT_LOCAL_EAGLE_MODEL`` to point at a directory containing the
  drafter, and skip otherwise. The exception is EAGLE3 for Kimi K2.6, which is declared in
  ``test_kimi`` without that gate — which is also why it is the one row with vLLM coverage.
* **Medusa is marked ⚠ because the case does not exercise Medusa.** The shared harness builds a
  speculative-decoding configuration only when the model ID contains ``eagle``, so the Medusa entry
  performs ordinary generation. It shows the checkpoint loads and serves; it does not validate
  Medusa decoding.

============================================================  ============  ============  ======  ========
Drafter                                                       Quant format  TensorRT-LLM  vLLM    SGLang
============================================================  ============  ============  ======  ========
EAGLE3 for Llama 3.3 70B, Llama 4 Maverick                    FP8           ✅             \-      ✅
EAGLE3 for Qwen 3 235B-A22B (incl. Thinking-2507, FP4)        BF16, NVFP4   ✅             \-      ✅
EAGLE3 for Qwen 3 30B-A3B-Thinking-2507                       BF16          ✅             \-      ✅
EAGLE3 for Kimi K2-Thinking, K2.5                             NVFP4         ✅             \-      ✅
EAGLE3 for Kimi K2.6                                          NVFP4         ✅             ✅      ✅
EAGLE3 for gpt-oss-120b                                       BF16          ✅             \-      ✅
Medusa for Llama 3.1 8B                                       FP8           ⚠             \-      ⚠
============================================================  ============  ============  ======  ========

Diffusion models
~~~~~~~~~~~~~~~~

============================================  ==============  ============  ======  ========
Model                                         Quant format    TensorRT-LLM  vLLM    SGLang
============================================  ==============  ============  ======  ========
Wan 2.2 T2V A14B                              FP8, NVFP4      ⚠             \-      ⚠
DiffusionGemma 26B-A4B                        NVFP4           ✅             ✅       ✅
============================================  ==============  ============  ======  ========

Wan 2.2 is marked ⚠ because its cases run through the same autoregressive text helper as the
language models and assert on generated text. They never call a diffusion or video serving API, so
they do not substantiate text-to-video deployment.

.. note::
   NVFP4 inference requires Blackwell GPUs. Hopper can produce an NVFP4 checkpoint but cannot serve
   it. On B300/GB300 (``sm_103``) use a CUDA-13 build of the serving framework; CUDA-12 builds lack
   the ``sm_103`` FP4 kernels.

Models not listed here
~~~~~~~~~~~~~~~~~~~~~~

This matrix records the combinations modelopt validates. It is not an exhaustive list of what will
run: vLLM, SGLang, and TensorRT-LLM load unified HF checkpoints generically, so a model built from
standard ``nn.Linear`` layers with an ``hf_quant_config.json`` will often deploy without any modelopt
change. Check the serving framework's own model support list first, then try it.

The exact checkpoints behind every ✅ above, including tensor-parallel size and minimum SM
version, are listed in
`tests/examples/hf_ptq/test_deploy.py <https://github.com/NVIDIA/Model-Optimizer/blob/main/tests/examples/hf_ptq/test_deploy.py>`__;
most are published under the
`NVIDIA Hugging Face organization <https://huggingface.co/nvidia>`_.


Deployment with Selected Inference Frameworks
==============================================

.. tab:: TensorRT-LLM

    Follow the `TensorRT-LLM installation instructions. <https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#installation>`_

    FP8 and NVFP4 quantized models are supported; you need v1.2.0 or later version of TensorRT-LLM.

    To run modelopt quantized model from Huggingface model hub, e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`_, refer to the sample code below:

    .. code-block:: python

        from tensorrt_llm import LLM, SamplingParams

        def main():

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

            llm = LLM(model="nvidia/Llama-3.1-8B-Instruct-FP8")

            outputs = llm.generate(prompts, sampling_params)

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        if __name__ == '__main__':
            main()

.. tab:: vLLM

    Follow `vLLM installation instructions. <https://github.com/vllm-project/vllm?tab=readme-ov-file#getting-started>`_

    FP8 and NVFP4 quantized models are supported; you need v0.10.1 or later version of vLLM. Pass
    ``quantization="modelopt"`` for FP8 and ``quantization="modelopt_fp4"`` for NVFP4.

    To run modelopt quantized model from Huggingface model hub, e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`_, refer to the sample code below:

    .. code-block:: python

        from vllm import LLM, SamplingParams

        def main():

            model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"
            sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]

            llm = LLM(model=model_id, quantization="modelopt")
            outputs = llm.generate(prompts, sampling_params)

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        if __name__ == "__main__":
            main()

.. tab:: SGLang

    Follow the `SGLang installation instructions. <https://docs.sglang.ai/get_started/install.html>`_

    FP8 and NVFP4 quantized models are supported; you need v0.4.10 or later version of SGLang. Pass
    ``quantization="modelopt"`` for FP8 and ``quantization="modelopt_fp4"`` for NVFP4.

    To run modelopt quantized model from Huggingface model hub, e.g., `nvidia/Llama-3.1-8B-Instruct-FP8`_, refer to the sample code below:

    .. code-block:: python

        import sglang as sgl

        def main():

            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            llm = sgl.Engine(model_path="nvidia/Llama-3.1-8B-Instruct-FP8", quantization="modelopt")

            outputs = llm.generate(prompts, sampling_params)
            for prompt, output in zip(prompts, outputs):
                print("===============================")
                print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

        if __name__ == "__main__":
            main()

.. _nvidia/Llama-3.1-8B-Instruct-FP8: https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8

.. =================================================================
.. TODO: Add sample usage for Autodeploy when it's public
.. =================================================================
