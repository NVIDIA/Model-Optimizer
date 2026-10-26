# Deploy quantized models from Nvidia's Hugging Face model hub with TensorRT-LLM, vLLM, and SGLang

The provided bash scripts show an example to deploy and run the [quantized LLama 3.1 8B Instruct FP8 model](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8) from Nvidia's Hugging Face model hub on TensorRT-LLM, vLLM and SGLang respectively.

Before running the bash scripts, please make sure you have setup the environment properly:

- Make sure you are authenticated with a Hugging Face account to interact with the Hub, e.g., use `huggingface-cli login` to save the access token on your machine.
- Install TensorRT-LLM properly by following instructions [here](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#installation).
- Install vLLM properly by following instructions [here](https://docs.vllm.ai/en/latest/getting_started/installation.html#install-released-versions).
- Install SGLang properly by following instructions [here](https://docs.sglang.ai/get_started/install.html)

Then, to deploy and run on TensorRT-LLM:

```sh
python run_llama_fp8_trtllm.py
```

To deploy and run on vLLM:

```sh
python run_llama_fp8_vllm.py
```

To deploy and run on SGLang:

```sh
python run_llama_fp8_sglang.py
```

If you want to run post-training quantization with Model Optimizer for your selected models, check [here](../llm_ptq/README.md).
