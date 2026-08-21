# ONNX Export and TensorRT Engine Build

This page covers the optional ONNX export + TensorRT engine workflow for diffusion models.
For quantization-only workflows, refer to `../README.md`.

## Quantize and export ONNX

### 8-bit Quantize and ONNX Export [Script](./build_sdxl_8bit_engine.sh)

You can run the following script to quantize SDXL backbone to INT8 or FP8 and generate an ONNX model built with default settings for SDXL. You can then directly head to the [Build the TRT engine for the Quantized ONNX Backbone](#build-the-trt-engine-for-the-quantized-onnx-backbone) section to run E2E pipeline and generate images.

```sh
bash build_sdxl_8bit_engine.sh --format {FORMAT} # FORMAT can be int8 or fp8
```

If you prefer to customize parameters in calibration or run other models, please follow the instructions below.

#### FLUX-Dev|SD3-Medium|SDXL|SDXL-Turbo INT8 [Script](./quantize.py)

```sh
python quantize.py \
    --model {flux-dev|sdxl-1.0|sdxl-turbo|sd3-medium} \
    --format int8 --batch-size 2 \
    --calib-size 32 --alpha 0.8 --n-steps 20 \
    --model-dtype {Half/BFloat16} --trt-high-precision-dtype {Half|BFloat16} \
    --quantized-torch-ckpt-save-path ./{MODEL_NAME}.pt --onnx-dir {ONNX_DIR}
```

#### FLUX-Dev|SDXL|SDXL-Turbo|LTX-Video FP8/FP4 [Script](./quantize.py)

*In our example code, FP4 is only supported for Flux. However, you can modify our script to enable FP4 format support for your own model.*

```sh
python quantize.py \
    --model {flux-dev|sdxl-1.0|sdxl-turbo|ltx-video-dev} --model-dtype {Half|BFloat16} --trt-high-precision-dtype {Half|BFloat16} \
    --format {fp8|fp4} --batch-size 2 --calib-size {128|256} --quantize-mha \
    --n-steps 20 --quantized-torch-ckpt-save-path ./{MODEL_NAME}.pt --collect-method default \
    --onnx-dir {ONNX_DIR}
```

We recommend using a device with a minimum of 48GB of combined CPU and GPU memory for exporting ONNX models. If not, please use CPU for ONNX export.

## Build the TRT engine for the Quantized ONNX Backbone

> [!IMPORTANT]
> TensorRT environment must be setup prior -- Please see [Pre-Requisites](../README.md#pre-requisites)
> INT8 requires **TensorRT version >= 9.2.0**. If you prefer to use the FP8 TensorRT, ensure you have **TensorRT version 10.2.0 or higher**. You can download the latest version of TensorRT at [here](https://developer.nvidia.com/tensorrt/download). Deployment of SVDQuant is currently not supported.

Generate INT8/FP8 Backbone Engine

```bash
# For SDXL
trtexec --builderOptimizationLevel=4 --stronglyTyped --onnx=./model.onnx \
    --minShapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 \
    --optShapes=sample:16x4x128x128,timestep:1,encoder_hidden_states:16x77x2048,text_embeds:16x1280,time_ids:16x6 \
    --maxShapes=sample:16x4x128x128,timestep:1,encoder_hidden_states:16x77x2048,text_embeds:16x1280,time_ids:16x6 \
    --saveEngine=model.plan

# For SD3-Medium
trtexec --builderOptimizationLevel=4 --stronglyTyped --onnx=./model.onnx \
    --minShapes=hidden_states:2x16x128x128,timestep:2,encoder_hidden_states:2x333x4096,pooled_projections:2x2048 \
    --optShapes=hidden_states:16x16x128x128,timestep:16,encoder_hidden_states:16x333x4096,pooled_projections:16x2048 \
    --maxShapes=hidden_states:16x16x128x128,timestep:16,encoder_hidden_states:16x333x4096,pooled_projections:16x2048 \
    --saveEngine=model.plan

# For FLUX-Dev FP8
trtexec --onnx=./model.onnx --fp8 --bf16 --stronglyTyped \
    --minShapes=hidden_states:1x4096x64,img_ids:4096x3,encoder_hidden_states:1x512x4096,txt_ids:512x3,timestep:1,pooled_projections:1x768,guidance:1 \
    --optShapes=hidden_states:1x4096x64,img_ids:4096x3,encoder_hidden_states:1x512x4096,txt_ids:512x3,timestep:1,pooled_projections:1x768,guidance:1 \
    --maxShapes=hidden_states:1x4096x64,img_ids:4096x3,encoder_hidden_states:1x512x4096,txt_ids:512x3,timestep:1,pooled_projections:1x768,guidance:1 \
    --saveEngine=model.plan
```

**Please note that `maxShapes` represents the maximum shape of the given tensor. If you want to use a larger batch size or any other dimensions, feel free to adjust the value accordingly.**

## Run End-to-end Stable Diffusion Pipeline with Model Optimizer Quantized ONNX Model and demoDiffusion

### demoDiffusion

If you want to run end-to-end SD/SDXL pipeline with Model Optimizer quantized UNet to generate images and measure latency on target GPUs, here are the steps:

- Clone a copy of [demo/Diffusion repo](https://github.com/NVIDIA/TensorRT/tree/release/10.2/demo/Diffusion).

- Following the README from demoDiffusion to set up the pipeline, and run a baseline txt2img example (fp16):

```sh
# SDXL
python demo_txt2img_xl.py "enchanted winter forest, soft diffuse light on a snow-filled day, serene nature scene, the forest is illuminated by the snow" --negative-prompt "normal quality, low quality, worst quality, low res, blurry, nsfw, nude" --version xl-1.0 --scheduler Euler --denoising-steps 30 --seed 2946901
# Please refer to the examples provided in the demoDiffusion SD/SDXL pipeline.
```

Note, it will take some time to build TRT engines for the first time

- Replace the fp16 backbone TRT engine with int8 engine generated in [Build the TRT engine for the Quantized ONNX Backbone](#build-the-trt-engine-for-the-quantized-onnx-backbone), e.g.,:

```sh
cp -r {YOUR_UNETXL}.plan ./engine/
```

Note, the engines must be built on the same GPU, and ensure that the INT8 engine name matches the names of the FP16 engines to enable compatibility with the demoDiffusion pipeline.

- Run the above txt2img example command again. You can compare the generated images and latency for fp16 vs int8.
  Similarly, you could run end-to-end pipeline with Model Optimizer quantized backbone and corresponding examples in demoDiffusion with other diffusion models.

## Running the inference pipeline with DeviceModel

DeviceModel is an interface designed to run TensorRT engines like torch models. It takes torch inputs and returns torch outputs. Under the hood, DeviceModel exports a torch checkpoint to ONNX and then generates a TensorRT engine from it. This allows you to swap the backbone of the diffusion pipeline with DeviceModel and execute the pipeline for your desired prompt.

Generate a quantized torch checkpoint using the [Script](./quantize.py) shown below:

```bash
python quantize.py \
    --model {sdxl-1.0|sdxl-turbo|sd3-medium|flux-dev} \
    --format fp8 \
    --batch-size {1|2} \
    --calib-size 128 \
    --n-steps 20 \
    --quantized-torch-ckpt-save-path ./{MODEL}_fp8.pt \
    --collect-method default
```

Generate images for the quantized checkpoint with the following [Script](./diffusion_trt.py):

```bash
python diffusion_trt.py \
    --model {sdxl-1.0|sdxl-turbo|sd3-medium|flux-dev} \
    --prompt "A cat holding a sign that says hello world" \
    [--override-model-path /path/to/model] \
    [--restore-from ./{MODEL}_fp8.pt] \
    [--onnx-load-path {ONNX_DIR}] \
    [--trt-engine-load-path {ENGINE_DIR}] \
    [--dq-only] \
    [--torch] \
    [--save-image-as /path/to/image] \
    [--benchmark] \
    [--torch-compile] \
    [--skip-image]
```

This script will save the output image as `./{MODEL}.png` and report the latency of the TensorRT backbone.
To generate the image with FP16|BF16 precision, you can run the command shown above without the `--restore-from` argument.

While loading a TensorRT engine using the --trt-engine-load-path argument, it is recommended to load only engines generated using this pipeline.

### Demo Images

| SDXL FP16 | SDXL INT8 |
|:---------:|:---------:|
| ![FP16](./assets/xl_base-fp16.png) | ![INT8](./assets/xl_base-int8.png) |
