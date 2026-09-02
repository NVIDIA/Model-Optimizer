# Adapted from https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/far3d-trt/tools/test_tensorrt.py
# which was modified from https://github.com/megvii-research/Far3D/blob/5efb9d73a246c39fac79b3cf8c20a8e059611c3f/tools/test.py.
# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Zhiqi Li.
#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import importlib
import os
import warnings

import tensorrt as trt
import torch
from mmcv import Config, DictAction
from mmcv.utils import import_modules_from_strings
from mmdet.apis import set_random_seed
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from tqdm import tqdm

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}
if int(trt.__version__.split(".")[0]) >= 10:
    TRT_TO_TORCH[trt.DataType.INT64] = torch.int64

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def aligned_tensor(shape, dtype, device, alignment=256):
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = int(torch.tensor(shape).prod().item())
    storage = torch.empty(element_count + alignment // element_size, dtype=dtype, device=device)
    offset_bytes = (-storage.data_ptr()) % alignment
    offset = offset_bytes // element_size
    return storage[offset : offset + element_count].view(shape)


class TensorRTRunner:
    def __init__(self, engine_path, state_names=()):
        with open(engine_path, "rb") as engine_file:
            engine_bytes = engine_file.read()
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create an execution context for {engine_path}")
        self.tensor_names = [
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        ]
        self.input_shapes = {}
        self.output_shapes = {}
        self.tensor_dtypes = {}
        for name in self.tensor_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            self.tensor_dtypes[name] = dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shapes[name] = shape
            else:
                self.output_shapes[name] = shape

        self.state = {}
        for base_name in state_names:
            name = self.resolve_name(base_name)
            if name in self.input_shapes:
                tensor = aligned_tensor(self.input_shapes[name], self.tensor_dtypes[name], "cuda")
                tensor.zero_()
                self.state[name] = tensor
                self.context.set_tensor_address(name, tensor.data_ptr())
        if self.state:
            torch.cuda.synchronize()

    def resolve_name(self, base_name):
        if base_name in self.tensor_names:
            return base_name
        suffixed_name = f"{base_name}.1"
        return suffixed_name if suffixed_name in self.tensor_names else base_name

    def reset_state(self):
        for tensor in self.state.values():
            tensor.zero_()

    def prepare_input(self, name, inputs):
        shape = self.input_shapes[name]
        base_name = name.rsplit(".1", maxsplit=1)[0] if name.endswith(".1") else name
        if base_name not in inputs:
            raise KeyError(f"Missing TensorRT input {base_name}")
        value = inputs[base_name].to(device="cuda", dtype=self.tensor_dtypes[name])
        if tuple(value.shape) != shape:
            if tuple(value.shape[1:]) == shape:
                value = value.squeeze(0)
            elif tuple(shape[1:]) == tuple(value.shape):
                value = value.unsqueeze(0)
            else:
                raise ValueError(
                    f"Input {base_name} has shape {tuple(value.shape)}, expected {shape}"
                )
        return value

    def __call__(self, stream, **inputs):
        input_buffers = {}
        for name, shape in self.input_shapes.items():
            if name in self.state:
                continue
            value = self.prepare_input(name, inputs)
            buffer = aligned_tensor(shape, value.dtype, value.device)
            buffer.copy_(value)
            input_buffers[name] = buffer
            self.context.set_tensor_address(name, buffer.data_ptr())

        outputs = {}
        for name, shape in self.output_shapes.items():
            output = aligned_tensor(shape, self.tensor_dtypes[name], "cuda")
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())

        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        return outputs


STATE_NAMES = (
    "memory_embedding",
    "memory_reference_point",
    "memory_egopose",
    "memory_velo",
    "memory_timestamp",
)


class Far3DDecoderRunner(TensorRTRunner):
    def __init__(self, engine_path, input_callback=None):
        super().__init__(engine_path, STATE_NAMES)
        self.input_callback = input_callback
        self.scene_token = None
        self.timestamp_offset = None

    def __call__(self, stream, img_metas, timestamp, **inputs):
        scene_token = img_metas[0].data[0][0]["scene_token"]
        new_scene = self.scene_token != scene_token
        if new_scene:
            self.reset_state()
            self.scene_token = scene_token
            self.timestamp_offset = timestamp.clone()
        prev_exists_name = self.resolve_name("prev_exists")
        if prev_exists_name in self.input_shapes:
            inputs["prev_exists"] = torch.full(
                self.input_shapes[prev_exists_name],
                not new_scene,
                dtype=self.tensor_dtypes[prev_exists_name],
                device="cuda",
            )
        inputs["timestamp"] = (timestamp - self.timestamp_offset).float()
        if self.input_callback:
            calibration_inputs = {}
            for name in self.input_shapes:
                base_name = name.rsplit(".1", maxsplit=1)[0] if name.endswith(".1") else name
                if name in self.state:
                    value = self.state[name]
                else:
                    value = self.prepare_input(name, inputs)
                calibration_inputs[base_name] = value
            self.input_callback(calibration_inputs)
        outputs = super().__call__(stream, **inputs)
        for base_name in STATE_NAMES:
            input_name = self.resolve_name(base_name)
            output_name = f"{base_name}_out"
            if input_name in self.state and output_name in outputs:
                state_length = self.state[input_name].shape[1]
                self.state[input_name].copy_(outputs[output_name][:, :state_length])
        return outputs


class Far3DPipeline:
    def __init__(self, encoder_engine, decoder_engine, decoder_input_callback=None):
        self.encoder = TensorRTRunner(encoder_engine)
        self.decoder = Far3DDecoderRunner(decoder_engine, decoder_input_callback)

    @staticmethod
    def unpack(data):
        lidar2img = data["lidar2img"][0].data[0][0].unsqueeze(0).cuda()
        return {
            "img": data["img"][0].data[0].flip(2).permute(0, 1, 3, 4, 2).contiguous().cuda(),
            "intrinsics": data["intrinsics"][0].data[0][0].unsqueeze(0).cuda(),
            "extrinsics": data["extrinsics"][0].data[0][0].unsqueeze(0).cuda(),
            "lidar2img": lidar2img,
            "img2lidar": lidar2img.inverse(),
            "ego_pose": data["ego_pose"][0].data[0][0].unsqueeze(0).cuda(),
            "ego_pose_inv": data["ego_pose_inv"][0].data[0][0].unsqueeze(0).cuda(),
            "pad_shape": torch.tensor(data["img_metas"][0].data[0][0]["pad_shape"][0]).cuda(),
            "timestamp": torch.tensor(data["timestamp"][0].data[0][0]).cuda(),
        }

    def __call__(self, stream, data):
        with torch.cuda.stream(stream):
            inputs = self.unpack(data)
            image_features = self.encoder(stream, **inputs)
            decoder_inputs = dict(data)
            decoder_inputs.update(inputs)
            decoder_inputs.update(image_features)
            return self.decoder(stream, **decoder_inputs)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FAR3D TensorRT engines on Argoverse 2")
    parser.add_argument("config", help="Path to the FAR3D configuration file")
    parser.add_argument("encoder_engine")
    parser.add_argument("decoder_engine")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--eval-options", nargs="+", action=DictAction)
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.options and args.eval_options:
        raise ValueError("--options and --eval-options cannot both be specified")
    if args.options:
        warnings.warn("--options is deprecated; use --eval-options", stacklevel=2)
        args.eval_options = args.options
    return args


def import_plugin(cfg):
    plugin_dir = os.path.dirname(cfg.plugin_dir).split("/")
    importlib.import_module(".".join(plugin_dir))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    import_plugin(cfg)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    set_random_seed(0, deterministic=False)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    pipeline = Far3DPipeline(args.encoder_engine, args.decoder_engine)
    stream = torch.cuda.Stream()
    outputs = []
    for data in tqdm(data_loader):
        result = pipeline(stream, data)
        boxes = LiDARInstance3DBoxes(result["bboxes"].cpu())
        outputs.append(
            {
                "pts_bbox": {
                    "boxes_3d": boxes,
                    "scores_3d": result["scores"].cpu(),
                    "labels_3d": result["labels"].cpu(),
                }
            }
        )
        if args.max_samples is not None and len(outputs) == args.max_samples:
            break

    if len(outputs) < len(dataset):
        print(f"Processed {len(outputs)} samples; skipping dataset metrics")
        return

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
        eval_kwargs.pop(key, None)
    if args.eval_options:
        eval_kwargs.update(args.eval_options)
    print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
