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

# Adapted from https://github.com/NVIDIA/DL4AGX/tree/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/petr-trt/export_eval.
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import importlib
import os

import tensorrt as trt
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import import_modules_from_strings
from mmdet.apis import set_random_seed
from mmdet3d.core import bbox3d2result
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
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
    offset = ((-storage.data_ptr()) % alignment) // element_size
    return storage[offset : offset + element_count].view(shape)


class TensorRTRunner:
    def __init__(self, engine_path, input_callback=None):
        with open(engine_path, "rb") as engine_file:
            engine_bytes = engine_file.read()
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_callback = input_callback
        self.input_names = []
        self.output_names = []
        self.shapes = {}
        self.dtypes = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.shapes[name] = tuple(self.engine.get_tensor_shape(name))
            self.dtypes[name] = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            names = (
                self.input_names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else self.output_names
            )
            names.append(name)

    @staticmethod
    def fit_shape(value, shape):
        if tuple(value.shape) == shape:
            return value
        if tuple(value.shape[1:]) == shape:
            return value.squeeze(0)
        if tuple(shape[1:]) == tuple(value.shape):
            return value.unsqueeze(0)
        raise ValueError(f"Input has shape {tuple(value.shape)}, expected {shape}")

    def __call__(self, stream, values):
        if len(values) != len(self.input_names):
            raise ValueError(f"Received {len(values)} inputs, expected {len(self.input_names)}")
        inputs = []
        for name, value in zip(self.input_names, values):
            value = value.to(device="cuda", dtype=self.dtypes[name]).contiguous()
            value = self.fit_shape(value, self.shapes[name])
            buffer = aligned_tensor(self.shapes[name], value.dtype, value.device)
            buffer.copy_(value)
            inputs.append(buffer)
            self.context.set_tensor_address(name, buffer.data_ptr())
        if self.input_callback:
            self.input_callback(inputs)

        outputs = []
        for name in self.output_names:
            output = aligned_tensor(self.shapes[name], self.dtypes[name], "cuda")
            outputs.append(output)
            self.context.set_tensor_address(name, output.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        return outputs


def import_plugin(cfg):
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get("plugin"):
        plugin_dir = os.path.dirname(cfg.plugin_dir).split("/")
        importlib.import_module(".".join(plugin_dir))


def build_runtime(config_path, checkpoint_path, cfg_options=None):
    cfg = Config.fromfile(config_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16"):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if hasattr(dataset, "PALETTE"):
        model.PALETTE = checkpoint.get("meta", {}).get("PALETTE", dataset.PALETTE)
    model = model.cuda().eval()
    return cfg, dataset, loader, model


class PETRPipeline:
    def __init__(self, version, model, backbone_engine, head_engine, callbacks=(None, None)):
        self.version = version
        self.model = model
        self.backbone = TensorRTRunner(backbone_engine, callbacks[0])
        self.head = TensorRTRunner(head_engine, callbacks[1])

    @staticmethod
    def masks(features, img_metas):
        batch_size, num_cams = features[0].shape[:2]
        input_h, input_w, _ = img_metas[0]["pad_shape"][0]
        masks = features[0].new_ones((batch_size, num_cams, input_h, input_w))
        for image_id in range(batch_size):
            for camera_id in range(num_cams):
                image_h, image_w, _ = img_metas[image_id]["img_shape"][camera_id]
                masks[image_id, camera_id, :image_h, :image_w] = 0
        return F.interpolate(masks, size=features[0].shape[-2:]).to(torch.bool)

    def backbone_inputs(self, images, img_metas):
        if self.version == "v1":
            return [images]
        current = images[:, :6].contiguous()
        previous = images[:, 6:12].contiguous()
        previous_features = self.model.extract_img_feat(previous, img_metas)
        return [current, *previous_features]

    def head_inputs(self, features, img_metas):
        masks = self.masks(features, img_metas)
        coords, _ = self.model.pts_bbox_head.position_embeding(features, img_metas, masks)
        values = [features[0]]
        if self.version == "v2":
            timestamps = features[0].new_tensor([meta["timestamp"] for meta in img_metas])
            timestamps = timestamps.view(1, -1, 6)
            values.append((timestamps[:, 1] - timestamps[:, 0]).mean(-1))
        values.append(coords)
        return values

    def __call__(self, stream, data):
        images = data["img"][0].data[0].cuda()
        img_metas = data["img_metas"][0].data[0]
        with torch.cuda.stream(stream), torch.no_grad():
            features = self.backbone(stream, self.backbone_inputs(images, img_metas))
            camera_count = 6 if self.version == "v1" else 12
            features = [value.reshape(1, camera_count, *value.shape[-3:]) for value in features]
            outputs = self.head(stream, self.head_inputs(features, img_metas))
            outputs = [value.float() for value in outputs]
            head_outputs = {
                "all_cls_scores": outputs[0],
                "all_bbox_preds": outputs[1],
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
            }
            boxes = self.model.pts_bbox_head.get_bboxes(head_outputs, img_metas, rescale=True)
        return [
            {"pts_bbox": bbox3d2result(boxes_3d, scores_3d, labels_3d)}
            for boxes_3d, scores_3d, labels_3d in boxes
        ]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PETR TensorRT engines on nuScenes")
    parser.add_argument("version", choices=("v1", "v2"))
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("backbone_engine")
    parser.add_argument("head_engine")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--eval-options", nargs="+", action=DictAction)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    return args


def main():
    args = parse_args()
    set_random_seed(0, deterministic=False)
    cfg, dataset, loader, model = build_runtime(args.config, args.checkpoint, args.cfg_options)
    pipeline = PETRPipeline(args.version, model, args.backbone_engine, args.head_engine)
    stream = torch.cuda.Stream()
    outputs = []
    for data in tqdm(loader):
        outputs.extend(pipeline(stream, data))
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
    main()
