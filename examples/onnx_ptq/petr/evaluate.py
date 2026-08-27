# Adapted from https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/petr-trt/export_eval/v1/v1_evaluate_trt.py
# and https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/petr-trt/export_eval/v2/v2_evaluate_trt.py.
# Copyright (c) OpenMMLab. All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.onnx_ptq.trt_runner import TensorRTRunner

__all__ = ["PETRPipeline", "build_runtime"]


def import_plugin(cfg):
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    plugin_dir = cfg.get("plugin_dir")
    if cfg.get("plugin") and plugin_dir:
        importlib.import_module(".".join(os.path.dirname(plugin_dir).split("/")))


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
    def __init__(
        self,
        version,
        model,
        backbone_engine,
        head_engine,
        backbone_input_callback=None,
        head_input_callback=None,
    ):
        self.version = version
        self.model = model
        self.backbone = TensorRTRunner(backbone_engine, input_callback=backbone_input_callback)
        self.head = TensorRTRunner(head_engine, input_callback=head_input_callback)

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
            return {"img": images}
        current = images[:, :6].contiguous()
        previous = images[:, 6:12].contiguous()
        previous_features = self.model.extract_img_feat(previous, img_metas)
        return {
            "img": current,
            **{f"prev.{index}": value for index, value in enumerate(previous_features)},
        }

    def head_inputs(self, features, img_metas):
        masks = self.masks(features, img_metas)
        coords, _ = self.model.pts_bbox_head.position_embeding(features, img_metas, masks)
        inputs = {"mlvl_feats.0": features[0]}
        if self.version == "v2":
            timestamps = features[0].new_tensor([meta["timestamp"] for meta in img_metas])
            timestamps = timestamps.view(1, -1, 6)
            inputs["img_metas.0[mean_time_stamp]"] = (timestamps[:, 1] - timestamps[:, 0]).mean(-1)
        inputs["img_metas.0[coords_position_embeding]"] = coords
        return inputs

    def __call__(self, stream, data):
        images = data["img"][0].data[0].cuda()
        img_metas = data["img_metas"][0].data[0]
        with torch.cuda.stream(stream), torch.no_grad():
            feature_outputs = self.backbone(stream, **self.backbone_inputs(images, img_metas))
            camera_count = 6 if self.version == "v1" else 12
            features = [
                feature_outputs[name].reshape(1, camera_count, *feature_outputs[name].shape[-3:])
                for name in ("out.0", "out.1")
            ]
            outputs = self.head(stream, **self.head_inputs(features, img_metas))
            head_outputs = {
                "all_cls_scores": outputs["out.all_cls_scores"].float(),
                "all_bbox_preds": outputs["out.all_bbox_preds"].float(),
                "enc_cls_scores": None,
                "enc_bbox_preds": None,
            }
            boxes = self.model.pts_bbox_head.get_bboxes(head_outputs, img_metas, rescale=True)
        torch.cuda.current_stream().wait_stream(stream)
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
        if args.max_samples is not None and len(outputs) >= args.max_samples:
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
