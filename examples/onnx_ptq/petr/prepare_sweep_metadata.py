# Adapted from https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/tools/generate_sweep_pkl.py.
# Copyright (c) 2022 megvii-model. All Rights Reserved.
#
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

import argparse
import os

import mmcv
import numpy as np
import tqdm
from nuscenes import NuScenes
from pyquaternion import Quaternion

SENSORS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
NUM_PREV = 5
NUM_SWEEPS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Add PETRv2 camera sweeps to nuScenes metadata")
    parser.add_argument("data_root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output")
    return parser.parse_args()


def add_frame(nuscenes, data_root, sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat):
    sweep_cam = {
        "is_key_frame": sample_data["is_key_frame"],
        "data_path": os.path.join(data_root, sample_data["filename"]),
        "type": "camera",
        "timestamp": sample_data["timestamp"],
        "sample_data_token": sample_data["sample_token"],
    }
    pose_record = nuscenes.get("ego_pose", sample_data["ego_pose_token"])
    calibrated_sensor_record = nuscenes.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )

    sweep_cam["ego2global_translation"] = pose_record["translation"]
    sweep_cam["ego2global_rotation"] = pose_record["rotation"]
    sweep_cam["sensor2ego_translation"] = calibrated_sensor_record["translation"]
    sweep_cam["sensor2ego_rotation"] = calibrated_sensor_record["rotation"]
    sweep_cam["cam_intrinsic"] = calibrated_sensor_record["camera_intrinsic"]

    l2e_r_s_mat = Quaternion(sweep_cam["sensor2ego_rotation"]).rotation_matrix
    e2g_r_s_mat = Quaternion(sweep_cam["ego2global_rotation"]).rotation_matrix
    e2g_t_s = sweep_cam["ego2global_translation"]
    l2e_t_s = sweep_cam["sensor2ego_translation"]
    rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    translation = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    translation -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep_cam["sensor2lidar_rotation"] = rotation.T
    sweep_cam["sensor2lidar_translation"] = translation

    lidar2cam_r = np.linalg.inv(sweep_cam["sensor2lidar_rotation"])
    lidar2cam_t = sweep_cam["sensor2lidar_translation"] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.array(sweep_cam["cam_intrinsic"])
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    sweep_cam["intrinsics"] = viewpad.astype(np.float32)
    sweep_cam["extrinsics"] = lidar2cam_rt.astype(np.float32)
    sweep_cam["lidar2img"] = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    for key in (
        "ego2global_translation",
        "ego2global_rotation",
        "sensor2ego_translation",
        "sensor2ego_rotation",
        "cam_intrinsic",
    ):
        sweep_cam.pop(key)

    return sweep_cam


def add_sweeps(key_infos, nuscenes, data_root):
    for current_id in tqdm.tqdm(range(len(key_infos["infos"]))):
        info = key_infos["infos"][current_id]
        e2g_t = info["ego2global_translation"]
        l2e_t = info["lidar2ego_translation"]
        l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix

        sample = nuscenes.get("sample", info["token"])
        current_cams = {cam: nuscenes.get("sample_data", sample["data"][cam]) for cam in SENSORS}
        sweep_lists = []
        for _ in range(NUM_PREV):
            if not sample["prev"]:
                break
            for _ in range(NUM_SWEEPS):
                sweep_cams = {}
                for cam in SENSORS:
                    if not current_cams[cam]["prev"]:
                        sweep_cams = sweep_lists[-1] if sweep_lists else {}
                        break
                    sample_data = nuscenes.get("sample_data", current_cams[cam]["prev"])
                    sweep_cams[cam] = add_frame(
                        nuscenes,
                        data_root,
                        sample_data,
                        e2g_t,
                        l2e_t,
                        l2e_r_mat,
                        e2g_r_mat,
                    )
                    current_cams[cam] = sample_data
                sweep_lists.append(sweep_cams)

            sample = nuscenes.get("sample", sample["prev"])
            sweep_cams = {}
            for cam in SENSORS:
                sample_data = nuscenes.get("sample_data", sample["data"][cam])
                sweep_cams[cam] = add_frame(
                    nuscenes,
                    data_root,
                    sample_data,
                    e2g_t,
                    l2e_t,
                    l2e_r_mat,
                    e2g_r_mat,
                )
                current_cams[cam] = sample_data
            sweep_lists.append(sweep_cams)
        info["sweeps"] = sweep_lists


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root) + os.sep
    input_path = os.path.join(data_root, f"nuscenes_infos_{args.split}.pkl")
    output_path = args.output or os.path.join(
        data_root, f"mmdet3d_nuscenes_30f_infos_{args.split}.pkl"
    )
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    # This metadata must be generated locally by the documented MMDetection3D data-prep step.
    key_infos = mmcv.load(input_path)

    version = "v1.0-test" if args.split == "test" else "v1.0-trainval"
    nuscenes = NuScenes(version, data_root)
    add_sweeps(key_infos, nuscenes, data_root)
    mmcv.dump(key_infos, output_path)


if __name__ == "__main__":
    main()
