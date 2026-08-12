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

# Adapted from PETR tools/generate_sweep_pkl.py.
# Copyright (c) 2022 Megvii Inc. All rights reserved.

import argparse
import os
import pickle

import mmcv
import numpy as np
import tqdm
from nuscenes import NuScenes
from pyquaternion import Quaternion

sensors = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Add PETRv2 camera sweeps to nuScenes metadata")
    parser.add_argument("data_root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output")
    return parser.parse_args()


args = parse_args()
info_prefix = args.split
data_root = os.path.abspath(args.data_root) + os.sep
num_prev = 5
num_sweep = 5

info_path = args.output or os.path.join(data_root, f"mmdet3d_nuscenes_30f_infos_{info_prefix}.pkl")
key_infos = pickle.load(open(os.path.join(data_root, f"nuscenes_infos_{info_prefix}.pkl"), "rb"))
if info_prefix == "test":
    nuscenes_version = "v1.0-test"
else:
    nuscenes_version = "v1.0-trainval"
nuscenes = NuScenes(nuscenes_version, data_root)


def add_frame(sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat):
    sweep_cam = {}
    sweep_cam["is_key_frame"] = sample_data["is_key_frame"]
    sweep_cam["data_path"] = os.path.join(data_root, sample_data["filename"])
    sweep_cam["type"] = "camera"
    sweep_cam["timestamp"] = sample_data["timestamp"]
    sweep_cam["sample_data_token"] = sample_data["sample_token"]
    pose_record = nuscenes.get("ego_pose", sample_data["ego_pose_token"])
    calibrated_sensor_record = nuscenes.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )

    sweep_cam["ego2global_translation"] = pose_record["translation"]
    sweep_cam["ego2global_rotation"] = pose_record["rotation"]
    sweep_cam["sensor2ego_translation"] = calibrated_sensor_record["translation"]
    sweep_cam["sensor2ego_rotation"] = calibrated_sensor_record["rotation"]
    sweep_cam["cam_intrinsic"] = calibrated_sensor_record["camera_intrinsic"]

    l2e_r_s = sweep_cam["sensor2ego_rotation"]
    l2e_t_s = sweep_cam["sensor2ego_translation"]
    e2g_r_s = sweep_cam["ego2global_rotation"]
    e2g_t_s = sweep_cam["ego2global_translation"]

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
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
    lidar2img_rt = viewpad @ lidar2cam_rt.T
    sweep_cam["intrinsics"] = viewpad.astype(np.float32)
    sweep_cam["extrinsics"] = lidar2cam_rt.astype(np.float32)
    sweep_cam["lidar2img"] = lidar2img_rt.astype(np.float32)

    pop_keys = [
        "ego2global_translation",
        "ego2global_rotation",
        "sensor2ego_translation",
        "sensor2ego_rotation",
        "cam_intrinsic",
    ]
    for key in pop_keys:
        sweep_cam.pop(key)

    return sweep_cam


for current_id in tqdm.tqdm(range(len(key_infos["infos"]))):
    e2g_t = key_infos["infos"][current_id]["ego2global_translation"]
    e2g_r = key_infos["infos"][current_id]["ego2global_rotation"]
    l2e_t = key_infos["infos"][current_id]["lidar2ego_translation"]
    l2e_r = key_infos["infos"][current_id]["lidar2ego_rotation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    sample = nuscenes.get("sample", key_infos["infos"][current_id]["token"])
    current_cams = {}
    for cam in sensors:
        current_cams[cam] = nuscenes.get("sample_data", sample["data"][cam])

    sweep_lists = []
    for _ in range(num_prev):
        if sample["prev"] == "":
            break
        for _ in range(num_sweep):
            sweep_cams = {}
            for cam in sensors:
                if current_cams[cam]["prev"] == "":
                    sweep_cams = sweep_lists[-1]
                    break
                sample_data = nuscenes.get("sample_data", current_cams[cam]["prev"])
                sweep_cam = add_frame(sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat)
                current_cams[cam] = sample_data
                sweep_cams[cam] = sweep_cam
            sweep_lists.append(sweep_cams)
        sample = nuscenes.get("sample", sample["prev"])
        sweep_cams = {}
        for cam in sensors:
            sample_data = nuscenes.get("sample_data", sample["data"][cam])
            sweep_cam = add_frame(sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat)
            current_cams[cam] = sample_data
            sweep_cams[cam] = sweep_cam
        sweep_lists.append(sweep_cams)
    key_infos["infos"][current_id]["sweeps"] = sweep_lists

mmcv.dump(key_infos, info_path)
