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
from pathlib import Path

import pandas as pd
from av2.utils.io import read_feather
from tools.create_infos_av2.create_av2_infos import create_av2_infos


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare FAR3D Argoverse 2 validation metadata")
    parser.add_argument("dataset_dir", type=Path, help="Argoverse 2 root containing val/")
    return parser.parse_args()


def main():
    args = parse_args()
    info_path = args.dataset_dir / "av2_val_infos.pkl"
    annotation_path = args.dataset_dir / "val_anno.feather"
    for output_path in (info_path, annotation_path):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")

    create_av2_infos(dataset_dir=args.dataset_dir, split="val", out_dir=args.dataset_dir)
    generated_info_path = args.dataset_dir / "av2_val_infos_mini.pkl"
    generated_info_path.replace(info_path)

    annotations = []
    for path in sorted((args.dataset_dir / "val").glob("*/annotations.feather")):
        frame = read_feather(path)
        frame["log_id"] = path.parent.name
        annotations.append(frame)
    if not annotations:
        raise RuntimeError(f"No validation annotations found under {args.dataset_dir / 'val'}")
    pd.concat(annotations).reset_index().to_feather(annotation_path)
    print(f"Saved {info_path} and {annotation_path}")


if __name__ == "__main__":
    main()
