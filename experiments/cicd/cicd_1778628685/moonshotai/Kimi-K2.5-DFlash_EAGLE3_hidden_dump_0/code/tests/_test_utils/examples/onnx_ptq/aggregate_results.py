# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import csv
import os


def get_metrics_from_csv(file_path):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        top1_accuracy, top5_accuracy, latency = None, None, None
        for row in csv_reader:
            if row[0] == "Top 1":
                top1_accuracy = float(row[1])
            elif row[0] == "Top 5":
                top5_accuracy = float(row[1])
            elif row[0] == "Latency":
                latency = float(row[1])
        return top1_accuracy, top5_accuracy, latency


def extract_model_name_and_precision(file_name, split_index):
    if split_index == 1:
        return file_name.rsplit("_", 1)
    elif split_index == 2:
        result = file_name.rsplit("_", 2)
        return result[0], "_".join(result[1:])
    else:
        raise ValueError("Invalid split index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help=(
            "Path to the folder containing results for all models."
            "The folder should have the following structure:"
            "results_dir"
            "|- model1"
            "|  |- precision1"
            "|  |  |- model1_precision1.csv"
            "|  |- precision2"
            "|  |  |- model1_precision2.csv"
            "|- model2"
            "|  |- precision1"
            "|  |  |- model2_precision1.csv"
            "|  |- precision2"
            "|  |  |- model2_precision2.csv"
            "|- ..."
            "Every model folder must contain a subfolder for fp16 precision to calculate the relative difference."
        ),
    )

    args = parser.parse_args()
    build_folder_path = args.results_dir
    data = []
    fp16_metrics = {}

    # Traverse each folder and subfolder
    for model_folder in os.listdir(build_folder_path):
        if model_folder == "calib" or not os.path.isdir(
            os.path.join(build_folder_path, model_folder)
        ):
            continue
        model_folder_path = os.path.join(build_folder_path, model_folder)
        sorted_precision_folders = sorted(os.listdir(model_folder_path))

        # Load FP16 precision accuracies first
        for precision_folder in sorted_precision_folders:
            if precision_folder != "fp16":
                continue
            precision_folder_path = os.path.join(model_folder_path, precision_folder)
            for file in os.listdir(precision_folder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(precision_folder_path, file)
                    top1_accuracy, top5_accuracy, latency = get_metrics_from_csv(file_path)
                    if (
                        top1_accuracy is not None
                        and top5_accuracy is not None
                        and latency is not None
                    ):
                        # Save FP16 accuracies for later reference and add to data with 0 diffs
                        fp16_metrics[model_folder] = (top1_accuracy, top5_accuracy, latency)
                        data.append(
                            [
                                model_folder,
                                "fp16",
                                top1_accuracy,
                                top5_accuracy,
                                latency,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ]
                        )

        # Calculate accuracies for other precision levels and compute differences
        for precision_folder in sorted_precision_folders:
            if precision_folder == "fp16":
                continue
            precision_folder_path = os.path.join(model_folder_path, precision_folder)
            for file in os.listdir(precision_folder_path):
                if file.endswith(".csv"):
                    file_name = file.split(".")[0]
                    split_index = 2 if precision_folder == "int8_iq" else 1
                    model_name, precision = extract_model_name_and_precision(file_name, split_index)
                    file_path = os.path.join(precision_folder_path, file)
                    top1_accuracy, top5_accuracy, latency = get_metrics_from_csv(file_path)
                    if (
                        top1_accuracy is not None
                        and top5_accuracy is not None
                        and latency is not None
                    ):
                        # Retrieve FP16 accuracies
                        fp16_top1, fp16_top5, fp16_latency = fp16_metrics.get(
                            model_folder, (None, None, None)
                        )
                        if fp16_top1 is not None and fp16_top5 is not None and latency is not None:
                            # Calculate absolute and relative differences
                            abs_diff_top1 = top1_accuracy - fp16_top1
                            abs_diff_top5 = top5_accuracy - fp16_top5
                            abs_diff_latency = latency - fp16_latency
                            rel_diff_top1 = (
                                (abs_diff_top1 / fp16_top1) * 100 if fp16_top1 != 0 else None
                            )
                            rel_diff_top5 = (
                                (abs_diff_top5 / fp16_top5) * 100 if fp16_top5 != 0 else None
                            )
                            rel_diff_latency = (
                                (abs_diff_latency / fp16_latency) * 100
                                if fp16_latency != 0
                                else None
                            )
                            data.append(
                                [
                                    model_name,
                                    precision,
                                    top1_accuracy,
                                    top5_accuracy,
                                    latency,
                                    abs_diff_top1,
                                    abs_diff_top5,
                                    abs_diff_latency,
                                    rel_diff_top1,
                                    rel_diff_top5,
                                    rel_diff_latency,
                                ]
                            )

    # Define output file path
    output_file_path = os.path.join(build_folder_path, "aggregated_results.csv")

    # Write aggregated data to a new CSV file
    with open(output_file_path, mode="w", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        # Write header
        csv_writer.writerow(
            [
                "Model Name",
                "Precision",
                "Top1 Accuracy",
                "Top5 Accuracy",
                "Latency",
                "Absolute Difference Top1",
                "Absolute Difference Top5",
                "Absolute Difference Latency",
                "Relative Difference Top1",
                "Relative Difference Top5",
                "Relative Difference Latency",
            ]
        )
        # Write data rows
        csv_writer.writerows(data)

    print(f"Aggregated data with differences saved to {output_file_path}")


if __name__ == "__main__":
    main()
