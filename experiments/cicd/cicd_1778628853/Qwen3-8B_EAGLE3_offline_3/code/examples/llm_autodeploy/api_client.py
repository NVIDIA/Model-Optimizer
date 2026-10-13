# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import requests


def get_server_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--prompt",
        nargs="+",  # Allows multiple inputs
        required=True,
        help="List of prompts to send (e.g., --prompt 'What is AI?' 'What is golf?')",
    )
    parser.add_argument(
        "--stop",
        nargs="+",  # Allows multiple inputs
        default=None,
        help="List of stop words.",
    )

    return parser.parse_args()


def send_request(host, port, prompts, stop):
    url = f"http://{host}:{port}/v1/completions"
    data = {"prompt": prompts, "stop": stop, "model": "autodeploy_demo"}

    try:
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()  # Check for HTTP errors
        if response.status_code == 200:
            response_dict = response.json()
            for prompt, output in zip(prompts, response_dict["choices"]):
                print(f"{prompt}{output['text']}")
        else:
            print(f"Error: {response.status_code}, {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    args = get_server_args()
    send_request(args.host, args.port, args.prompt, args.stop)
