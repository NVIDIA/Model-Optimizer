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

from pathlib import Path

SKILLS = Path(__file__).parents[2]
CREDENTIALS = SKILLS / "common" / "credentials.md"
SLURM_SETUP = SKILLS / "common" / "slurm-setup.md"


def test_gitlab_enroot_guidance_checks_layers_and_keeps_secrets_out_of_arguments():
    credentials = CREDENTIALS.read_text()
    slurm_setup = SLURM_SETUP.read_text()

    assert "printf 'machine %s login %s password %s\\n'" in credentials
    assert (
        '"$registry_host" "$registry_user" "$gitlab_token" > "$auth_dir/.credentials"'
        in credentials
    )
    assert (
        '"docker://${registry_user}@${registry_host}:${registry_port}#${repository}:${tag}"'
        in credentials
    )
    assert "`machine` value is the hostname **without a port**" in credentials
    assert "Do not append `@sha256:...`" in credentials

    assert "a real import verifies blob/layer access" in slurm_setup
    assert 'test -s "$probe_dir/image.sqsh"' in slurm_setup
    assert "enroot import --output /dev/null" not in slurm_setup
    assert "`CONNECT tunnel failed, response 403`" in slurm_setup
