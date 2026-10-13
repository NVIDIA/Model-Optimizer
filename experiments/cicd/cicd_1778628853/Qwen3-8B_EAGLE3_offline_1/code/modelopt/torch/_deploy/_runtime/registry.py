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

from collections.abc import Callable

from .runtime_client import Deployment, RuntimeClient

__all__ = ["RuntimeRegistry"]


class RuntimeRegistry:
    """Registry to store and retrieve various runtime client implementations."""

    _runtime_client_lookup: dict[str, type[RuntimeClient]] = {}

    @classmethod
    def register(cls, runtime: str) -> Callable[[type[RuntimeClient]], type[RuntimeClient]]:
        """A decorator to register a RuntimeClient with its relevant runtime.

        For example:

        .. code-block:: python

            @RuntimeRegistry.register("my_runtime")
            class MyRuntimeClient(RuntimeClient):
                pass
        """

        def _register_runtime_client(new_type: type[RuntimeClient]) -> type[RuntimeClient]:
            cls._runtime_client_lookup[runtime] = new_type
            new_type._runtime = runtime
            return new_type

        return _register_runtime_client

    @classmethod
    def unregister(cls, runtime: str) -> None:
        """A helper to unregister a RuntimeClient

        For example:

        .. code-block:: python

            @RuntimeRegistry.register("my_runtime")
            class MyRuntimeClient(RuntimeClient):
                pass


            # later
            RuntimeRegistry.unregister("my_runtime")
        """
        cls._runtime_client_lookup.pop(runtime)

    @classmethod
    def get(cls, deployment: Deployment) -> RuntimeClient:
        """Get the runtime client for the given deployment.

        Args:
            deployment: Deployment configuration.

        Returns:
            The runtime client for the given deployment.
        """
        # check for valid runtime
        if "runtime" not in deployment:
            raise KeyError("Runtime was not set.")
        if deployment["runtime"] not in cls._runtime_client_lookup:
            raise ValueError(f"Runtime {deployment['runtime']} is not supported.")

        # initialize runtime client
        return cls._runtime_client_lookup[deployment["runtime"]](deployment)
