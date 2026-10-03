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

from abc import abstractmethod

import torch
import torch.nn as nn

from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


class GetT:
    def __init__(self):
        self.counter = 1
        self.active = 0
        self.num_choices = 3

    def set_default_counter(self):
        self.counter = 1

    def __call__(self):
        options = {
            0: torch.ones(3, 3) * self.counter,
            1: torch.tensor(1) * self.counter,
            2: torch.tensor(1.0) * self.counter,
        }
        return options[self.active]


class BaseDeployModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.get = GetT()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Implement the forward method."""

    @abstractmethod
    def onnx_input_names():
        """Return expected onnx input names."""

    @abstractmethod
    def onnx_output_names():
        """Return expected onnx output names."""

    @abstractmethod
    def output_spec():
        """Return expected output spec."""

    @abstractmethod
    def get_args(self):
        """Return expected input."""

    @property
    def compile_fail(self):
        return False

    @property
    def invalid_device_input(self):
        return False

    def check_input_option(self, active):
        return True


class TensorModel(BaseDeployModel):
    def forward(self, x: torch.Tensor):
        return torch.add(x, x) - x

    def onnx_input_names(self):
        return ["x"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return self.get()


class ListModel1(BaseDeployModel):
    def forward(self, x: list[torch.Tensor]):
        return torch.add(x[0], x[0]) - x[0]

    def onnx_input_names(self):
        return ["x.0"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return [self.get()]


class ListModel2(BaseDeployModel):
    def forward(self, x: list[torch.Tensor]):
        return torch.add(x[0], x[1]) - x[0]

    def onnx_input_names(self):
        return ["x.0", "x.1"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return [self.get(), self.get()]


class ListMultiModel(BaseDeployModel):
    def forward(self, x: list[torch.Tensor], y: torch.Tensor):
        return torch.add(x[0], x[1]) - x[0] + y

    def onnx_input_names(self):
        return ["x.0", "x.1", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return ([self.get(), self.get()], self.get())


class ListListModel(BaseDeployModel):
    def forward(self, x: list[list[torch.Tensor]], y: torch.Tensor):
        return torch.add(x[0][0], x[0][1]) - x[1] + y

    def onnx_input_names(self):
        return ["x.0.0", "x.0.1", "x.1", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return ([[self.get(), self.get()], self.get()], self.get())


class ListListListModel(BaseDeployModel):
    def forward(self, x: list[list[torch.Tensor]], y: torch.Tensor):
        return torch.add(x[0][0][0], x[0][0][1]) - x[1] + y

    def onnx_input_names(self):
        return ["x.0.0.0", "x.0.0.1", "x.1", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return ([[[self.get(), self.get()]], self.get()], self.get())


class ListDictModel(BaseDeployModel):
    def forward(self, x: list[list[torch.Tensor]], t: dict[str, torch.Tensor], y: torch.Tensor):
        return torch.add(x[0][0], x[0][1]) - x[1] + y + t["x"] - t["y"]

    def onnx_input_names(self):
        return ["x.0.0", "x.0.1", "x.1", "t.x", "t.y", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (
            [[self.get(), self.get()], self.get()],
            {"x": self.get(), "y": self.get()},
            self.get(),
        )


class NestedModel(BaseDeployModel):
    def __init__(self) -> None:
        super().__init__()
        self.nested_list = ListListModel()
        self.nested_dict = DictModel()

    def forward(self, x: list[list[torch.Tensor]], t: dict[str, torch.Tensor], y: torch.Tensor):
        return self.nested_list(x, y) - self.nested_dict(t)

    def onnx_input_names(self):
        return ["x.0.0", "x.0.1", "x.1", "t.x", "t.y", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (
            [[self.get(), self.get()], self.get()],
            {"x": self.get(), "y": self.get()},
            self.get(),
        )


class DictModel(BaseDeployModel):
    def forward(self, t: dict[str, torch.Tensor]):
        return torch.add(t["x"], t["y"]) - t["x"]

    def onnx_input_names(self):
        return ["t.x", "t.y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return ({"x": self.get(), "y": self.get()}, {})


class DictModel2(DictModel):
    def onnx_input_names(self):
        # For DictModel2 we pass in the input dict with reverse order of keys
        return ["t.y", "t.x"]

    def get_args(self):
        return ({"y": self.get(), "x": self.get()}, {})


class DictMultiModel(BaseDeployModel):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y) - x

    def onnx_input_names(self):
        return ["x", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), self.get())


class TwoOutModel(BaseDeployModel):
    def forward(self, x: torch.Tensor):
        return torch.add(x, x), torch.add(x, x)

    def onnx_input_names(self):
        return ["x"]

    def onnx_output_names(self):
        return ["out.0", "out.1"]

    def output_spec(self):
        return [None, None]

    def get_args(self):
        return self.get()


class TupleOutModel(BaseDeployModel):
    def forward(self, x: torch.Tensor):
        return (torch.add(x, x),)

    def onnx_input_names(self):
        return ["x"]

    def onnx_output_names(self):
        return ["out.0"]

    def output_spec(self):
        return (None,)

    def get_args(self):
        return self.get()


class NestedOutModel(BaseDeployModel):
    def forward(self, x: torch.Tensor):
        return [
            x,
            [
                x + x,
                x * x,
                x - x,
                [x * 2 + 4],
                {"a": x * 3},
                {"b": x - 10},
            ],
        ]

    def onnx_input_names(self):
        return ["x"]

    def onnx_output_names(self):
        return ["out.0", "out.1.0", "out.1.1", "out.1.2", "out.1.3.0", "out.1.4.a", "out.1.5.b"]

    def output_spec(self):
        return [
            None,
            [
                None,
                None,
                None,
                [None],
                {"a": None},
                {"b": None},
            ],
        ]

    def get_args(self):
        return self.get()


class DefaultModel1(BaseDeployModel):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = torch.ones(3, 3),
        b: bool = False,
        f: float = 0.5,
    ):
        out = x + t if b else x + f

        return out

    def onnx_input_names(self):
        return ["x", "t"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), {"f": 1.4, "b": True})

    @property
    def invalid_device_input(self):
        return True

    def check_input_option(self, active):
        return active == 0


class DefaultModel2(DefaultModel1):
    def onnx_input_names(self):
        return ["x", "f"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), {"f": 1.4, "t": torch.ones(3, 3)})

    def check_input_option(self, active):
        return False


class DefaultModel3(DefaultModel1):
    def onnx_input_names(self):
        return ["x", "t"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), {"b": True})

    def check_input_option(self, active):
        return False


class ArgsKwargsModel1(BaseDeployModel):
    def forward(self, x, y, *args, z=5.0, **kwargs):
        return torch.add(x, y) + z + sum(args) + sum(kwargs.values())

    def onnx_input_names(self):
        return ["x", "y", "args.0", "args.1"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), self.get(), self.get(), self.get(), {})


class ArgsKwargsModel2(ArgsKwargsModel1):
    def get_args(self):
        return (self.get(), self.get(), self.get(), self.get(), {"z": self.get()})

    @property
    def compile_fail(self):
        return True


class ArgsKwargsModel3(ArgsKwargsModel1):
    def get_args(self):
        return (self.get(), self.get(), self.get(), self.get(), {"zz": self.get()})

    @property
    def compile_fail(self):
        return True


class SkipArgsKwargsModel1(BaseDeployModel):
    """A model that has args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = set()

    def forward(self, x, y, a=None, b=None, c=None):
        res = torch.add(x, y)
        if a is None:
            assert "a" not in self._provided_args
        else:
            assert "a" in self._provided_args
            res += 100 * a
        if b is None:
            assert "b" not in self._provided_args
        else:
            assert "b" in self._provided_args
            res += 1000 * b
        if c is None:
            assert "c" not in self._provided_args
        else:
            assert "c" in self._provided_args
            res += 10000 * c
        return res

    def onnx_input_names(self):
        return ["x", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), self.get())


class SkipArgsKwargsModel2(SkipArgsKwargsModel1):
    """A model that has args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = {"c"}

    def onnx_input_names(self):
        return ["x", "y", "c"]

    def get_args(self):
        return (self.get(), self.get(), {"c": self.get()})


class SkipArgsKwargsModel3(SkipArgsKwargsModel1):
    """A model that has args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = {"a"}

    def onnx_input_names(self):
        return ["x", "y", "a"]

    def get_args(self):
        return (self.get(), self.get(), {"a": self.get()})


class SkipArgsKwargsModel4(SkipArgsKwargsModel1):
    """A model that has args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = {"a", "b", "c"}

    def onnx_input_names(self):
        return ["x", "y", "a", "b", "c"]

    def get_args(self):
        return (self.get(), self.get(), {"a": self.get(), "b": self.get(), "c": self.get()})


class SkipArgsKwargsModel5(SkipArgsKwargsModel1):
    """A model that has args/kwargs that are skipped because provided args have None."""

    _provided_args = {"b"}

    def forward(self, x, y, a=None, b=torch.tensor(1), c=None):
        return super().forward(x, y, a, b, c)

    def onnx_input_names(self):
        return ["x", "y", "b"]

    def get_args(self):
        return (self.get(), self.get())

    def check_input_option(self, active):
        return active == 1


class SkipArgsKwargsModel6(SkipArgsKwargsModel5):
    """A model that has args/kwargs that are skipped because provided args have None."""

    _provided_args = set()

    def onnx_input_names(self):
        return ["x", "y"]

    def get_args(self):
        return (self.get(), self.get(), {"b": None})

    def check_input_option(self, active):
        return True


class SkipArgsKwargsModel7(SkipArgsKwargsModel6):
    """A model that has args/kwargs that are skipped because provided args have None."""

    _provided_args = {"c"}

    def onnx_input_names(self):
        return ["x", "y", "c"]

    def get_args(self):
        return (self.get(), self.get(), {"b": None, "c": self.get()})


class SkipNestedArgsKwargsModel1(BaseDeployModel):
    """A model that has nested args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = set()

    def forward(self, x, y, z=None, lookup=None):
        res = torch.add(x, y)
        lookup = lookup or {}
        if "a" in lookup:
            assert "a" in self._provided_args
            res += 100 * lookup["a"]
        else:
            assert "a" not in self._provided_args
        if "b" in lookup:
            assert "b" in self._provided_args
            res += 1000 * lookup["b"]
        else:
            assert "b" not in self._provided_args
        if "c" in lookup:
            assert "c" in self._provided_args
            res += 10000 * lookup["c"]
        else:
            assert "c" not in self._provided_args
        if z is not None:
            res += 100000 * z
        return res

    def onnx_input_names(self):
        return ["x", "y"]

    def onnx_output_names(self):
        return ["out"]

    def output_spec(self):
        return None

    def get_args(self):
        return (self.get(), self.get())


class SkipNestedArgsKwargsModel2(SkipNestedArgsKwargsModel1):
    _provided_args = {"c"}

    def onnx_input_names(self):
        return ["x", "y", "lookup.c"]

    def get_args(self):
        return (self.get(), self.get(), {"lookup": {"c": self.get()}})


class SkipNestedArgsKwargsModel3(SkipNestedArgsKwargsModel1):
    _provided_args = {"a"}

    def onnx_input_names(self):
        return ["x", "y", "lookup.a"]

    def get_args(self):
        return (self.get(), self.get(), {"lookup": {"a": self.get()}})


class SkipNestedArgsKwargsModel4(SkipNestedArgsKwargsModel1):
    """A model that has args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = {"a", "b", "c"}

    def onnx_input_names(self):
        return ["x", "y", "lookup.a", "lookup.b", "lookup.c"]

    def get_args(self):
        return (
            self.get(),
            self.get(),
            {"lookup": {"a": self.get(), "b": self.get(), "c": self.get()}},
        )


class SkipNestedArgsKwargsModel5(SkipNestedArgsKwargsModel1):
    """A model that has args and kwargs that will be skipped in the onnx conversion."""

    _provided_args = {"a", "b", "c"}

    def onnx_input_names(self):
        return ["x", "y", "z", "lookup.a", "lookup.b", "lookup.c"]

    def get_args(self):
        return (
            self.get(),
            self.get(),
            self.get(),
            {"a": self.get(), "b": self.get(), "c": self.get()},
            {},
        )


def get_deploy_models(dynamic_control_flow=True):
    models = [
        TensorModel(),
        ListModel1(),
        ListModel2(),
        ListMultiModel(),
        ListListModel(),
        ListListListModel(),
        ListDictModel(),
        NestedModel(),
        DictModel(),
        DictModel2(),
        DictMultiModel(),
        TwoOutModel(),
        TupleOutModel(),
        NestedOutModel(),
        ArgsKwargsModel1(),
        ArgsKwargsModel2(),
        ArgsKwargsModel3(),
    ]
    if dynamic_control_flow:
        models.extend(
            [
                DefaultModel1(),
                DefaultModel2(),
                DefaultModel3(),
                SkipArgsKwargsModel1(),
                SkipArgsKwargsModel2(),
                SkipArgsKwargsModel3(),
                SkipArgsKwargsModel4(),
                SkipArgsKwargsModel5(),
                SkipArgsKwargsModel6(),
                SkipArgsKwargsModel7(),
                SkipNestedArgsKwargsModel1(),
                SkipNestedArgsKwargsModel2(),
                SkipNestedArgsKwargsModel3(),
                SkipNestedArgsKwargsModel4(),
                SkipNestedArgsKwargsModel5(),
            ]
        )
    return {type(model).__name__: model for model in models}


# TODO: check if we want to standardize these models with the BaseDeployModel above!


class LeNet5(nn.Module):
    """Model that uses functional modules instead of nn.Modules. Expects input of shape (1, 3, 32, 32)."""

    def __init__(self, num_classes=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(400, 12)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(12, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    @staticmethod
    def gen_input(batch_size: int = 1):
        return torch.randn([batch_size, 3, 32, 32])

    def to_onnx_bytes(self, batch_size: int = 1):
        onnx_bytes, _ = get_onnx_bytes_and_metadata(self, LeNet5.gen_input(batch_size))
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        return onnx_bytes_obj.onnx_model["LeNet5.onnx"]


class LeNet5TwoInputs(torch.nn.Module):
    def __init__(self, num_classes=3, seed=0):
        super().__init__()
        self.model = LeNet5(num_classes, seed)

        self.conv1 = nn.Conv2d(5, 3, kernel_size=1, stride=2, padding=0)

    def forward(self, x, x1):
        y = self.model(x)
        y2 = self.model.relu1(self.conv1(x1))
        return y, y2

    @staticmethod
    def gen_input(input_id, batch_size: int = 1):
        if input_id == 0:
            return torch.randn([batch_size, 3, 32, 32])
        return torch.randn([batch_size, 5, 64, 64])

    def to_onnx_bytes(self, batch_size: int = 1):
        onnx_bytes, _ = get_onnx_bytes_and_metadata(
            self,
            (LeNet5TwoInputs.gen_input(0, batch_size), LeNet5TwoInputs.gen_input(1, batch_size)),
        )
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        return onnx_bytes_obj.onnx_model["LeNet5TwoInputs.onnx"]


class LeNet5TwoOutputs(torch.nn.Module):
    def __init__(self, num_classes=3, seed=0):
        super().__init__()
        self.model = LeNet5(num_classes, seed)

    def forward(self, x):
        y = self.model(x)
        y2 = self.model.relu1(x)
        return y, y2

    @staticmethod
    def gen_input(batch_size: int = 1):
        return torch.randn([batch_size, 3, 32, 32])

    def to_onnx_bytes(self, batch_size: int = 1):
        onnx_bytes, _ = get_onnx_bytes_and_metadata(self, LeNet5TwoOutputs.gen_input(batch_size))
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        return onnx_bytes_obj.onnx_model["LeNet5TwoOutputs.onnx"]


class LeNet5Ooo(torch.nn.Module):
    def __init__(self, num_classes=3, seed=0):
        super().__init__()
        self.model = LeNet5(num_classes, seed)

        self.conv1 = nn.Conv2d(5, 3, kernel_size=1, stride=2, padding=0)

    def forward(self, x2, x1):
        y2 = self.model(x2)
        y1 = self.model.relu1(self.conv1(x1))
        return {"y2": y2, "y1": y1}

    @staticmethod
    def gen_input(input_id, batch_size: int = 1):
        if input_id == 0:
            return torch.randn([batch_size, 3, 32, 32])
        return torch.randn([batch_size, 5, 64, 64])

    def to_onnx_bytes(self, batch_size: int = 1):
        onnx_bytes, _ = get_onnx_bytes_and_metadata(
            self,
            (LeNet5Ooo.gen_input(0, batch_size), LeNet5Ooo.gen_input(1, batch_size)),
        )
        onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
        return onnx_bytes_obj.onnx_model["LeNet5Ooo.onnx"]
