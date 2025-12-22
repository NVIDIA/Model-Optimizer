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

"""A collection of hand-crafted, tricky models for testing mostly tracing and conversion."""

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from modelopt.torch.utils import random

try:
    import transformers as tf
except ImportError:  # unavailable for external unit tests
    tf = None

DependencyDict = dict[str, set[str]]
DepthDict = dict[str, list[int]]

__all__ = ["get_example_models"]


class FakeConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        return self.conv(x)


class BaseExampleModel(nn.Module):
    """A simple base class for example models that require some constants required for testing."""

    @abstractmethod
    def get_num_searchable_symbols(self):
        raise NotImplementedError

    def get_expected_dependencies(self) -> DependencyDict:
        """Specify dependencies for each symbol unless they are trivial (=0 dependencies)."""
        return {}

    def get_skippable_for_depths(self) -> DepthDict:
        """Specify non-trivial (i.e. min_depth != max_depth) for variable depth."""
        return {}

    def get_unsortable_searchable_symbols(self) -> set[str]:
        """Specify symbols that cannot be sorted."""
        return set()

    def get_sample_func(self):
        return random.choice

    @abstractmethod
    def get_args(self):
        raise NotImplementedError

    @abstractmethod
    def get_num_configurable_hparams(self):
        # sometimes, it can be less if searchable symbols are trivial (only one choice)
        return self.get_num_searchable_symbols()

    def assert_order(self, hp_name, hp_root):
        """Check order of the hparam after sorting."""
        importance = hp_root.importance
        if importance is None:
            return True
        order = torch.argsort(importance, descending=True)
        assert torch.equal(order, torch.arange(len(order), device=order.device))

    def get_num_sortable_hparams(self) -> int:
        """Hparams that are sortable and have registered importance method."""
        raise NotImplementedError


class ExampleModel1(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.conv2 = FakeConv(3, 16, 3, padding="same")
        self.conv3 = nn.Conv2d(16, 1, 3, padding="same")
        self.layers = nn.Sequential(
            nn.Conv2d(8, 8, 1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding="same"),
        )

    def forward(self, x):
        a = self.layers(self.conv1(x))
        b = self.conv2(x)
        c = a + b
        d = F.interpolate(c, scale_factor=0.5)
        e = d + F.max_pool2d(b.abs(), (2, 2), (2, 2))
        f = self.conv3(e)

        return f

    def get_num_searchable_symbols(self):
        return 8  # 5x ks + 3x out_channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv1.out_channels": {"layers.0.in_channels"},
            "layers.0.out_channels": {"layers.2.in_channels"},
            "layers.2.out_channels": {"conv2.conv.out_channels", "conv3.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 7

    def get_num_sortable_hparams(self) -> int:
        return 3


class ExampleModel2(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.branch1 = nn.Sequential(
            BasicBlock(8, 8),
            BasicBlock(8, 8),
        )
        self.branch2 = nn.Conv2d(8, 8, 1, padding="same")
        self.branch3 = BasicBlock(8, 8)
        self.last_layer = nn.Conv2d(8, 16, 1, padding="same")

    def forward(self, x, training=True):
        x = self.conv1(x)
        a = self.branch1(x)
        b = self.branch2(x)
        c = self.branch3(x)
        d = a + b + c if training else a + b + c + 1
        e = self.last_layer(d)
        return e

    def get_num_searchable_symbols(self):
        return 14

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv1.out_channels": {
                "branch1.0.conv2.out_channels",
                "branch2.out_channels",
                "branch1.0.conv1.in_channels",
                "branch1.1.conv1.in_channels",
                "branch1.1.bn2.num_features",
                "branch2.in_channels",
                "branch3.conv2.out_channels",
                "last_layer.in_channels",
                "branch3.bn2.num_features",
                "branch1.0.bn2.num_features",
                "branch1.1.conv2.out_channels",
                "branch3.conv1.in_channels",
            },
            "branch1.0.conv1.out_channels": {
                "branch1.0.bn1.num_features",
                "branch1.0.conv2.in_channels",
            },
            "branch1.1.conv1.out_channels": {
                "branch1.1.bn1.num_features",
                "branch1.1.conv2.in_channels",
            },
            "branch3.conv1.out_channels": {"branch3.bn1.num_features", "branch3.conv2.in_channels"},
        }

    def get_skippable_for_depths(self) -> DepthDict:
        return {"branch1.depth": [0, 1]}

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 12

    def get_num_sortable_hparams(self) -> int:
        return 4


class ExampleModel3(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.relu(x)
        return x

    def get_num_searchable_symbols(self):
        return 1

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 0


class ExampleModel4(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, 3, padding="same")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x[:, :8, :-1, :-1])
        return x

    def get_num_searchable_symbols(self):
        return 2

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 0


class ExampleModel5(BaseExampleModel):
    """A tricky model that has dependencies through the network because of residual connections.

    Specifically, conv1 will change from free --> dependent --> first node --> boundary constraint.
    Moreover, due to residual connections everything will be bound at the end.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3, padding="same") for _ in range(4)])

    def forward(self, x):
        out0 = self.convs[0](x)
        out1 = self.convs[1](out0)
        out2 = out0 + out1  # linking convs[0] and convs[1]
        out3 = self.convs[2](out2)
        out1 += out3  # linking convs[1] and convs[2]
        return self.convs[3](out0 + out1 + out2 + out3 + x)  # input linking

    def get_num_searchable_symbols(self):
        return 4

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 0


class ExampleModel6(BaseExampleModel):
    """A model with one searchable out_channels parameter in convs[0].

    This model tests the recursive solution of root dependencies. Specifically, in the add2 node
    we need to recursively resolve conv3's dependency to conv1.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3) for _ in range(5)])

    def forward(self, x):
        conv0 = self.convs[0](x)
        conv1 = self.convs[1](x)
        conv2 = self.convs[2](x)
        conv3 = self.convs[3](x)

        add0 = conv2 + conv3  # link 3 to 2
        add1 = conv1 + conv2  # link 2 to 1
        add2 = conv0 + conv3  # link 1 (from conv3) to 0

        return self.convs[4](add0 + add1 + add2)

    def get_num_searchable_symbols(self):
        return 6

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs.0.out_channels": {
                "convs.1.out_channels",
                "convs.2.out_channels",
                "convs.3.out_channels",
                "convs.4.in_channels",
            },
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel7(BaseExampleModel):
    """ExampleModel6 that also uses torch's out kwarg to store output.

    We can test the ability of our tracer to correctly re-used out nodes in the graph.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3) for _ in range(5)])

    @torch.no_grad()  # required since specifying out in torch.add would break grad.
    def forward(self, x):
        conv0 = self.convs[0](x)
        conv1 = self.convs[1](x)
        conv2 = self.convs[2](x)
        conv3 = self.convs[3](x)

        add0 = conv2 + conv3  # link 3 to 2
        add1 = conv1 + conv2  # link 2 to 1

        # link 1 (from conv3) to 0 and store result in conv1
        torch.add(conv0, conv3, alpha=2.0, out=conv1)

        return self.convs[4](add0 + add1 + conv1)

    def get_num_searchable_symbols(self):
        return 6

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs.0.out_channels": {
                "convs.1.out_channels",
                "convs.2.out_channels",
                "convs.3.out_channels",
                "convs.4.in_channels",
            },
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel8(BaseExampleModel):
    """ExampleModel6 that also uses torch's in-place operators to store result.

    We can test the ability of our tracer to correctly re-used out nodes in the graph.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3) for _ in range(5)])

    @torch.no_grad()  # required since specifying out in torch.add would break grad.
    def forward(self, x):
        conv0 = self.convs[0](x)
        conv1 = self.convs[1](x)
        conv2 = self.convs[2](x)
        conv3 = self.convs[3](x)

        add0 = conv2 + conv3  # link 3 to 2
        add1 = conv1 + conv2  # link 2 to 1

        # link 1 (from conv3) to 0 and store result in conv0
        _ = conv0.add_(conv3, alpha=2.0)

        return self.convs[4](add0 + add1 + conv0)

    def get_num_searchable_symbols(self):
        return 6

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs.0.out_channels": {
                "convs.1.out_channels",
                "convs.2.out_channels",
                "convs.3.out_channels",
                "convs.4.in_channels",
            },
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel9(BaseExampleModel):
    """A model with a boundary that spreads.

    Hereby, we can test whether boundaries are propagated correctly.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3) for _ in range(5)])

    def forward(self, x):
        conv0 = self.convs[0](x)
        conv1 = self.convs[1](x)
        conv2 = self.convs[2](x)
        conv3 = self.convs[3](x)

        add0 = conv2 + conv3  # link 3 to 2
        add1 = conv1 + conv2  # link 2 to 1

        # make 3 a boundary which will also make 2/1 a boundary
        boundary = torch.sum(conv3)

        # try link 1 (from conv3) to 0 but it will fail and we need to propagate boundary to 0
        add2 = conv0 + conv3

        return self.convs[4](add0 + add1 + add2 + boundary)

    def get_num_searchable_symbols(self):
        return 5

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 0

    def get_num_sortable_hparams(self) -> int:
        return 0


class ExampleModel10(BaseExampleModel):
    """A simple cycle that might cause some bugs.

    A few potential bugs with this architecture:
      * The first node in multi-variate constraints is varying --> need to go to root every time.
      * It also links dangling hparams multiple times since convs are used multiple times.
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3, padding="same") for _ in range(3)])

    def forward(self, x):
        out1 = self.convs[0](x) + self.convs[1](x)
        out2 = self.convs[1](x) - self.convs[0](x)
        return self.convs[2](out1 + out2)

    def get_num_searchable_symbols(self):
        return 4

    def get_expected_dependencies(self) -> DependencyDict:
        return {"convs.0.out_channels": {"convs.1.out_channels", "convs.2.in_channels"}}

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel11(BaseExampleModel):
    """A model that reuses convs and thus requires proper linking of **all** input nodes to conv."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, 3, 3, padding="same"), nn.BatchNorm2d(3)) for _ in range(5)]
        )

    def forward(self, x):
        out0 = self.convs[0](x)

        # both convs[1] and convs[2] are now linked to convs[0]'s outgoing hparams
        out1 = self.convs[1](out0)
        out2 = self.convs[2](out0)

        # this gets confusing because convs[2] should be linked to both convs[3] and convs[0]
        # so in reality convs[3] outgoing should be linked to convs[0] as well now.
        out3 = self.convs[3](out0)
        out2_2 = self.convs[2](out3)

        out_sum = out0 + out1 + out2 + out2_2  # now everything linked to convs[0]

        return self.convs[4](out_sum)

    def get_num_searchable_symbols(self):
        return 6  # 5x ks + 1x channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs.0.0.out_channels": {
                "convs.0.1.num_features",
                "convs.1.1.num_features",
                "convs.2.1.num_features",
                "convs.3.1.num_features",
                "convs.1.0.in_channels",
                "convs.2.0.in_channels",
                "convs.3.0.in_channels",
                "convs.4.0.in_channels",
                "convs.1.0.out_channels",
                "convs.2.0.out_channels",
                "convs.3.0.out_channels",
            }
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel12(BaseExampleModel):
    """A model with multiple inputs."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding="same")
        self.conv2 = nn.Conv2d(3, 8, 3, padding="same")
        self.conv3 = nn.Conv2d(8, 8, 3, padding="same")

    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        out2 = self.conv3(self.conv2(x2))
        return out1 + out2

    def get_num_searchable_symbols(self):
        return 4

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv2.out_channels": {"conv3.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8))

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel13(BaseExampleModel):
    """A model with a submodule that fails during tracing and needs to be wrapped into a leaf.

    Since the submodule that fails contains further sub-modules we need to repeat the tracing
    process. Otherwise, we have nodes in the graph that are inside the leaf node, which renders the
    graph invalid. This model tests whether we correctly repeat the tracing process.
    """

    class FailingSubModel(nn.Module):
        """A module that fails after first tracing through a bunch of valid code."""

        def __init__(self, recurse=True) -> None:
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3, padding="same") for _ in range(2)])
            self.more_failing = type(self)(recurse=False) if recurse else nn.Identity()

        def forward(self, x):
            out1 = self.convs[0](x)
            out2 = self.convs[1](x)

            # link both convs together making conv1 have dynamic hparams
            # if we don't retrace the entire graph we will get an error during processing the
            # dependencies since we will first link convs[1] to convs[0] and then later try
            # de-activating convs[1] which will fail since it is linked to convs[0].
            out3 = out1 + out2

            # potentially recurse into another failing module
            out4 = self.more_failing(out3)

            # now make the model fail as a whole
            out5 = 2 * out3 if out4.shape[0] > 10 else out3
            return out5

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(3, 3, 3, padding="same") for _ in range(4)])
        self.failing = self.FailingSubModel()

    def forward(self, x):
        out1 = self.convs[1](self.convs[0](x) + self.failing(x)) + self.convs[2](x)
        return self.convs[3](out1)

    def get_num_searchable_symbols(self):
        return 9

    def get_expected_dependencies(self) -> DependencyDict:
        return {"convs.1.out_channels": {"convs.3.in_channels", "convs.2.out_channels"}}

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel14(BaseExampleModel):
    """A model with different choices for some convs.

    conv1 can have [8, 16, 24, 32] but conv2 only [16, 32]. Since we choose min sample function,
    we will choose 8 from conv1 and try to match it to conv2. Hence we need to "sync" the choices
    during the convert process.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(3, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 16, 3, padding="same")

    def forward(self, x1):
        out1 = self.conv1(x1)
        out2 = self.conv2(x1)
        out3 = out1 + out2
        return self.conv3(out3)

    def get_num_searchable_symbols(self):
        return 4

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv1.out_channels": {"conv3.in_channels", "conv2.out_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel15(BaseExampleModel):
    """A model that with special arguments for conversion."""

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(3, 3, 3) for _ in range(2)])

    def forward(self, x, is_identity=False):
        return x if is_identity else self.convs(x)

    def get_num_searchable_symbols(self):
        return 3  # 2x ks + 1x out_channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs.0.out_channels": {"convs.1.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8), False)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel16(BaseExampleModel):
    """A model that will fail tracing but submodules can still be traced."""

    def __init__(self):
        super().__init__()
        self.convs1 = nn.Sequential(*[nn.Conv2d(3, 3, 3) for _ in range(2)])
        self.convs2 = nn.Sequential(*[nn.Conv2d(3, 3, 3) for _ in range(2)])

    def forward(self, x):
        out = self.convs1(x)
        out = self.convs2(x) if out.shape[0] > 10 else self.convs2(x)  # noqa: RUF034
        return out

    def get_num_searchable_symbols(self):
        return 6  # 4x ks + 2x out_channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs1.0.out_channels": {"convs1.1.in_channels"},
            "convs2.0.out_channels": {"convs2.1.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 2

    def get_num_sortable_hparams(self) -> int:
        return 2


class ExampleModel17(BaseExampleModel):
    """A model with unvisited sub-modules that are frozen."""

    def __init__(self):
        super().__init__()
        self.skip = True
        self.convs1 = nn.Sequential(*[nn.Conv2d(3, 3, 3) for _ in range(2)])
        self.convs2 = nn.Sequential(*[nn.Conv2d(3, 3, 3) for _ in range(2)])

    def forward(self, x):
        out = self.convs1(x)
        if not self.skip:
            out = self.convs2(x)
        return out

    def get_num_searchable_symbols(self):
        return 5  # 4x ks + 1x out_channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "convs1.0.out_channels": {"convs1.1.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 1


class ExampleModel18(BaseExampleModel):
    """A model with ConvTranspose2d layer."""

    def __init__(self) -> None:
        super().__init__()
        oc = 16
        self.conv_transpose = nn.Sequential(
            *[nn.ConvTranspose2d(3 if i == 0 else oc, oc, 3, padding=1) for i in range(3)]
        )

    def forward(self, x):
        out = self.conv_transpose(x)
        return out

    def get_num_searchable_symbols(self):
        return 5  # 3x ks + 2x out_channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv_transpose.0.out_channels": {"conv_transpose.1.in_channels"},
            "conv_transpose.1.out_channels": {"conv_transpose.2.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 2


class ExampleModel19(BaseExampleModel):
    """A model with InstanceNorm layer."""

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding="same"),
            nn.InstanceNorm2d(8, affine=True),
            nn.Conv2d(8, 8, 3, padding="same"),
            nn.InstanceNorm2d(8, affine=True),
            nn.Conv2d(8, 8, 3, padding="same"),
            nn.InstanceNorm2d(8, affine=True),
            nn.Conv2d(8, 1, 3, padding="same"),
            nn.InstanceNorm2d(1, affine=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out

    def get_num_searchable_symbols(self):
        return 7  # 4x ks + 3x out_channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "net.0.out_channels": {"net.2.in_channels", "net.1.num_features"},
            "net.2.out_channels": {"net.4.in_channels", "net.3.num_features"},
            "net.4.out_channels": {"net.6.in_channels", "net.5.num_features"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 3


class ExampleModel20(BaseExampleModel):
    """This model fails as a whole and has multiple inputs/outputs.

    This checks that our tracing of leaf modules works correctly since after it fails it will be
    traced as a leaf module.
    """

    class Submodel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.conv1 = nn.Conv2d(2, 8, 3, padding="same")
            self.conv2 = nn.Conv2d(2, 16, 3, padding="same")
            self.conv3 = nn.Conv2d(16, 8, 3, padding="same")

        def forward(self, x, y):
            x = self.conv1(x)
            y = self.conv2(y)

            out1 = F.relu(x) if x.max() < 0 else x
            out2 = x + self.conv3(y)

            return out1, out2

    def __init__(self):
        super().__init__()

        self.m = self.Submodel()
        self.conv4 = nn.Conv2d(8, 1, 3, padding="same")

    def forward(self, x, y):
        out1, out2 = self.m(x, y)
        out2 = F.softplus(out2) if out2.max() > 0 else F.tanh(out2)
        out2 = self.conv4(out2)
        return {"out1": out1, "out2": out2}

    def get_num_searchable_symbols(self):
        return 4  # 4x ks

    def get_args(self):
        return (torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8))

    def get_num_sortable_hparams(self) -> int:
        return 0


class ExampleModel21(ExampleModel19):
    """Just like 19 but using a GroupNorm to block sorting."""

    def __init__(self) -> None:
        super().__init__()
        self.net[3] = nn.GroupNorm(2, 8)

    def get_expected_dependencies(self) -> DependencyDict:
        # in group norm it's num_channels instead of num_features
        deps = super().get_expected_dependencies()
        deps2 = deps["net.2.out_channels"]
        deps2.remove("net.3.num_features")
        deps2.add("net.3.num_channels")
        return deps

    def get_unsortable_searchable_symbols(self) -> set[str]:
        return {"net.2.out_channels"}  # should come from unsortable group norm in net.3

    def get_num_sortable_hparams(self) -> int:
        return 2


class LinearModel1(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Linear(3, 8),
                nn.Linear(8, 16),
                nn.Linear(16, 16),
                nn.Linear(16, 16),
                nn.Linear(16, 1),
            ]
        )

    def forward(self, x):
        for block in self.model:
            x = block(x)
        return x

    def get_num_searchable_symbols(self):
        return 4

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "model.0.out_features": {"model.1.in_features"},
            "model.1.out_features": {"model.2.in_features"},
            "model.2.out_features": {"model.3.in_features"},
            "model.3.out_features": {"model.4.in_features"},
        }

    def get_args(self):
        return (torch.randn(1, 3),)

    def get_num_sortable_hparams(self) -> int:
        return 4


class LinearModel2(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.Linear(8, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)

    def get_num_searchable_symbols(self):
        return 4

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "model.0.out_features": {"model.1.in_features"},
            "model.1.out_features": {"model.2.in_features"},
            "model.2.out_features": {"model.3.in_features"},
            "model.3.out_features": {"model.4.in_features"},
        }

    def get_args(self):
        return (torch.randn(1, 3),)

    def get_num_sortable_hparams(self) -> int:
        return 4


class LinearModel3(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 8),  # 0
            nn.LayerNorm(8),  # 1
            nn.Linear(8, 16),  # 2
            nn.ReLU(),  # 3
            nn.LayerNorm(16),  # 4
            nn.Linear(16, 16),  # 5
            nn.LayerNorm(16),  # 6
            nn.Linear(16, 16),  # 7
            nn.LayerNorm(16),  # 8
            nn.Linear(16, 1),  # 9
        )

    def forward(self, x):
        return self.model(x)

    def get_num_searchable_symbols(self):
        return 4  # 4x features

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "model.0.out_features": {"model.1.num_features", "model.2.in_features"},
            "model.2.out_features": {"model.4.num_features", "model.5.in_features"},
            "model.5.out_features": {"model.6.num_features", "model.7.in_features"},
            "model.7.out_features": {"model.8.num_features", "model.9.in_features"},
        }

    def get_args(self):
        return (torch.randn(4, 3),)

    def get_num_configurable_hparams(self):
        return 4

    def get_num_sortable_hparams(self) -> int:
        return 4


class LinearModel4(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Conv2d(3, 8, 3, padding="same"),  # 0
                nn.Conv2d(8, 8, 3, padding="same"),  # 1
                nn.BatchNorm2d(8),  # 2
                nn.Conv2d(8, 8, 3, padding="same"),  # 3
                nn.Flatten(),  # 4
                nn.Linear(8 * 32 * 32, 8),  # 5
                nn.LayerNorm(8),  # 6
                nn.Linear(8, 2),  # 7
            ]
        )

    def forward(self, x):
        for block in self.model:
            x = block(x)
        return x

    def get_num_searchable_symbols(self):
        return 6  # 3x ks + 3x features

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "model.0.out_channels": {"model.1.in_channels"},
            "model.1.out_channels": {"model.2.num_features", "model.3.in_channels"},
            "model.5.out_features": {"model.6.num_features", "model.7.in_features"},
        }

    def get_args(self):
        return (torch.randn(4, 3, 32, 32),)

    def get_num_sortable_hparams(self) -> int:
        return 3


class LinearModel5(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(4, 16),
            nn.LayerNorm(16),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.lin(x)
        return x

    def get_num_searchable_symbols(self):
        return 2

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "lin.0.out_features": {"lin.1.num_features", "lin.2.in_features"},
            "lin.4.out_features": {"lin.5.num_features", "lin.6.in_features"},
        }

    def get_args(self):
        return (torch.randn(2, 4),)

    def get_num_sortable_hparams(self) -> int:
        return 2


class LinearModel6(BaseExampleModel):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding="same"),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, 3, padding="same"),
        )

        self.lin1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(8),
            nn.LayerNorm(8),
        )
        self.lin2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.transpose(x, 1, 2)
        x = self.lin1(x)
        x = torch.transpose(x, 1, 2)
        x = self.lin2(x)
        return x

    def get_num_searchable_symbols(self):
        return 3  # 2x ks + 1x features

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "cnn.0.out_channels": {"cnn.1.num_features", "cnn.2.in_channels"},
        }

    def get_args(self):
        return (torch.randn(4, 3, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class LinearModel7(BaseExampleModel):
    """Residuals in a sequential that make symbols dependent and enable variable depth."""

    class Residual(nn.Module):
        def __init__(self, lin_cls, in_features, out_features):
            super().__init__()
            self.lin = lin_cls(in_features, out_features)
            self.bn = nn.LayerNorm(out_features)

        def forward(self, x):
            return x + F.relu(self.bn(self.lin(x)))

    # so we more easily inherit...
    Linear = nn.Linear
    Sequential = nn.Sequential

    def __init__(self) -> None:
        super().__init__()
        self.initial = self.Linear(4, 4)
        self.model = self.Sequential(*[self.Residual(self.Linear, 4, 4) for _ in range(5)])
        self.final = self.Linear(4, 1)

    def forward(self, x):
        return self.final(self.model(self.initial(x)))

    def get_num_searchable_symbols(self):
        return 2  # 1x features + 1x depth

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "initial.out_features": {
                "model.0.lin.in_features",
                "model.0.bn.num_features",
                "model.0.lin.out_features",
                "model.1.lin.in_features",
                "model.1.bn.num_features",
                "model.1.lin.out_features",
                "model.2.lin.in_features",
                "model.2.bn.num_features",
                "model.2.lin.out_features",
                "model.3.lin.in_features",
                "model.3.bn.num_features",
                "model.3.lin.out_features",
                "model.4.lin.in_features",
                "model.4.bn.num_features",
                "model.4.lin.out_features",
                "final.in_features",
            },
        }

    def get_skippable_for_depths(self) -> dict[str, int]:
        return {"model.depth": [0, 1, 2, 3, 4]}

    def get_args(self):
        return (torch.randn(1, 4),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class LinearModel8(LinearModel7):
    """Same as LinearModel7, but some blocks in the sequential are non-residual."""

    def __init__(self) -> None:
        super().__init__()
        self.model[1] = nn.Linear(4, 4)
        self.model[3] = nn.Linear(4, 4)

    def get_num_searchable_symbols(self):
        return 4  # 3x features + 1x depth

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "initial.out_features": {
                "model.0.lin.in_features",
                "model.0.bn.num_features",
                "model.0.lin.out_features",
                "model.1.in_features",
            },
            "model.1.out_features": {
                "model.2.lin.in_features",
                "model.2.bn.num_features",
                "model.2.lin.out_features",
                "model.3.in_features",
            },
            "model.3.out_features": {
                "model.4.lin.in_features",
                "model.4.bn.num_features",
                "model.4.lin.out_features",
                "final.in_features",
            },
        }

    def get_skippable_for_depths(self) -> dict[str, int]:
        return {"model.depth": [0, 2, 4]}  # some blocks are non-residual

    def get_num_sortable_hparams(self) -> int:
        return 3


class LinearModel9(LinearModel8):
    """Same as LinearModel8, but skipping self.model[2] will cause an error (keep it around!)."""

    class Residual2(LinearModel8.Residual):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._executed = False

        def forward(self, x):
            self._executed = True
            return super().forward(x)

    def __init__(self) -> None:
        super().__init__()
        self.model[2] = self.Residual2(self.Linear, 4, 4)
        self._hidden_storage = [self.model[2]]

    def forward(self, x):
        self._hidden_storage[0]._executed = False
        ret = super().forward(x)
        assert self._hidden_storage[0]._executed  # this will fail for the min_depth model
        return ret

    def get_skippable_for_depths(self) -> dict[str, int]:
        return {"model.depth": [0, 4]}  # can't skip 1,2,3

    def get_num_sortable_hparams(self) -> int:
        return 3


class LinearModel10(LinearModel7):
    """A model that has modules inheriting from another module class w/o modifying forward.

    In this instance, we should treat the inherited modules just like regular modules and thus it
    should behave just like LinearModel7.
    """

    class InheritedLinear(nn.Linear):
        pass

    class InheritedSequential(nn.Sequential):
        pass

    Linear = InheritedLinear
    Sequential = InheritedSequential

    def __init__(self) -> None:
        super().__init__()

        # just a sanity check
        assert type(self.initial) is self.InheritedLinear
        assert type(self.model) is self.InheritedSequential
        assert type(self.model[0].lin) is self.InheritedLinear

    def get_num_sortable_hparams(self) -> int:
        return 1


class LinearModel11(LinearModel7):
    """A model that has modules inheriting from another module class w modifying forward.

    In this instance, we should NOT treat the inherited modules like regular modules and thus it
    should not detect any searchable symbols...
    """

    class InheritedLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x)

    class InheritedSequential(nn.Sequential):
        def forward(self, x):
            return super().forward(x)

    Linear = InheritedLinear
    Sequential = InheritedSequential

    def __init__(self) -> None:
        super().__init__()

        # just a sanity check
        assert type(self.initial) is self.InheritedLinear
        assert type(self.model) is self.InheritedSequential

    def get_num_searchable_symbols(self):
        return 0

    def get_expected_dependencies(self) -> DependencyDict:
        return {}

    def get_skippable_for_depths(self) -> dict[str, int]:
        return {}

    def get_num_sortable_hparams(self) -> int:
        return 0


class PassthroughNoBoundaryModel(BaseExampleModel):
    """A model where a passthrough node is treated as it should be.

    This can happen when the elastic dimensions of two layers are compatible with each other!
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding="same")
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 2, 3, padding="same")

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x

    def get_num_searchable_symbols(self):
        return 3

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv1.out_channels": {"conv2.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 3, 3),)

    def get_num_sortable_hparams(self) -> int:
        return 1


class PassthroughAsBoundaryModel(BaseExampleModel):
    """A model where a passthrough node is treated as boundary node.

    This can happen when the elastic dimensions of two layers are incompatible with each other!
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.pool = nn.AvgPool2d(2)
        self.linear2 = nn.Linear(2, 4)

    def forward(self, x):
        x = self.linear(x)
        x = self.pool(x)
        x = self.linear2(x)
        return x

    def get_num_searchable_symbols(self):
        return 0

    def get_args(self):
        return (torch.randn(1, 3, 3, 3),)

    def get_num_sortable_hparams(self) -> int:
        return 0


class GetattrShapeModel(BaseExampleModel):
    """A model that has tensor shape related getattr and getitem nodes in forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.conv2 = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        o1 = self.conv1(x)  # stays searchable

        shape1 = o1.shape
        shape2 = o1.size()
        shape3 = getattr(o1, "shape")

        # Conv2d searchable_tensor_dim=1 not accessed anywhere
        getitem1 = shape1[:1]  # noqa: F841
        getitem2 = shape2[2:]  # noqa: F841
        getitem3 = shape3[2:3]  # noqa: F841
        getitem4 = shape1[-1]  # noqa: F841

        return self.conv2(o1)

    def get_num_searchable_symbols(self):
        return 3

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv1.out_channels": {"conv2.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 1


class GroupConvModel1(BaseExampleModel):
    """A model with group convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 3, groups=4)
        self.conv3 = nn.Conv2d(8, 4, 3)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        c = self.conv3(b)
        return c

    def get_num_searchable_symbols(self):
        return 4

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv1.out_channels": {"conv2.in_channels", "conv2.out_channels", "conv3.in_channels"},
        }

    def get_unsortable_searchable_symbols(self) -> set[str]:
        return {"conv1.out_channels"}

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self) -> int:
        return 0


class GroupConvModel2(BaseExampleModel):
    """A model with group convolution that cannot be handled as passthrough."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3, groups=4)
        self.conv3 = nn.Conv2d(16, 4, 3)
        self._b_shape = None

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        # get image shape of b... (caused a bug in older versions during handling because we did not
        # add the out_channels/in_channels symbols in this case...)
        b_shape = b.shape[2:]
        self._b_shape = b_shape
        c = self.conv3(b)
        return c

    def get_num_searchable_symbols(self):
        return 3  # 3x ks

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 0

    def get_num_sortable_hparams(self) -> int:
        return 0


class GroupConvModel3(GroupConvModel1):
    """Same as GroupConvModel1 but conv2 is sortable since it's depthwise."""

    def __init__(self) -> None:
        super().__init__()
        self.conv2 = nn.Conv2d(8, 8, 3, groups=8)

    def get_unsortable_searchable_symbols(self) -> set[str]:
        return set()

    def get_num_sortable_hparams(self) -> int:
        return 1


class BaseConcatModel(BaseExampleModel):
    hps_cat = []

    def assert_order(self, hp_name, hp_root) -> bool:
        if hp_name not in self.hps_cat:
            return super().assert_order(hp_name, hp_root)

        # special check for concat hparam
        ord_split = hp_root._split_order(torch.argsort(hp_root.importance, descending=True))
        assert all(
            torch.equal(ord_, torch.arange(len(ord_), device=ord_.device)) for ord_ in ord_split
        )


class ConcatModel1(BaseConcatModel):
    """A model with concat operations along dim=0. We support along channel dim only i.e. dim=1."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(3, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x)], dim=0)  # not supported
        out = self.conv3(out)
        return out

    def get_num_searchable_symbols(self):
        return 3

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 0

    def get_num_sortable_hparams(self):
        return 0


class ConcatModel2(BaseConcatModel):
    """A model with concat operations on repeated tensor."""

    hps_cat = ["conv2.in_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(12, 8, 3)

    def forward(self, x):
        o1 = self.conv1(x)
        out = torch.cat(tensors=[o1, o1, o1], dim=1)
        out = self.conv2(out)
        return out

    def get_num_searchable_symbols(self):
        return 3

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv2.in_channels": {"conv1.out_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self):
        return 1


class ConcatModel3(BaseConcatModel):
    """A model with concat added to conv."""

    hps_cat = ["conv3.out_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.conv3 = nn.Conv2d(3, 8, 3)
        self.conv4 = nn.Conv2d(8, 8, 3)

    def forward(self, x):
        o1 = F.relu(self.bn1(self.conv1(x)))
        o2 = self.conv2(x)
        out = self.conv3(x) + torch.cat([o1, o2], 1)  # cat searchable because of higher priority
        out = self.conv4(out)
        return out

    def get_num_searchable_symbols(self):
        return 5

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv4.in_channels": {
                "conv1.out_channels",
                "conv2.out_channels",
                "conv3.out_channels",
                "bn1.num_features",
            },
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self):
        return 1


class ConcatModel4(BaseConcatModel):
    """A model with concat added to another concat."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.conv3 = nn.Conv2d(3, 4, 3)
        self.conv4 = nn.Conv2d(3, 4, 3)
        self.conv5 = nn.Conv2d(8, 8, 3)

    def forward(self, x):
        o1 = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        o2 = torch.cat([self.conv3(x), self.conv4(x)], dim=1)
        out = self.conv5(o1 + F.relu(o2))  # disable both inputs as both are concats
        return out

    def get_num_searchable_symbols(self):
        return 5

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 0

    def get_num_sortable_hparams(self):
        return 0


class ConcatModel5(BaseConcatModel):
    """A model with concat on another concat."""

    hps_cat = ["conv5.in_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.conv3 = nn.Conv2d(3, 4, 3)
        self.conv4 = nn.Conv2d(3, 4, 3)
        self.conv5 = nn.Conv2d(16, 16, 3)

    def forward(self, x):
        cat1 = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)
        cat2 = torch.cat([self.conv4(x), cat1], dim=1)  # cat1 disabled, conv4 still searchable
        out = self.conv5(cat2)
        return out

    def get_num_searchable_symbols(self):
        return 6  # 5 kernel sizes + conv4.out_channels packaged into conv5.in_channels as concat

    def get_expected_dependencies(self) -> DependencyDict:
        return {"conv5.in_channels": {"conv4.out_channels"}}

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self):
        return 1


class ConcatModel6(BaseConcatModel):
    """A model with a conv input to 2 different concats."""

    hps_cat = ["conv4.in_channels", "conv5.in_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.conv3 = nn.Conv2d(3, 4, 3)
        self.conv4 = nn.Conv2d(8, 8, 3)
        self.conv5 = nn.Conv2d(8, 8, 3)

    def forward(self, x):
        o1 = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        o2 = torch.cat([self.conv3(x), F.relu(self.conv1(x))], dim=1)
        out = self.conv4(o1) + self.conv5(o2)
        return out

    def get_num_searchable_symbols(self):
        return 7  # 5x kernel size + conv2, conv3 out_channels packaged into cats

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv4.in_channels": {"conv2.out_channels"},
            "conv5.in_channels": {"conv3.out_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 2

    def get_num_sortable_hparams(self):
        return 2


class ConcatModel7(BaseConcatModel):
    """A model with all concat inputs already dynamic."""

    hps_cat = ["conv5.in_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.conv3 = nn.Conv2d(3, 4, 3)
        self.conv4 = nn.Conv2d(4, 4, 3)
        self.conv5 = nn.Conv2d(8, 4, 3)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o3 = self.conv3(x)
        out1 = o1 + o2 + o3  # only conv1 stays searchable
        out2 = torch.cat([o2, o3], dim=1)
        out = self.conv4(out1) + self.conv5(out2)
        return out

    def get_num_searchable_symbols(self):
        return 6

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv5.in_channels": {
                "conv1.out_channels",
                "conv2.out_channels",
                "conv3.out_channels",
                "conv4.in_channels",
            },
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self):
        return 1


class ConcatModel8(BaseConcatModel):
    """A model with two identical concat operations."""

    hps_cat = ["conv3.in_channels", "conv4.in_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(3, 4, 3)
        self.conv3 = nn.Conv2d(8, 4, 3)
        self.conv4 = nn.Conv2d(8, 4, 3)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2(x)
        cat1 = torch.cat([o1, o2], dim=1)
        cat2 = torch.cat([o1, o2], dim=1)  # Reuse node target from previous concat
        out = self.conv3(cat1) + self.conv4(cat2)
        return out

    def get_num_searchable_symbols(self):
        return 5

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv3.in_channels": {"conv1.out_channels", "conv2.out_channels", "conv4.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 1

    def get_num_sortable_hparams(self):
        return 1


class ConcatModel9(BaseConcatModel):
    """A model with a linear layer inputs to 2 different concats."""

    hps_cat = ["lin4.in_features", "lin5.in_features"]

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2, 4)
        self.lin2 = nn.Linear(2, 4)
        self.lin3 = nn.Linear(2, 4)
        self.lin4 = nn.Linear(8, 8)
        self.lin5 = nn.Linear(8, 8)

    def forward(self, x):
        o1 = torch.cat([self.lin1(x), self.lin2(x)], dim=-1)
        o2 = torch.cat([self.lin3(x), F.relu(self.lin1(x))], dim=-1)
        out = self.lin4(o1) + self.lin5(o2)
        return out

    def get_num_searchable_symbols(self):
        return 2  # lin2, lin3 out_features packaged into cats

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "lin4.in_features": {"lin2.out_features"},
            "lin5.in_features": {"lin3.out_features"},
        }

    def get_args(self):
        return (torch.randn(2, 2),)

    def get_num_sortable_hparams(self):
        return 2


class ConcatModel10(ConcatModel9):
    """Same as 9 but with different args/kwargs combinations."""

    def forward(self, x):
        o1 = torch.cat(dim=-1, tensors=[self.lin1(x), self.lin2(x)])
        o2 = torch.cat([self.lin3(x), F.relu(self.lin1(x))], -1)
        out = self.lin4(o1) + self.lin5(o2)
        return out


class ConcatModel11(BaseConcatModel):
    """A concat with inputs that aren't all searchable.

    The concat is still searchable though if at least one input is. However, in this case we cannot
    assign symbols to every input and thus we need to disable it...
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2, 2)
        self.conv1 = nn.Conv2d(3, 3, 3, padding="same")
        self.conv2 = nn.Conv2d(3, 3, 3, padding="same")
        self.conv3 = nn.Conv2d(3, 3, 3, padding="same")
        self.conv4 = nn.Conv2d(12, 1, 3, padding="same")

    def forward(self, x):
        const = torch.ones_like(x)
        a = self.conv1(x)
        b = self.conv2(x) + x
        c = a + self.conv3(a)

        # This concat is searchable because at least one input is searchable. Specifically,
        # a is searchable.
        # b has an outgoing symbol but is constant because of the residual connection with x.
        # c has an outgoing symbol but it is dynamic because of the residual connection with a.
        # const does not have an outgoing symbol and will be simulated with a fake constant symbol.
        e = torch.cat([a, b, c, const], dim=1)
        f = self.conv4(e)
        return f

    def get_num_searchable_symbols(self):
        return 4  # 4x kernel_size

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self):
        return 0


class ConcatModel12(ConcatModel1):
    """ConcatModel1 without specifying cat-dim (0 is already the default)."""

    def forward(self, x):
        # tracer needs to correctly retrieve cat-dim from default value
        out = torch.cat([self.conv1(x), self.conv2(x)])
        out = self.conv3(out)
        return out

    def get_num_searchable_symbols(self):
        return 3


class MeanConcatModel(BaseConcatModel):
    """Taking a mean over a non-elastic dimension."""

    hps_cat = ["conv3.in_channels"]

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding="same")
        self.conv2 = nn.Conv2d(3, 4, 3, padding="same")
        self.conv3 = nn.Conv1d(8, 8, 3, padding="same")
        self.conv4 = nn.Conv1d(8, 1, 3, padding="same")

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        cat1 = torch.cat([c1.mean(dim=(2,)), torch.mean(c2, 3)], dim=1)
        return self.conv4(self.conv3(cat1))

    def get_num_searchable_symbols(self):
        return 4 + 2  # 4x kernel size + conv3 in and out channels

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            "conv3.in_channels": {"conv1.out_channels", "conv2.out_channels"},
            "conv3.out_channels": {"conv4.in_channels"},
        }

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_sortable_hparams(self):
        return 2


class MeanModel2(BaseExampleModel):
    """Taking a mean over elastic dimensions --> will be blocked."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding="same")
        self.conv2 = nn.Conv2d(3, 4, 3, padding="same")
        self.conv3 = nn.Conv2d(1, 2, 1, padding="same")

    def forward(self, x):
        # b x 1 x 1 x 1 after mean ...
        c1 = self.conv1(x).mean(dim=(1, 2, 3), keepdim=True)
        c2 = torch.mean(self.conv2(x), dim=(-1, -2, -3), keepdim=True)

        return self.conv3(c1 + c2)

    def get_num_searchable_symbols(self):
        return 3  # 3x kernel size

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 2

    def get_num_sortable_hparams(self) -> int:
        return 0


_cache: dict[nn.Module, Any] = {}


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, cached: bool = False) -> None:
        super().__init__()
        self.cached = cached
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x) + self.main(torch.cos(x))
        x = torch.relu(x)
        if self.cached:
            _cache[self] = x
        return x


class FailingModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 if x.shape[0] > 1 else x


class DepthModel1(BaseExampleModel):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.Conv2d(8, 4, 1),
            ResidualBlock(4),
            ResidualBlock(4),
            ResidualBlock(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def get_num_searchable_symbols(self):
        return 5 + 1 + 1  # kernel size + conv0 out + depth (no channels since blocked by output)

    def get_skippable_for_depths(self) -> DepthDict:
        return {"main.depth": [2, 3, 4]}

    def get_expected_dependencies(self) -> DependencyDict:
        return {"main.0.out_channels": {"main.1.in_channels"}}

    def get_args(self):
        return (torch.randn(1, 3, 8, 8),)

    def get_num_configurable_hparams(self):
        return 5

    def get_num_sortable_hparams(self) -> int:
        return 1


class DepthModel2(BaseExampleModel):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ResidualBlock(4),
            ResidualBlock(4),
            ResidualBlock(4),
            ResidualBlock(4),
        )
        self.failing = FailingModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.main(x)
            + self.main[0](x)
            + self.main[1](x)
            + self.failing(x)
            + self.main[0].main[0](x)
            + self.main[2].main[0](x)
        )

    def get_num_searchable_symbols(self):
        return 4 + 1  # ks + depth

    def get_skippable_for_depths(self) -> DepthDict:
        return {"main.depth": [1, 3]}  # can't skip 0, 2 because we access conv inside

    def get_args(self):
        return (torch.rand(2, 4, 16, 16),)

    def get_num_sortable_hparams(self) -> int:
        return 0


class DepthModel3(BaseExampleModel):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ResidualBlock(4, cached=True),
            ResidualBlock(4, cached=True),
            ResidualBlock(4, cached=True),
            ResidualBlock(4, cached=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _cache.clear()
        return self.main(x) + _cache[self.main[0]] + _cache[self.main[1]] + _cache[self.main[3]]

    def get_num_searchable_symbols(self):
        return 4 + 1  # ks + depth

    def get_skippable_for_depths(self) -> DepthDict:
        return {"main.depth": [2]}  # only 2 since that's the only cache we don't access

    def get_args(self):
        return (torch.randn(4, 4, 8, 8),)

    def get_num_configurable_hparams(self):
        return 4

    def get_num_sortable_hparams(self) -> int:
        return 0


class DepthModel4(BaseExampleModel):
    def __init__(self) -> None:
        super().__init__()
        block = ResidualBlock(4)
        self.main = nn.Sequential(*[block for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def get_num_searchable_symbols(self):
        return 1 + 1  # ks + depth --> only 1 ks since Residual is shared!!

    def get_skippable_for_depths(self) -> DepthDict:
        return {"main.depth": [0, 1, 2, 3]}

    def get_args(self):
        return (torch.randn(4, 4, 8, 8),)

    def get_num_sortable_hparams(self) -> int:
        return 0


class BaseTransformerModel(BaseExampleModel):
    def __init__(self, model_type, config):
        super().__init__()
        self.config = config
        self.main = model_type(self.config).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def get_args(self):
        return (self.main.dummy_inputs["input_ids"],)

    def get_num_searchable_symbols(self):
        return len(self.get_expected_dependencies())


class BertQAModel(BaseTransformerModel):
    def __init__(self) -> None:
        config = tf.BertConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=16,
            max_position_embeddings=32,
        )
        super().__init__(tf.BertForQuestionAnswering, config)

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            **{
                f"main.bert.encoder.layer.{i}.attention.num_attention_heads": {
                    f"main.bert.encoder.layer.{i}.attention.hidden_dim"
                }
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"main.bert.encoder.layer.{i}.intermediate.dense.out_features": {
                    f"main.bert.encoder.layer.{i}.output.dense.in_features"
                }
                for i in range(self.config.num_hidden_layers)
            },
        }

    def get_num_configurable_hparams(self):
        return 2 * self.config.num_hidden_layers  # mlp intermediate size and attn heads per layer

    def get_num_sortable_hparams(self):
        return 2 * self.config.num_hidden_layers  # mlp intermediate size and attn heads per layer


class GPTJModel(BaseTransformerModel):
    def __init__(self) -> None:
        config = tf.GPTJConfig(
            vocab_size=64,
            n_positions=8,
            n_embd=16,
            n_layer=2,
            n_head=4,
            rotary_dim=4,
            bos_token_id=0,
            eos_token_id=1,
            use_cache=False,  # Else sorting will not be correct
        )
        super().__init__(tf.GPTJForCausalLM, config)

    def get_expected_dependencies(self) -> DependencyDict:
        return {
            **{
                f"main.transformer.h.{i}.attn.num_attention_heads": {
                    f"main.transformer.h.{i}.attn.hidden_dim"
                }
                for i in range(self.config.n_layer)
            },
            **{
                f"main.transformer.h.{i}.mlp.fc_in.out_features": {
                    f"main.transformer.h.{i}.mlp.fc_out.in_features"
                }
                for i in range(self.config.n_layer)
            },
        }

    def get_num_configurable_hparams(self):
        return 2 * self.config.num_hidden_layers  # mlp intermediate size and attn heads per layer

    def get_num_sortable_hparams(self):
        return 2 * self.config.num_hidden_layers  # mlp intermediate size and attn heads per layer


def get_example_models():
    return {
        type(model).__name__: model
        for model in [
            ExampleModel1(),
            ExampleModel2(),
            ExampleModel3(),
            ExampleModel4(),
            ExampleModel5(),
            ExampleModel6(),
            ExampleModel7(),
            ExampleModel8(),
            ExampleModel9(),
            ExampleModel10(),
            ExampleModel11(),
            ExampleModel12(),
            ExampleModel13(),
            ExampleModel14(),
            ExampleModel15(),
            ExampleModel16(),
            ExampleModel17(),
            ExampleModel18(),
            ExampleModel19(),
            ExampleModel20(),
            ExampleModel21(),
            LinearModel1(),
            LinearModel2(),
            LinearModel3(),
            LinearModel4(),
            LinearModel5(),
            LinearModel6(),
            LinearModel7(),
            LinearModel8(),
            LinearModel9(),
            LinearModel10(),
            LinearModel11(),
            PassthroughNoBoundaryModel(),
            PassthroughAsBoundaryModel(),
            GetattrShapeModel(),
            GroupConvModel1(),
            GroupConvModel2(),
            GroupConvModel3(),
            ConcatModel1(),
            ConcatModel2(),
            ConcatModel3(),
            ConcatModel4(),
            ConcatModel5(),
            ConcatModel6(),
            ConcatModel7(),
            ConcatModel8(),
            ConcatModel9(),
            ConcatModel10(),
            ConcatModel11(),
            ConcatModel12(),
            MeanConcatModel(),
            MeanModel2(),
            DepthModel1(),
            DepthModel2(),
            DepthModel3(),
            DepthModel4(),
        ]
        + (
            [
                BertQAModel(),
                GPTJModel(),
            ]
            if tf is not None
            else []
        )
    }
