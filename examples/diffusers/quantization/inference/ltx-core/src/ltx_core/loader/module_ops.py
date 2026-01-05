# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Alexey Kravtsov
from typing import NamedTuple, Protocol

import torch


class ModuleMatcher(Protocol):
    def match(self, module: torch.nn.Module) -> bool:
        pass


class ModuleMutator(Protocol):
    def mutate(self, module: torch.nn.Module) -> torch.nn.Module:
        pass


class ModuleOps(NamedTuple):
    name: str
    matcher: ModuleMatcher
    mutator: ModuleMutator
