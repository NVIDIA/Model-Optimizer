# Copyright (c) 2025 Lightricks. All rights reserved.
# Created by Alexey Kravtsov
import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_keys_ops import SDKeyOps


class Registry(Protocol):
    def add(self, paths: list[str], sd_key_ops: SDKeyOps | None, state_dict: StateDict) -> None: ...

    def pop(self, paths: list[str], sd_key_ops: SDKeyOps | None) -> StateDict | None: ...

    def get(self, paths: list[str], sd_key_ops: SDKeyOps | None) -> StateDict | None: ...


class DummyRegistry(Registry):
    def add(self, paths: list[str], sd_key_ops: SDKeyOps | None, state_dict: StateDict) -> None:
        pass

    def pop(self, paths: list[str], sd_key_ops: SDKeyOps | None) -> StateDict | None:
        pass

    def get(self, paths: list[str], sd_key_ops: SDKeyOps | None) -> StateDict | None:
        pass


@dataclass
class StateDictRegistry(Registry):
    _state_dicts: dict[str, StateDict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _generate_id(self, paths: list[str], sd_key_ops: SDKeyOps) -> str:
        m = hashlib.sha256()
        parts = [str(Path(p).resolve()) for p in paths]
        if sd_key_ops is not None:
            parts.append(sd_key_ops.name)
        m.update("\0".join(parts).encode("utf-8"))
        return m.hexdigest()

    def add(self, paths: list[str], sd_key_ops: SDKeyOps | None, state_dict: StateDict) -> str:
        sd_id = self._generate_id(paths, sd_key_ops)
        with self._lock:
            if sd_id in self._state_dicts:
                raise ValueError(
                    f"State dict retrieved from {paths} with {sd_key_ops} already added, check with get first"
                )
            self._state_dicts[sd_id] = state_dict
        return sd_id

    def pop(self, paths: list[str], sd_key_ops: SDKeyOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.pop(self._generate_id(paths, sd_key_ops), None)

    def get(self, paths: list[str], sd_key_ops: SDKeyOps | None) -> StateDict | None:
        with self._lock:
            return self._state_dicts.get(self._generate_id(paths, sd_key_ops), None)
