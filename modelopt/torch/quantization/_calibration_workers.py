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

"""Isolated AutoQuant calibration artifacts with a single atomic coordinator."""

import hashlib
import inspect
import itertools
import json
import sys
from argparse import Namespace
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from modelopt.torch.opt.utils import named_hparams

from ._calibration_store import _CalibrationStore, _collective_check
from .algorithms import QuantRecipe, QuantRecipeHparam
from .nn import TensorQuantizer

__all__ = []


@lru_cache(maxsize=1)
def _implementation_digest():
    digest = hashlib.sha256()
    root = Path(__file__).parents[1]
    for path in sorted(root.rglob("*")):
        if path.relative_to(root).parts[0] not in {
            "__init__.py",
            "quantization",
            "kernels",
            "opt",
            "utils",
            "models",
            "trace",
        }:
            continue
        if path.suffix not in {".py", ".cpp", ".cu", ".h", ".cuh"}:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _digest(value):
    digest = hashlib.sha256()

    def update(item):
        kind = next(
            (
                name
                for cls, name in (
                    (torch.Tensor, "tensor"),
                    (dict, "dict"),
                    (list, "list"),
                    (tuple, "tuple"),
                )
                if isinstance(item, cls)
            ),
            type(item).__name__,
        )
        digest.update(kind.encode())
        if isinstance(item, torch.Tensor):
            tensor = item.to_local() if isinstance(item, DTensor) else item
            if tensor.is_meta or not tensor.is_contiguous():
                raise ValueError(
                    "Calibration workers require materialized contiguous source tensors"
                )
            digest.update(str((tuple(item.shape), str(item.dtype))).encode())
            if isinstance(item, DTensor):
                digest.update(str((item.placements, item.device_mesh.mesh.tolist())).encode())
            flat = tensor.detach().view(-1)
            for chunk in flat.split(max(1, 8 * 1024**2 // tensor.element_size())):
                digest.update(chunk.cpu().view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                update(key)
                update(item[key])
        elif isinstance(item, list | tuple):
            digest.update(str(len(item)).encode())
            for child in item:
                update(child)
        elif isinstance(item, torch.dtype):
            digest.update(str(item).encode())
        elif isinstance(item, Namespace | SimpleNamespace):
            update(vars(item))
        elif isinstance(item, Path):
            update(str(item))
        elif isinstance(item, torch.nn.Module):
            update([inspect.getsource(type(item)), item.state_dict()])
        elif inspect.isfunction(item):
            closure = inspect.getclosurevars(item)
            update(
                [
                    inspect.getsource(item),
                    item.__defaults__,
                    item.__kwdefaults__,
                    closure.nonlocals,
                    {k: v for k, v in closure.globals.items() if not inspect.ismodule(v)},
                ]
            )
        else:
            try:
                digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
            except TypeError as exc:
                raise ValueError(
                    f"Unsupported calibration identity value {type(item).__qualname__}; "
                    "capture serializable callback configuration instead of opaque runtime objects"
                ) from exc
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _check_provenance(actual, expected):
    if actual == expected:
        return
    previous = (actual or {}).get("contract", {})
    current = expected["contract"]
    fields = []
    for section in sorted(previous.keys() | current.keys()):
        left, right = previous.get(section), current.get(section)
        if section == "identity" and isinstance(left, dict) and isinstance(right, dict):
            fields.extend(
                f"identity.{key}"
                for key in sorted(left.keys() | right.keys())
                if left.get(key) != right.get(key)
            )
        elif left != right:
            fields.append(section)
    raise ValueError("Calibration provenance differs: " + ", ".join(fields or ["digest"]))


class _CalibrationSession(_CalibrationStore):
    def __init__(self, options, model, data_loader, steps, callbacks):
        if not isinstance(options, dict) or set(options) - {
            "directory",
            "candidate",
            "max_input_bytes",
            "collect_scores",
            "max_artifact_bytes",
            "max_candidate_bytes",
        }:
            raise ValueError("Invalid calibration worker options")
        super().__init__(options["directory"], options.get("candidate"))
        self.max_artifact_bytes = options.get("max_artifact_bytes")
        self.max_candidate_bytes = options.get("max_candidate_bytes", self.max_artifact_bytes)
        for limit in (self.max_artifact_bytes, self.max_candidate_bytes):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Artifact byte limits must be positive")
        self.collect_scores = options.get("collect_scores", False)
        if type(self.collect_scores) is not bool:
            raise ValueError("collect_scores must be a bool")
        self.scores_ready = False
        self.raw_scores, self.fixed_digests = {}, {}
        self.fixed_recipes = set()
        if self.candidate is not None and (type(self.candidate) is not int or self.candidate < 0):
            raise ValueError("candidate must be a nonnegative index or None for the coordinator")
        limit = options.get("max_input_bytes", 64 * 1024**2)
        if type(limit) is not int or limit < 1:
            raise ValueError("max_input_bytes must be positive")
        self.batches, used = [], 0
        for batch in itertools.islice(data_loader, steps):
            leaves, spec = tree_flatten(batch)
            copied: list[tuple[Any, torch.device | None]] = []
            for leaf in leaves:
                if isinstance(leaf, torch.Tensor):
                    used += leaf.numel() * leaf.element_size()
                    if used > limit:
                        raise ValueError("Calibration input snapshot exceeds max_input_bytes")
                    copied.append((leaf.detach().cpu().clone(), leaf.device))
                else:
                    copied.append((leaf, None))
            self.batches.append((copied, spec))
        if len(self.batches) != steps:
            raise ValueError("Calibration workers require the declared number of input batches")
        self.identity = {
            "source": _digest(
                dict(itertools.chain(model.named_parameters(), model.named_buffers()))
            ),
            "inputs": _digest([[x for x, _ in leaves] for leaves, _ in self.batches]),
            "input_structure": [str(spec) for _, spec in self.batches],
            "model_type": type(model).__qualname__,
            "training": {name: m.training for name, m in model.named_modules()},
            "forwards": _digest(sorted({inspect.getsource(m.forward) for m in model.modules()})),
            "model_config": _digest(model.config.to_dict()) if hasattr(model, "config") else None,
            "callbacks": _digest(callbacks),
            "numerics": [
                torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            ],
            "rng": _digest(
                [
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                ]
            ),
            "world_size": self.world_size,
            "rank": self.rank,
            "torch": str(torch.__version__),
            "dependencies": {
                name: str(getattr(sys.modules[name], "__version__", None))
                for name in {type(m).__module__.split(".")[0] for m in model.modules()}
            },
            "cuda": torch.version.cuda,
            "device_capability": torch.cuda.get_device_capability()
            if torch.cuda.is_available()
            else None,
            "code": _implementation_digest(),
        }
        self.contract = None
        self.recipes = []

    def __iter__(self):
        for leaves, spec in self.batches:
            yield tree_unflatten(
                [
                    value.to(device, copy=True) if device is not None else value
                    for value, device in leaves
                ],
                spec,
            )

    def _path(self, recipe):
        return self.directory / hashlib.sha256(recipe.checkpoint_signature.encode()).hexdigest()

    def prepare(self, searcher, recipes, schemas):
        if self.collect_scores:
            self.hparams = {
                name: hparam
                for name, hparam in named_hparams(searcher.model, unique=True)
                if isinstance(hparam, QuantRecipeHparam)
            }
            self.module_names = {module: name for name, module in searcher.model.named_modules()}
            self.fixed_recipes = {
                hparam.active for hparam in self.hparams.values() if not hparam.is_configurable
            }
            self.fixed_recipes = {recipe for recipe in self.fixed_recipes if not recipe.is_no_quant}
            error = None
            for name, hparam in self.hparams.items():
                for recipe, choices in hparam._all_quantizer_choices.items():
                    assert isinstance(recipe, QuantRecipe)
                    if recipe.is_no_quant:
                        continue
                    if not any(
                        isinstance(q, TensorQuantizer) and q.is_enabled
                        for attrs in choices.values()
                        for quantizer in attrs.values()
                        for q in quantizer.modules()
                    ):
                        error = (
                            f"Candidate {recipe} enables no quantizers in group {name}; "
                            "candidate quantizer patterns must be module-relative"
                        )
            _collective_check(error)
        self.recipes = [recipe for recipe in recipes if not recipe.is_no_quant]
        state = searcher.state_dict()
        self.contract = {
            "version": 2 if self.collect_scores else 1,
            "identity": self.identity,
            "setup": {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "quantizer_states",
                    "candidate_stats",
                    "best",
                    "calibration_provenance",
                    "cost_denominator",
                }
            },
            "steps": [searcher.config["num_calib_steps"], searcher.config["num_score_steps"]],
            "method_options": {key: searcher.config[key] for key in searcher.method_config_keys},
        }
        # Compare a canonical digest because setup signatures contain tuples and recipe metadata.
        provenance = {"digest": _digest(self.contract), "contract": self.contract}
        restored = state.get("calibration_provenance")
        error, accepted = None, {}
        _collective_check(
            None,
            _digest(
                [
                    self.contract["setup"],
                    self.contract["steps"],
                    self.contract["method_options"],
                    self.candidate,
                    str(self.directory),
                ]
            ),
        )
        locations: list[str | None] = [None] * len(self.recipes)
        if self.rank == 0:
            try:
                locations = [
                    str(self._path(r).resolve()) if self._path(r).exists() else None
                    for r in self.recipes
                ]
            except Exception as exc:
                error = str(exc)
        _collective_check(error)
        if dist.is_initialized():
            dist.broadcast_object_list(locations, src=0)
        try:
            if searcher.quantizer_states or searcher.candidate_stats:
                _check_provenance(restored, provenance)
            if self.candidate is not None and self.candidate >= len(self.recipes):
                raise ValueError("Candidate index is outside the complete search space")
            selected = self.recipes if self.candidate is None else [self.recipes[self.candidate]]
            for recipe in selected:
                path = locations[self.recipes.index(recipe)]
                if path is None and self.candidate is not None:
                    continue
                if path is None:
                    raise ValueError("A candidate calibration result is missing")
                artifact = self.load(Path(path) / f"rank{self.rank}.pth", self.max_candidate_bytes)
                _check_provenance(artifact["provenance"], provenance)
                if artifact["recipe"] != recipe.checkpoint_signature:
                    raise ValueError("Candidate recipe differs")
                entry = artifact["state"]
                if (
                    set(entry) != {"metadata", "state_dict"}
                    or set(entry["metadata"]) != schemas[recipe]
                    or set(entry["state_dict"]) != schemas[recipe]
                ):
                    raise ValueError("Candidate quantizer mapping is incomplete")
                for name, metadata in entry["metadata"].items():
                    description = metadata["_pytorch_state_metadata"]
                    tensors = {**description["buffers"], **description["params"]}
                    if metadata.get("backend") == "ggml" and not metadata.get("_disabled", False):
                        algorithms = recipe.config.algorithm
                        algorithms = algorithms if isinstance(algorithms, list) else [algorithms]
                        if any(
                            (isinstance(a, dict) and a["method"] in {"gptq", "imatrix"})
                            or a == "imatrix"
                            for a in algorithms
                        ):
                            key = "_ggml_pinned_" + metadata["_num_bits"]
                            if key not in tensors or tensors[key]["dtype"] != torch.uint8:
                                raise ValueError("Candidate is missing its selected packed weight")
                    if set(tensors) != set(entry["state_dict"][name]):
                        raise ValueError("Candidate quantizer buffers are incomplete")
                    for key, spec in tensors.items():
                        tensor = entry["state_dict"][name][key]
                        if (
                            not isinstance(tensor, torch.Tensor)
                            or tuple(tensor.shape) != tuple(spec["shape"])
                            or tensor.dtype != spec["dtype"]
                        ):
                            raise ValueError("Candidate buffer layout differs from metadata")
                if _digest(entry) != artifact["digest"]:
                    raise ValueError("Candidate payload is incomplete or corrupt")
                if self.collect_scores:
                    self._validate_scores(recipe, artifact)
                if (
                    recipe in searcher.quantizer_states
                    and _digest(searcher.quantizer_states[recipe]) != artifact["digest"]
                ):
                    raise ValueError("Candidate conflicts with the shared checkpoint")
                accepted[recipe] = entry
            if self.collect_scores and self.candidate is None:
                expected_fixed = {
                    r.checkpoint_signature: _digest(accepted[r]) for r in self.fixed_recipes
                }
                if any(value != expected_fixed for value in self.fixed_digests.values()):
                    raise ValueError("Format workers used different fixed reference states")
        except Exception as exc:
            error = str(exc)
        _collective_check(error)
        already_complete = all(recipe in searcher.quantizer_states for recipe in self.recipes)
        _collective_check(None, str((already_complete, bool(searcher.candidate_stats))))
        searcher.quantizer_states.update(accepted)
        searcher.calibration_provenance = provenance
        if self.candidate is None and not already_complete:
            searcher.save_search_checkpoint()

    def includes(self, recipe):
        return (
            self.candidate is None
            or recipe == self.recipes[self.candidate]
            or (self.collect_scores and recipe in self.fixed_recipes)
        )

    @property
    def scoring_candidate(self):
        return (
            self.recipes[self.candidate]
            if self.collect_scores and self.candidate is not None
            else None
        )

    @property
    def reused_scoring_candidate(self):
        return self.scoring_candidate is not None and self.scoring_candidate in self.raw_scores

    def _score_mapping(self, recipe):
        return {
            name: {self.module_names[module]: module for module in hparam.score_modules}
            for name, hparam in self.hparams.items()
            if hparam.is_configurable and recipe in hparam.choices
        }

    def _validate_scores(self, recipe, artifact):
        scores = artifact["raw_scores"]
        mapping = self._score_mapping(recipe)
        if set(scores) != set(mapping):
            raise ValueError("Format scoring group coverage differs")
        for name, modules in mapping.items():
            if set(scores[name]) != set(modules):
                raise ValueError("Format scoring boundaries differ")
            for value in scores[name].values():
                if value is not None and (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                    or not value.is_floating_point()
                    or not torch.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Invalid raw gradient sensitivity contribution")
        fixed = artifact["fixed_digests"]
        if set(fixed) != {r.checkpoint_signature for r in self.fixed_recipes}:
            raise ValueError("Format fixed-reference coverage differs")
        if _digest([scores, fixed]) != artifact["scoring_digest"]:
            raise ValueError("Format scoring artifact is incomplete or corrupt")
        self.raw_scores[recipe], self.fixed_digests[recipe] = scores, fixed

    def finish_scoring(self, searcher):
        if self.candidate is not None:
            if not self.reused_scoring_candidate:
                searcher.estimate_sensitivity_scores()
                self.scores_ready = True
                searcher.save_search_checkpoint()
            return
        for recipe, scores in self.raw_scores.items():
            for name, modules in self._score_mapping(recipe).items():
                for module_name, module in modules.items():
                    self.hparams[name]._importance_dict[recipe][module] = scores[name][module_name]
        previous = searcher.candidate_stats
        searcher.candidate_stats = {}
        searcher.initialize_candidate_stats()
        error = (
            "Format scores conflict with the shared checkpoint"
            if previous and previous != searcher.candidate_stats
            else None
        )
        try:
            _collective_check(error)
        except ValueError:
            searcher.candidate_stats = previous
            raise
        self.scores_ready = True
        if not previous:
            searcher.save_search_checkpoint()

    def save(self, searcher):
        if self.collect_scores and not self.scores_ready:
            return
        error = None
        destination = searcher.config["checkpoint"]
        if self.candidate is None and destination is None:
            return
        try:
            payload = searcher.state_dict()
            if self.candidate is not None:
                recipe = self.recipes[self.candidate]
                entry = searcher.quantizer_states[recipe]
                destination = self._path(recipe)
                payload = {
                    "provenance": searcher.calibration_provenance,
                    "recipe": recipe.checkpoint_signature,
                    "state": entry,
                    "digest": _digest(entry),
                }
                if self.collect_scores:
                    scores = {
                        name: {
                            key: value.detach().cpu() if value is not None else None
                            for key, module in modules.items()
                            for value in [self.hparams[name]._importance_dict[recipe][module]]
                        }
                        for name, modules in self._score_mapping(recipe).items()
                    }
                    fixed = {
                        r.checkpoint_signature: _digest(searcher.quantizer_states[r])
                        for r in self.fixed_recipes
                    }
                    payload.update(
                        raw_scores=scores,
                        fixed_digests=fixed,
                        scoring_digest=_digest([scores, fixed]),
                    )
                    self._validate_scores(recipe, payload)
        except Exception as exc:
            error = str(exc)
        _collective_check(error)
        with self.exclusive(destination) if self.candidate is not None else nullcontext():
            error = None
            exists = False
            try:
                if self.candidate is not None and Path(destination).exists():
                    exists = True
                    old = self.load(
                        Path(destination).resolve() / f"rank{self.rank}.pth",
                        self.max_candidate_bytes,
                    )
                    if (
                        old["provenance"] != payload["provenance"]
                        or old["digest"] != payload["digest"]
                        or old.get("scoring_digest") != payload.get("scoring_digest")
                    ):
                        raise ValueError("A conflicting candidate was published concurrently")
            except Exception as exc:
                error = str(exc)
            _collective_check(error, str(exists))
            # Candidate artifacts are immutable. Readers may use them without taking the writer lock.
            if not exists:
                self._publish(Path(destination).absolute(), payload)
