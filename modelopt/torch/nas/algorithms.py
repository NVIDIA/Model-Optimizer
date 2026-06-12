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

"""High-level search and model design algorithms to help you optimize your model."""

import copy
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import QhullError
from torch import nn

from modelopt.torch.opt.conversion import get_mode
from modelopt.torch.opt.searcher import ConstraintsDict, Deployment, LimitsTuple, SearchConfig
from modelopt.torch.opt.utils import is_configurable, search_space_size
from modelopt.torch.utils import (
    clear_cuda_cache,
    num2hrb,
    param_num_from_forward,
    random,
    unwrap_model,
)
from modelopt.torch.utils import distributed as dist

from .conversion import export
from .search_space import SampleFunc
from .utils import (
    _SearchSpaceUnwrapped,
    get_subnet_config,
    inference_flops,
    no_modelopt_patches,
    print_search_space_summary,
    sample,
    sample_and_reset,
)

__all__ = ["profile", "search"]

# Mapping module type to a specific constraint function. Plugins may change this mapping.
MODULE_TYPE_TO_CONSTRAINTS_FUNC = {}

ConstraintsRes = dict[str, float]
ConstraintEvalFunc = Callable[[ConstraintsRes | None], float]


class ConstraintInterpolator:
    """A piece-wise, linear, d-dim interpolator to interpolate some model property like latency.

    The interpolator is based on scipy.interpolate.LinearNDInterpolator, which is based on an
    d-dimensional linear interpolation from unstructured d-dimensional data points. The interpolator
    uses a Delaunay triangulation as intermediate data representation.

    Note that for a successful triangulation/interpolation, at least d+1 linearly independent points
    are needed that form a convex hull around the desired query point.

    Every time a new data point is queried, the interpolator will be dynamically re-generated as
    needed.
    """

    def __init__(
        self,
        model: nn.Module,
        points_funcs: dict[str, ConstraintEvalFunc],
        value_func: ConstraintEvalFunc,
    ):
        """Initialize interpolator with points_func and value_func to interpolate."""
        assert len(points_funcs) > 0, "Need at least one point function to interpolate from!"
        self.model = model
        self._points_funcs = points_funcs
        self._value_func = value_func
        self.points = np.zeros((0, len(points_funcs)))
        self._vals = np.zeros(0)
        self._interpolator = None

        # In collect mode, data points are added to the grid even if the interpolator could be used
        # in order to improve the interpolator's quality.
        self.collect_mode = False

    def _add_point(self, point: list | NDArray) -> float:
        """Add point to interpolation table and return value."""
        # compute new value
        val = self._value_func(self.model)

        # add point/va. to our database of points
        self.points = np.vstack((self.points, np.asarray(point)[None]))
        self._vals = np.hstack((self._vals, val))

        # dynamically re-generate the interpolator
        try:
            if len(self._points_funcs) > 1:
                self._interpolator = LinearNDInterpolator(self.points, self._vals)
            elif len(self._vals) > 1:  # can only use interp1d with 2+ points collected
                self._interpolator = interp1d(self.points[:, 0], self._vals, bounds_error=False)
            else:
                self._interpolator = None  # not enough points collected yet
        except QhullError:
            self._interpolator = None  # not enough linearly independent points collected yet

        # return new point
        return val

    def __call__(self, pre_computed: dict[str, float] | None = None) -> float:
        """Interpolate based on the provided model query or compute a new, exact data point."""
        # compute new point but only query as necessary
        pre_computed = pre_computed or {}
        point = [
            pre_computed[key] if key in pre_computed else func(self.model)
            for key, func in self._points_funcs.items()
        ]

        # check if we can use the interpolator or need/want to query the value directly
        if self._interpolator is None or self.collect_mode:
            val = self._add_point(point)
        else:
            val = self._interpolator(point)
            # check if we landed outside the convex hull (val == NaN) and if so add it as new point
            if np.isnan(val):
                val = self._add_point(point)

        # return value
        return np.asarray(val).item()


class ConstraintsFunc:
    """A Functor class to check if sub-net satisfied all provided constraints.

    We intentionally expose some attributes like `limits` s.t. we can modify it manually.
    """

    _sample_points_dict: dict[tuple[str, ...], dict[str, SampleFunc]] = {
        ("flops",): {"min": min, "centroid": random.centroid, "max": max},
        ("flops", "flops_min_depth"): {
            "min": min,
            "min_regular_centroid_depth": {"*": min, "*.depth": random.centroid},
            "min_regular_max_depth": {"*": min, "*.depth": max},
            "centroid_regular_min_depth": {"*": random.centroid, "*.depth": min},
            "centroid": random.centroid,
            "centroid_regular_max_depth": {"*": random.centroid, "*.depth": max},
            "max_regular_min_depth": {"*": max, "*.depth": min},
            "max_regular_centroid_depth": {"*": max, "*.depth": random.centroid},
            "max": max,
        },
    }

    def __init__(
        self,
        model: nn.Module,
        constraints: ConstraintsDict,
        dummy_input: Any | tuple[Any, ...],
        deployment: Deployment | None = None,
        fast_eval: bool = True,
        dp_group=None,
    ):
        self.model = model
        self.dummy_input = dummy_input
        self.deployment = deployment
        self._fast_eval = fast_eval

        # Getting data parallel group for
        self.dp_group = dp_group

        # initialize latency interpolator
        keys_for_interpolation = ("flops",)
        if ConstraintsFunc.is_configurable(self.model, "depth"):
            keys_for_interpolation += ("flops_min_depth",)
        self._latency_interpolator = ConstraintInterpolator(
            self.model,
            points_funcs={k: self.constraint_eval_funcs[k] for k in keys_for_interpolation},
            value_func=self._get_true_latency,
        )
        # set fast/regular mode for latency interpolator
        self._latency_interpolator.collect_mode = not self.fast_eval

        # set limit at the end with setter to use sanity checks on constraints
        self._limits = {}
        self.limits = constraints

    @staticmethod
    def is_configurable(model: nn.Module, hparam_suffix: str | None = None):
        return hparam_suffix is None or any(
            k.rpartition(".")[-1] == hparam_suffix for k in get_subnet_config(model, True)
        )

    def get_sample_points(self) -> dict[str, SampleFunc]:
        """Get the sample points required for latency interpolator."""
        return self._sample_points_dict[tuple(self._latency_interpolator._points_funcs)]

    @property
    def fast_eval(self) -> bool:
        """Return boolean indicator whether we use faster interpolation for latency."""
        return self._fast_eval

    @fast_eval.setter
    def fast_eval(self, value: bool):
        """Set bool whether we should use faster interpolation for latency."""
        self._fast_eval = value
        self._latency_interpolator.collect_mode = not value

    @property
    def limits(self) -> dict[str, LimitsTuple]:
        """Get limits for all constraints."""
        return self._limits

    @limits.setter
    def limits(self, constraints: ConstraintsDict) -> None:
        # check that no unexpected keys are found in provided constraints dict.
        if not constraints.keys() <= self.constraint_eval_funcs.keys():
            raise KeyError(
                f"Found unexpected keys:\n{constraints.keys() - self.constraint_eval_funcs.keys()}",
                f"\nCurrently supported constraint keys:\n{self.constraint_eval_funcs.keys()}",
            )

        # check that deployment is provided when latency is used as a constraint.
        if "latency" in constraints and self.deployment is None:
            raise ValueError(
                "Deployment config must be provided when latency is used as a constraint."
            )

        # making sure we add both "flops" and "params" by default as fake constraints
        # since it's cheap and can provide useful insights during constraint evaluation.
        # Moreover, we need it for latency interpolation anyway.
        constraints_complete: ConstraintsDict = {
            k: None for k in ("flops", "params") if k in self.constraint_eval_funcs
        }
        constraints_complete.update(**constraints)

        if "latency" in constraints:
            fake_constraints = dict.fromkeys(self._latency_interpolator._points_funcs.keys())
            constraints_complete.update(fake_constraints)

        self._limits = self.get_expanded_limits(constraints_complete)

    @property
    def effective_limits(self) -> dict[str, LimitsTuple]:
        return {k: v for k, v in self.limits.items() if v != self.trivial_limit}

    @property
    def trivial_limit(self) -> LimitsTuple:
        return (0, float("inf"))

    def get_expanded_limits(self, constraints: ConstraintsDict) -> dict[str, LimitsTuple]:
        """Extract and standardize limits from constraint dictionary."""
        # standardize limits to LimitsTuple: (lower_bound, upper_bound)
        expanded_limits = {}
        for k, lim in copy.deepcopy(constraints).items():
            if lim is None:
                expanded_limits[k] = self.trivial_limit
                continue

            if isinstance(lim, str):
                assert lim.endswith("%")
                fraction_value = float(lim[:-1]) / 100

                # Use _SearchSpaceUnwrapped to avoid resetting the calibration statistics
                ss = _SearchSpaceUnwrapped(self.model)
                config = ss.config()
                ss.sample(random.original)
                orig_value = self.constraint_eval_funcs[k]()  # type: ignore[call-arg]
                ss.select(config)

                lim = orig_value * fraction_value

            expanded_limits[k] = (0, lim)  # lower bound is always 0 by default

        return expanded_limits

    def set_rel_lower_bounds(self, rel_lower: float = 0.85) -> None:
        """Set relative lower bounds for all constraints."""
        for k, (lb, ub) in self._limits.items():
            assert lb == 0, "Lower bound has already been set!"
            if self._limits[k] != self.trivial_limit:
                self._limits[k] = (rel_lower * ub, ub)

    def _get_params(self, _: ConstraintsRes | None = None) -> float:
        """Get number of model parameters from forward pass."""
        return param_num_from_forward(self.model, args=self.dummy_input, unit=1.0)

    def _get_flops(self, _: ConstraintsRes | None = None) -> float:
        """Get inference FLOPs."""
        return float(inference_flops(self.model, dummy_input=self.dummy_input, unit=1.0))

    def _get_flops_min_depth(self, _: ConstraintsRes | None = None) -> float:
        """Get inference FLOPs with depth set to minimum."""
        with sample_and_reset(self.model, {"*.depth": min}):
            flops = float(inference_flops(self.model, dummy_input=self.dummy_input, unit=1.0))
        return flops

    def _get_true_latency(self, _: ConstraintsRes | None = None) -> float:
        """Get true inference latency."""
        from modelopt.torch._deploy import get_latency

        assert self.deployment is not None, "Deployment config must be provided for latency!"
        return get_latency(self.model, self.dummy_input, self.deployment)

    def _get_latency(self, precomputed: ConstraintsRes | None = None) -> float:
        """Get inference latency from interpolator."""
        return self._latency_interpolator(precomputed)

    @property
    def constraint_eval_funcs(self) -> dict[str, ConstraintEvalFunc]:
        """Get constraint eval fns."""
        return {
            "params": self._get_params,
            "flops": self._get_flops,
            "flops_min_depth": self._get_flops_min_depth,
            "latency": self._get_latency,
        }

    def __call__(self) -> tuple[bool, ConstraintsRes]:
        return self.evaluate_constraints()

    def evaluate_constraints(self) -> tuple[bool, ConstraintsRes]:
        """Check if sub-net satisfied all provided constraints.

        Returns: A tuple (is_satisfied, vals) where
            is_satisfied is an indicator (bool) whether all constraints are satisfied,
            vals is a dictionary containing all evaluated constraints up to 1st unsatisfied constraint.

        The sub-net is profiled after exporting it in eval mode without BN.
        """

        def check_constraint(val: float, limits: LimitsTuple) -> bool:
            """Check if val falls within limits. `None` indicates open-ended intervals."""
            return val >= limits[0] and val <= limits[1]

        is_satisfied = True
        vals: ConstraintsRes = {}

        if dist.is_master(group=self.dp_group):
            # construct sub-net without DDP and BN for more accurate profiling
            model = unwrap_model(self.model)

            # put model into eval mode
            is_training = model.training
            model.eval()

            # check constraint (with early stopping if not satisfied)
            # follow the order of the constraint_eval_funcs
            for criterion, score_func in self.constraint_eval_funcs.items():
                if criterion not in self.limits:
                    continue
                val = score_func(vals)
                vals[criterion] = val
                is_satisfied &= check_constraint(val, self.limits[criterion])

            # put back into original mode
            model.train(is_training)

        is_satisfied, vals = dist.broadcast([is_satisfied, vals])
        return is_satisfied, vals


class StatisticsEstimator:
    """A class to randomly sample subnets & estimate statistics for the provided constraints.

    The class can:
        Estimate median and standard deviation of the constraints
        Plot the distribution of constraints

    Args:
        constraints_func: An object of class ``ConstraintsFunc`` used to define and evaluate the
            constraints.
        num_of_samples: An integer that specifies the number of random samples of model
            used to estimate the constraint statsitics.
    """

    def __init__(
        self,
        constraints_func: ConstraintsFunc,
        num_of_samples: int = 100,
    ) -> None:
        self.constraints_func = constraints_func
        self.num_of_samples = num_of_samples
        self._sample_dict = {}
        self.reset()

    def reset(self):
        """Resets statistics estimator history."""
        self._sample_dict = {k: [] for k in self.constraints_func.limits}

    @property
    def num_of_collected_samples(self):
        return len(next(iter(self._sample_dict.values())))

    def _sample_random_subnets(self, model) -> dict:
        with random._deterministic_seed():
            max_num_samples = self.num_of_samples if is_configurable(model) else 1
            for _ in range(self.num_of_collected_samples, max_num_samples):
                sample(model)
                info = self.constraints_func()[1]
                for constraint in self.constraints_func.limits:
                    self._sample_dict[constraint].append(info[constraint])

        return self._sample_dict

    def median(self, model: nn.Module) -> dict[str, float]:
        """Estimates median constraint values for the model."""
        median_dict = {}
        sample_dict = self._sample_random_subnets(model)
        for constraint, constraint_samples in sample_dict.items():
            median_dict[constraint] = float(np.median(constraint_samples))
        return median_dict

    def std(self, model: nn.Module) -> dict[str, float]:
        """Estimates std constraint values for the model."""
        std_dict = {}
        sample_dict = self._sample_random_subnets(model)
        for constraint, constraint_samples in sample_dict.items():
            std_dict[constraint] = float(np.std(constraint_samples))
        return std_dict

    def median_subnet(self, model: nn.Module) -> dict:
        """Estimates a median/representative subnet for the model.

        Returns model nas.config dict
        """
        raise NotImplementedError


def get_constraints_func(model, *args, **kwargs) -> ConstraintsFunc:
    """Getting the standard constraints function or a special one from the plugins."""
    # Initialize special constraints functor if MODULE_TYPE_TO_CONSTRAINTS_FUNC is not empty.
    for module_type, func in MODULE_TYPE_TO_CONSTRAINTS_FUNC.items():
        if isinstance(model, module_type):
            return func(model, *args, **kwargs)
    # Initialize the standard constraints functor
    return ConstraintsFunc(model, *args, **kwargs)


def search(
    model: nn.Module,
    constraints: ConstraintsDict,
    dummy_input: Any | tuple[Any, ...],
    config: SearchConfig | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Search a given prunable model for the best sub-net and return the search model.

    The best sub-net maximizes the score given by ``score_func`` while satisfying the
    ``constraints``.

    Args:
        model: The converted model to be searched.
        constraints: The dictionary from constraint name to upper bound the searched model has to satisfy.
            Currently, we support ``flops`` and ``params`` as constraints.
            The constraints dictionary generally takes the following form:

            .. code-block:: python

                constraints = {"params": 5.0e6, "flops": 4.5e8}

            We recommend to simply provide the most relevant constraint, e.g., flops:

            .. code-block:: python

                constraints = {"flops": 4.5e8}

            You can also provide a percentage value instead of absolute value, e.g.,

            .. code-block:: python

                # search for a model with <= 60% of the original model flops
                constraints = {"flops": "60%"}

        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        config: Additional optional arguments to configure the search. Currently, we support:

            * ``checkpoint``: Path to save/restore checkpoint with dictionary containing intermediate
              search state. If provided, the intermediate search state will be automatically
              restored before search (if exists) and stored/saved during search.
            * ``verbose``: Whether to print detailed search space profiling and search stats during search.
            * ``forward_loop``: A ``Callable`` that takes a model as input and runs a forward loop
              on it. It is recommended to choose the data loader used inside the forward loop
              carefully to reduce the runtime. Cannot be provided at the same time as
              ``data_loader`` and ``collect_func``.
            * ``data_loader``: An iterator yielding batches of data for calibrating the
              normalization layers in the model or compute gradient scores. It is recommended to use
              the same data loader as for training but with significantly fewer iterations. Cannot
              be provided at the same time as ``forward_loop``.
            * ``collect_func``: A ``Callable`` that takes a batch of data from the data loader as
              input and returns the input to ``model.forward()`` as described in
              :meth:`run_forward_loop <modelopt.torch.utils.network.run_forward_loop>`. Cannot
              be provided at the same time as ``forward_loop``.
            * ``max_iter_data_loader``: Maximum number of iterations to run the data loader.
            * ``score_func``: A callable taking the model as input and returning a single accuracy/score
              metric (float). This metric will be maximized during search.

                .. note::

                    The ``score_func`` is required for ``autonas`` and ``fastnas`` modes. It will be
                    evaluated on models in eval mode (``model.eval()``).
            * ``loss_func``: A ``Callable`` which takes the model output (i.e output of ``model.forward()``)
              and the batch of data as its inputs and returns a scalar loss.
              This is a required argument if the model is converted via ``gradnas`` mode.

              It should be possible to run a backward pass on the loss value returned by this method.

              ``collect_func`` will be used to gather the inputs to ``model.forward()``
              from a batch of data yielded by``data_loader``.

              ``loss_func`` should support the following usage:

                .. code-block:: python

                    for i, batch in enumerate(data_loader):
                        if i >= max_iter_data_loader:
                            break

                        # Assuming collect_func returns a tuple of arguments
                        output = model(*collect_func(batch))

                        loss = loss_func(output, batch)
                        loss.backward()

            .. note::

                Additional configuration options may be added by individual algorithms. Please
                refer to the documentation of the individual algorithms for more information.

    Returns: A tuple (subnet, state_dict) where
        subnet is the searched subnet (nn.Module), which can be used for subsequent tasks like
        fine-tuning, state_dict contains the history and detailed stats of the search procedure.

    .. note::

        The given model is modified (exported) in-place to match the best subnet found by the
        search algorithm. The returned subnet is thus a reference to the same model instance as the
        input model.
    """
    # retrieve searcher class
    mode = get_mode(model)
    assert mode is not None, "Model is not converted to a search space. Please convert first."
    searcher_cls = mode.search_algorithm

    # run search, select best, and get config, metric, constraint of search result
    searcher = searcher_cls()

    searcher.search(model, constraints, dummy_input, config)

    # export model in-place
    model = export(model)

    clear_cuda_cache()

    # return the optimized & exported model and searcher's state_dict (detailed stats)
    return model, searcher.state_dict()


@torch.no_grad()
@no_modelopt_patches()
def profile(
    model: nn.Module,
    dummy_input: Any | tuple[Any, ...] | None = None,
    constraints: ConstraintsDict | ConstraintsFunc | None = None,
    deployment: Deployment | None = None,
    strict: bool = False,
    verbose: bool = True,
    use_centroid: bool = False,
) -> tuple[bool, dict[str, dict]]:
    """Profile statistics of the search space of a converted model or a regular model.

    Args:
        model: The model to be profiled. Can be converted or not.
        dummy_input: Arguments of ``model.forward()``. This is used for exporting and calculating
            inference-based metrics, such as latency/FLOPs. The format of ``dummy_inputs`` follows
            the convention of the ``args`` argument in
            `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_.
            Specifically, ``dummy_input`` can be:

            #. a single argument (``type(dummy_input) != tuple``) corresponding to

               .. code-block:: python

                    model.forward(dummy_input)

            #. a tuple of arguments corresponding to

               .. code-block:: python

                    model.forward(*dummy_input)

            #. a tuple of arguments such that ``type(dummy_input[-1]) == dict`` corresponding to

               .. code-block:: python

                    model.forward(*dummy_input[:-1], **dummy_input[-1])

               .. warning::

                   In this case the model's ``forward()`` method **cannot** contain keyword-only
                   arguments (e.g. ``forward(..., *, kw_only_args)``) or variable keyword arguments
                   (e.g. ``forward(..., **kwargs)``) since these cannot be sorted into positional
                   arguments.

            .. note::

                In order to pass a dict as last non-keyword argument, you need to use a tuple as
                ``dummy_input`` and add an *empty* dict as the last element, e.g.,

                .. code-block:: python

                    dummy_input = (x, {"y": y, "z": z}, {})

                The empty dict at the end will then be interpreted as the keyword args.

            See `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html#torch.onnx.export>`_
            for more info.

            Note that if you provide a ``{arg_name}`` with batch size ``b``, the results will be
            computed based on batch size ``b``.
        constraints: The dictionary from constraint name to upper bound the searched model has to satisfy.
            Currently, we support ``flops`` and ``params`` as constraints.
            The constraints dictionary generally takes the following form:

            .. code-block:: python

                constraints = {"params": 5.0e6, "flops": 4.5e8}

            .. note::

                We recommend to simply provide the most relevant constraint, e.g., flops:

                .. code-block:: python

                    constraints = {"flops": 4.5e8}

            Note that you can also provide a percentage value instead of absolute value, e.g.,

            .. code-block:: python

                # search for a model with <= 60% of the original model flops
                constraints = {"flops": "60%"}

        strict: Raise an error if constraints are not satisfiable.
        verbose: Print detailed profiling results.
        use_centroid: By default, profile reports median of the evaluation results from
            randomly sampled subnets (instead of the evaluation result from deterministic
            centroid subnet). Set use_centroid to True to use the deterministic centroid for
            profiling.

    Returns: A tuple (is_all_sat, stats) where
        is_all_sat is a bool indicating whether all constraints can be satisfied.
        stats is a dictionary containing statistics for the search space if the model is converted,
        e.g., the FLOPs and params for the min, centroid, max subnets and their max/min ratios,
        size of the search space, number of configurable hparams.
    """
    # pre-fetch some low-level properties properties of the search space
    is_configurable_ = is_configurable(model)
    size_of_search_space = search_space_size(model)
    num_configurable = len(get_subnet_config(model, configurable=True))

    # custom print function that tracks print statements
    detailed_msg = ""

    def _print(msg: str, verb: bool = verbose):
        nonlocal detailed_msg
        if detailed_msg:
            detailed_msg += "\n"
        detailed_msg += msg
        if verb:
            print(msg)

    # construct constraints checker function from constraints dict
    if isinstance(constraints, ConstraintsFunc):
        assert_msg = "No deployment or dummy_input allowed for constraints as ConstraintFunc obj!"
        assert deployment is None and dummy_input is None, assert_msg
        assert constraints.model is model, "Model of constraints does not match the given model!"
        constraints_func = constraints
    else:
        assert dummy_input is not None, "Please provide a dummy_input for the constraints."
        constraints = copy.deepcopy(constraints) or {}
        if deployment is not None and "latency" not in constraints:
            constraints["latency"] = None
        constraints_func = get_constraints_func(model, constraints, dummy_input, deployment, False)

    # retrieve sample points and print message
    options: dict[str, SampleFunc]
    if is_configurable_:
        options = constraints_func.get_sample_points()
        msg = f"Profiling the following subnets from the given model: {tuple(options)}."
    else:
        msg = "Profiling full net, since the model is not a converted model."
        options = OrderedDict([("full", max)])
    _print(msg="\n" + msg + "\n" + "-" * 80)

    # extract limits
    limits = constraints_func.limits

    # sample through some distinct networks and collect some stats
    # note that during that process we definitely don't want fast constraint evaluation!
    fast_eval = constraints_func.fast_eval
    constraints_func.fast_eval = False
    info: dict[str, ConstraintsRes] = {}
    for subnet_key, sample_method in options.items():
        # avoid changing the active subnet before and after nas.profile
        with sample_and_reset(model, sample_method):
            info[subnet_key] = constraints_func()[1]

    # Use statistical median values.
    # it does not have a pre-defined sampling method.
    # To further speed up, we can reduce the number of samples.
    if "centroid" in options and use_centroid is False:
        constraints_func.fast_eval = True
        statistics_estimator = StatisticsEstimator(constraints_func, num_of_samples=30)
        with sample_and_reset(model):
            info["centroid"] = statistics_estimator.median(model)

    constraints_func.fast_eval = fast_eval

    # for each sub net and constraint report the constraint evaluation and where it falls within the
    # limits
    is_all_sat = True

    stats = info.copy()
    if is_configurable_:
        stats["max/min ratio"] = {}

    # options that we also print
    options_keys_for_print = [k for k in options if k in {"min", "centroid", "max", "full"}]

    console = Console()
    with console.capture() as capture:
        table_profile = Table(title="\nProfiling Results")
        table_profile.add_column("Constraint", width=12)

        table_constraints = Table(title="\nConstraints Evaluation")
        table_constraints.add_column("Constraint", width=12)
        table_constraints.add_column("Upper Bound", width=12)
        table_constraints.add_column("Satisfiable Upper Bound", width=12)

        for subnet in options_keys_for_print:
            table_profile.add_column(subnet, width=12)

        if is_configurable_:
            table_profile.add_column("max/min ratio")

        for constraint, (lb, ub) in limits.items():
            print_info = [f"{constraint:8}"]
            is_sat_lower = False
            is_sat_upper = False
            # check constraint satisfaction
            for subnet in options_keys_for_print:
                subnet_info = info[subnet]
                val = subnet_info[constraint]
                is_sat_lower |= val >= lb
                is_sat_upper |= val <= ub
                print_info.append(f"{num2hrb(val)}")

            if is_configurable_:
                # add max/min ratio
                ratio_val = info["max"][constraint] / info["min"][constraint]
                print_info.append(f"{ratio_val:4.2f}")
                stats["max/min ratio"].update({constraint: ratio_val})
            # table_profile.add_row()
            table_profile.add_row(*print_info)

            is_all_sat &= is_sat_lower and is_sat_upper
            # print update
            if constraint in constraints_func.effective_limits:
                print_info = [f"{constraint:8}"]
                print_info.extend([f"{num2hrb(ub)}", f"{is_sat_upper!s:5}"])
                table_constraints.add_row(*print_info)

        console.print(table_profile)
        if len(constraints_func.effective_limits) > 0:
            console.print(table_constraints)

    print_info = str(capture.get())
    _print(print_info)  # type: ignore[arg-type]

    if is_configurable_:
        if verbose:
            print_search_space_summary(model)
        # add search space info
        _print(f"Number of configurable hparams: {num_configurable}")
        stats["number of configurable hparams"] = num_configurable
        _print(f"Total size of the search space: {size_of_search_space:8.2e}")
        stats["search space size"] = size_of_search_space

    # check if all constraints are satisfiable and raise error if not and strict
    if is_all_sat:
        if constraints_func.effective_limits:
            _print("Note: all constraints can be satisfied within the search space!\n")
    elif is_configurable_:
        warn_msg = "" if verbose else f"\n{detailed_msg}\n"
        warn_msg += "NOT all constraints can be satisfied within the search space, see above!"
        if strict:
            raise ValueError(warn_msg)
        else:
            _print(f"Warning: {warn_msg}\n", verb=True)

    return is_all_sat, stats
