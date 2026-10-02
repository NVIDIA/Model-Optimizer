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

"""Recipe loading utilities."""

import warnings

try:
    from importlib.resources.abc import Traversable
except ImportError:  # Python < 3.11
    from importlib.abc import Traversable
from pathlib import Path

from omegaconf import OmegaConf

from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT as BUILTIN_RECIPES_LIB
from modelopt.torch.opt.config_loader import (
    _alias_builtin_recipe_prefix,
    load_config,
    peek_declared_schema,
)
from modelopt.torch.quantization.config import QuantizeConfig

from .config import (
    RECIPE_TYPE_TO_CLASS,
    ModelOptPTQRecipe,
    ModelOptRecipeBase,
    RecipeMetadataConfig,
    RecipeType,
)

__all__ = ["load_config", "load_recipe"]

# Each recipe type's mandatory top-level body section.  Checked at the loader level (on the
# raw YAML, before pydantic fills in defaults) so the user sees a clear "PTQ recipe file X
# must contain 'quantize'" instead of pydantic's generic missing-field error.
_REQUIRED_SECTION_PER_RECIPE_TYPE: dict[RecipeType, str] = {
    RecipeType.PTQ: "quantize",
    RecipeType.AUTO_QUANTIZE: "auto_quantize",
    RecipeType.SPECULATIVE_EAGLE: "eagle",
    RecipeType.SPECULATIVE_DFLASH: "dflash",
    RecipeType.SPECULATIVE_MEDUSA: "medusa",
}


def _resolve_recipe_path(recipe_path: str | Path | Traversable) -> Path | Traversable:
    """Resolve a recipe path, checking the filesystem first then the built-in library.

    Returns the resolved path (file or directory).
    """
    if isinstance(recipe_path, (str, Path)) and not (
        isinstance(recipe_path, Path) and recipe_path.is_absolute()
    ):
        rp_str = str(recipe_path)
        # Backward-compat aliases for the recipe-library restructure. A source checkout keeps
        # ``huggingface`` -> ``model_type`` (and the nested ``model_type/models`` -> ``../models``)
        # symlinks, but symlinks don't survive into built wheels, so the deprecated tier
        # prefixes are rewritten (see ``_alias_builtin_recipe_prefix``) for the built-in lookup,
        # keeping saved ``--recipe huggingface/...`` paths working for pip-installed users.
        aliased = _alias_builtin_recipe_prefix(rp_str)

        def _suffixes(s: str) -> list[str]:
            return [""] if s.endswith((".yml", ".yaml")) else ["", ".yml", ".yaml"]

        # Filesystem first, probing the path exactly as given (then its aliased form), so a
        # user's own local recipe tree overrides a built-in of the same name -- the same
        # precedence as ``config_loader._resolve_config_path``. A local ``huggingface/<type>/...``
        # file therefore still wins even when ``<type>`` collides with a shipped ``model_type``.
        for probe in dict.fromkeys((rp_str, aliased)):
            for suffix in _suffixes(probe):
                fs_candidate = Path(probe + suffix)
                if fs_candidate.is_file() or fs_candidate.is_dir():
                    return fs_candidate
        # Then the built-in library, using the aliased (renamed-tier) form so old
        # ``huggingface/...`` paths resolve from wheels where the compat symlink is gone.
        # Resolving via a rewritten prefix means the user passed a deprecated tier name, so
        # nudge them off it (only here, not when a local file above already won by its own name).
        for suffix in _suffixes(aliased):
            candidate = BUILTIN_RECIPES_LIB.joinpath(aliased + suffix)
            if candidate.is_file() or candidate.is_dir():
                if aliased != rp_str:
                    warnings.warn(
                        f"Recipe path {rp_str!r} uses a deprecated recipe-tier prefix; it "
                        f"resolved to the built-in {aliased!r}. Update saved ``--recipe`` paths "
                        "to the new prefix -- the deprecated one will be removed in a future "
                        "release.",
                        FutureWarning,
                        # Point at the caller of the public ``load_recipe`` (one internal frame
                        # above this one), the common entry point through which a path arrives.
                        stacklevel=3,
                    )
                return candidate
        return Path(rp_str)
    return recipe_path


def load_recipe(
    recipe_path: str | Path | Traversable,
    overrides: list[str] | None = None,
) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file or directory, with optional CLI-style overrides.

    ``recipe_path`` can be:

    * A ``.yml`` / ``.yaml`` file with ``metadata`` and one of ``quantize`` (PTQ),
      ``eagle`` (EAGLE speculative decoding), ``dflash`` (DFlash speculative
      decoding) or ``medusa`` (Medusa speculative decoding) sections. The suffix
      may be omitted and will be probed automatically.

      .. _recipe-alias:

      A recipe can instead **alias an existing recipe**. A top-level ``$import``
      brings the imported recipe in whole, so the file needs nothing else::

          imports:
            base: general/ptq/nvfp4_default-kv_fp8_cast

          $import: base

      That resolves to exactly the imported recipe -- its body, its algorithm and
      its metadata alike -- under a second name.

      Keys given alongside the ``$import`` override the imported ones, so an alias
      can say what it is for while inheriting everything else::

          imports:
            base: general/ptq/nvfp4_default-kv_fp8_cast

          $import: base
          metadata:
            description: What this checkpoint uses the base recipe for.

      An override replaces a top-level key outright rather than merging into it, so
      a partial ``metadata`` supplies the whole section. Nothing needs restating in
      either form: the recipe's kind comes from the imported recipe (see
      :func:`_peek_recipe_type`), and the two must agree if the alias states one.

      The one requirement is on the other side -- the imported recipe must carry a
      ``# modelopt-schema:`` comment naming its schema class, as every
      ``$import``-able file must.

      This is how the checkpoint entries under ``models/<org>/<model_id>/`` record
      that a released checkpoint is reproduced by an existing recipe without
      duplicating it.
    * A directory containing ``metadata.yml`` and ``quantize.yml`` —
      **PTQ recipes only**. Speculative-decoding recipes are always single YAML files.

    The path may be relative to the built-in recipes library or an absolute /
    relative filesystem path.

    ``overrides`` is an optional list of ``key.path=value`` dotlist entries applied
    on top of the YAML before Pydantic validation. Values are parsed with
    ``yaml.safe_load`` so they get proper types (``foo.bar=true`` → bool, ``foo=1``
    → int, ``foo=[1,2]`` → list, etc.). Only supported when *recipe_path* is a
    single YAML file.
    """
    resolved = _resolve_recipe_path(recipe_path)

    _builtin_prefix = str(BUILTIN_RECIPES_LIB)
    _resolved_str = str(resolved)
    if _resolved_str.startswith(_builtin_prefix):
        _display = "<builtin>/" + _resolved_str[len(_builtin_prefix) :].lstrip("/\\")
    else:
        _display = _resolved_str
    print(f"[load_recipe] loading: {_display}")

    if resolved.is_file():
        return _load_recipe_from_file(resolved, overrides=overrides)

    if resolved.is_dir():
        if overrides:
            raise ValueError(
                "overrides are not supported for directory-format recipes; "
                "use the single-YAML-file form instead."
            )
        return _load_recipe_from_dir(resolved)

    raise ValueError(f"Recipe path {recipe_path!r} is not a valid YAML file or directory.")


def _apply_dotlist(data: dict, overrides: list[str]) -> dict:
    """Merge ``a.b.c=value`` command line overrides on top of ``data`` via OmegaConf."""
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Invalid override (missing '='): {entry!r}")
    merged = OmegaConf.merge(
        OmegaConf.create(data),
        OmegaConf.from_dotlist(list(overrides)),
    )
    return OmegaConf.to_container(merged, resolve=False)


#: Recipe schema classes by their fully-qualified path, for resolving a
#: ``# modelopt-schema:`` comment to the recipe kind it names.
_RECIPE_SCHEMA_PATHS: dict[str, RecipeType] = {
    f"{cls.__module__}.{cls.__qualname__}": rtype for rtype, cls in RECIPE_TYPE_TO_CLASS.items()
}


_UNPARSED = object()  # "caller supplied no pre-parsed body", distinct from a body of ``None``


def _peek_recipe_type(
    recipe_file: Path | Traversable,
    _seen: frozenset[str] | None = None,
    _cycle: list[tuple[str, str]] | None = None,
    _raw: object = _UNPARSED,
) -> RecipeType | None:
    """Determine a recipe's kind without resolving its ``$import`` references.

    Needed so :func:`load_config` can be called with the correct ``schema_type`` for
    typed-list ``$import`` resolution before the full recipe is constructed -- which is
    why this cannot simply wait for the imports to resolve.

    Neither way of saying it is mandatory; a recipe just has to say it *somehow*, and
    the three sources are checked in this order:

    1. a ``# modelopt-schema:`` comment naming the recipe's schema class. Only files
       that are **imported** by another one need this -- it is what
       ``$import`` resolution requires of any snippet -- so a leaf recipe never has to
       carry it;
    2. ``metadata.recipe_type`` in the YAML body. **Deprecated** -- still read and still
       honoured, so no existing recipe has to change, but a new one should declare its
       schema instead;
    3. the recipe this one **delegates to** via a top-level ``$import``. A checkpoint
       alias states neither of the above: its kind is whatever its base is, and the base
       must declare a schema to be importable at all, so the walk terminates.

    When more than one source is present they must agree --
    :class:`~modelopt.recipe.config.ModelOptRecipeBase` rejects a recipe whose
    ``metadata.recipe_type`` contradicts its schema class.
    """
    import yaml

    key = str(recipe_file)
    _seen = (_seen or frozenset()) | {key}

    try:
        declared = peek_declared_schema(recipe_file)
    except ValueError:  # multiple modelopt-schema comments; load_config reports it
        declared = None
    if declared in _RECIPE_SCHEMA_PATHS:
        return _RECIPE_SCHEMA_PATHS[declared]

    if _raw is not _UNPARSED:
        # Supplied by _load_recipe_from_file, which has already parsed this file; only the
        # top-level call can reuse it, the recursion below still reads each import itself.
        raw = _raw
    else:
        try:
            raw = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            return None
    if not isinstance(raw, dict):
        return None

    try:
        return RecipeType(raw["metadata"]["recipe_type"])
    except (TypeError, KeyError, ValueError):
        pass

    for base in _delegated_recipe_paths(raw):
        if str(base) in _seen:
            # Skipping is what stops the recursion, but a caller that ends up with no kind
            # at all needs to know a cycle is *why* -- otherwise it reports "nothing
            # declared anywhere" and sends the author to fix something that is not wrong.
            # Recorded rather than raised: another `$import` may still resolve the kind,
            # and a cycle that does not prevent resolution is not worth mentioning.
            if _cycle is not None:
                _cycle.append((key, str(base)))
            continue
        rtype = _peek_recipe_type(base, _seen, _cycle)
        if rtype is not None:
            return rtype
    return None


def _delegated_recipe_paths(raw: dict) -> list[Path | Traversable]:
    """Resolve the recipe files a raw recipe body delegates to via top-level ``$import``.

    Returns an empty list for an ordinary recipe. Names that do not appear in the
    ``imports`` section, or that do not resolve to a file, are skipped here and reported
    by ``$import`` resolution itself with a better message.
    """
    ref = raw.get("$import")
    imports = raw.get("imports") or {}
    if ref is None or not isinstance(imports, dict):
        return []
    paths = []
    for name in ref if isinstance(ref, list) else [ref]:
        target = imports.get(name)
        if not target:
            continue
        resolved = _resolve_recipe_path(target)
        if resolved.is_file():
            paths.append(resolved)
    return paths


def _load_recipe_from_file(
    recipe_file: Path | Traversable,
    overrides: list[str] | None = None,
) -> ModelOptRecipeBase:
    """Load a recipe from a YAML file, optionally applying dotlist overrides.

    The file has to say what kind of recipe it is -- see :func:`_peek_recipe_type` for
    the three ways to do that -- and to supply the matching body section (``quantize`` /
    ``eagle`` / ``dflash`` / ``medusa``), either directly or through a top-level
    ``$import`` of a recipe that has one.
    """
    # Parsed once and threaded into the peek below, which would otherwise read and parse
    # the same file a second time. A YAMLError is held rather than raised here so the
    # kind-resolution errors below still come first, as they did when the peek owned the
    # only parse -- a malformed file that nonetheless declares its schema in a comment
    # keeps resolving, and one that does not still reports the parse error.
    import yaml

    yaml_error: Exception | None = None
    try:
        raw_top: object = yaml.safe_load(recipe_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raw_top, yaml_error = None, exc

    cycle: list[tuple[str, str]] = []
    rtype = _peek_recipe_type(recipe_file, _cycle=cycle, _raw=raw_top)
    if rtype is None:
        if cycle:
            # The delegation remedy below is the one thing the author cannot do here --
            # they already delegated, and that is the problem. Name the cycle instead.
            edges = ", ".join(f"{src} -> {dst}" for src, dst in cycle)
            raise ValueError(
                f"Recipe file {recipe_file} delegates to a recipe that delegates back to "
                f"it, so its kind cannot be resolved (cycle: {edges}). Break the cycle, or "
                "declare the kind directly with a '# modelopt-schema: "
                "modelopt.recipe.config.ModelOpt<Kind>Recipe' comment."
            )
        raise ValueError(
            f"Recipe file {recipe_file} does not say what kind of recipe it is. Set "
            "'metadata.recipe_type', or declare a '# modelopt-schema: "
            "modelopt.recipe.config.ModelOpt<Kind>Recipe' comment, or delegate to a recipe "
            "that does with a top-level '$import'."
        )
    schema_class = RECIPE_TYPE_TO_CLASS.get(rtype)
    if schema_class is None:
        raise ValueError(f"Unsupported recipe type: {rtype!r}")

    if yaml_error is not None:
        raise yaml_error
    raw = raw_top if isinstance(raw_top, dict) else {}

    # A recipe that delegates inherits the imported recipe's body wholesale, so the two
    # have to be the same kind. Checked here, and against the *declared* kind on both
    # sides, so a mismatch reads as a mismatch -- otherwise it surfaces as whatever
    # pydantic makes of, say, an ``eagle`` section spliced into a PTQ schema.
    for base in _delegated_recipe_paths(raw):
        base_type = _peek_recipe_type(base)
        if base_type is not None and base_type != rtype:
            raise ValueError(
                f"Recipe file {recipe_file} is a {rtype.value!r} recipe but imports "
                f"{base}, which is a {base_type.value!r} recipe. A top-level '$import' "
                "takes over the whole body, so both must be the same kind."
            )

    # Pre-flight check on the *raw* YAML so the user sees a clear loader-level error
    # rather than a generic pydantic missing-field error.  Speculative recipes' body
    # sections have field-level defaults, so this check is what keeps their loader
    # semantics consistent with PTQ.
    required_section = _REQUIRED_SECTION_PER_RECIPE_TYPE.get(rtype)
    if required_section is not None:
        # A recipe may delegate its whole body to another recipe with a top-level
        # ``$import`` and override only ``metadata`` -- see :ref:`recipe-alias`. The
        # body section then arrives during import resolution, so it cannot be required
        # in the raw YAML; pydantic still rejects the result if the import does not
        # supply one.
        if "$import" not in raw and required_section not in raw:
            # Strip only the ``speculative_`` prefix so multi-word non-speculative types
            # (e.g. ``auto_quantize``) keep their full name: AUTO_QUANTIZE, not QUANTIZE.
            kind = rtype.value.removeprefix("speculative_").upper()
            raise ValueError(f"{kind} recipe file {recipe_file} must contain {required_section!r}.")

    # Passing ``schema_type=schema_class`` to ``load_config`` enables typed-list
    # ``$import`` resolution (e.g. ``$import: disable_all`` spliced into
    # ``quantize.quant_cfg`` needs to know the list's element schema is
    # :class:`QuantizerCfgEntry`).  The return value is already a validated schema
    # instance.
    if overrides:
        # Overrides have to be applied before pydantic validation.  Round-trip through
        # ``model_dump()`` so $imports are resolved and the dict has the resolved shape;
        # then splice the dotlist values and re-validate.
        recipe = load_config(recipe_file, schema_type=schema_class)
        data = recipe.model_dump()
        data = _apply_dotlist(data, overrides)
        return schema_class.model_validate(data)

    recipe = load_config(recipe_file, schema_type=schema_class)
    if not isinstance(recipe, schema_class):
        raise ValueError(
            f"Recipe file {recipe_file} must produce a {schema_class.__name__}, "
            f"got {type(recipe).__name__}."
        )
    return recipe


def _find_recipe_section_file(
    recipe_dir: Path | Traversable, section_name: str
) -> Path | Traversable:
    """Find ``<section_name>.yml`` or ``<section_name>.yaml`` in a recipe directory."""
    for suffix in (".yml", ".yaml"):
        candidate = recipe_dir.joinpath(section_name + suffix)
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"Cannot find {section_name} in {recipe_dir}. "
        f"Looked for: {section_name}.yml, {section_name}.yaml"
    )


def _load_recipe_from_dir(recipe_dir: Path | Traversable) -> ModelOptRecipeBase:
    """Load a recipe from a directory containing ``metadata.yml`` and ``quantize.yml``.

    Each file is loaded independently. The file name provides the recipe
    section key: ``metadata.yml`` becomes metadata, and ``quantize.yml`` becomes
    quantize.

    ``metadata.yml``'s kind is resolved the same way a single-file recipe's is (see
    :func:`_peek_recipe_type`), minus the ``$import``-delegation source: a directory
    recipe's metadata has nowhere to delegate from. A ``# modelopt-schema:`` comment is
    checked first, then the (deprecated) ``recipe_type`` field; if both are present,
    :class:`~modelopt.recipe.config.ModelOptRecipeBase` rejects a disagreement.
    """
    metadata_file = _find_recipe_section_file(recipe_dir, "metadata")
    metadata = load_config(metadata_file, schema_type=RecipeMetadataConfig)

    try:
        declared = peek_declared_schema(metadata_file)
    except ValueError:  # multiple modelopt-schema comments; load_config reports it
        declared = None
    # dict.get(declared, ...) is what ruff (SIM401) wants here, but its key type is str,
    # not str | None like `declared` -- mypy rejects that call, so the if/else stays.
    if declared in _RECIPE_SCHEMA_PATHS:  # noqa: SIM401
        rtype = _RECIPE_SCHEMA_PATHS[declared]
    else:
        rtype = metadata.recipe_type
    if rtype is None:
        raise ValueError(
            f"Recipe directory {recipe_dir}: {metadata_file} must set 'recipe_type' or "
            "declare a '# modelopt-schema: modelopt.recipe.config.ModelOpt<Kind>Recipe' "
            "comment. A directory recipe's metadata.yml is the only place its kind can "
            "come from."
        )
    if rtype == RecipeType.PTQ:
        quantize_file = _find_recipe_section_file(recipe_dir, "quantize")
        quantize_cfg = load_config(quantize_file, schema_type=QuantizeConfig)
        return ModelOptPTQRecipe(metadata=metadata, quantize=quantize_cfg)
    raise ValueError(f"Unsupported recipe type: {rtype!r}")
