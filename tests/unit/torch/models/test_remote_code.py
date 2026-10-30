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

"""Staging every module of a local checkpoint's remote code, and keeping a patch on it."""

import importlib
import os
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

dmu = pytest.importorskip("transformers.dynamic_module_utils")

from modelopt.torch.models import _remote_code
from modelopt.torch.models._remote_code import keep_patched, stage_full_remote_closure


def test_every_module_of_a_local_checkpoint_is_staged(monkeypatch, tmp_path):
    """transformers stages a remote module and its direct imports only, so the third module of
    a -> b -> c is missing until the wrap copies the checkpoint's other modules. Installing again
    wraps the loader once."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    ckpt.mkdir()
    stage.mkdir()
    for name, body in (
        ("a.py", "from .b import B\n"),
        ("b.py", "from .c import C\n"),
        ("c.py", ""),
    ):
        (ckpt / name).write_text(body)

    def stage_one_hop(path, module_file, *args, **kwargs):
        for name in (module_file, "b.py"):
            shutil.copy(ckpt / name, stage / name)
        return str(stage / module_file)

    monkeypatch.setattr(dmu, "get_cached_module_file", stage_one_hop)
    stage_full_remote_closure()
    wrapped = dmu.get_cached_module_file
    stage_full_remote_closure()

    assert dmu.get_cached_module_file is wrapped
    assert dmu.get_cached_module_file(str(ckpt), "a.py") == str(stage / "a.py")
    assert sorted(p.name for p in stage.iterdir()) == ["a.py", "b.py", "c.py"]


def _stage_only_the_module(monkeypatch, ckpt, stage):
    """Stand in for transformers' staging of the module itself, without its imports."""

    def stage_only_the_module(path, module_file, *args, **kwargs):
        (stage / module_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt / module_file, stage / module_file)
        return str(stage / module_file)

    monkeypatch.setattr(dmu, "get_cached_module_file", stage_only_the_module)
    stage_full_remote_closure()


def test_a_module_is_staged_whole_or_not_at_all(monkeypatch, tmp_path):
    """Every rank stages into the same dir and imports what it finds there, so a module must not
    appear before its copy completes: a copy cut short leaves nothing behind for a rank to take as
    staged, not even its temp file, the other modules are still staged, and the next load stages
    the whole module."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    ckpt.mkdir()
    stage.mkdir()
    (ckpt / "a.py").write_text("from .b import B\n")
    (ckpt / "b.py").write_text("B = 1\n" * 100)
    (ckpt / "c.py").write_text("C = 1\n")
    copy = shutil.copy

    def copy_cut_short(src, dst):
        if not src.endswith("b.py"):
            return copy(src, dst)
        with open(src) as f, open(dst, "w") as out:
            out.write(f.read()[:10])
        raise OSError(28, "No space left on device")

    _stage_only_the_module(monkeypatch, ckpt, stage)
    with monkeypatch.context() as m:
        m.setattr(_remote_code, "shutil", SimpleNamespace(copy=copy_cut_short))
        with pytest.warns(UserWarning, match=r"b\.py .*No space left"):
            dmu.get_cached_module_file(str(ckpt), "a.py")
    assert sorted(p.name for p in stage.iterdir()) == ["a.py", "c.py"]

    dmu.get_cached_module_file(str(ckpt), "a.py")
    assert (stage / "b.py").read_text() == (ckpt / "b.py").read_text()


def test_a_staged_module_that_differs_from_the_checkpoints_is_staged_again(monkeypatch, tmp_path):
    """As transformers does for the modules it stages itself: a stage dir can hold another copy of
    a module, from a checkpoint staged there before or from before the checkpoint's own was
    edited, and the load must import the checkpoint's."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    ckpt.mkdir()
    stage.mkdir()
    (ckpt / "a.py").write_text("from .b import B\n")
    (ckpt / "b.py").write_text("B = 2\n")
    (stage / "b.py").write_text("B = 1\n")

    _stage_only_the_module(monkeypatch, ckpt, stage)
    dmu.get_cached_module_file(str(ckpt), "a.py")

    assert (stage / "b.py").read_text() == "B = 2\n"


@pytest.mark.parametrize("layout", ["glob-metacharacters", "package-marker", "module-in-subdir"])
def test_the_modules_staged_are_the_ones_beside_the_module(monkeypatch, tmp_path, layout):
    """Wherever the module is and whatever its path holds, the modules staged with it are the ones
    in its own directory, and never the checkpoint's ``__init__.py``: the stage dir's package
    marker is transformers', and a checkpoint's own would run as the package of every staged
    module."""
    ckpt = tmp_path / ("ckpt[v2]" if layout == "glob-metacharacters" else "ckpt")
    stage = tmp_path / "stage"
    module = "sub/a.py" if layout == "module-in-subdir" else "a.py"
    (ckpt / "sub").mkdir(parents=True)
    stage.mkdir()
    (stage / "__init__.py").write_text("")
    (ckpt / "__init__.py").write_text("import a_module_only_the_checkpoint_has\n")
    (ckpt / "a.py").write_text("from .b import B\n")
    (ckpt / "b.py").write_text("B = 1\n")
    (ckpt / "sub" / "a.py").write_text("from .b import B\n")
    (ckpt / "sub" / "b.py").write_text("B = 2\n")

    _stage_only_the_module(monkeypatch, ckpt, stage)
    dmu.get_cached_module_file(str(ckpt), module)

    staged_b = stage / module.replace("a.py", "b.py")
    assert staged_b.read_text() == (ckpt / module.replace("a.py", "b.py")).read_text()
    assert (stage / "__init__.py").read_text() == ""


def test_a_module_staged_after_its_dir_was_imported_from_is_found(monkeypatch, tmp_path):
    """The import system caches what a directory holds, and only notices a change when the
    directory's modification time does; on a filesystem that stamps it coarsely, a module staged
    within the same tick stays hidden from the import that needs it, unless the caches are
    invalidated, as transformers does after its own copies."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "staged_pkg"
    ckpt.mkdir()
    stage.mkdir()
    (stage / "__init__.py").write_text("")
    (stage / "a.py").write_text("")
    (ckpt / "a.py").write_text("")
    (ckpt / "b.py").write_text("B = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "staged_pkg", raising=False)
    try:
        importlib.import_module("staged_pkg.a")  # caches what the stage dir holds
        mtime = os.stat(stage).st_mtime_ns

        _stage_only_the_module(monkeypatch, ckpt, stage)
        dmu.get_cached_module_file(str(ckpt), "a.py")
        os.utime(stage, ns=(mtime, mtime))  # a stamp too coarse to tell the copy apart

        assert importlib.import_module("staged_pkg.b").B == 1
    finally:
        for name in [n for n in sys.modules if n == "staged_pkg" or n.startswith("staged_pkg.")]:
            del sys.modules[name]


@pytest.fixture
def remote_kimi(monkeypatch, tmp_path):
    """A checkpoint whose remote modeling module defines ``DeepseekV3ForCausalLM`` and the VLM class
    and imports a class from a sibling module, plus a patch pointing the former at a native
    stand-in; the modules transformers loads from it are dropped afterwards."""
    modules = tmp_path / "modules"
    modules.mkdir()
    monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(modules))
    monkeypatch.setattr(dmu, "get_class_in_module", dmu.get_class_in_module)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    # The staged code imports as the package transformers_modules, from the modules cache on
    # sys.path: neither another test's cache there nor the package imported from it may shadow
    # this one.
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.syspath_prepend(str(modules))
    for name in [
        n for n in sys.modules if n.partition(".")[0] == dmu.TRANSFORMERS_DYNAMIC_MODULE_NAME
    ]:
        monkeypatch.delitem(sys.modules, name)
    loaded = set(sys.modules)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "modeling_kimi_k25.py").write_text(
        "from .modeling_vision import VisionTower\n\n\n"
        "class DeepseekV3ForCausalLM:\n    pass\n\n\nclass KimiK25ForConditionalGeneration:\n    pass\n"
    )
    (ckpt / "modeling_vision.py").write_text("class VisionTower:\n    pass\n")
    (tmp_path / "same_ckpt").symlink_to(ckpt)
    native = type("DeepseekV3ForCausalLM", (), {})

    def to_native(module):
        changed = module.DeepseekV3ForCausalLM is not native
        module.DeepseekV3ForCausalLM = native
        return changed

    yield SimpleNamespace(
        ckpt=ckpt, same_ckpt=tmp_path / "same_ckpt", native=native, patch=to_native
    )
    for name in set(sys.modules) - loaded:
        if name.partition(".")[0] == dmu.TRANSFORMERS_DYNAMIC_MODULE_NAME:
            del sys.modules[name]


@pytest.mark.parametrize("load_again", ["forced", "another-path"])
def test_a_patch_stays_on_a_module_transformers_loads_again(remote_kimi, capsys, load_again):
    """transformers re-executes a remote module it loads again (when the files it hashes change, or
    forced), and loads a second copy under another name for another path to the same checkpoint;
    either way the module's globals are fresh. The patch must be back on the module the class is
    resolved from, which is the one from_pretrained builds from, and the re-patch is reported."""
    ref = "modeling_kimi_k25.KimiK25ForConditionalGeneration"
    first = dmu.get_class_from_dynamic_module(ref, str(remote_kimi.ckpt))
    patched = keep_patched(".modeling_kimi_k25", remote_kimi.patch)
    if load_again == "forced":
        again = dmu.get_class_from_dynamic_module(ref, str(remote_kimi.ckpt), force_download=True)
    else:
        again = dmu.get_class_from_dynamic_module(ref, str(remote_kimi.same_ckpt))

    assert patched == [first.__module__]
    assert again is not first  # a fresh module, not the one patched above
    assert sys.modules[again.__module__].DeepseekV3ForCausalLM is remote_kimi.native
    assert f"re-patched {again.__module__}" in capsys.readouterr().out


@pytest.mark.parametrize(
    "resolved", ["DeepseekV3ForCausalLM", "VisionTower"], ids=["patched-name", "imported-class"]
)
def test_resolving_any_class_from_a_module_loaded_again_puts_its_patch_back(remote_kimi, resolved):
    """The class handed back is the module's as patched, even when the patch rebinds that very name,
    which transformers read before the patch ran; and a class the module imports from a sibling,
    which another module defines, still puts the patch back on the module loaded again."""
    vlm = dmu.get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration", str(remote_kimi.ckpt)
    )
    keep_patched(".modeling_kimi_k25", remote_kimi.patch)

    cls = dmu.get_class_from_dynamic_module(
        f"modeling_kimi_k25.{resolved}", str(remote_kimi.ckpt), force_download=True
    )

    module = sys.modules[vlm.__module__]
    assert module.DeepseekV3ForCausalLM is remote_kimi.native
    assert cls is getattr(module, resolved)


@pytest.mark.parametrize(
    "defined_in",
    [
        "transformers.models.kimi_k25.modeling_kimi_k25",
        "transformers_modules.Kimi.4567cdef.modeling_kimi_k25",  # not in sys.modules
    ],
    ids=["port", "not-loaded"],
)
def test_only_remote_modules_are_patched(monkeypatch, defined_in):
    """transformers' own port of a model can share its remote module's name (from 5.13,
    ``transformers.models.kimi_k25.modeling_kimi_k25``); a patch for the checkpoint's code must
    leave it alone. So must the re-patch when transformers resolves a class, which reaches the
    module defining it: a remote module can import its class from the port, or from a module
    sys.modules does not hold."""
    remote = ModuleType("transformers_modules.Kimi.0123abcd.modeling_kimi_k25")
    port = ModuleType("transformers.models.kimi_k25.modeling_kimi_k25")
    for module in (remote, port):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    cls = type("KimiK25ForConditionalGeneration", (), {"__module__": defined_in})
    monkeypatch.setattr(dmu, "get_class_in_module", lambda *args, **kwargs: cls)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    seen = []

    assert keep_patched(".modeling_kimi_k25", seen.append) == [remote.__name__]
    module_path = f"{remote.__name__.replace('.', '/')}.py"  # the remote module, as loaded
    assert dmu.get_class_in_module(cls.__name__, module_path) is cls
    assert seen and all(module is remote for module in seen)


@pytest.mark.parametrize("loaded", [True, False], ids=["patch-refuses", "nothing-loaded"])
def test_a_patch_that_does_not_apply_leaves_the_loader_alone(monkeypatch, loaded):
    """A patch that refuses the remote code it finds, as the Kimi adapters' does for code it cannot
    patch, or no loaded module to patch, installs nothing: a later load in the process, which a
    caller may retry another way, is not refused again from inside transformers' loader."""
    monkeypatch.setattr(dmu, "get_class_in_module", dmu.get_class_in_module)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    loader = dmu.get_class_in_module
    if loaded:
        remote = ModuleType("transformers_modules.Repo.0123abcd.modeling_refused")
        monkeypatch.setitem(sys.modules, remote.__name__, remote)

    def refuse(module):
        raise RuntimeError("not the remote code this patch changes")

    if loaded:
        with pytest.raises(RuntimeError, match="not the remote code"):
            keep_patched(".modeling_refused", refuse)
    else:
        assert keep_patched(".modeling_refused", refuse) == []
    assert dmu.get_class_in_module is loader
    assert _remote_code._KEPT_PATCHES == {}
