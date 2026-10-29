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


def test_a_module_is_staged_whole_or_not_at_all(monkeypatch, tmp_path):
    """Every rank stages into the same dir and imports what it finds there, so a module must not
    appear before its copy completes: a copy cut short leaves nothing behind for a rank to take as
    staged, and the next load stages the whole module."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    ckpt.mkdir()
    stage.mkdir()
    (ckpt / "a.py").write_text("from .b import B\n")
    (ckpt / "b.py").write_text("B = 1\n" * 100)

    def stage_only_the_module(path, module_file, *args, **kwargs):
        shutil.copy(ckpt / module_file, stage / module_file)
        return str(stage / module_file)

    def copy_cut_short(src, dst):
        with open(src) as f, open(dst, "w") as out:
            out.write(f.read()[:10])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dmu, "get_cached_module_file", stage_only_the_module)
    stage_full_remote_closure()
    with monkeypatch.context() as m:
        m.setattr(_remote_code, "shutil", SimpleNamespace(copy=copy_cut_short))
        with pytest.warns(UserWarning, match="No space left"):
            dmu.get_cached_module_file(str(ckpt), "a.py")
    assert not (stage / "b.py").exists()

    dmu.get_cached_module_file(str(ckpt), "a.py")
    assert (stage / "b.py").read_text() == (ckpt / "b.py").read_text()


@pytest.mark.parametrize("load_again", ["forced", "another-path"])
def test_a_patch_stays_on_a_module_transformers_loads_again(
    monkeypatch, tmp_path, capsys, load_again
):
    """transformers re-executes a remote module it loads again (when the files it hashes change, or
    forced), and loads a second copy under another name for another path to the same checkpoint;
    either way the module's globals are fresh. The patch must be back on the module the class is
    resolved from, which is the one from_pretrained builds from, and the re-patch is reported."""
    monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    monkeypatch.setattr(dmu, "get_class_in_module", dmu.get_class_in_module)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    loaded = set(sys.modules)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "modeling_kimi_k25.py").write_text(
        "class DeepseekV3ForCausalLM:\n    pass\n\n\nclass KimiK25ForConditionalGeneration:\n    pass\n"
    )
    (tmp_path / "same_ckpt").symlink_to(ckpt)
    native = type("DeepseekV3ForCausalLM", (), {})

    def to_native(module):
        changed = module.DeepseekV3ForCausalLM is not native
        module.DeepseekV3ForCausalLM = native
        return changed

    ref = "modeling_kimi_k25.KimiK25ForConditionalGeneration"
    try:
        first = dmu.get_class_from_dynamic_module(ref, str(ckpt))
        patched = keep_patched(".modeling_kimi_k25", to_native)
        if load_again == "forced":
            again = dmu.get_class_from_dynamic_module(ref, str(ckpt), force_download=True)
        else:
            again = dmu.get_class_from_dynamic_module(ref, str(tmp_path / "same_ckpt"))

        assert patched == [first.__module__]
        assert again is not first  # a fresh module, not the one patched above
        assert sys.modules[again.__module__].DeepseekV3ForCausalLM is native
        assert f"re-patched {again.__module__}" in capsys.readouterr().out
    finally:
        for name in set(sys.modules) - loaded:
            del sys.modules[name]


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
    assert seen == [remote]


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
