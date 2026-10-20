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

"""Freeze a drafter export: a read-only, checksummed copy the benchmark serves.

Copies <src> to a ``.partial`` sibling of <dst>, re-hashes it against the source, writes
``SHA256SUMS`` in ``sha256sum -c`` format, makes the tree world-readable and its files
read-only, and renames it to <dst>. A ``dst`` already holding the same files is accepted as
is, so a resumed run can re-freeze safely. ``--verify`` re-hashes ``dst`` against its
``SHA256SUMS``, after checking that file against ``--manifest-sha256`` when given (before a
benchmark that follows a resume). Both return ``manifest_sha256``, the hash of ``SHA256SUMS``
itself, for the caller to record outside the copy.

Same contract as the gates: JSON ``{pass, failure_class, detail, notes, inputs, ...}``;
exit 0 pass, 1 failed, 2 ``USER_CONFIG_ERROR`` (including unreadable or unwritable paths).
Stdlib only. Pure comparison in ``evaluate_copy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

SUMS = "SHA256SUMS"


def hash_file(path):
    """sha256 hex digest of one file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root):
    """{relative path: sha256} of every file under root except SHA256SUMS."""
    hashes = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            if rel != SUMS:
                hashes[rel] = hash_file(path)
    return hashes


def read_sums(root):
    """{relative path: sha256} from root/SHA256SUMS, or None if absent."""
    path = os.path.join(root, SUMS)
    if not os.path.exists(path):
        return None
    sums = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                digest, sep, rel = line.rstrip("\n").partition("  ")
                if not sep:
                    raise ValueError(f"malformed {SUMS} line: {line.rstrip()!r}")
                sums[rel] = digest
    return sums


def evaluate_copy(expected, actual):
    """Problems between two {relative path: sha256} maps; empty if identical."""
    problems = [f"missing {rel}" for rel in sorted(set(expected) - set(actual))]
    problems += [f"unexpected {rel}" for rel in sorted(set(actual) - set(expected))]
    problems += [
        f"checksum differs: {rel}"
        for rel in sorted(set(expected) & set(actual))
        if expected[rel] != actual[rel]
    ]
    return problems


def _is_export(hashes, root):
    weights = [
        rel
        for rel in hashes
        if rel.endswith(".safetensors") and os.path.getsize(os.path.join(root, rel)) > 0
    ]
    return "config.json" in hashes and bool(weights)


def _make_read_only(root):
    # Directories stay writable so a failed or rejected copy can still be removed.
    for dirpath, _, filenames in os.walk(root):
        os.chmod(dirpath, os.stat(dirpath).st_mode | 0o555)
        for name in filenames:
            path = os.path.join(dirpath, name)
            os.chmod(path, (os.stat(path).st_mode | 0o444) & ~0o222)


def _write_sums(root, hashes):
    with open(os.path.join(root, SUMS), "w") as f:
        f.writelines(f"{hashes[rel]}  {rel}\n" for rel in sorted(hashes))


def _result(problems, failure_class, detail, **extra):
    if problems:
        return {
            "pass": False,
            "failure_class": failure_class,
            "detail": "; ".join(problems),
            **extra,
        }
    return {"pass": True, "failure_class": None, "detail": detail, **extra}


def freeze(src, dst):
    """Copy src to dst with checksums. Returns the verdict dict (without inputs)."""
    notes: list[str] = []
    src_hashes = hash_tree(src)
    if not _is_export(src_hashes, src):
        return _result(
            ["source has no config.json and non-empty *.safetensors"],
            "EXPORT_MISSING",
            "",
            notes=notes,
        )
    if os.path.exists(dst):
        recorded = read_sums(dst)
        problems = evaluate_copy(src_hashes, hash_tree(dst))
        if recorded is not None:
            problems += [f"{SUMS}: {p}" for p in evaluate_copy(src_hashes, recorded)]
        else:
            problems.append(f"{dst} exists without {SUMS}")
        if problems:
            return _result(problems, "FREEZE_MISMATCH", "", notes=notes, path=dst)
        notes.append("destination already frozen; not copied again")
        return _result(
            [],
            None,
            "already frozen",
            notes=notes,
            path=dst,
            sha256=src_hashes,
            manifest_sha256=hash_file(os.path.join(dst, SUMS)),
        )

    partial = dst.rstrip("/") + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(src, partial)
    problems = evaluate_copy(src_hashes, hash_tree(partial))
    if problems:
        notes.append(f"bad copy left at {partial}, not published; re-run to retry")
        return _result(problems, "FREEZE_MISMATCH", "", notes=notes, path=dst)
    _write_sums(partial, src_hashes)
    _make_read_only(partial)
    os.rename(partial, dst)
    return _result(
        [],
        None,
        "frozen",
        notes=notes,
        path=dst,
        sha256=src_hashes,
        manifest_sha256=hash_file(os.path.join(dst, SUMS)),
    )


def verify(dst, manifest_sha256=None):
    """Re-hash dst against its SHA256SUMS, first checking SHA256SUMS against manifest_sha256."""
    sums_path = os.path.join(dst, SUMS)
    if not os.path.exists(sums_path):
        return _result([f"no {SUMS} in {dst}"], "FREEZE_MISMATCH", "", notes=[], path=dst)
    actual_manifest = hash_file(sums_path)
    if manifest_sha256 and actual_manifest != manifest_sha256.lower():
        return _result(
            [f"{SUMS} differs from the recorded manifest_sha256"],
            "FREEZE_MISMATCH",
            "",
            notes=[],
            path=dst,
        )
    recorded = read_sums(dst)
    actual = hash_tree(dst)
    problems = evaluate_copy(recorded, actual)
    if not _is_export(actual, dst):
        problems.append("no config.json and non-empty *.safetensors")
    return _result(
        problems,
        "FREEZE_MISMATCH",
        "verified",
        notes=[] if manifest_sha256 else [f"{SUMS} not checked against a recorded manifest_sha256"],
        path=dst,
        sha256=recorded,
        manifest_sha256=actual_manifest,
    )


def main(argv=None):
    """CLI entry point: freeze (or verify) and print the verdict."""
    p = argparse.ArgumentParser(description="Freeze a drafter export with SHA256 checksums")
    p.add_argument("--src", help="exported drafter directory (exported-checkpoint-*)")
    p.add_argument("--dst", required=True, help="frozen destination; must not be inside --src")
    p.add_argument("--verify", action="store_true", help="only re-hash --dst against SHA256SUMS")
    p.add_argument(
        "--manifest-sha256",
        help="with --verify: the manifest_sha256 freeze returned; SHA256SUMS must hash to it",
    )
    args = p.parse_args(argv)
    inputs = {"src": args.src, "dst": args.dst, "manifest_sha256": args.manifest_sha256}

    error = None
    if args.verify:
        if args.src:
            error = "--src is not used with --verify"
        elif not os.path.isdir(args.dst):
            error = f"{args.dst} does not exist"
        elif args.manifest_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.manifest_sha256):
            error = "--manifest-sha256 is not a 64-character hex digest"
    elif args.manifest_sha256:
        error = "--manifest-sha256 is only used with --verify"
    elif not args.src or not os.path.isdir(args.src):
        error = f"source directory {args.src} does not exist"
    else:
        real_src, real_dst = os.path.realpath(args.src), os.path.realpath(args.dst)
        if real_dst == real_src or real_dst.startswith(real_src + os.sep):
            error = "--dst is --src or inside it"
        elif not os.path.isdir(os.path.dirname(os.path.abspath(args.dst))):
            error = f"parent of --dst {args.dst} does not exist"
        elif os.path.isdir(args.dst) and not os.listdir(args.dst):
            error = (
                f"destination {args.dst} is an empty directory; "
                "remove it or pass the path without creating it"
            )

    if not error:
        try:
            if args.verify:
                result = verify(args.dst, args.manifest_sha256)
            else:
                result = freeze(args.src, args.dst)
        except OSError as e:
            error = str(e)
        except ValueError as e:
            result = _result([str(e)], "FREEZE_MISMATCH", "", notes=[], path=args.dst)
    if error:
        result = {"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": error, "notes": []}
    result["inputs"] = inputs
    print(json.dumps(result, indent=2))
    if error:
        return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
