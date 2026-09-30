#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish exact GitHub Python blobs through a single, additive Hub writer.

Only the fixed package slots and a source binding are managed. Curated cards,
quarantined serialization, weights, and other Hub assets retain their Git blobs.
Ordinary fast-forward Git pushes support both legacy model and kernel repos;
no force flags, deletion operations, repository creation, or settings changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable

SOURCE_REPOSITORY = "szl-holdings/szl-formulas"
SOURCE_URL = "https://github.com/szl-holdings/szl-formulas.git"
TARGETS = (
    ("model", "https://huggingface.co/SZLHOLDINGS/szl-formulas"),
    ("kernel", "https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas"),
)
PACKAGE_FILES = (
    "__init__.py", "_formulas.py", "_composer.py", "_aggregators.py",
    "atlas.py", "formula_atlas.v1.json", "metadata.json",
)
VARIANTS = ("torch-universal", "torch-cpu")
BINDING_PATH = "SZL_SOURCE_BINDING.json"
SHA40 = re.compile(r"[0-9a-f]{40}")


class PublicationError(RuntimeError):
    """A missing source, changed parent, or failed readback is never publication."""


def require_sha(value: str) -> str:
    if not isinstance(value, str) or not SHA40.fullmatch(value) or value == "0" * 40:
        raise PublicationError("an exact nonzero 40-character source/parent SHA is required")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def git(root: Path, *args: str, data: bytes | None = None, env: dict | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null", *args],
        cwd=root, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, timeout=180, check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace")[-2000:]
        token = (env or os.environ).get("HF_TOKEN", "")
        if token:
            message = message.replace(token, "[REDACTED]")
        raise PublicationError(f"Git operation failed: {message}")
    return result.stdout


def build_payload(root: Path, revision: str) -> dict[str, bytes]:
    require_sha(revision)
    if git(root, "rev-parse", "HEAD").decode().strip() != revision:
        raise PublicationError("checkout differs from the admitted source revision")
    payload, sources = {}, {}
    for leaf in PACKAGE_FILES:
        source = f"torch-ext/szl_formulas/{leaf}"
        value = git(root, "show", f"{revision}:{source}")
        if not value or len(value) > 2_000_000:
            raise PublicationError(f"missing or oversized source blob: {source}")
        for variant in VARIANTS:
            destination = f"build/{variant}/szl_formulas/{leaf}"
            payload[destination] = value
            sources[destination] = source
    for variant in VARIANTS:
        destination = f"build/{variant}/metadata.json"
        payload[destination] = encoded({
            "name": "szl-formulas", "version": 1,
            "id": f"_szl_formulas_{variant.replace('-', '_')}_{revision[:12]}",
            "license": "Apache-2.0", "python-depends": [],
            "backend": {"type": "cpu"}, "universal": variant == "torch-universal",
            "kernel-sha": revision, "kernel-dirty": False,
        })
        sources[destination] = "generated:source-bound-variant-metadata/v1"
    payload[BINDING_PATH] = encoded({
        "schema": "szl.hf-python-source-binding/v1",
        "source_repository": SOURCE_REPOSITORY, "source_revision": revision,
        "source_url": f"https://github.com/{SOURCE_REPOSITORY}/tree/{revision}",
        "artifact_kind": "python-software-kernel", "trained_weights_present": False,
        "locked_proven_count": 8, "lambda_status": "Conjecture 1; advisory only",
        "files": {path: {"sha256": digest(value), "source": sources[path]}
                  for path, value in sorted(payload.items())},
    })
    return payload


class GitTarget:
    def __init__(self, kind: str, url: str, root: Path, env: dict | None = None):
        self.kind, self.url, self.root = kind, url, root
        self.env = dict(os.environ if env is None else env)
        self.env.update(
            GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1",
            GIT_AUTHOR_NAME="SZL canonical publisher", GIT_COMMITTER_NAME="SZL canonical publisher",
            GIT_AUTHOR_EMAIL="noreply@szl-holdings.invalid", GIT_COMMITTER_EMAIL="noreply@szl-holdings.invalid",
        )
        root.mkdir(parents=True)
        git(root, "init", env=self.env)
        git(root, "remote", "add", "origin", url, env=self.env)

    def run(self, *args: str, data: bytes | None = None) -> bytes:
        return git(self.root, *args, data=data, env=self.env)

    def remote_head(self) -> str:
        lines = self.run("ls-remote", "origin", "refs/heads/main").decode().splitlines()
        if len(lines) != 1:
            raise PublicationError("Hub main is missing or ambiguous")
        return require_sha(lines[0].split()[0])

    def tree(self, revision: str) -> dict[str, tuple[str, str, str]]:
        entries = {}
        for item in self.run("ls-tree", "-rz", revision).split(b"\0"):
            if item:
                header, path = item.split(b"\t", 1)
                entries[path.decode("utf-8")] = tuple(header.decode().split())
        return entries

    def prepare(self, parent: str, payload: dict[str, bytes], source: str) -> dict:
        require_sha(parent)
        require_sha(source)
        if self.remote_head() != parent:
            raise PublicationError(f"{self.kind} parent moved before preparation")
        self.run("fetch", "--depth=1", "origin", "refs/heads/main")
        if self.run("rev-parse", "FETCH_HEAD").decode().strip() != parent:
            raise PublicationError(f"{self.kind} parent moved during fetch")
        before = self.tree(parent)
        self.run("read-tree", parent)
        changes = []
        for path, value in sorted(payload.items()):
            old = before.get(path)
            if old is not None and (old[0] not in {"100644", "100755"} or old[1] != "blob"):
                raise PublicationError("managed slot is not a regular file")
            oid = self.run("hash-object", "-w", "--stdin", data=value).decode().strip()
            mode = old[0] if old else "100644"
            if old != (mode, "blob", oid):
                changes.append({"path": path, "before_blob": old[2] if old else None,
                                "after_blob": oid, "after_sha256": digest(value)})
            self.run("update-index", "--add", "--cacheinfo", f"{mode},{oid},{path}")
        tree = self.run("write-tree").decode().strip()
        if changes:
            message = (
                f"Publish szl-formulas Python source {source}\n\n"
                f"GitHub source: {SOURCE_REPOSITORY}@{source}\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
            ).encode()
            candidate = self.run("-c", "commit.gpgsign=false", "commit-tree", tree, "-p", parent, data=message).decode().strip()
        else:
            candidate = parent
        unmanaged = {path: value for path, value in before.items() if path not in payload}
        after = self.tree(candidate)
        if {path: value for path, value in after.items() if path not in payload} != unmanaged:
            raise PublicationError("unmanaged Hub tree changed during preparation")
        return {
            "target": self.kind, "url": self.url, "parent_revision": parent,
            "candidate_revision": candidate, "changes": changes,
            "unmanaged_blobs": unmanaged, "delete_operations": 0,
            "readback_verified": False, "unmanaged_blobs_preserved": False,
        }

    def publish(self, row: dict, payload: dict[str, bytes]) -> None:
        if self.remote_head() != row["parent_revision"]:
            raise PublicationError(f"{self.kind} parent moved before push")
        candidate = row["candidate_revision"]
        if candidate != row["parent_revision"]:
            self.run("push", "origin", f"{candidate}:refs/heads/main")
        row["push_completed"] = True
        self.run("fetch", "--depth=1", "origin", "refs/heads/main")
        observed = self.run("rev-parse", "FETCH_HEAD").decode().strip()
        row["observed_revision"] = observed
        if observed != candidate or self.remote_head() != candidate:
            raise PublicationError("Hub head differs from the published revision on readback")
        after = self.tree(observed)
        if {path: value for path, value in after.items() if path not in payload} != row["unmanaged_blobs"]:
            raise PublicationError("unmanaged Hub blobs differ on readback")
        readback = {}
        for path, expected in sorted(payload.items()):
            actual = self.run("show", f"{observed}:{path}")
            if actual != expected:
                raise PublicationError(f"published readback differs: {path}")
            readback[path] = digest(actual)
        row.update(readback_verified=True, unmanaged_blobs_preserved=True, readback_sha256=readback)

    def preflight_write(self, row: dict) -> None:
        # Dry-run exercises the authenticated receive-pack endpoint for each
        # target before either target's refs are updated. It changes no Hub ref.
        self.run("push", "--dry-run", "origin", f"{row['candidate_revision']}:refs/heads/main")
        row["write_preflight_passed"] = True


def publish_targets(targets, parents, payload, source, source_guard: Callable, rows: list) -> None:
    # Complete both target preflights before the first remote write. A later
    # provider failure is PARTIAL/failed, with all already-written rows retained.
    prepared = [target.prepare(parent, payload, source) for target, parent in zip(targets, parents, strict=True)]
    rows.extend(prepared)
    for target, row in zip(targets, prepared, strict=True):
        target.preflight_write(row)
    for target, row in zip(targets, prepared, strict=True):
        source_guard()
        target.publish(row, payload)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--model-parent", required=True)
    parser.add_argument("--kernel-parent", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, default=Path("reports/hf-runtime-publication.json"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    receipt = {"schema": "szl.hf-python-publication-receipt/v1", "complete": False,
               "state": "HOLD", "source_revision": args.expected_source_revision,
               "targets": [], "delete_operations": 0, "secret_values_recorded": False}
    try:
        source = require_sha(args.expected_source_revision)
        payload = build_payload(root, source)
        receipt["payload_sha256"] = {path: digest(value) for path, value in sorted(payload.items())}
        if args.apply and (
            os.environ.get("GITHUB_REPOSITORY") != SOURCE_REPOSITORY
            or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.environ.get("GITHUB_SHA") != source
            or not os.environ.get("HF_TOKEN")
        ):
            raise PublicationError("apply requires the admitted source-owned main workflow and HF credential")
        def guard():
            current = git(root, "ls-remote", SOURCE_URL, "refs/heads/main").decode().split()
            if not current or current[0] != source:
                raise PublicationError("GitHub source moved before publication")
        guard()
        with tempfile.TemporaryDirectory(prefix="szl-formulas-publish-") as scratch:
            directory = Path(scratch)
            env = dict(os.environ)
            if args.apply:
                askpass = directory / "askpass.sh"
                askpass.write_text('#!/bin/sh\ncase "$1" in\n*Username*) printf "%s\\n" hf_user;;\n*Password*) printf "%s\\n" "$HF_TOKEN";;\n*) exit 1;;\nesac\n')
                askpass.chmod(0o700)
                env["GIT_ASKPASS"] = str(askpass)
            targets = [GitTarget(kind, url, directory / kind, env) for kind, url in TARGETS]
            parents = [args.model_parent, args.kernel_parent]
            if args.apply:
                publish_targets(targets, parents, payload, source, guard, receipt["targets"])
                guard()
                receipt.update(complete=True, state="MEASURED")
            else:
                receipt["targets"] = [target.prepare(parent, payload, source)
                                      for target, parent in zip(targets, parents, strict=True)]
                receipt["state"] = "PLAN_ONLY"
    except Exception as exc:
        receipt["error"] = type(exc).__name__
        receipt["detail"] = str(exc).replace(os.environ.get("HF_TOKEN") or "\0", "[REDACTED]")
        receipt["state"] = "PARTIAL" if any(row.get("push_completed") for row in receipt["targets"]) else "HOLD"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_bytes(encoded(receipt))
    print(json.dumps({"state": receipt["state"], "complete": receipt["complete"],
                      "receipt": str(args.receipt), "source_revision": args.expected_source_revision}))
    return 0 if receipt["state"] in {"MEASURED", "PLAN_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
