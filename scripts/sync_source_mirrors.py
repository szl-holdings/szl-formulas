#!/usr/bin/env python3
"""Synchronize checked-in package mirrors from the canonical torch-ext source.

The checked-in build/torch-universal tree is a source mirror only. Hub variant
metadata and source bindings are generated separately by publish_hf_runtime.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "torch-ext" / "szl_formulas"
MIRROR = ROOT / "build" / "torch-universal" / "szl_formulas"
FILES = (
    "__init__.py",
    "_aggregators.py",
    "_composer.py",
    "_formulas.py",
    "atlas.py",
    "formula_atlas.v1.json",
    "metadata.json",
)


def drift() -> list[str]:
    problems: list[str] = []
    for name in FILES:
        source = SOURCE / name
        mirror = MIRROR / name
        if not source.is_file():
            problems.append(f"missing canonical source: {source.relative_to(ROOT)}")
            continue
        if not mirror.is_file():
            problems.append(f"missing checked-in mirror: {mirror.relative_to(ROOT)}")
            continue
        if source.read_bytes() != mirror.read_bytes():
            problems.append(f"mirror differs: {mirror.relative_to(ROOT)}")
    extras = sorted(
        path.name
        for path in MIRROR.iterdir()
        if path.is_file() and path.name not in FILES
    )
    problems.extend(f"unexpected mirror file: build/torch-universal/szl_formulas/{name}" for name in extras)
    return problems


def sync() -> None:
    MIRROR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (MIRROR / name).write_bytes((SOURCE / name).read_bytes())
    for path in MIRROR.iterdir():
        if path.is_file() and path.name not in FILES:
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        problems = drift()
        if problems:
            raise SystemExit("\n".join(problems))
        print("checked-in source mirror matches canonical torch-ext bytes")
        return 0
    sync()
    problems = drift()
    if problems:
        raise SystemExit("\n".join(problems))
    print("synchronized checked-in source mirror")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
