"""Read-only access to the attributed SZL formula and quant atlas."""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_formula_atlas() -> dict[str, Any]:
    resource = files(__package__).joinpath("formula_atlas.v1.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if value.get("schema") != "szl.formula-quant-atlas/v1":
        raise ValueError("unsupported formula atlas schema")
    return value


def load_locked_formula_reference() -> dict[str, Any]:
    """Load the revision-bound external Lean reference shipped with the kernel.

    The reference reports axiom evidence for named Lean declarations.  It does
    not establish a Lean-to-Python refinement proof or execution authority.
    """
    resource = files(__package__).joinpath("proof_reference.v1.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if value.get("schema") != "szl.formulas.locked-formula-reference/v1":
        raise ValueError("unsupported locked-formula reference schema")
    return value


def atlas_summary() -> dict[str, Any]:
    return dict(load_formula_atlas()["summary"])


def quant_domains() -> list[dict[str, Any]]:
    return [dict(row) for row in load_formula_atlas()["quant_domains"]]


def list_attributed_formulas() -> list[dict[str, Any]]:
    return [dict(row) for row in load_formula_atlas()["attributed_formulas"]]
