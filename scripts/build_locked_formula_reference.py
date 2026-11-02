#!/usr/bin/env python3
"""Build a fail-closed reference to exact external Lean axiom evidence.

This script does not run Lean and does not claim that a Python function refines
the named theorem.  It validates two byte-pinned files exported from the
``lutar-lean`` workflow, records the validation level actually reached, and
keeps stronger proof checks explicitly ``NOT_RUN``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA = "szl.formulas.locked-formula-reference/v1"
EVIDENCE_SCHEMA = "szl.lutar.locked-formula-evidence/v1"
RECEIPT_SCHEMA = "szl.papers.external-evidence-receipt/v1"
SOURCE_REPOSITORY = "szl-holdings/lutar-lean"
SOURCE_COMMIT = "c4c42baf107f2cd5ec11ac3958fbd8fd06a875c8"
SOURCE_TREE = "d6959371d2654c87dce8b18d5e21d40d0c7a6b57"
EVIDENCE_FILE = (
    "evidence/lutar-lean/"
    "locked-formulas.c4c42baf107f2cd5ec11ac3958fbd8fd06a875c8.json"
)
RECEIPT_FILE = EVIDENCE_FILE.removesuffix(".json") + ".receipt.json"
EVIDENCE_SHA256 = "sha256:10da7a74357f5d23dee02f19dbc8a5475d10f336b227d7b909eb83ffd29ca33c"
RECEIPT_SHA256 = "sha256:e732886028b4a53714ef2931fad7a9b4809e6a4ae6909f91b0eb2514afec6e78"
EVIDENCE_DIGEST = "sha256:d99da0d0215263681991d90a3b382c3247f6f4feb1cfc5ae4b55370c73f68bc9"
LOCKED_IDS = ["F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"]
TRUST_BASE = ["propext", "Classical.choice", "Quot.sound"]
CARRIER_COUNT = 24


class ReferenceError(ValueError):
    """Pinned evidence does not satisfy the reference contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceError(f"{path} root must be an object")
    return value


def _sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReferenceError(f"cannot hash {path}: {exc}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_evidence(path: Path) -> dict[str, Any]:
    actual_sha = _sha256(path)
    if actual_sha != EVIDENCE_SHA256:
        raise ReferenceError(
            f"vendored Lean evidence SHA-256 drift: {actual_sha} != {EVIDENCE_SHA256}"
        )
    evidence = _load_json(path)
    expected_fields = {
        "schema": EVIDENCE_SCHEMA,
        "repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "input_files_match_head": True,
        "lean_toolchain": "leanprover/lean4:v4.34.1",
        "kernel_query_tool": "Lean.collectAxioms",
        "carrier_theorem_count": CARRIER_COUNT,
        "locked_ids": LOCKED_IDS,
        "trust_base": TRUST_BASE,
        "evidence_digest": EVIDENCE_DIGEST,
    }
    for field, expected in expected_fields.items():
        if evidence.get(field) != expected:
            raise ReferenceError(
                f"Lean evidence {field} drift: {evidence.get(field)!r} != {expected!r}"
            )

    core = deepcopy(evidence)
    observed_digest = core.pop("evidence_digest", None)
    if observed_digest != _canonical_digest(core):
        raise ReferenceError("Lean evidence canonical digest does not verify")

    claims = evidence.get("claims")
    if not isinstance(claims, list):
        raise ReferenceError("Lean evidence claims must be a list")
    if [row.get("claim_id") for row in claims if isinstance(row, dict)] != LOCKED_IDS:
        raise ReferenceError("Lean evidence claims do not exactly match the locked IDs")

    seen: set[str] = set()
    carrier_total = 0
    for row in claims:
        if not isinstance(row, dict) or not isinstance(row.get("carriers"), list):
            raise ReferenceError("each Lean evidence claim must have a carrier list")
        for carrier in row["carriers"]:
            carrier_total += 1
            if not isinstance(carrier, dict):
                raise ReferenceError("carrier must be an object")
            theorem = carrier.get("theorem")
            axioms = carrier.get("axioms")
            if (
                not isinstance(theorem, str)
                or not theorem.startswith("Puriq.Formula.Proved.")
                or theorem in seen
            ):
                raise ReferenceError(f"invalid or duplicate carrier theorem {theorem!r}")
            seen.add(theorem)
            if carrier.get("allowed") is not True:
                raise ReferenceError(f"{theorem}: carrier is not allowed")
            if not isinstance(axioms, list) or not all(
                isinstance(axiom, str) for axiom in axioms
            ):
                raise ReferenceError(f"{theorem}: axioms must be a string list")
            if len(axioms) != len(set(axioms)) or not set(axioms).issubset(TRUST_BASE):
                raise ReferenceError(f"{theorem}: axiom set is outside the pinned trust base")
            if "sorryAx" in axioms or any("_native" in axiom for axiom in axioms):
                raise ReferenceError(f"{theorem}: incomplete or native axiom is not admitted")
    if carrier_total != CARRIER_COUNT:
        raise ReferenceError(
            f"expected {CARRIER_COUNT} carriers, found {carrier_total}"
        )
    return evidence


def _validate_receipt(path: Path) -> dict[str, Any]:
    actual_sha = _sha256(path)
    if actual_sha != RECEIPT_SHA256:
        raise ReferenceError(
            f"vendored evidence receipt SHA-256 drift: {actual_sha} != {RECEIPT_SHA256}"
        )
    receipt = _load_json(path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ReferenceError("unexpected external evidence receipt schema")
    if receipt.get("source_repository") != SOURCE_REPOSITORY:
        raise ReferenceError("receipt source repository drift")
    if receipt.get("source_commit") != SOURCE_COMMIT:
        raise ReferenceError("receipt source commit drift")
    if receipt.get("source_tree") != SOURCE_TREE:
        raise ReferenceError("receipt source tree drift")
    workflow = receipt.get("workflow")
    artifact = receipt.get("artifact")
    if not isinstance(workflow, dict) or workflow.get("conclusion") != "success":
        raise ReferenceError("receipt does not record a successful source workflow")
    if not isinstance(artifact, dict):
        raise ReferenceError("receipt artifact must be an object")
    if artifact.get("member_sha256") != EVIDENCE_SHA256:
        raise ReferenceError("receipt member digest does not bind the evidence bytes")
    if artifact.get("member_path") != "locked-formulas.evidence.json":
        raise ReferenceError("receipt artifact member path drift")
    authority = receipt.get("authority")
    if authority != {
        "publication": False,
        "deployment": False,
        "runtime": False,
        "paper_claim_promotion": False,
    }:
        raise ReferenceError("receipt must grant no publication, deployment, runtime, or paper authority")
    return receipt


def build_contract(evidence_path: Path, receipt_path: Path) -> dict[str, Any]:
    evidence = _validate_evidence(evidence_path)
    receipt = _validate_receipt(receipt_path)
    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "EXTERNAL_LEAN_AXIOM_EVIDENCE_OBSERVED_AT_PIN",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
            "lean_toolchain": evidence["lean_toolchain"],
            "evidence_file": EVIDENCE_FILE,
            "evidence_file_sha256": EVIDENCE_SHA256,
            "evidence_digest": EVIDENCE_DIGEST,
            "receipt_file": RECEIPT_FILE,
            "receipt_file_sha256": RECEIPT_SHA256,
            "workflow": {
                "name": receipt["workflow"]["name"],
                "run_id": receipt["workflow"]["run_id"],
                "conclusion": receipt["workflow"]["conclusion"],
            },
            "artifact": {
                "id": receipt["artifact"]["id"],
                "archive_digest": receipt["artifact"]["archive_digest"],
                "member_sha256": receipt["artifact"]["member_sha256"],
            },
        },
        "validation_ladder": {
            "build_and_elaboration": "OBSERVED_AT_PIN_IN_REFERENCED_WORKFLOW",
            "axiom_enumeration": "OBSERVED_AT_PIN_VIA_LEAN_COLLECT_AXIOMS",
            "lean4checker_fresh": "NOT_RUN",
            "comparator_external_checkers": "NOT_RUN",
            "guidance": [
                "https://lean-lang.org/doc/reference/latest/ValidatingProofs/",
                "https://lean-lang.org/doc/reference/latest/Axioms/",
            ],
        },
        "locked_ids": list(LOCKED_IDS),
        "carrier_theorem_count": CARRIER_COUNT,
        "trust_base": list(TRUST_BASE),
        "claims": deepcopy(evidence["claims"]),
        "mapping": {
            "f_number_to_executable_registry": "UNVERIFIED",
            "lean_theorem_to_python_function": "UNVERIFIED",
            "proof_term_to_informal_claim_semantics": "UNVERIFIED",
        },
        "authority": {
            "execution": False,
            "python_formula_proof_promotion": False,
            "paper_claim_promotion": False,
            "publication": False,
            "deployment": False,
            "runtime": False,
        },
        "limitations": [
            "This contract references axiom evidence for named Lean declarations at one exact revision.",
            "It does not prove that any Python function refines a Lean theorem.",
            "It does not establish informal claim semantics, empirical validity, publication, deployment, runtime behavior, or authorization.",
            "Lean4checker --fresh and comparator/external-checker validation were not run for this contract.",
        ],
    }
    contract["payload_sha256"] = _canonical_digest(contract)
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(EVIDENCE_FILE))
    parser.add_argument("--receipt", type=Path, default=Path(RECEIPT_FILE))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        contract = build_contract(args.evidence, args.receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
    except (OSError, ReferenceError) as exc:
        print(f"locked-formula reference: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "locked-formula reference: OK: "
        f"{len(contract['locked_ids'])} claims / "
        f"{contract['carrier_theorem_count']} carriers / {contract['payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
