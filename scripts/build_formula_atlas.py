#!/usr/bin/env python3
"""Build the deterministic SZL formula/quant atlas from attributed sources.

The atlas preserves each source record's original class and reported status.
It never promotes a record into the locked-proven set and never guesses the
unknown mapping between the F-number corpus and the 21 executable functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import szl_formulas as executable  # noqa: E402

SOURCE = {
    "repository": "szl-holdings/szl-formula-ledger",
    "revision": "ceaef540eba6c5acf85091faf4a20cd9aef480f9",
    "path": "formulas/corpus.json",
    "git_blob_sha": "600654f89cb062320acae4284cb5220ff881eb3c",
    "state": "ARCHIVED_ATTRIBUTED_SOURCE",
}
ALLOWED_CLASSES = {
    "SYMBOLIC",
    "DIMENSIONAL",
    "EMPIRICAL",
    "DEFINITIONAL",
    "CONJECTURE",
}
QUANT_DOMAIN_BY_ID = {
    "A2-homogeneity": "trust_aggregation",
    "A4-bounded-amgm": "trust_aggregation",
    "lambda-score-dimensionless": "trust_aggregation",
    "TH_L1-lambda-uniqueness": "trust_aggregation",
    "axis-schema-13": "trust_aggregation",
    "F18-reed-solomon-singleton": "coding_error_control",
    "TH_V18_03-kraft": "coding_error_control",
    "shor-913-distance": "coding_error_control",
    "F1-euler-khipu-chi": "topology_geometry",
    "TH_V18_04-egyptian-horus": "algebra_number_theory",
    "quadratic-completion": "algebra_number_theory",
    "cauchy-schwarz-2d": "algebra_number_theory",
    "madhava-leibniz-atan": "algebra_number_theory",
    "F19-bekenstein-additive": "energy_entropy_physics",
    "bekenstein-dimensional": "energy_entropy_physics",
    "landauer-energy": "energy_entropy_physics",
    "K13-bekenstein-fire": "energy_entropy_physics",
    "F12-kuramoto-additive": "dynamics_consensus",
    "byzantine-n3f1": "dynamics_consensus",
    "K06-rho-closure": "dynamics_consensus",
    "conjecture-2-khipu-safety": "dynamics_consensus",
    "conjecture-3-khipu-liveness": "dynamics_consensus",
    "fisher-rao-identity": "information_geometry",
    "pinsker-2pt": "information_geometry",
    "K01-receipt-build-latency": "governance_receipts",
    "k-verify-accuracy": "governance_receipts",
    "F0001-system-tuple": "governance_receipts",
    "F0003-receipt-edge": "governance_receipts",
    "dsse-envelope-struct": "governance_receipts",
    "code-of-reality-lineage": "narrative_lineage",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def admission_for(record_class: str) -> str:
    if record_class == "CONJECTURE":
        return "OPEN_NOT_EXECUTION_AUTHORITY"
    if record_class == "EMPIRICAL":
        return "SOURCE_RECEIPT_REQUIRED_FOR_CURRENT_CLAIM"
    return "REFERENCE_ONLY_UNLESS_EXECUTABLE_MATCH_IS_EXPLICIT"


def build(source_path: Path) -> dict[str, Any]:
    raw = source_path.read_bytes()
    source = json.loads(raw)
    formulas = source.get("formulas")
    if not isinstance(formulas, list) or len(formulas) != 30:
        raise ValueError("attributed formula corpus must contain exactly 30 records")

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in formulas:
        if not isinstance(row, dict):
            raise TypeError("each formula record must be an object")
        missing = {"id", "source", "statement", "class", "reported_status"} - set(row)
        if missing:
            raise ValueError(f"formula record missing fields: {sorted(missing)}")
        formula_id = str(row["id"])
        if formula_id in seen:
            raise ValueError(f"duplicate formula id: {formula_id}")
        seen.add(formula_id)
        record_class = str(row["class"])
        if record_class not in ALLOWED_CLASSES:
            raise ValueError(f"unsupported formula class: {record_class}")
        if formula_id not in QUANT_DOMAIN_BY_ID:
            raise ValueError(f"formula lacks an explicit quant domain: {formula_id}")
        records.append(
            {
                "id": formula_id,
                "source": str(row["source"]),
                "statement": str(row["statement"]),
                "class": record_class,
                "reported_status": str(row["reported_status"]),
                "quant_domain": QUANT_DOMAIN_BY_ID[formula_id],
                "admission": admission_for(record_class),
                "locked_proven_membership": (
                    "UNKNOWN_NOT_INFERRED_FROM_REPORTED_STATUS"
                ),
            }
        )

    executable_rows = [
        {
            "name": name,
            "proof_status": executable.PROOF_STATUS[name],
        }
        for name in sorted(executable.REGISTRY)
    ]
    if len(executable_rows) != 21:
        raise ValueError("executable registry must remain exactly 21 functions")
    locked = sorted(executable.LOCKED_PROVEN_FORMULA_IDS)
    if locked != ["F1", "F11", "F12", "F18", "F19", "F22", "F4", "F7"]:
        raise ValueError("locked-proven formula set drifted")

    class_counts = Counter(row["class"] for row in records)
    domain_counts = Counter(row["quant_domain"] for row in records)
    core: dict[str, Any] = {
        "schema": "szl.formula-quant-atlas/v1",
        "state": "ATTRIBUTED_REFERENCE_PLUS_EXECUTABLE_KERNEL",
        "source": {
            **SOURCE,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "authority": {
            "executable_registry_repository": "szl-holdings/szl-formulas",
            "executable_registry_count": 21,
            "locked_proven_count": 8,
            "locked_proven_ids": locked,
            "lambda_status": "CONJECTURE_1_OPEN_ADVISORY_ONLY",
            "f_number_to_executable_registry_mapping": "UNKNOWN_NOT_INFERRED",
            "rule": (
                "Per-obligation PROOF_STATUS, corpus reported_status, and locked-proven "
                "membership are distinct dimensions. No status string promotes a formula."
            ),
        },
        "summary": {
            "attributed_formula_count": len(records),
            "executable_formula_count": len(executable_rows),
            "class_counts": dict(sorted(class_counts.items())),
            "quant_domain_counts": dict(sorted(domain_counts.items())),
        },
        "quant_domains": [
            {
                "id": domain,
                "formula_count": domain_counts[domain],
                "authority": "REFERENCE_AND_CONSTRAINT_INPUT_ONLY",
            }
            for domain in sorted(domain_counts)
        ],
        "executable_formulas": executable_rows,
        "attributed_formulas": sorted(records, key=lambda row: row["id"]),
    }
    core["payload_sha256"] = hashlib.sha256(canonical_bytes(core)).hexdigest()
    return core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
