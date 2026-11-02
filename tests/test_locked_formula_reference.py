from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import szl_formulas as formulas  # noqa: E402


EVIDENCE = (
    ROOT
    / "evidence"
    / "lutar-lean"
    / "locked-formulas.c4c42baf107f2cd5ec11ac3958fbd8fd06a875c8.json"
)
RECEIPT = EVIDENCE.with_name(EVIDENCE.stem + ".receipt.json")
CONTRACT = ROOT / "contracts" / "locked-formula-reference.v1.json"
SCRIPT = ROOT / "scripts" / "build_locked_formula_reference.py"
EVIDENCE_SHA256 = "10da7a74357f5d23dee02f19dbc8a5475d10f336b227d7b909eb83ffd29ca33c"
RECEIPT_SHA256 = "e732886028b4a53714ef2931fad7a9b4809e6a4ae6909f91b0eb2514afec6e78"
LOCKED_IDS = ["F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"]
COPIES = ["torch-ext", "build/torch-cpu", "build/torch-universal"]


def test_reference_rebuild_is_byte_deterministic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "reference.json"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--evidence",
                str(EVIDENCE),
                "--receipt",
                str(RECEIPT),
                "--output",
                str(output),
            ],
            check=True,
        )
        assert output.read_bytes() == CONTRACT.read_bytes()


def test_external_evidence_bytes_and_source_are_exact() -> None:
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == EVIDENCE_SHA256
    assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest() == RECEIPT_SHA256
    reference = formulas.load_locked_formula_reference()
    source = reference["source"]
    assert source["repository"] == "szl-holdings/lutar-lean"
    assert source["commit"] == "c4c42baf107f2cd5ec11ac3958fbd8fd06a875c8"
    assert source["tree"] == "d6959371d2654c87dce8b18d5e21d40d0c7a6b57"
    assert source["evidence_file_sha256"] == "sha256:" + EVIDENCE_SHA256
    assert source["receipt_file_sha256"] == "sha256:" + RECEIPT_SHA256


def test_validation_ladder_stops_at_observed_axiom_enumeration() -> None:
    reference = formulas.load_locked_formula_reference()
    ladder = reference["validation_ladder"]
    assert ladder["build_and_elaboration"] == "OBSERVED_AT_PIN_IN_REFERENCED_WORKFLOW"
    assert ladder["axiom_enumeration"] == "OBSERVED_AT_PIN_VIA_LEAN_COLLECT_AXIOMS"
    assert ladder["lean4checker_fresh"] == "NOT_RUN"
    assert ladder["comparator_external_checkers"] == "NOT_RUN"
    assert reference["mapping"] == {
        "f_number_to_executable_registry": "UNVERIFIED",
        "lean_theorem_to_python_function": "UNVERIFIED",
        "proof_term_to_informal_claim_semantics": "UNVERIFIED",
    }
    assert all(value is False for value in reference["authority"].values())


def test_locked_claims_and_axioms_are_bounded() -> None:
    reference = formulas.load_locked_formula_reference()
    assert reference["locked_ids"] == LOCKED_IDS
    assert reference["carrier_theorem_count"] == 24
    assert reference["trust_base"] == ["propext", "Classical.choice", "Quot.sound"]
    claims = reference["claims"]
    assert [row["claim_id"] for row in claims] == LOCKED_IDS
    carriers = [carrier for row in claims for carrier in row["carriers"]]
    assert len(carriers) == 24
    assert len({carrier["theorem"] for carrier in carriers}) == 24
    for carrier in carriers:
        assert carrier["allowed"] is True
        assert set(carrier["axioms"]) <= set(reference["trust_base"])
        assert "sorryAx" not in carrier["axioms"]
        assert not any("_native" in axiom for axiom in carrier["axioms"])


def test_reference_payload_digest_covers_the_canonical_core() -> None:
    reference = json.loads(CONTRACT.read_text(encoding="utf-8"))
    expected = reference.pop("payload_sha256")
    encoded = json.dumps(
        reference, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    assert "sha256:" + hashlib.sha256(encoded).hexdigest() == expected


def test_builder_rejects_any_vendored_evidence_drift() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        mutated = directory / "evidence.json"
        output = directory / "reference.json"
        mutated.write_bytes(EVIDENCE.read_bytes() + b"\n")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--evidence",
                str(mutated),
                "--receipt",
                str(RECEIPT),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert not output.exists()
        assert "evidence SHA-256 drift" in completed.stderr


def test_packaged_reference_matches_canonical_bytes(shipped_root) -> None:
    for copy in COPIES:
        packaged = shipped_root(copy) / "szl_formulas" / "proof_reference.v1.json"
        assert packaged.read_bytes() == CONTRACT.read_bytes()
