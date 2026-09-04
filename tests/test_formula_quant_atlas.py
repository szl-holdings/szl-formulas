from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import szl_formulas as formulas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "atlas" / "source-formula-ledger-corpus.json"
ATLAS = ROOT / "atlas" / "formula-atlas.v1.json"


def test_atlas_rebuild_is_byte_deterministic() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "atlas.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_formula_atlas.py"),
                "--source",
                str(SOURCE),
                "--output",
                str(output),
            ],
            check=True,
        )
        assert output.read_bytes() == ATLAS.read_bytes()


def test_attributed_source_and_counts_are_exact() -> None:
    atlas = formulas.load_formula_atlas()
    assert atlas["source"]["revision"] == "ceaef540eba6c5acf85091faf4a20cd9aef480f9"
    assert atlas["source"]["git_blob_sha"] == "600654f89cb062320acae4284cb5220ff881eb3c"
    assert atlas["source"]["sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert atlas["summary"]["attributed_formula_count"] == 30
    assert atlas["summary"]["executable_formula_count"] == 21
    assert sum(atlas["summary"]["class_counts"].values()) == 30
    assert sum(atlas["summary"]["quant_domain_counts"].values()) == 30


def test_locked_eight_remains_separate_from_all_status_strings() -> None:
    atlas = formulas.load_formula_atlas()
    authority = atlas["authority"]
    assert authority["locked_proven_count"] == 8
    assert set(authority["locked_proven_ids"]) == {
        "F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"
    }
    assert authority["f_number_to_executable_registry_mapping"] == "UNKNOWN_NOT_INFERRED"
    for row in atlas["attributed_formulas"]:
        assert row["locked_proven_membership"] == "UNKNOWN_NOT_INFERRED_FROM_REPORTED_STATUS"
        if row["class"] == "CONJECTURE":
            assert row["admission"] == "OPEN_NOT_EXECUTION_AUTHORITY"


def test_every_formula_has_one_explicit_quant_domain() -> None:
    atlas = formulas.load_formula_atlas()
    ids = [row["id"] for row in atlas["attributed_formulas"]]
    assert len(ids) == len(set(ids)) == 30
    assert all(row["quant_domain"] for row in atlas["attributed_formulas"])
    assert len(formulas.quant_domains()) == 9
    assert formulas.atlas_summary() == atlas["summary"]
    assert formulas.list_attributed_formulas() == atlas["attributed_formulas"]


def test_payload_digest_covers_the_canonical_core() -> None:
    atlas = json.loads(ATLAS.read_text(encoding="utf-8"))
    expected = atlas.pop("payload_sha256")
    core = (
        json.dumps(atlas, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(core).hexdigest() == expected
