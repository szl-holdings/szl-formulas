# SPDX-License-Identifier: Apache-2.0
"""The Memory Covenant wedge formulas stay candidates.

F23 is the Λ uniqueness id. These sources must not reuse it or describe
themselves as locked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "contracts" / "wedge-formula-candidates.v1.json"
LEAN = ROOT / "lean"


def test_wedge_candidates_keep_a_non_f23_id_and_candidate_status() -> None:
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    assert payload["authority"]["promotion_to_locked"] is False
    rows = payload["candidates"]
    assert [row["claim_id"] for row in rows] == ["MC1", "F24", "F25", "F26"]
    by_id = {row["claim_id"]: row for row in rows}
    assert by_id["MC1"]["status"] == "CANDIDATE"
    assert by_id["F24"]["status"] == "CANDIDATE"
    assert by_id["F25"]["status"] == "CANDIDATE"
    assert by_id["F26"]["status"].startswith("CONJECTURE")
    assert "F23" in by_id["MC1"]["id_note"]
    assert "Conjecture 1" in by_id["MC1"]["id_note"]


def test_wedge_lean_sources_are_candidates_under_the_pinned_toolchain() -> None:
    assert (LEAN / "lean-toolchain").read_text(encoding="utf-8").strip() == (
        "leanprover/lean4:v4.34.1"
    )
    assert not (LEAN / "F23_deny_by_default.lean").exists()
    challenge = (LEAN / "CHALLENGE_wedge.v1.lean").read_text(encoding="utf-8")
    assert "EXIT=0" not in challenge
    assert "F23_deny_by_default.lean" not in challenge
    assert "locked wedge" not in challenge.lower()
    for name in ("MC1_deny_by_default.lean", "F24_F25_covenant.lean"):
        text = (LEAN / name).read_text(encoding="utf-8")
        assert "namespace Puriq.Formula.Candidate" in text
        assert "namespace Puriq.Formula.Proved" not in text
        assert re.search(r"(?m)^\s*sorry\b", text) is None
