from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_WORKFLOW = ROOT / ".github" / "workflows" / "cpu-contract.yml"
SOURCE_PACKAGE = ROOT / "torch-ext" / "szl_formulas"
MIRROR_PACKAGE = ROOT / "build" / "torch-universal" / "szl_formulas"
PACKAGE_FILES = (
    "__init__.py",
    "_aggregators.py",
    "_composer.py",
    "_formulas.py",
    "atlas.py",
    "formula_atlas.v1.json",
    "metadata.json",
)


def test_pr_qualification_checks_out_exact_owner_head() -> None:
    text = CPU_WORKFLOW.read_text(encoding="utf-8")
    assert "SOURCE_REVISION: ${{ github.event.pull_request.head.sha || github.sha }}" in text
    assert "ref: ${{ env.SOURCE_REVISION }}" in text
    assert "persist-credentials: false" in text
    assert "fetch-depth: 1" in text
    assert 'test "$(git rev-parse HEAD)" = "$SOURCE_REVISION"' in text
    assert "refs/pull" not in text
    assert "merge_commit_sha" not in text
    assert "pull_request_target" not in text


def test_checked_in_universal_mirror_matches_canonical_source() -> None:
    for name in PACKAGE_FILES:
        assert (MIRROR_PACKAGE / name).read_bytes() == (SOURCE_PACKAGE / name).read_bytes()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_source_mirrors.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_metadata_refuses_a_frozen_test_count() -> None:
    metadata = json.loads((SOURCE_PACKAGE / "metadata.json").read_text(encoding="utf-8"))
    assert "tests" not in metadata
    evidence = metadata["test_evidence"]
    assert evidence["state"] == "REVISION_SCOPED_CI_REQUIRED"
    assert "frozen pass count" in evidence["rule"]


def test_runtime_does_not_promote_historical_counts_to_current_measurements() -> None:
    text = (SOURCE_PACKAGE / "_aggregators.py").read_text(encoding="utf-8")
    assert "LEDGER_CHECKED_N" not in text
    assert "ledger_checked_claimed" not in text
    assert "~185 CI-green" not in text
    assert '"current_evidence": "SOURCE_RECEIPT_REQUIRED"' in text
    assert '"state": "UNOBSERVED"' in text
