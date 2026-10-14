# SPDX-License-Identifier: Apache-2.0
"""Kernel Hub load guidance must name an immutable revision (README "Load")."""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COPIES = ["torch-ext", "build/torch-cpu", "build/torch-universal"]  # see conftest.py
MUTABLE = re.compile(r"""revision\s*=\s*["'](?:main|master|HEAD|latest)["']""")
GUARD = 're.fullmatch(r"[0-9a-f]{40}", hf_revision)'


@pytest.mark.parametrize("copy", COPIES)
def test_quickstart_never_loads_a_mutable_revision(copy, shipped_root):
    text = (shipped_root(copy) / "szl_formulas" / "__init__.py").read_text(encoding="utf-8")
    assert "get_kernel(" in text
    assert MUTABLE.search(text) is None, "quickstart must not load a branch name"
    assert GUARD in text, "quickstart must reject non-40-hex revisions before loading"
    assert "revision=hf_revision" in text


def test_readme_and_quickstart_share_the_same_guard():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert GUARD in readme
    assert MUTABLE.search(readme) is None
