# SPDX-License-Identifier: Apache-2.0
"""The real szl-formulas config builds the Hub tree get_kernel loads (stdlib + git).

Runs scripts/szl_kernel_publish.py's build against this repository's own HEAD: both
noarch variants carry szl_formulas (including its atlas data and package metadata), byte
for byte from torch-ext (LF), with no root shim, with NOTICE and LICENSE beside the card,
and with the kernel card and the model-twin pointer card passing the lint.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_szl_kernel_publish_layout",
                                               ROOT / "scripts" / "szl_kernel_publish.py")
kp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kp
_spec.loader.exec_module(kp)

CFG = kp.find_kernel(ROOT, "SZLHOLDINGS/szl-formulas")
HEAD = kp.resolve_commit(ROOT, "HEAD")


def test_config_maps_the_single_package_and_the_pointer_twin():
    assert [(p.src, p.name) for p in CFG.packages] == [("torch-ext/szl_formulas", "szl_formulas")]
    assert CFG.version == 1 and CFG.version_branch == "v1"
    assert CFG.extra_files == (("NOTICE", "NOTICE"),)
    assert {"OPERATIONAL.json", "BENCH.laptop-blackwell.json", "SZL_SOURCE_BINDING.json"} == set(CFG.prune_paths)
    assert CFG.twin is not None and CFG.twin.repo_id == "SZLHOLDINGS/szl-formulas"
    assert {"build/", "torch-ext/", "scripts/", "tests/", "build.toml", "model.joblib"} <= set(CFG.twin.prune_paths)


def test_head_build_is_the_torch_ext_source():
    payload = kp.build_payload(CFG, ROOT, HEAD)
    for variant in kp.VARIANTS:
        prefix = f"build/{variant}/"
        served = {p[len(prefix):]: d for p, d in payload.items() if p.startswith(prefix)}
        expected = {}
        for pkg in CFG.packages:
            for rel, data in kp.read_tree_files(ROOT, HEAD, pkg.src).items():
                expected[f"{pkg.name}/{rel}"] = data.replace(b"\r\n", b"\n")
        assert set(served) == set(expected) | {"metadata.json"}, variant
        for rel, data in expected.items():
            assert served[rel] == data, f"{variant}:{rel}"
        assert "__init__.py" not in served  # no shim: the loaded module is szl_formulas
        assert {"szl_formulas/formula_atlas.v1.json", "szl_formulas/metadata.json"} <= set(served)
        meta = json.loads(served["metadata.json"])
        assert meta["name"] == "szl-formulas" and meta["version"] == 1
        assert meta["backend"] == {"type": "cpu"} and meta["python-depends"] == []
        assert set(meta["digest"]["files"]) == set(expected)


def test_card_notice_and_binding_name_the_version_channel():
    payload = kp.build_payload(CFG, ROOT, HEAD)
    card = payload["README.md"].decode("utf-8")
    assert 'get_kernel("SZLHOLDINGS/szl-formulas", version=1, trust_remote_code=True)' in card
    assert "{{" not in card
    assert payload["NOTICE"] == (ROOT / "NOTICE").read_bytes().replace(b"\r\n", b"\n")
    assert payload["LICENSE"] == (ROOT / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
    binding = json.loads(payload[kp.BINDING_PATH])
    assert binding["source_commit"] == HEAD
    assert binding["source_repository"] == "szl-holdings/szl-formulas"
    assert set(payload) - {p for p in payload if p.startswith("build/")} == {
        "README.md", "LICENSE", "NOTICE", kp.BINDING_PATH}


def test_model_twin_pointer_card_passes_the_lint():
    text = kp.normalize_text(CFG.twin.card, kp.read_blob(ROOT, HEAD, CFG.twin.card)).decode("utf-8")
    text = kp.render_card(CFG, text, CFG.twin.card)
    kp.lint_card(CFG, text, CFG.twin.card, require_usage=True)
    assert "https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas" in text
    assert 'get_kernel("SZLHOLDINGS/szl-formulas", version=1, trust_remote_code=True)' in text
