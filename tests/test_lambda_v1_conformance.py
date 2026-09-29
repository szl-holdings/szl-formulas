# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl.lambda/v1 conformance for ``szl_formulas.lambda_aggregate`` (FF-04).

The golden vectors are a byte copy of ``spec/lambda_v1_vectors.json`` from the
szl-lambda-gate merge recorded in ``fixtures/lambda_v1_vectors.SOURCE``. Every
vector runs through BOTH package copies: ``torch-ext`` (what this repo imports)
and ``build/torch-universal`` (what the Hub loads). Error codes are exact;
values are compared within each vector's ``value_tol`` because this is not the
reference implementation.

szl-formulas has no τ gate, so each vector's ``verdict``/``code`` (gate_v1
outputs) is not applicable here; only Λ's value or error is checked.

Λ is advisory. Λ uniqueness is Conjecture 1 (open); nothing here depends on it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import szl_formulas as F  # noqa: E402
from szl_formulas import _formulas as TORCH_EXT  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
VECTORS_PATH = FIXTURES / "lambda_v1_vectors.json"
SOURCE_PATH = FIXTURES / "lambda_v1_vectors.SOURCE"
TORCH_EXT_FILE = ROOT / "torch-ext" / "szl_formulas" / "_formulas.py"
BUILD_FILE = ROOT / "build" / "torch-universal" / "szl_formulas" / "_formulas.py"

# The FF-01 merge of szl-lambda-gate and the digest its spec records.
PINNED_COMMIT = "d3443b0539ad9fdbd407a0b0bf0454b416102089"
PINNED_CANONICAL_SHA256 = "2a3fef3d17ca36142139fa6bd08b7f0e41526c749abc7cfd810b77cc50ab5d1f"

# Every code lambda_aggregate may raise, in the v1 precedence order.
# LAMBDA_TAU_INVALID is gate-only and szl-formulas has no gate.
V1_AGGREGATE_CODES = (
    "LAMBDA_TYPE_INVALID",
    "LAMBDA_EMPTY",
    "LAMBDA_LENGTH_MISMATCH",
    "LAMBDA_NONFINITE_AXIS",
    "LAMBDA_AXIS_OUT_OF_RANGE",
    "LAMBDA_NONFINITE_WEIGHT",
    "LAMBDA_WEIGHT_NONPOSITIVE",
    "LAMBDA_WEIGHT_SUM",
)


def _load_by_path(name: str, path: Path):
    # _formulas.py imports only the stdlib, so the Hub copy loads standalone.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_by_path("szl_formulas_build_universal_formulas", BUILD_FILE)
IMPLS = {"torch-ext": TORCH_EXT, "build/torch-universal": BUILD}


def _canonical_sha256(obj) -> str:
    body = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _decode(value):
    """'f64:<16 hex>' -> float; any other JSON value is passed through unchanged."""
    if isinstance(value, str) and value.startswith("f64:") and len(value) == 20:
        return struct.unpack(">d", bytes.fromhex(value[4:]))[0]
    return value


def _decode_seq(value):
    return [_decode(v) for v in value] if isinstance(value, list) else value


VECTORS_DOC = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
VECTORS = VECTORS_DOC["vectors"]


def _code_of(impl, *args, **kwargs) -> str:
    with pytest.raises(ValueError) as caught:
        impl.lambda_aggregate(*args, **kwargs)
    err = caught.value
    assert isinstance(err, impl.LambdaV1Error), type(err)
    assert err.code in V1_AGGREGATE_CODES
    assert str(err).startswith(err.code)
    return err.code


# --------------------------------------------------------------------------- #
# Provenance of the vendored vectors                                          #
# --------------------------------------------------------------------------- #
def _source_fields() -> dict:
    lines = SOURCE_PATH.read_text(encoding="utf-8").splitlines()
    fields = {"pin": lines[0].strip()}
    for line in lines[1:]:
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def test_source_pins_the_ff01_merge_and_digest():
    fields = _source_fields()
    assert fields["pin"] == f"szl-lambda-gate@{PINNED_COMMIT}:spec/lambda_v1_vectors.json"
    assert fields["repository"] == "szl-holdings/szl-lambda-gate"
    assert fields["commit"] == PINNED_COMMIT
    assert fields["path"] == "spec/lambda_v1_vectors.json"
    assert fields["canonical_sha256"] == PINNED_CANONICAL_SHA256
    # Canonical JSON bytes: identical under LF and CRLF checkouts.
    assert _canonical_sha256(VECTORS_DOC) == PINNED_CANONICAL_SHA256


def test_fixture_is_a_byte_copy_of_the_pinned_blob():
    # git stores the blob with LF endings; normalise a CRLF checkout back to it.
    raw = VECTORS_PATH.read_bytes().replace(b"\r\n", b"\n")
    blob = f"blob {len(raw)}\0".encode("ascii") + raw
    assert hashlib.sha1(blob, usedforsecurity=False).hexdigest() == _source_fields()["git_blob_sha1"]


def test_vectors_are_well_formed():
    assert VECTORS_DOC["schema"] == "szl.lambda/v1.vectors"
    ids = [v["id"] for v in VECTORS]
    assert len(ids) == len(set(ids)) == 50
    for v in VECTORS:
        expect = v["expect"]
        assert ("error" in expect) != ("value_f64" in expect), v["id"]
        if "error" in expect:
            assert expect["error"] in V1_AGGREGATE_CODES, v["id"]
        assert v["value_tol"] > 0


# --------------------------------------------------------------------------- #
# Every vector, through both package copies                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("impl_name", sorted(IMPLS))
@pytest.mark.parametrize("vector", VECTORS, ids=[v["id"] for v in VECTORS])
def test_lambda_aggregate_matches_vector(vector, impl_name):
    impl = IMPLS[impl_name]
    axes = _decode_seq(vector["axes"])
    weights = _decode_seq(vector["weights"])
    expect = vector["expect"]
    if "error" in expect:
        assert _code_of(impl, axes, weights) == expect["error"]
        return
    value = impl.lambda_aggregate(axes, weights)
    want = _decode(expect["value_f64"])
    assert type(value) is float
    assert 0.0 <= value <= 1.0
    assert abs(value - want) <= vector["value_tol"], (value, want)
    if want == 0.0:
        assert value == 0.0  # a zero axis is a veto: exactly 0, never an error


def test_every_aggregate_error_code_is_exercised_by_the_vectors():
    seen = {v["expect"]["error"] for v in VECTORS if "error" in v["expect"]}
    assert seen == set(V1_AGGREGATE_CODES)


# --------------------------------------------------------------------------- #
# Acceptance, stated directly (default uniform weights: the E5 call shape)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("impl_name", sorted(IMPLS))
def test_x_above_one_no_longer_returns_1_1619(impl_name):
    # E5: main returned 1.1619 (above 1, above the 0.97 ceiling) for this call.
    assert _code_of(IMPLS[impl_name], [1.5, 0.9]) == "LAMBDA_AXIS_OUT_OF_RANGE"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "pos_inf", "neg_inf"])
def test_nonfinite_axis_raises(impl_name, bad):
    # E5: main returned Infinity for +Inf and NaN for NaN.
    assert _code_of(IMPLS[impl_name], [bad, 0.9]) == "LAMBDA_NONFINITE_AXIS"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
@pytest.mark.parametrize("bad", [-0.1, -5e-324, 1.0000000000000002, 2.0])
def test_axis_outside_unit_interval_raises(impl_name, bad):
    assert _code_of(IMPLS[impl_name], [bad, 0.9]) == "LAMBDA_AXIS_OUT_OF_RANGE"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
@pytest.mark.parametrize(
    "weights", [(1.5, -0.5), [1.0, 0.0], [1.0, -0.0]], ids=["negative", "zero", "negative_zero"]
)
def test_nonpositive_weight_raises(impl_name, weights):
    # E5: w = (1.5, -0.5) broke monotonicity on main (Λ(0.5, 0.9) > Λ(0.5, 1.0)).
    impl = IMPLS[impl_name]
    assert _code_of(impl, [0.5, 0.9], weights) == "LAMBDA_WEIGHT_NONPOSITIVE"
    assert _code_of(impl, [0.5, 1.0], weights) == "LAMBDA_WEIGHT_NONPOSITIVE"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
def test_weight_sum_tolerance_is_1e_12_absolute(impl_name):
    impl = IMPLS[impl_name]
    # Inside 1e-12: accepted, used as declared (no renormalisation).
    assert impl.lambda_aggregate([0.9, 0.5], [0.5 + 4e-13, 0.5]) == pytest.approx(
        math.sqrt(0.9 * 0.5), abs=1e-12
    )
    # Outside 1e-12 but inside main's old relative 1e-9: now an error.
    assert _code_of(impl, [0.9, 0.5], [0.5 + 3e-12, 0.5]) == "LAMBDA_WEIGHT_SUM"
    assert _code_of(impl, [0.9, 0.5], [0.5 + 5e-10, 0.5]) == "LAMBDA_WEIGHT_SUM"
    assert _code_of(impl, [0.9, 0.5], [2.0, 2.0]) == "LAMBDA_WEIGHT_SUM"
    assert _code_of(impl, [0.9, 0.5], [1e308, 1e308]) == "LAMBDA_WEIGHT_SUM"  # fsum overflow


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
def test_empty_input_raises(impl_name):
    impl = IMPLS[impl_name]
    assert _code_of(impl, []) == "LAMBDA_EMPTY"
    assert _code_of(impl, (), ()) == "LAMBDA_EMPTY"
    assert _code_of(impl, [], [1.0]) == "LAMBDA_EMPTY"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
def test_nonfinite_weight_raises(impl_name):
    impl = IMPLS[impl_name]
    for bad in (math.nan, math.inf, -math.inf):
        assert _code_of(impl, [0.5, 0.9], [bad, 0.5]) == "LAMBDA_NONFINITE_WEIGHT"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
def test_non_numbers_are_type_errors_not_coerced(impl_name):
    impl = IMPLS[impl_name]
    # main coerced these with float(): "0.9" parsed and True counted as 1.0.
    assert _code_of(impl, ["0.9", 0.9]) == "LAMBDA_TYPE_INVALID"
    assert _code_of(impl, [True, 0.9]) == "LAMBDA_TYPE_INVALID"
    assert _code_of(impl, [None, 0.9]) == "LAMBDA_TYPE_INVALID"
    assert _code_of(impl, None) == "LAMBDA_TYPE_INVALID"
    assert _code_of(impl, [0.5, 0.9], {"a": 0.5, "b": 0.5}) == "LAMBDA_TYPE_INVALID"


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
@pytest.mark.parametrize("k", [1, 2, 3, 7, F.LEGACY_AXIS_COUNT, F.DEFAULT_AXIS_COUNT, 97, 1000, 4099])
def test_default_uniform_weights_stay_inside_the_sum_tolerance(impl_name, k):
    # weights=None keeps its meaning (w_i = 1/k) and must never trip LAMBDA_WEIGHT_SUM.
    impl = IMPLS[impl_name]
    assert impl.lambda_aggregate([0.81] * k) == pytest.approx(0.81, abs=1e-12)
    assert impl.lambda_aggregate([1.0] * k) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("impl_name", sorted(IMPLS))
def test_value_is_bounded_by_its_axes(impl_name):
    impl = IMPLS[impl_name]
    x = [0.3, 0.7, 0.9]
    lam = impl.lambda_aggregate(x, [0.2, 0.3, 0.5])
    assert min(x) <= lam <= max(x) <= 1.0


def test_error_type_is_a_value_error_with_a_code():
    assert issubclass(TORCH_EXT.LambdaV1Error, ValueError)
    err = TORCH_EXT.LambdaV1Error("LAMBDA_EMPTY", "detail")
    assert err.code == "LAMBDA_EMPTY"
    assert str(err) == "LAMBDA_EMPTY: detail"
    assert TORCH_EXT.LAMBDA_V1_ERROR_CODES == V1_AGGREGATE_CODES
    assert TORCH_EXT.LAMBDA_V1_WEIGHT_SUM_TOL == 1e-12
    assert BUILD.LAMBDA_V1_ERROR_CODES == V1_AGGREGATE_CODES


# --------------------------------------------------------------------------- #
# Lockstep: the Hub-loaded copy is the same file                               #
# --------------------------------------------------------------------------- #
def test_torch_ext_and_build_copies_are_byte_identical():
    assert TORCH_EXT_FILE.read_bytes() == BUILD_FILE.read_bytes()


# --------------------------------------------------------------------------- #
# Callers inside the package                                                  #
# --------------------------------------------------------------------------- #
def test_lambda_homogeneous_c_above_one_now_raises():
    # Documented behaviour change: c * x leaving [0, 1] is outside the v1 domain.
    assert F.lambda_homogeneous(0.5, [0.9, 0.8]) is True
    assert F.lambda_homogeneous(1.0, [0.9, 0.8]) is True
    with pytest.raises(ValueError) as caught:
        F.lambda_homogeneous(2.0, [0.9, 0.8])
    assert caught.value.code == "LAMBDA_AXIS_OUT_OF_RANGE"


def test_lambda_bounded_and_schur_reject_out_of_range_axes():
    with pytest.raises(ValueError, match="LAMBDA_AXIS_OUT_OF_RANGE"):
        F.lambda_bounded([1.5, 0.9])
    with pytest.raises(ValueError, match="LAMBDA_AXIS_OUT_OF_RANGE"):
        F.schur_concave_lambda_two_axis(1.5, 0.9)
    assert F.schur_concave_lambda_two_axis(0.2, 0.8) is True


def test_governed_loop_never_feeds_lambda_an_out_of_domain_scalar():
    # _raw_scalar maps every output into [0, 1] (NaN -> 0), so the loop halts
    # on a NaN output instead of raising, and clean runs still replay.
    chain = F.run_governed_loop([
        {"formula_name": "bekenstein_cascade", "args": [1.0, 1.0]},
        {"formula_name": "gleason_quantum_lambda", "args": [[[math.nan]]]},
    ])
    assert chain["halted"] is True
    assert "axis_floor" in (chain["halt_reason"] or "")
    assert chain["replay_ok"] is True
    assert 0.0 <= chain["lambda_aggregate"] <= 1.0


def test_forged_chain_with_lambda_above_one_no_longer_verifies():
    # On main, a forged receipt with scalar 1.5 (hash recomputed, validators set
    # true) and lambda_aggregate 1.5 made verify_chain return True.
    from szl_formulas import _composer as C

    calls = [{"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]}]
    chain = F.run_governed_loop(calls)
    receipt = chain["receipts"][0]
    receipt["scalar"] = 1.5
    receipt["receipt_hash"] = C._receipt_hash(
        C.GENESIS, 0, receipt["formula_name"], receipt["args_digest"], 1.5
    )
    chain["lambda_aggregate"] = 1.5
    with pytest.raises(ValueError, match="LAMBDA_AXIS_OUT_OF_RANGE"):
        F.verify_chain(chain, calls)
