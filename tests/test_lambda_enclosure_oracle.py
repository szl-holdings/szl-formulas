# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Independent decimal enclosure oracle for ``lambda_aggregate`` (R05 / P-MATH-03).

``tests/lambda_enclosure_oracle.py`` brackets the exact real ``∏ xᵢ^{wᵢ}`` of
the exact binary64 inputs with stdlib ``decimal`` directed rounding and bounds
the kernel's float error in exact ``Fraction`` arithmetic under ONE labelled
libm assumption. Every check runs through BOTH package copies: ``torch-ext``
(what this repo imports) and ``build/torch-universal`` (what the Hub loads),
the same IMPLS pattern as ``test_lambda_v1_conformance.py``.

Containment is asserted as ``out ∈ [lo - eps, hi + eps]``. The pinned fixture
``value_f64`` are float reference outputs, not exact reals (two of them are
not the correctly rounded exact value), so ``value_f64 ∈ [lo, hi]`` is NOT
asserted.

This is numeric certification only, NOT a gate verdict: it does not model
szl-lambda-gate's NUMERIC_TIE policy band. Λ uniqueness remains Conjecture 1
(open); nothing here depends on it.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lambda_enclosure_oracle as O  # noqa: E402
from szl_formulas import _composer as C  # noqa: E402
from szl_formulas import _formulas as TORCH_EXT  # noqa: E402

ORACLE_FILE = ROOT / "tests" / "lambda_enclosure_oracle.py"
VECTORS_PATH = ROOT / "tests" / "fixtures" / "lambda_v1_vectors.json"
BUILD_FILE = ROOT / "build" / "torch-universal" / "szl_formulas" / "_formulas.py"


def _load_by_path(name: str, path: Path):
    # _formulas.py imports only the stdlib, so the Hub copy loads standalone.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_by_path("szl_formulas_build_universal_formulas_for_oracle", BUILD_FILE)
IMPLS = {"torch-ext": TORCH_EXT, "build/torch-universal": BUILD}
IMPL_NAMES = sorted(IMPLS)

VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]
VALUE_VECTORS = [v for v in VECTORS if "value_f64" in v["expect"]]
ERROR_VECTORS = [v for v in VECTORS if "error" in v["expect"]]
assert len(VALUE_VECTORS) + len(ERROR_VECTORS) == len(VECTORS) == 60
assert VALUE_VECTORS and ERROR_VECTORS


def _seq(value):
    return [O.decode_f64(v) for v in value] if isinstance(value, list) else value


def _inputs(vector):
    return _seq(vector["axes"]), _seq(vector["weights"])


def _is_binary64_tau_in_unit_interval(tau) -> bool:
    return (
        isinstance(tau, float)
        and not isinstance(tau, bool)
        and math.isfinite(tau)
        and 0.0 < tau <= 1.0
    )


TAU_VECTORS = [
    v for v in VALUE_VECTORS if _is_binary64_tau_in_unit_interval(O.decode_f64(v.get("tau")))
]
assert TAU_VECTORS

WIDTH_RATIO = Fraction(1, 10**50)  # enclosure width must stay below 1e-50 * hi


def _check_enclosure(impl, axes, weights):
    """Shared assertions: nesting, containment in [lo-eps, hi+eps], exact zero, width."""
    out = impl.lambda_aggregate(axes, weights)
    lo, hi, lo2, hi2 = O.enclose_nested(axes, weights, impl=impl)
    assert lo <= lo2 <= hi2 <= hi
    ws = O.kernel_weights(axes, weights)
    eps = O.float_error_bound(axes, ws, hi)
    assert eps >= 0
    v = Fraction(out)
    assert Fraction(lo) - eps <= v <= Fraction(hi) + eps, (axes, weights, out, lo, hi, eps)
    if lo == 0 == hi:
        assert out == 0.0 and eps == 0
    else:
        assert 0 < lo <= hi <= 1
        assert Fraction(hi) - Fraction(lo) < WIDTH_RATIO * Fraction(hi)
        assert Fraction(hi2) - Fraction(lo2) < Fraction(hi) - Fraction(lo)  # 90 digits tighten
    return out, lo, hi, eps


# --------------------------------------------------------------------------- #
# Platform identities and honesty strings                                      #
# --------------------------------------------------------------------------- #
def test_platform_identities_the_error_model_relies_on():
    assert all(O.check_platform_identities().values()), O.check_platform_identities()


def test_oracle_is_stdlib_only_and_independent_of_the_kernel_at_import():
    tree = ast.parse(ORACLE_FILE.read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    allowed = {"__future__", "math", "struct", "sys", "decimal", "fractions", "pathlib", "typing"}
    assert top_level <= allowed, top_level - allowed
    assert "szl_formulas" not in top_level  # the kernel is injected, never imported at module level


def test_oracle_states_its_scope_honestly():
    doc = O.__doc__
    for phrase in ("NOT a gate verdict", "NUMERIC_TIE", "Conjecture 1", "NEVER normalised"):
        assert phrase in doc, phrase
    for phrase in ("NOT a gate verdict", "NUMERIC_TIE", "Conjecture 1"):
        assert phrase in O.SCOPE
        assert phrase in O.classify.__doc__
        assert phrase in O.certify.__doc__
    assert "ASSUMPTION" in O.float_error_bound.__doc__
    assert O.LIBM_ULP_ASSUMPTION == 4
    assert any("not IEEE-754 mandated" in a for a in O.ASSUMPTIONS)
    assert "theorem" not in doc.lower()  # Λ uniqueness is a conjecture, never a theorem here


# --------------------------------------------------------------------------- #
# Closed forms                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize(
    "axes, weights, exact",
    [
        ([0.25], [1.0], Fraction(1, 4)),
        ([0.25, 1.0], [0.5, 0.5], Fraction(1, 2)),
        ([0.5, 0.125], [0.5, 0.5], Fraction(1, 4)),
        ([0.5, 0.5], None, Fraction(1, 2)),
        ([0.0625, 1.0, 1.0, 1.0], [0.5, 0.25, 0.125, 0.125], Fraction(1, 4)),
    ],
    ids=["quarter", "sqrt_quarter_times_one", "sqrt_sixteenth", "uniform_half", "ones_contribute_zero"],
)
def test_closed_forms_are_enclosed(impl_name, axes, weights, exact):
    out, lo, hi, eps = _check_enclosure(IMPLS[impl_name], axes, weights)
    assert Fraction(lo) <= exact <= Fraction(hi)
    assert Fraction(lo) < exact < Fraction(hi)  # strictly, because each bound is widened


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize("k", [1, 2, 3, 7, 13, 97])
def test_all_ones_give_hi_exactly_one(impl_name, k):
    impl = IMPLS[impl_name]
    lo, hi = O.enclose([1.0] * k, impl=impl)
    assert hi == 1 and isinstance(hi, Decimal)
    assert 0 < Fraction(1) - Fraction(lo) < WIDTH_RATIO
    assert impl.lambda_aggregate([1.0] * k) == 1.0
    _check_enclosure(impl, [1.0] * k, None)
    lo_i, hi_i = O.enclose([1] * k, [1.0 / k] * k, impl=impl)  # int axes are exact too
    assert (lo_i, hi_i) == (lo, hi)


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize(
    "axes, weights",
    [
        ([0.0, 0.9], None),
        ([-0.0, 0.9], None),
        ([0.9, 0.0], [0.5, 0.5]),
        ([0.5, -0.0, 0.7], [0.2, 0.3, 0.5]),
        ([0], [1.0]),
        ([0.0], None),
    ],
    ids=["zero", "negative_zero", "zero_last", "negative_zero_middle", "int_zero", "single_zero"],
)
def test_zero_axis_gives_exactly_zero(impl_name, axes, weights):
    impl = IMPLS[impl_name]
    lo, hi = O.enclose(axes, weights, impl=impl)
    assert lo == 0 and hi == 0
    assert str(lo) == "0" and str(hi) == "0"
    assert impl.lambda_aggregate(axes, weights) == 0.0
    assert O.float_error_bound(axes, O.kernel_weights(axes, weights), hi) == 0
    report = O.certify(axes, weights, impl=impl)
    assert report["status"] == O.ENCLOSED
    assert report["float_value"] == 0.0 and report["contained"] is True
    assert report["lo"] == report["hi"] == "0" and report["eps"] == "0"


# --------------------------------------------------------------------------- #
# Weights: binary64 defaults reproduced, never normalised                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("impl_name", IMPL_NAMES)
def test_weights_none_reproduces_the_kernels_binary64_uniform_weights(impl_name):
    impl = IMPLS[impl_name]
    for k in (1, 2, 3, 7, 13):
        axes = [0.81] * k
        assert O.enclose(axes, impl=impl) == O.enclose(axes, [1.0 / k] * k, impl=impl)
    # k = 3: fsum([1.0/3]*3) = 1 - 2**-54, so the exact value of the binary64 call is
    # 0.81**(1 - 2**-54) > 0.81. The oracle must see that (no rational 1/k shortcut)...
    lo, hi = O.enclose([0.81] * 3, impl=impl)
    assert Fraction(lo) > Fraction(0.81)
    assert math.fsum([1.0 / 3] * 3) == 1.0 - 2.0**-54
    # ...while exact thirds enclose the exact binary64 0.81 itself.
    lo_r, hi_r = O.enclose_rational_weights([0.81] * 3, [Fraction(1, 3)] * 3)
    assert Fraction(lo_r) < Fraction(0.81) < Fraction(hi_r)
    assert hi_r < lo  # the two enclosures are disjoint at 60 digits
    _check_enclosure(impl, [0.81] * 3, None)


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
def test_weights_inside_the_sum_tolerance_are_not_normalised(impl_name):
    impl = IMPLS[impl_name]
    axes = [0.9, 0.5]
    weights = [0.5 + 4e-13, 0.5]  # fsum = 1 + 4e-13, inside the v1 1e-12 tolerance
    assert abs(math.fsum(weights) - 1.0) <= impl.LAMBDA_V1_WEIGHT_SUM_TOL
    out, lo, hi, eps = _check_enclosure(impl, axes, weights)
    total = Fraction(weights[0]) + Fraction(weights[1])
    lo_n, hi_n = O.enclose_rational_weights(axes, [Fraction(w) / total for w in weights])
    # Raw-weight and Fraction-normalised enclosures are disjoint (~1e-13 apart vs 1e-60 wide).
    assert hi < lo_n
    assert Fraction(lo_n) - Fraction(hi) > Fraction(1, 10**14)
    # The kernel output sits in the raw band and outside the normalised band.
    assert not (Fraction(lo_n) - eps <= Fraction(out) <= Fraction(hi_n) + eps)
    # Closed forms straddle sqrt(0.45): raw = sqrt(0.45) * 0.9**4e-13 (below it by ~2.8e-14),
    # normalised = sqrt(0.45) * 1.8**2e-13 (above it by ~7.9e-14); fl(sqrt(0.45)) is within
    # 1 ulp (1.2e-16) of the real, so both comparisons have a wide margin.
    root = Fraction(math.sqrt(0.9 * 0.5))
    assert Fraction(hi) < root < Fraction(lo_n)


# --------------------------------------------------------------------------- #
# Error precedence is the kernel's, by construction                             #
# --------------------------------------------------------------------------- #
def _oracle_code(impl, axes, weights=None) -> str:
    with pytest.raises(ValueError) as caught:
        O.enclose(axes, weights, impl=impl)
    assert isinstance(caught.value, impl.LambdaV1Error)
    return caught.value.code


def _kernel_code(impl, axes, weights=None) -> str:
    with pytest.raises(ValueError) as caught:
        impl.lambda_aggregate(axes, weights)
    assert isinstance(caught.value, impl.LambdaV1Error)
    return caught.value.code


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize("vector", ERROR_VECTORS, ids=[v["id"] for v in ERROR_VECTORS])
def test_fixture_error_vectors_raise_the_kernel_code(impl_name, vector):
    impl = IMPLS[impl_name]
    axes, weights = _inputs(vector)
    want = vector["expect"]["error"]
    assert _oracle_code(impl, axes, weights) == want
    assert _kernel_code(impl, axes, weights) == want
    with pytest.raises(impl.LambdaV1Error) as caught:
        O.certify(axes, weights, impl=impl)
    assert caught.value.code == want


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize(
    "axes, weights, want",
    [
        ([], None, "LAMBDA_EMPTY"),
        ((), (), "LAMBDA_EMPTY"),
        (None, None, "LAMBDA_TYPE_INVALID"),
        ("0.9", None, "LAMBDA_TYPE_INVALID"),
        ([True, 0.9], None, "LAMBDA_TYPE_INVALID"),
        ([0.5, 0.9], {"a": 0.5}, "LAMBDA_TYPE_INVALID"),
        ([1.5, 0.9], None, "LAMBDA_AXIS_OUT_OF_RANGE"),
        ([-5e-324, 0.9], None, "LAMBDA_AXIS_OUT_OF_RANGE"),
        ([0.0, math.nan], None, "LAMBDA_NONFINITE_AXIS"),
        ([0.0, 1.5], None, "LAMBDA_AXIS_OUT_OF_RANGE"),
        ([0.9], [0.5, 0.5], "LAMBDA_LENGTH_MISMATCH"),
        ([0.5, 0.9], [math.inf, 0.5], "LAMBDA_NONFINITE_WEIGHT"),
        ([0.5, 0.9], [1.0, -0.0], "LAMBDA_WEIGHT_NONPOSITIVE"),
        ([0.9, 0.5], [0.5 + 3e-12, 0.5], "LAMBDA_WEIGHT_SUM"),
        ([0.9, 0.5], [1e308, 1e308], "LAMBDA_WEIGHT_SUM"),
    ],
)
def test_weights_none_and_edge_errors_match_the_kernel(impl_name, axes, weights, want):
    impl = IMPLS[impl_name]
    assert _oracle_code(impl, axes, weights) == want
    assert _kernel_code(impl, axes, weights) == want


# --------------------------------------------------------------------------- #
# Every fixture value vector, both copies                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize("vector", VALUE_VECTORS, ids=[v["id"] for v in VALUE_VECTORS])
def test_fixture_value_vectors_are_enclosed(impl_name, vector):
    impl = IMPLS[impl_name]
    axes, weights = _inputs(vector)
    out, lo, hi, eps = _check_enclosure(impl, axes, weights)
    want = O.decode_f64(vector["expect"]["value_f64"])
    if want == 0.0:
        assert lo == hi == 0 and out == 0.0
    # The pinned reference is a float output; it is within the fixture's own tolerance of
    # the exact real, but NOT asserted to lie inside [lo, hi] (two references are not
    # correctly rounded).
    tol = Fraction(vector["value_tol"])
    assert Fraction(lo) - tol <= Fraction(want) <= Fraction(hi) + tol
    report = O.certify(axes, weights, impl=impl)
    assert report["status"] == O.ENCLOSED and report["contained"] is True
    assert report["threshold_relation"] is None and report["tau"] is None
    assert Decimal(report["lo"]) == lo and Decimal(report["hi"]) == hi
    assert report["float_value"] == out


# --------------------------------------------------------------------------- #
# Seeded sweep                                                                  #
# --------------------------------------------------------------------------- #
SWEEP_SEED = 20261006
SWEEP_N = 1000


def _sweep_cases():
    rng = random.Random(SWEEP_SEED)
    cases = []
    for i in range(SWEEP_N):
        k = 97 if i % 250 == 249 else rng.randint(1, 13)
        axes = []
        for _ in range(k):
            r = rng.random()
            if r < 0.05:
                axes.append(5e-324)
            elif r < 0.10:
                axes.append(1.0)
            elif r < 0.15:
                axes.append(math.nextafter(1.0, 0.0))
            elif r < 0.20:
                axes.append(rng.uniform(0.0, 1.0) * 1e-300)
            else:
                axes.append(rng.uniform(1e-6, 1.0))
        mode = rng.random()
        if k == 1:
            weights = None if mode < 0.5 else [1.0]
        elif mode < 0.4:
            weights = None
        elif mode < 0.5:
            tiny = rng.choice([5e-324, 1e-300, 1e-200, 1e-17])
            weights = [tiny] + [(1.0 - tiny) / (k - 1)] * (k - 1)
        else:
            raw = [rng.random() + 1e-3 for _ in range(k)]
            total = math.fsum(raw)
            weights = [r / total for r in raw]
        if weights is not None and abs(math.fsum(weights) - 1.0) > 1e-12:
            weights = None  # keep the case count; the kernel would reject these
        cases.append((axes, weights))
    return cases


def test_sweep_cases_are_deterministic_and_cover_the_edges():
    cases = _sweep_cases()
    assert cases == _sweep_cases()
    assert len(cases) == SWEEP_N
    assert sum(1 for axes, _ in cases if len(axes) == 97) == 4
    flat = [x for axes, _ in cases for x in axes]
    assert 5e-324 in flat and 1.0 in flat and math.nextafter(1.0, 0.0) in flat
    assert any(0 < x < 1e-300 for x in flat)
    assert sum(1 for _, w in cases if w is None) > 300
    assert sum(1 for _, w in cases if w is not None) > 300


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
def test_seeded_sweep_is_enclosed_and_nested(impl_name):
    impl = IMPLS[impl_name]
    evaluated = 0
    failures = []
    for axes, weights in _sweep_cases():
        try:
            _check_enclosure(impl, axes, weights)
        except AssertionError as exc:  # collect, then fail loudly with the first few
            failures.append((axes, weights, str(exc)[:200]))
        evaluated += 1
    assert evaluated == SWEEP_N
    assert failures == [], failures[:5]


# --------------------------------------------------------------------------- #
# Composer floor: the binary64 threshold the code compares against              #
# --------------------------------------------------------------------------- #
THR = C.AXIS_FLOOR - TORCH_EXT.EPS  # fl(0.5 - 1e-9): exactly what _validate compares


def _relation(impl, axes, tau):
    out = impl.lambda_aggregate(axes)
    lo, hi, _, _ = O.enclose_nested(axes, impl=impl)
    eps = O.float_error_bound(axes, None, hi)
    return out, lo, hi, eps, O.classify(lo, hi, eps, tau)


def _axis_floor_validator(running_lambda: float) -> bool:
    return C._validate({"formula_name": "lambda_bounded"}, 0.5, running_lambda)["axis_floor"]


def test_composer_threshold_is_the_binary64_value_not_the_real():
    assert THR == 0.499999999
    assert Fraction(THR) != Fraction(1, 2) - Fraction(1, 10**9)
    assert BUILD.EPS == TORCH_EXT.EPS == 1e-9 and C.AXIS_FLOOR == 0.5


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
def test_composer_floor_relations(impl_name):
    impl = IMPLS[impl_name]
    out, lo, hi, eps, rel = _relation(impl, [0.5, 0.5], THR)
    assert rel == O.SEPARATED_ABOVE and out >= THR and _axis_floor_validator(out) is True
    assert O.certify([0.5, 0.5], None, THR, impl=impl)["status"] == O.ENCLOSED

    out, lo, hi, eps, rel = _relation(impl, [THR], THR)
    assert rel == O.UNKNOWN
    report = O.certify([THR], None, THR, impl=impl)
    assert report["status"] == O.UNKNOWN and report["contained"] is True
    assert report["threshold_relation"] == O.UNKNOWN

    out, lo, hi, eps, rel = _relation(impl, [0.2, 0.3], THR)
    assert rel == O.SEPARATED_BELOW and out < THR and _axis_floor_validator(out) is False
    assert O.certify([0.2, 0.3], None, THR, impl=impl)["status"] == O.ENCLOSED


def _near_threshold_cases():
    rng = random.Random(SWEEP_SEED + 1)
    ulp = math.ulp(THR)
    cases = [[THR + i * ulp] for i in range(-40, 41)]
    two = 2.0 * THR * THR  # [0.5, b] has geometric mean sqrt(0.5 b) ~ THR when b ~ 2 THR²
    cases += [[0.5, two + j * math.ulp(two)] for j in range(-40, 41)]
    for _ in range(300):
        a = rng.uniform(0.3, 0.8)
        b = rng.uniform(0.3, 0.8)
        c = THR**3 / (a * b)
        if 0.0 < c <= 1.0:
            cases.append([a, b, c])
        cases.append([rng.uniform(0.0, 1.0) for _ in range(rng.randint(1, 5))])
    return cases


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
def test_every_separated_relation_agrees_with_the_float_comparison(impl_name):
    impl = IMPLS[impl_name]
    seen = {O.SEPARATED_ABOVE: 0, O.SEPARATED_BELOW: 0, O.UNKNOWN: 0}
    for axes in _near_threshold_cases():
        out, lo, hi, eps, rel = _relation(impl, axes, THR)
        seen[rel] += 1
        if rel == O.SEPARATED_ABOVE:
            assert out >= THR, (axes, out)
        elif rel == O.SEPARATED_BELOW:
            assert out < THR, (axes, out)
        else:
            # UNKNOWN is honest, not lazy: it only happens when the float itself sits
            # within one enclosure width plus two error bands of the threshold.
            slack = Fraction(hi) - Fraction(lo) + 2 * eps
            assert abs(Fraction(out) - Fraction(THR)) <= slack, (axes, out)
    assert all(count > 0 for count in seen.values()), seen


# --------------------------------------------------------------------------- #
# Fixture taus: certification, not gate verdicts                                #
# --------------------------------------------------------------------------- #
EXPECTED_RELATION = {
    "nominal": O.SEPARATED_ABOVE,
    "hidden_weak": O.SEPARATED_BELOW,
    "tie_exact": O.UNKNOWN,
    "tie_multi_axis_at_own_value": O.UNKNOWN,
    # Numerically separated by 4e-10, far outside any float error. szl-lambda-gate still
    # ABSTAINs on these by its NUMERIC_TIE policy band, which the oracle does not model.
    "tie_inside_above": O.SEPARATED_ABOVE,
    "tie_inside_below": O.SEPARATED_BELOW,
    "tie_outside_above": O.SEPARATED_ABOVE,
    "tie_outside_below": O.SEPARATED_BELOW,
    "tau_one": O.SEPARATED_BELOW,
    "zero_axis_min_subnormal_tau": O.SEPARATED_BELOW,
}


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
@pytest.mark.parametrize("vector", TAU_VECTORS, ids=[v["id"] for v in TAU_VECTORS])
def test_fixture_taus_in_unit_interval(impl_name, vector):
    impl = IMPLS[impl_name]
    axes, weights = _inputs(vector)
    tau = O.decode_f64(vector["tau"])
    out, lo, hi, eps = _check_enclosure(impl, axes, weights)
    rel = O.classify(lo, hi, eps, tau)
    if rel == O.SEPARATED_ABOVE:
        assert out >= tau
    elif rel == O.SEPARATED_BELOW:
        assert out < tau
    else:
        assert rel == O.UNKNOWN
        assert abs(Fraction(out) - Fraction(tau)) <= Fraction(hi) - Fraction(lo) + 2 * eps
    if vector["id"] in EXPECTED_RELATION:
        assert rel == EXPECTED_RELATION[vector["id"]], vector["id"]
    report = O.certify(axes, weights, tau, impl=impl)
    assert report["threshold_relation"] == rel
    assert report["status"] == (O.UNKNOWN if rel == O.UNKNOWN else O.ENCLOSED)
    assert report["tau"] == repr(tau)


def test_fixture_tau_filter_covers_the_tie_family():
    ids = {v["id"] for v in TAU_VECTORS}
    assert {"tie_exact", "tie_inside_above", "tie_inside_below", "tie_multi_axis_at_own_value",
            "tau_one", "zero_axis_min_subnormal_tau"} <= ids
    # Non-binary64 or out-of-(0,1] taus are gate-only cases and are not classified here.
    assert not {"tau_nan", "tau_bool", "tau_null", "tau_above_one", "tau_negative",
                "tau_zero_would_admit_veto"} & ids


# --------------------------------------------------------------------------- #
# classify / certify API                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tau", [True, False, None, math.nan, math.inf, "0.8", Decimal("0.8")])
def test_classify_requires_the_exact_binary64_tau(tau):
    with pytest.raises(ValueError):
        O.classify(Decimal("0.4"), Decimal("0.6"), Fraction(0), tau)


def test_classify_uses_exact_comparison_against_the_code_threshold():
    # The code compares against the binary64 fl(0.5 - 1e-9), which is 2.7e-17 BELOW the
    # decimal literal 0.499999999. A degenerate enclosure at the decimal literal is
    # therefore already SEPARATED_ABOVE: the oracle never substitutes the real threshold.
    literal = Decimal("0.499999999")
    assert Fraction(THR) < Fraction(literal)
    assert O.classify(literal, literal, Fraction(0), THR) == O.SEPARATED_ABOVE
    exact = Fraction(THR)
    assert O.classify(exact, exact, Fraction(0), THR) == O.SEPARATED_ABOVE  # lo - 0 >= tau: ">="
    tiny = Fraction(1, 10**40)
    assert O.classify(exact - tiny, exact - tiny, Fraction(0), THR) == O.SEPARATED_BELOW
    assert O.classify(exact, exact, tiny, THR) == O.UNKNOWN
    assert O.classify(exact - tiny, exact - tiny, tiny, THR) == O.UNKNOWN
    with pytest.raises(ValueError):
        O.classify(exact, exact, Fraction(-1), THR)


@pytest.mark.parametrize("impl_name", IMPL_NAMES)
def test_certify_report_shape(impl_name):
    report = O.certify([0.95, 0.92, 0.88, 0.9], [0.25] * 4, 0.8, impl=IMPLS[impl_name])
    assert set(report) >= {
        "scope", "kernel", "prec", "check_prec", "nested", "lo", "hi", "width", "float_value",
        "eps", "contained", "ulp_outside", "tau", "threshold_relation", "assumptions", "status",
    }
    assert report["status"] == O.ENCLOSED and report["threshold_relation"] == O.SEPARATED_ABOVE
    assert report["prec"] == 60 and report["check_prec"] == 90 and report["nested"] is True
    assert isinstance(report["float_value"], float)
    assert Decimal(report["width"]) < Decimal("1e-50")
    assert Decimal(report["eps"]) < Decimal("1e-14")
    assert report["ulp_outside"] >= 0.0
    assert report["assumptions"] == list(O.ASSUMPTIONS)
    assert "NOT a gate verdict" in report["scope"] and "Conjecture 1" in report["scope"]
    json.dumps(report)  # plain dict, serialisable


def test_float_error_bound_scales_with_hi_and_floors_subnormals():
    eps_half = O.float_error_bound([0.5, 0.5], None, Decimal("0.5"))
    assert 0 < eps_half < Fraction(1, 10**14)
    eps_tiny = O.float_error_bound([5e-324], [1.0], Decimal(5e-324))
    assert eps_tiny >= O.LIBM_ULP_ASSUMPTION * O.TINY
    assert eps_tiny < Fraction(1, 10**300)
    assert O.float_error_bound([0.0, 0.5], None, Decimal(0)) == 0
    assert O.abs_ln_upper(1.0) == 0
    assert O.abs_ln_upper(0.5) > Fraction(6931471805599453, 10**16)  # > ln 2 lower digits


def test_nesting_failure_is_an_error_not_a_fallback():
    assert issubclass(O.OracleInconsistency, RuntimeError)
    with pytest.raises(O.OracleInconsistency):
        O.float_error_bound([0.5], [1.0], Decimal(1), libm_ulps=10**20)  # d_s > 1 is refused
