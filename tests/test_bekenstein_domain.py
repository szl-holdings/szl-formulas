# SPDX-License-Identifier: Apache-2.0
"""Finite physical-input contract; not the locked F19 or a physical proof.

Stdlib + pytest only. Exercise source and both generated Hub package variants.
The historical expression is retained as an independent bit-equivalence oracle.
"""
import importlib.util
import math
import random
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

COPIES = ["torch-ext", "build/torch-cpu", "build/torch-universal"]


@pytest.fixture(params=COPIES)
def package(request, shipped_root):
    path = shipped_root(request.param) / "szl_formulas"
    name = "_bekenstein_domain_" + str(COPIES.index(request.param))
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def historical(r, e):
    if r < 0.0 or e < 0.0:
        raise ValueError("R and E must be >= 0")
    return (2.0 * math.pi * r * e) / (1.054571817e-34 * 299792458.0)


@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"),
                                -float("inf"), -1.0, "1", None, 1j, Decimal("1")])
def test_invalid_input_is_value_error(package, field, bad):
    args = [1.0, 1.0]
    args[field] = bad
    with pytest.raises(ValueError):
        package.bekenstein_cascade(*args)


def test_true_float_output_overflow_rejected(package):
    with pytest.raises(ValueError, match="finite"):
        package.bekenstein_cascade(1e308, 1e308)


@pytest.mark.parametrize("r,e", [(1e308, 1e-308), (10**400, Fraction(1, 10**400)),
                                (Fraction(10**400), Fraction(1, 10**400)),
                                (10**400, 0.0)])
def test_avoidable_intermediate_overflow_recovers_finite_result(package, r, e):
    coefficient = Fraction(2.0 * math.pi)
    denominator = Fraction(1.054571817e-34 * 299792458.0)
    expected = float(coefficient * Fraction(r) * Fraction(e) / denominator)
    actual = package.bekenstein_cascade(r, e)
    assert isinstance(actual, float)
    assert math.isfinite(actual)
    assert actual == expected


@pytest.mark.parametrize("r,e", [(0.0, 1.0), (1.0, 0.0), (-0.0, 1.0),
                                (0, 0), (Fraction(1, 3), Fraction(2, 7)),
                                (math.ulp(0.0), 1.0), (1.0, math.ulp(0.0)),
                                (1e-200, 1e-200), (1e100, 1e100), (1e280, 1.0)])
def test_existing_finite_boundary_bits_preserved(package, r, e):
    expected = historical(r, e)
    assert math.isfinite(expected)
    actual = package.bekenstein_cascade(r, e)
    assert type(actual) is type(expected)
    assert actual.hex() == expected.hex()


def test_finite_forge_range_and_wide_domain_bits_preserved(package):
    rng = random.Random(20261103)
    pairs = [(rng.uniform(0, 10), rng.uniform(0, 1e-10)) for _ in range(250)]
    pairs += [(10**rng.uniform(-250, 100), 10**rng.uniform(-250, 100))
              for _ in range(250)]
    for r, e in pairs:
        expected = historical(r, e)
        assert math.isfinite(expected)
        assert package.bekenstein_cascade(r, e).hex() == expected.hex()


def test_float_max_exact_rational_boundary(package):
    coefficient = Fraction(2.0 * math.pi)
    denominator = Fraction(1.054571817e-34 * 299792458.0)
    r = Fraction(sys.float_info.max) * denominator / coefficient
    # Historical finite rounding is preserved if present; otherwise fallback
    # must not refuse the exactly representable mathematical boundary.
    old = historical(r, 1.0)
    expected = old if math.isfinite(old) else sys.float_info.max
    assert package.bekenstein_cascade(r, 1.0) == expected


def test_float_subclasses_remain_supported(package):
    class RealFloat(float):
        pass
    assert package.bekenstein_cascade(RealFloat(0.001), RealFloat(1e-9)) == historical(
        RealFloat(0.001), RealFloat(1e-9)
    )


def test_registry_identity_and_composer_failure_boundary(package):
    assert package.registry_count() == 21
    assert package.REGISTRY["bekenstein_cascade"] is package.bekenstein_cascade
    assert package.PROOF_STATUS["bekenstein_cascade"] == "PROVEN(TH6 DPI form); dimensional helper"
    assert package.LOCKED_PROVEN_COUNT == 8
    result = package.run_governed_loop([
        {"formula_name": "bekenstein_cascade", "args": [True, 1.0]},
    ])
    assert result["halted"] is True
    assert "bekenstein_cascade" in result["halt_reason"]
