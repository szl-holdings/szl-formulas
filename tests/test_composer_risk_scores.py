# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""A governed-loop step cannot buy advisory trust with a non-finite or vacuous output.

At 304e54b (#23) only ``pac_bayes_mcallester`` was fail-closed. Every other
RISK_LIKE formula still went through the generic projection ``1 - 1/(1+v)`` for
``v > 1``, so a larger risk earned a larger advisory scalar (2.0 -> 0.667,
1e17 -> 1.0), and +inf, -inf, NaN and negative outputs scored 1.0 (maximum)
with every validator passing and no halt. ``False`` and ``[]`` also scored
1.0, and ``1j``, ``Fraction`` and ``Decimal`` outputs scored 0.5 whatever their
value. STRUCTURAL steps scored 1.0 for every output, non-finite ones included.
Inputs the formulas accept reach this:
``bekenstein_cascade(inf, 1.0)`` and ``bekenstein_cascade(1e300, 1e300)``
return +inf, ``bekenstein_cascade(nan, 1.0)`` returns NaN,
``pinsker_kl_bound([1.7e308, -1.7e308, 1.0], [-1.7e308, 1.7e308, 1.0])``
returns +inf, and ``reed_solomon_singleton(inf, 1.0)`` and
``reed_solomon_singleton(Decimal("Infinity"), 1)`` return a non-finite value.

Every case runs through every shipped package copy (see conftest.py): torch-ext
and the publisher's build/torch-cpu and build/torch-universal. Stdlib + pytest
only. Λ remains Conjecture 1 (open).
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_package(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COPIES = ["torch-ext", "build/torch-cpu", "build/torch-universal"]  # see conftest.py
_PACKAGES: dict = {}


@pytest.fixture
def package(request, shipped_root):
    """One shipped copy of the package, loaded once under an isolated name."""
    copy = request.param
    if copy not in _PACKAGES:
        _PACKAGES[copy] = _load_package(f"_composer_risk_test_{COPIES.index(copy)}",
                                        shipped_root(copy) / "szl_formulas")
    return _PACKAGES[copy]


PACKAGE_PARAMS = pytest.mark.parametrize("package", COPIES, indirect=True)

INF = float("inf")
NAN = float("nan")
PREFIX = [{"formula_name": "reed_solomon_singleton", "args": [255, 223]}]


class _FloatLike:
    """A number that is not a float: it only converts through ``__float__``."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"_FloatLike({self.value!r})"


class _OwnTypeFloat(float):
    """A float subclass whose subtraction keeps its own type, as numpy.float64 does."""

    def __rsub__(self, other):
        return _OwnTypeFloat(float(other) - float(self))


def _composer(package):
    return importlib.import_module(f"{package.__name__}._composer")


@PACKAGE_PARAMS
@pytest.mark.parametrize(
    "name,args",
    [
        pytest.param("bekenstein_cascade", [INF, 1.0], id="bekenstein-inf"),
        pytest.param("bekenstein_cascade", [NAN, 1.0], id="bekenstein-nan"),
        pytest.param("bekenstein_cascade", [1e300, 1e300], id="bekenstein-overflow"),
        pytest.param(
            "pinsker_kl_bound",
            [[1.7e308, -1.7e308, 1.0], [-1.7e308, 1.7e308, 1.0]],
            id="pinsker-overflow",
        ),
        pytest.param("reed_solomon_singleton", [INF, 1.0], id="reed-solomon-inf"),
        pytest.param("reed_solomon_singleton", [NAN, 1.0], id="reed-solomon-nan"),
        pytest.param(
            "reed_solomon_singleton", [Decimal("Infinity"), 1], id="reed-solomon-decimal-inf"
        ),
    ],
)
def test_nonfinite_output_from_accepted_inputs_halts_before_a_receipt(package, name, args):
    out = package.REGISTRY[name](*args)
    assert not math.isfinite(out)
    prefix = package.run_governed_loop(PREFIX)
    calls = PREFIX + [{"formula_name": name, "args": args}]
    chain = package.run_governed_loop(calls)
    assert chain["halted"] is True
    assert chain["halt_reason"].startswith(f"step 1 ({name}) raised:")
    assert chain["receipts"] == prefix["receipts"]
    assert chain["root_hash"] == prefix["root_hash"]
    assert chain["lambda_aggregate"] == prefix["lambda_aggregate"]
    assert chain["replay_ok"] is True


@PACKAGE_PARAMS
@pytest.mark.parametrize(
    "output",
    [
        pytest.param(INF, id="inf"),
        pytest.param(-INF, id="-inf"),
        pytest.param(NAN, id="nan"),
        pytest.param(-5e-324, id="negative-subnormal"),
        pytest.param(-1.0, id="negative"),
        pytest.param(True, id="True"),
        pytest.param(False, id="False"),
        pytest.param(None, id="None"),
        pytest.param("0.1", id="str"),
        pytest.param([], id="empty-list"),
        pytest.param(1j, id="complex"),
        # Only int and float (and float subclasses) are read as risk values;
        # other numeric types fail closed instead of scoring a value-blind 0.5.
        pytest.param(Fraction(1, 4), id="Fraction"),
        pytest.param(Decimal("0.25"), id="Decimal"),
    ],
)
def test_invalid_risk_output_halts_before_a_receipt(package, monkeypatch, output):
    for name in sorted(_composer(package).RISK_LIKE):
        monkeypatch.setitem(package.REGISTRY, name, lambda: output)
        chain = package.run_governed_loop([{"formula_name": name, "args": []}])
        assert chain["halted"] is True, name
        assert chain["halt_reason"].startswith(f"step 0 ({name}) raised:"), name
        assert chain["receipts"] == []
        assert chain["lambda_aggregate"] == 0.0
        assert chain["root_hash"] == package.GENESIS
        assert chain["replay_ok"] is True


@PACKAGE_PARAMS
@pytest.mark.parametrize(
    "output",
    [
        pytest.param(INF, id="inf"),
        pytest.param(-INF, id="-inf"),
        pytest.param(NAN, id="nan"),
        pytest.param(Decimal("Infinity"), id="Decimal-inf"),
        pytest.param(Decimal("-Infinity"), id="Decimal-negative-inf"),
        pytest.param(Decimal("NaN"), id="Decimal-nan"),
        pytest.param(Decimal("sNaN"), id="Decimal-snan"),
        pytest.param(complex(INF, 0.0), id="complex-inf"),
        pytest.param(complex(0.0, NAN), id="complex-nan"),
        pytest.param(_FloatLike(INF), id="float-like-inf"),
        pytest.param(_FloatLike(NAN), id="float-like-nan"),
        # A number whose conversion fails cannot be shown to be finite either.
        pytest.param(_FloatLike("not a float"), id="float-like-broken"),
    ],
)
def test_nonfinite_structural_output_halts_before_a_receipt(package, monkeypatch, output):
    for name in sorted(_composer(package).STRUCTURAL):
        monkeypatch.setitem(package.REGISTRY, name, lambda: output)
        chain = package.run_governed_loop([{"formula_name": name, "args": []}])
        assert chain["halted"] is True, name
        assert chain["halt_reason"].startswith(f"step 0 ({name}) raised:"), name
        assert chain["receipts"] == []
        assert chain["replay_ok"] is True


@PACKAGE_PARAMS
@pytest.mark.parametrize(
    "output",
    [
        pytest.param(33, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(10**400, id="int-above-float-range"),
        pytest.param(Fraction(10**400, 3), id="Fraction-above-float-range"),
        pytest.param(Decimal("1e400"), id="Decimal-above-float-range"),
        pytest.param(Decimal("33"), id="Decimal"),
        pytest.param(_FloatLike(33.0), id="float-like"),
        pytest.param({"verdict": "test"}, id="dict"),
    ],
)
def test_finite_or_non_numeric_structural_output_still_scores_one(package, output):
    # Exact values are never converted to float, so a finite value above float
    # range is not mistaken for infinity; a non-number keeps the 1.0 that #23
    # pinned for a dict.
    composer = _composer(package)
    for name in sorted(composer.STRUCTURAL):
        assert composer._to_scalar(output, name) == 1.0, name


@PACKAGE_PARAMS
def test_risk_score_never_increases_with_risk(package):
    composer = _composer(package)
    risks = [
        0, 0.0, 0.25, 0.5, math.nextafter(1.0, 0.0),
        1, 1.0, math.nextafter(1.0, INF), 1.5, 2.0, math.pi, 10.0, 1e17,
        sys.float_info.max,
    ]
    for name in sorted(composer.RISK_LIKE):
        scores = [composer._to_scalar(risk, name) for risk in risks]
        assert all(0.0 <= score <= 1.0 for score in scores), name
        assert all(left >= right for left, right in zip(scores, scores[1:])), name
        assert scores[:5] == [1.0, 1.0, 0.75, 0.5, 2**-53], name
        assert scores[5:] == [0.0] * (len(risks) - 5), name
        # An integer above float range is compared exactly and saturates too.
        assert composer._to_scalar(10**400, name) == 0.0, name


@PACKAGE_PARAMS
def test_risk_score_is_a_builtin_float_for_float_subclasses(package, monkeypatch):
    # numpy.float64 subclasses float but keeps its own type in arithmetic, so
    # ``1.0 - out`` alone would put a non-builtin scalar into the receipt.
    composer = _composer(package)
    for name in sorted(composer.RISK_LIKE):
        for risk, expected in ((0.25, 0.75), (0.0, 1.0), (2.0, 0.0)):
            score = composer._to_scalar(_OwnTypeFloat(risk), name)
            assert type(score) is float and score == expected, (name, risk)
    monkeypatch.setitem(package.REGISTRY, "hoeffding_tail", lambda: _OwnTypeFloat(0.25))
    chain = package.run_governed_loop([{"formula_name": "hoeffding_tail", "args": []}])
    assert chain["halted"] is False
    assert type(chain["receipts"][0]["scalar"]) is float
    assert chain["receipts"][0]["scalar"] == 0.75
    assert chain["replay_ok"] is True


@PACKAGE_PARAMS
@pytest.mark.parametrize(
    "name,args",
    [
        pytest.param("bekenstein_cascade", [1.0, 1.0], id="bekenstein-huge"),
        pytest.param("pinsker_kl_bound", [[1.0, 0.0], [0.0, 1.0]], id="pinsker-disjoint"),
        pytest.param("fisher_rao_distance", [[1.0, 0.0], [0.0, 1.0]], id="fisher-rao-disjoint"),
    ],
)
def test_vacuous_risk_from_accepted_inputs_scores_zero_and_halts(package, name, args):
    out = package.REGISTRY[name](*args)
    assert math.isfinite(out) and out > 1
    chain = package.run_governed_loop([{"formula_name": name, "args": args}])
    assert chain["receipts"][-1]["scalar"] == 0.0
    assert chain["receipts"][-1]["output_repr"] == repr(out)[:120]
    assert chain["halted"] is True
    assert chain["halt_reason"] == f"step 0 ({name}) HALT on validators ['axis_floor']"
    assert chain["root_hash"] == package.GENESIS
    assert chain["replay_ok"] is True


@PACKAGE_PARAMS
def test_valid_structural_and_in_range_risk_steps_are_unchanged(package):
    chain = package.run_governed_loop(
        PREFIX + [{"formula_name": "hoeffding_tail", "args": [0.2, 100]}]
    )
    bound = package.hoeffding_tail(0.2, 100)
    assert chain["halted"] is False
    assert [r["scalar"] for r in chain["receipts"]] == [1.0, 1.0 - bound]
    assert chain["replay_ok"] is True
