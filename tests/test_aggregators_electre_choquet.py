"""ELECTRE III + Choquet sibling operators. Not locked-8. Conjecture 1 stays OPEN."""
from __future__ import annotations

import math

from szl_formulas._aggregators import (
    LOCKED_EIGHT,
    YUYAY_FLOORS,
    a2_homogeneity_degree_one,
    additive_capacity,
    choquet_discrete,
    choquet_mobius_2additive,
    default_yuyay_thresholds,
    electre_gate_vs_profile,
    electre_iii,
    min_capacity,
    reed_solomon_singleton,
)


def test_locked_eight_untouched() -> None:
    assert LOCKED_EIGHT == ("F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22")


def test_a2_degree_one_not_c_to_n() -> None:
    result = a2_homogeneity_degree_one([0.8, 0.5, 0.9], [1 / 3, 1 / 3, 1 / 3], c=2.0)
    assert result["ok"] is True
    assert result["status"] == "CHECKED"


def test_electre_vetoes_when_lambda_would_allow() -> None:
    scores = [0.95] * 12 + [0.30]
    r = electre_gate_vs_profile(scores)
    assert r.state == "MEASURED"
    assert r.credibility == 0.0
    assert r.veto_axes


def test_electre_shape_mismatch_blocks() -> None:
    th = default_yuyay_thresholds()
    r = electre_iii([0.9, 0.9], [0.9] * 13, th)
    assert r.state == "BLOCKED"
    assert "SHAPE_MISMATCH" in r.reason_codes


def test_choquet_additive_recovers_wam() -> None:
    x = [0.2, 0.4, 0.8]
    w = [0.2, 0.3, 0.5]
    cap = additive_capacity(w)
    c = choquet_discrete(x, cap)
    assert c.state == "MEASURED"
    assert abs(c.value - sum(a * b for a, b in zip(x, w))) < 1e-9


def test_choquet_min_capacity_is_min() -> None:
    x = [0.9, 0.4, 0.7]
    c = choquet_discrete(x, min_capacity(len(x)))
    assert abs(c.value - min(x)) < 1e-12


def test_choquet_2additive_zero_interaction_is_shapley() -> None:
    x = [0.6, 0.6, 0.6]
    shapley = [1 / 3, 1 / 3, 1 / 3]
    c = choquet_mobius_2additive(x, shapley, {})
    assert c.state == "MEASURED"
    assert abs(c.value - 0.6) < 1e-12


def test_uniform_floors_finite() -> None:
    r = electre_gate_vs_profile(list(YUYAY_FLOORS))
    assert r.state == "MEASURED"
    assert r.credibility is not None and math.isfinite(r.credibility)


def test_reed_solomon_operational_fixture_is_defined_and_bounded() -> None:
    measured = reed_solomon_singleton()
    assert measured["ok"] is True
    assert measured["status"] == "CHECKED"
    assert measured["d"] == 5

    unsupported = reed_solomon_singleton(n=11, k=6)
    assert unsupported["ok"] is False
    assert unsupported["status"] == "FAILED"
