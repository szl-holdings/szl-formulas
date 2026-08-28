# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Honest, FALSIFIABLE tests for szl_formulas.

Every test asserts a property that a real regression could break:
- the registry is exactly the canonical 21;
- the locked-proven canonical set is EXACTLY 8 (never inflated);
- PROOF_STATUS is mirrored verbatim from the dataset (never coerced);
- the formulas are numerically correct AND their guards actually reject bad input;
- the governed loop replays clean but detects tamper and halts below the Λ floor;
- Λ stays Conjecture 1 (open) — never labelled "proven".
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build", "torch-universal"))

import szl_formulas as F  # noqa: E402


# --------------------------------------------------------------------------- #
# Registry + locked-proven honesty                                            #
# --------------------------------------------------------------------------- #
def test_registry_is_exactly_21():
    assert F.registry_count() == 21
    assert len(F.REGISTRY) == 21


def test_proof_status_covers_every_formula_verbatim():
    # Every registry formula has a status, and there are no extra/orphan statuses.
    assert set(F.PROOF_STATUS) == set(F.REGISTRY)
    # Mirrored verbatim from the canonical dataset (spot-check exact strings).
    assert F.proof_status("madhava_series") == "PROVEN(alternating series)"
    assert F.proof_status("pinsker_kl_bound") == "AXIOM(pinsker)"
    assert F.proof_status("pac_bayes_mcallester") == "SORRY(PACBayes)"
    assert F.proof_status("lambda_aggregate") == "PROVEN(A1-A4); uniqueness CONJECTURE"


def test_proof_status_unknown_raises_never_coerced():
    with pytest.raises(KeyError):
        F.proof_status("not_a_real_formula")


def test_locked_proven_set_is_exactly_eight():
    assert F.LOCKED_PROVEN_COUNT == 8
    assert F.LOCKED_PROVEN_FORMULA_IDS == frozenset(
        {"F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"}
    )
    assert len(F.LOCKED_PROVEN_FORMULA_IDS) == 8


def test_lambda_stays_conjecture_one_everywhere():
    assert "Conjecture 1" in F.lambda_status()
    assert "proven" not in F.lambda_status().lower().replace("unproven", "")
    assert "ADVISORY" in F.LAMBDA_LABEL
    assert "Conjecture 1" in F.LAMBDA_LABEL


# --------------------------------------------------------------------------- #
# Formula correctness + falsifiable guards                                     #
# --------------------------------------------------------------------------- #
def test_lambda_aggregate_uniform_is_geomean():
    assert F.lambda_aggregate([0.9, 0.9, 0.9]) == pytest.approx(0.9)
    # weighted geometric mean, not arithmetic — a spread vector must sit BELOW
    # the arithmetic mean (this is what makes Λ non-compensatory).
    lam = F.lambda_aggregate([0.2, 0.8])
    assert lam < (0.2 + 0.8) / 2.0
    assert lam == pytest.approx(math.sqrt(0.2 * 0.8))


def test_lambda_aggregate_zero_pins_to_zero():
    # non-compensatory: a single zero axis drives the whole aggregate to 0.
    assert F.lambda_aggregate([0.99, 0.0, 0.99]) == 0.0


def test_lambda_aggregate_rejects_bad_weights():
    with pytest.raises(ValueError):
        F.lambda_aggregate([0.5, 0.5], weights=[0.3, 0.3])  # do not sum to 1
    with pytest.raises(ValueError):
        F.lambda_aggregate([])  # empty


def test_lambda_bounded_holds_and_is_falsifiable():
    # A4: Λ(x) <= max(x). True on any valid vector.
    assert F.lambda_bounded([0.2, 0.8, 0.5]) is True
    # Falsifiable framing: the aggregate is genuinely <= max, never above it.
    x = [0.3, 0.7, 0.9]
    assert F.lambda_aggregate(x) <= max(x) + F.EPS


def test_reed_solomon_singleton_exact():
    assert F.reed_solomon_singleton(255, 223) == 33
    with pytest.raises(ValueError):
        F.reed_solomon_singleton(10, 20)  # k > n


def test_madhava_series_approximates_pi():
    # 4 * atan(1) = pi; convergence is slow but monotone-ish — many terms needed.
    approx_pi = 4.0 * F.madhava_series(1.0, 20000)
    assert abs(approx_pi - math.pi) < 1e-3


def test_hoeffding_tail_is_a_probability():
    p = F.hoeffding_tail(0.1, 100)
    assert 0.0 <= p <= 1.0
    # larger deviation => smaller tail bound (falsifiable monotonicity).
    assert F.hoeffding_tail(0.2, 100) < F.hoeffding_tail(0.1, 100)


# --------------------------------------------------------------------------- #
# Governed-loop composer — replay + tamper + Λ-floor halt                      #
# --------------------------------------------------------------------------- #
def test_clean_loop_replays_ok():
    chain = F.run_governed_loop([
        {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
        {"formula_name": "reed_solomon_singleton", "args": [255, 223]},
    ])
    assert chain["halted"] is False
    assert chain["replay_ok"] is True
    assert "ADVISORY" in chain["lambda_label"]


def test_tampered_receipt_is_detected():
    calls = [{"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]}]
    chain = F.run_governed_loop(calls)
    assert F.verify_chain(chain, calls) is True
    # flip one receipt hash → verification MUST fail (falsifiable).
    chain["receipts"][0]["receipt_hash"] = "0" * 64
    assert F.verify_chain(chain, calls) is False


def test_axis_floor_halts_the_loop():
    # hoeffding_tail is RISK_LIKE => scalar = 1 - bound; a saturated bound gives
    # scalar 0, dropping the running Λ below AXIS_FLOOR and HALTING the loop.
    chain = F.run_governed_loop([
        {"formula_name": "hoeffding_tail", "args": [0.0, 1]},
    ])
    assert chain["halted"] is True
    assert "axis_floor" in (chain["halt_reason"] or "")


def test_unknown_formula_halts_not_crashes():
    chain = F.run_governed_loop([{"formula_name": "does_not_exist", "args": []}])
    assert chain["halted"] is True
    assert "unknown formula" in (chain["halt_reason"] or "")


# --------------------------------------------------------------------------- #
# selfcheck end-to-end                                                         #
# --------------------------------------------------------------------------- #
def test_selfcheck_demonstrates_falsifiability():
    sc = F.selfcheck()
    assert sc["registry_count"] == 21
    assert sc["locked_count_is_eight"] is True
    assert sc["proof_status_covers_all"] is True
    assert sc["clean_replay_ok"] is True
    assert sc["tamper_detected"] is True
    assert sc["falsifiable_demonstrated"] is True
    assert "Conjecture 1" in sc["lambda_status"]
