# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Fail-closed domain contract for ``szl_formulas.pac_bayes_mcallester``.

Before this contract (fca9707) the formula swallowed a NaN KL through
``max(0.0, complexity)`` and returned the bare empirical risk, accepted
``n=True`` (as 1), fractional and non-finite ``n``, a negative or >1 empirical
risk, and raised OverflowError (not ValueError) for ``n=10**400`` or an int
``kl`` above float range. The governed loop then produced a HIGHER advisory scalar for the NaN case (0.9)
than for the valid baseline (0.836...), and 1.0 for ``empirical_risk=-5.0``,
without halting.

Every case runs through BOTH package copies: ``torch-ext`` (what this repo
imports) and ``build/torch-universal`` (what the Hub loads), following the
IMPLS pattern of tests/test_lambda_v1_conformance.py. Stdlib + pytest only.

The float expression for a valid input is unchanged, so every input the new
domain accepts evaluates to the same float as before; only the domain moved.
PROOF_STATUS stays "SORRY(PACBayes)" verbatim. Λ remains Conjecture 1 (open).
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-ext"))

import szl_formulas as F  # noqa: E402
from szl_formulas import _formulas as TORCH_EXT  # noqa: E402

BUILD_FILE = ROOT / "build" / "torch-universal" / "szl_formulas" / "_formulas.py"


def _load_by_path(name: str, path: Path):
    # _formulas.py imports only the stdlib, so the Hub copy loads standalone.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load_by_path("szl_formulas_build_universal_formulas_pac_bayes", BUILD_FILE)
IMPLS = {"torch-ext": TORCH_EXT, "build/torch-universal": BUILD}
IMPL_PARAMS = pytest.mark.parametrize("impl", list(IMPLS.values()), ids=list(IMPLS))

NAN = float("nan")
INF = float("inf")

# Valid baseline recorded at fca9707; must stay bit-exact.
BASELINE_ARGS = (0.1, 1.0, 1000, 0.05)
BASELINE_VALUE = 0.1638073549585195
BASELINE_HEX = "0x1.4f7a3b0320fb8p-3"
# Smallest valid n (Maurer 2004); value recorded at fca9707 with the same float expression.
MIN_N_ARGS = (0.1, 1.0, 8, 0.05)
MIN_N_VALUE = 0.6983623601628163

# scripts/forge.py:52-54: the surrogate generator ranges for this formula.
FORGE_RISK = (0.0, 0.5)
FORGE_KL = (0.0, 5.0)
FORGE_N = (10, 5000)
FORGE_DELTA = (0.01, 0.5)
FORGE_SEED = 20260721
FORGE_SAMPLES = 500

# (args, exact message fragment, id). Each formerly returned a number or leaked
# a non-ValueError; each must now raise ValueError naming the offending field.
REJECTED = [
    ((0.1, NAN, 1000, 0.05), "kl must be finite", "kl-nan"),
    ((0.1, INF, 1000, 0.05), "kl must be finite", "kl-inf"),
    ((0.1, -1.0, 1000, 0.05), "KL divergence must be >= 0", "kl-negative"),
    ((0.1, 10**400, 1000, 0.05), "kl is too large for float arithmetic", "kl-10-pow-400"),
    ((0.1, 2**1024, 1000, 0.05), "kl is too large for float arithmetic", "kl-2-pow-1024"),
    ((0.1, 1.0, INF, 0.05), "n must be finite", "n-inf"),
    ((0.1, 1.0, NAN, 0.05), "n must be finite", "n-nan"),
    ((0.1, 1.0, True, 0.05), "n must be a real number", "n-bool"),
    ((0.1, 1.0, 1000.9, 0.05), "n must be an int", "n-fractional"),
    ((0.1, 1.0, 1000.0, 0.05), "n must be an int", "n-integral-float"),
    ((0.1, 1.0, 1e-6, 0.05), "n must be an int", "n-tiny-float"),
    ((0.1, 1.0, 0, 0.05), "n must be >= 8", "n-zero"),
    ((0.1, 1.0, -1000, 0.05), "n must be >= 8", "n-negative"),
    ((0.1, 1.0, 1, 0.05), "n must be >= 8", "n-one"),
    ((0.1, 1.0, 7, 0.05), "n must be >= 8", "n-seven"),
    ((0.1, 1.0, 10**400, 0.05), "n is too large for float arithmetic", "n-10-pow-400"),
    ((0.1, 1.0, 2**1023, 0.05), "n is too large for float arithmetic", "n-2-pow-1023"),
    ((NAN, 1.0, 1000, 0.05), "empirical_risk must be finite", "risk-nan"),
    ((INF, 1.0, 1000, 0.05), "empirical_risk must be finite", "risk-inf"),
    ((-INF, 1.0, 1000, 0.05), "empirical_risk must be finite", "risk-neg-inf"),
    ((-5.0, 1.0, 1000, 0.05), "empirical_risk must be in [0,1]", "risk-negative"),
    ((2.0, 1.0, 1000, 0.05), "empirical_risk must be in [0,1]", "risk-two"),
    ((1.0000001, 1.0, 1000, 0.05), "empirical_risk must be in [0,1]", "risk-just-above-one"),
    ((0.1, 1.0, 1000, 0.0), "delta must be in (0,1)", "delta-zero"),
    ((0.1, 1.0, 1000, 1.0), "delta must be in (0,1)", "delta-one"),
    ((0.1, 1.0, 1000, -0.05), "delta must be in (0,1)", "delta-negative"),
    ((0.1, 1.0, 1000, NAN), "delta must be finite", "delta-nan"),
    ((0.1, 1.0, 1000, INF), "delta must be finite", "delta-inf"),
    (("0.1", 1.0, 1000, 0.05), "empirical_risk must be a real number", "risk-str"),
    ((True, 1.0, 1000, 0.05), "empirical_risk must be a real number", "risk-bool"),
    ((0.1, None, 1000, 0.05), "kl must be a real number", "kl-none"),
    ((0.1, False, 1000, 0.05), "kl must be a real number", "kl-bool"),
    ((0.1, 1.0, "1000", 0.05), "n must be a real number", "n-str"),
    ((0.1, 1.0, 1000, True), "delta must be a real number", "delta-bool"),
    ((0.1, 1.0, 1000, [0.05]), "delta must be a real number", "delta-list"),
    # 2*sqrt(n)/delta overflows a float: formerly returned inf, now an error.
    ((0.1, 0.0, 5000, 5e-324), "2*sqrt(n)/delta overflowed", "delta-denormal-overflow"),
]

# A matrix of garbage and edge values in every argument position. Any
# exception other than ValueError escaping the formula fails the contract.
GARBAGE = [
    NAN, INF, -INF, True, False, None, "1", [1], -1, -0.5, 0, 1, 2, 7, 8, 1e-6,
    1000.0, 1000.9, 10**400, 2**1023, 2**1024, 5e-324, 1.0000001, 0.05, 0.5,
]

ACCEPTED_BOUNDARIES = [
    pytest.param((0.0, 0.0, 8, 0.5), id="risk0-kl0-n8"),
    pytest.param((1.0, 0.0, 8, 0.999), id="risk1-boundary"),
    pytest.param((0, 0, 8, 0.5), id="int-risk-and-kl"),
    pytest.param((0.5, 1e6, 10**6, 1e-300), id="large-kl-tiny-delta"),
    pytest.param((0.1, 2**1023, 1000, 0.05), id="huge-int-kl-still-a-finite-float"),
    pytest.param((0.1, 1.0, 2**1000, 0.05), id="huge-n-still-a-finite-float"),
]


def _forge_samples(count: int = FORGE_SAMPLES):
    rng = random.Random(FORGE_SEED)
    return [
        (
            rng.uniform(*FORGE_RISK),
            rng.uniform(*FORGE_KL),
            rng.randint(*FORGE_N),
            rng.uniform(*FORGE_DELTA),
        )
        for _ in range(count)
    ]


def _forge_corners():
    return [
        (risk, kl, n, delta)
        for risk in FORGE_RISK
        for kl in FORGE_KL
        for n in FORGE_N
        for delta in FORGE_DELTA
    ]


# --------------------------------------------------------------------------- #
# Rejections                                                                   #
# --------------------------------------------------------------------------- #
@IMPL_PARAMS
@pytest.mark.parametrize(
    "args,fragment",
    [pytest.param(args, fragment, id=case_id) for args, fragment, case_id in REJECTED],
)
def test_out_of_domain_input_raises_value_error(impl, args, fragment):
    # OverflowError is not a ValueError, so a leaked OverflowError fails this test.
    with pytest.raises(ValueError, match=re.escape(fragment)):
        impl.pac_bayes_mcallester(*args)


@IMPL_PARAMS
def test_garbage_matrix_raises_only_value_error_or_returns_a_finite_in_domain_bound(impl):
    def real(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def in_domain(risk, kl, n, delta):
        return (
            real(risk) and 0 <= risk <= 1
            and real(kl) and kl >= 0 and (isinstance(kl, int) or math.isfinite(kl))
            and type(n) is int and n >= 8
            and real(delta) and 0 < delta < 1
        )

    returned = 0
    for risk in GARBAGE:
        for kl in GARBAGE:
            for n in GARBAGE:
                for delta in GARBAGE:
                    try:
                        out = impl.pac_bayes_mcallester(risk, kl, n, delta)
                    except ValueError:
                        continue
                    # Any other exception type (OverflowError, TypeError, ...) propagates and fails.
                    returned += 1
                    assert in_domain(risk, kl, n, delta), (risk, kl, n, delta, out)
                    assert isinstance(out, float) and math.isfinite(out) and out >= risk
    assert returned > 0


@IMPL_PARAMS
def test_type_check_precedes_finiteness_and_domain_checks(impl):
    # Every field is type-checked before any finiteness or range check runs.
    with pytest.raises(ValueError, match="delta must be a real number"):
        impl.pac_bayes_mcallester(NAN, -1.0, 0, "0.05")


# --------------------------------------------------------------------------- #
# Valid inputs: bit-exact and strictly above the empirical risk                #
# --------------------------------------------------------------------------- #
@IMPL_PARAMS
def test_valid_baseline_is_bit_exact(impl):
    out = impl.pac_bayes_mcallester(*BASELINE_ARGS)
    assert out == BASELINE_VALUE
    assert out.hex() == BASELINE_HEX


@IMPL_PARAMS
def test_minimum_n_is_accepted_with_the_recorded_value(impl):
    out = impl.pac_bayes_mcallester(*MIN_N_ARGS)
    assert out == MIN_N_VALUE
    assert out > MIN_N_ARGS[0]


@IMPL_PARAMS
@pytest.mark.parametrize("args", ACCEPTED_BOUNDARIES)
def test_boundary_inputs_are_accepted_finite_and_not_below_risk(impl, args):
    out = impl.pac_bayes_mcallester(*args)
    assert isinstance(out, float)
    assert math.isfinite(out)
    # >= not >: for n = 2**1000 the positive penalty is below the float
    # resolution of the risk, so the sum rounds to the risk itself.
    assert out >= args[0]


@IMPL_PARAMS
def test_every_forge_generator_corner_and_sample_is_accepted(impl):
    for risk, kl, n, delta in _forge_corners() + _forge_samples():
        out = impl.pac_bayes_mcallester(risk, kl, n, delta)
        assert math.isfinite(out)
        # kl >= 0 and ln(2*sqrt(n)/delta) > ln 2, so the penalty exceeds sqrt(ln2 / 2n):
        # the complexity term is strictly positive and nothing was clamped.
        assert out - risk > math.sqrt(math.log(2.0) / (2.0 * n)), (risk, kl, n, delta, out)


def test_both_package_copies_agree_bit_for_bit_on_forge_samples():
    for args in _forge_corners() + _forge_samples():
        outs = {label: impl.pac_bayes_mcallester(*args) for label, impl in IMPLS.items()}
        assert len({out.hex() for out in outs.values()}) == 1, (args, outs)


# --------------------------------------------------------------------------- #
# Governed loop: a raised step halts before any receipt                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "args",
    [
        pytest.param([0.1, NAN, 1000, 0.05], id="nan-kl"),
        pytest.param([-5.0, 1.0, 1000, 0.05], id="negative-risk"),
        pytest.param([0.1, 1.0, True, 0.05], id="bool-n"),
        pytest.param([0.1, 1.0, 10**400, 0.05], id="huge-n"),
        pytest.param([0.1, 10**400, 1000, 0.05], id="huge-kl"),
    ],
)
def test_governed_loop_halts_on_raised_step_before_any_receipt(args):
    chain = F.run_governed_loop([{"formula_name": "pac_bayes_mcallester", "args": args}])
    assert chain["halted"] is True
    assert "raised" in chain["halt_reason"]
    assert "pac_bayes_mcallester" in chain["halt_reason"]
    assert chain["receipts"] == []
    assert chain["lambda_aggregate"] == 0.0
    assert chain["root_hash"] == F.GENESIS
    assert chain["replay_ok"] is True


def test_governed_loop_valid_step_scalar_is_one_minus_the_bound():
    chain = F.run_governed_loop(
        [{"formula_name": "pac_bayes_mcallester", "args": list(BASELINE_ARGS)}]
    )
    assert chain["halted"] is False
    assert chain["halt_reason"] is None
    assert len(chain["receipts"]) == 1
    assert chain["receipts"][0]["scalar"] == 1.0 - BASELINE_VALUE
    assert chain["lambda_aggregate"] == pytest.approx(1.0 - BASELINE_VALUE)
    assert chain["replay_ok"] is True


# --------------------------------------------------------------------------- #
# What did NOT change: status, registry, atlas, signature; and the clamp is gone #
# --------------------------------------------------------------------------- #
@IMPL_PARAMS
def test_proof_status_and_registry_are_untouched(impl):
    assert impl.PROOF_STATUS["pac_bayes_mcallester"] == "SORRY(PACBayes)"
    assert impl.REGISTRY["pac_bayes_mcallester"] is impl.pac_bayes_mcallester
    assert len(impl.REGISTRY) == 21
    assert len(impl.PROOF_STATUS) == 21


@IMPL_PARAMS
def test_signature_is_unchanged(impl):
    params = list(inspect.signature(impl.pac_bayes_mcallester).parameters)
    assert params == ["empirical_risk", "kl", "n", "delta"]


@IMPL_PARAMS
def test_docstring_states_attribution_domain_and_sorry_status(impl):
    doc = " ".join(impl.pac_bayes_mcallester.__doc__.split())
    assert "Maurer (2004) refinement of McAllester (1999)" in doc
    assert "requires n >= 8 and a loss in [0,1]" in doc
    assert "PROOF-STATUS: SORRY(PACBayes)" in doc
    assert "not exact real arithmetic" in doc


@IMPL_PARAMS
def test_nan_swallowing_clamp_is_gone_from_the_source(impl):
    source = inspect.getsource(impl.pac_bayes_mcallester)
    assert "max(0.0" not in source
    assert "math.sqrt(complexity)" in source


def test_atlas_executable_entry_and_locked_count_are_unchanged():
    atlas = F.load_formula_atlas()
    rows = [row for row in atlas["executable_formulas"] if row["name"] == "pac_bayes_mcallester"]
    assert rows == [{"name": "pac_bayes_mcallester", "proof_status": "SORRY(PACBayes)"}]
    assert F.proof_status("pac_bayes_mcallester") == "SORRY(PACBayes)"
    assert F.LOCKED_PROVEN_COUNT == 8
    assert F.registry_count() == 21
