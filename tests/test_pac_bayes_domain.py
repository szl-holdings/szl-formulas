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

Every case runs through every shipped package copy (see conftest.py):
``torch-ext`` (what this repo imports) and the publisher's ``build/torch-cpu``
and ``build/torch-universal`` (what the Hub loads). Stdlib + pytest only.

The float expression for a valid input is unchanged, so every input the new
domain accepts evaluates to the same float as before; only the domain moved.
PROOF_STATUS stays "SORRY(PACBayes)" verbatim. Λ remains Conjecture 1 (open).
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import random
import re
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_package(name: str, path: Path):
    # Isolated package names keep relative imports, registries and composers
    # bound to the copy under test, regardless of other tests' sys.path edits.
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COPIES = ["torch-ext", "build/torch-cpu", "build/torch-universal"]  # see conftest.py
_PACKAGES: dict = {}


def _package(shipped_root, copy: str):
    if copy not in _PACKAGES:
        _PACKAGES[copy] = _load_package(f"_pac_bayes_test_{COPIES.index(copy)}",
                                        shipped_root(copy) / "szl_formulas")
    return _PACKAGES[copy]


def _impl(shipped_root, copy: str):
    return importlib.import_module(f"{_package(shipped_root, copy).__name__}._formulas")


@pytest.fixture
def package(request, shipped_root):
    return _package(shipped_root, request.param)


@pytest.fixture
def impl(request, shipped_root):
    return _impl(shipped_root, request.param)


IMPL_PARAMS = pytest.mark.parametrize("impl", COPIES, indirect=True)
PACKAGE_PARAMS = pytest.mark.parametrize("package", COPIES, indirect=True)

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

# (args, exact message fragment, id). Each must raise ValueError naming the
# offending field. At fca9707 every row returned a number, leaked a
# non-ValueError or raised a ValueError with a different message, except the
# rows in REJECTED_REGRESSION_PINS below: fca9707 already raised those exact
# ValueErrors, so they pin the old contract and do not prove the fix
# (machine-checked by test_only_labelled_rejection_rows_pass_on_the_fca9707_formula).
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

# REGRESSION PIN rows of REJECTED: fca9707 already raised these with the same
# message ("KL divergence must be >= 0", "delta must be in (0,1)").
REJECTED_REGRESSION_PINS = {"kl-negative", "delta-zero", "delta-one", "delta-negative"}

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


def test_every_package_copy_agrees_bit_for_bit_on_forge_samples(shipped_root):
    impls = {copy: _impl(shipped_root, copy) for copy in COPIES}
    for args in _forge_corners() + _forge_samples():
        outs = {label: impl.pac_bayes_mcallester(*args) for label, impl in impls.items()}
        assert len({out.hex() for out in outs.values()}) == 1, (args, outs)


# --------------------------------------------------------------------------- #
# Governed loop: a raised step halts before any receipt                        #
# --------------------------------------------------------------------------- #
# (args, exact fragment of the formula's own ValueError message, id). Each case
# must halt on the formula's fail-closed ValueError, and the halt reason must
# carry that message. "raised" in the halt reason does not discriminate the
# domain fix: with the fca9707 formula, huge-n and huge-kl halt the loop through
# a leaked OverflowError, and under this composer negative-risk halts because
# the composer's output check rejects the fca9707 formula's negative bound
# inside the same try. Both start with the same prefix, so only the message
# after it tells them apart from the fix (pinned below in
# FCA9707_HALTS_THAT_SATISFIED_THE_FORMER_CHECK).
LOOP_HALT_PREFIX = "step 0 (pac_bayes_mcallester) raised: "
LOOP_HALT_CASES = [
    pytest.param([0.1, NAN, 1000, 0.05], "kl must be finite", id="nan-kl"),
    pytest.param([-5.0, 1.0, 1000, 0.05], "empirical_risk must be in [0,1]", id="negative-risk"),
    pytest.param([0.1, 1.0, True, 0.05], "n must be a real number", id="bool-n"),
    pytest.param([0.1, 1.0, 10**400, 0.05], "n is too large for float arithmetic", id="huge-n"),
    pytest.param([0.1, 10**400, 1000, 0.05], "kl is too large for float arithmetic", id="huge-kl"),
]


def _run_single_pac_step(package, args):
    return package.run_governed_loop([{"formula_name": "pac_bayes_mcallester", "args": args}])


def _registered_formula_value_error(package, args):
    # The registered formula's own ValueError message for args, or None when it
    # returns a value or raises anything else: a leaked OverflowError is not a
    # fail-closed domain error.
    try:
        package.REGISTRY["pac_bayes_mcallester"](*args)
    except ValueError as exc:
        return str(exc)
    except Exception:
        return None
    return None


def _assert_loop_halts_on_the_formula_value_error(package, args, fragment):
    chain = _run_single_pac_step(package, args)
    reason = chain["halt_reason"]
    assert chain["halted"] is True, reason
    assert isinstance(reason, str) and reason.startswith(LOOP_HALT_PREFIX), reason
    # The formula's own message follows the prefix. A leaked OverflowError or
    # the composer's output-validation error carries a different message.
    assert reason[len(LOOP_HALT_PREFIX):].startswith(fragment), reason
    own_message = _registered_formula_value_error(package, args)
    assert own_message is not None, reason
    assert reason == LOOP_HALT_PREFIX + own_message
    assert chain["receipts"] == []
    assert chain["lambda_aggregate"] == 0.0
    assert chain["root_hash"] == package.GENESIS
    assert chain["replay_ok"] is True


@PACKAGE_PARAMS
@pytest.mark.parametrize("args,fragment", LOOP_HALT_CASES)
def test_governed_loop_halts_on_raised_step_before_any_receipt(package, args, fragment):
    _assert_loop_halts_on_the_formula_value_error(package, args, fragment)


def _fca9707_pac_bayes_mcallester(empirical_risk, kl, n, delta):
    # The body of pac_bayes_mcallester at fca9707 (before the domain fix), kept
    # only as a mutant for the discrimination guards below.
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if kl < 0.0:
        raise ValueError("KL divergence must be >= 0")
    complexity = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
    return empirical_risk + math.sqrt(max(0.0, complexity))


@PACKAGE_PARAMS
@pytest.mark.parametrize("args,fragment", LOOP_HALT_CASES)
def test_loop_halt_cases_reject_the_fca9707_formula(package, monkeypatch, args, fragment):
    # Discrimination guard: with the pre-fix formula registered, the shared
    # loop-halt assertion fails for every case, so none of them is a
    # regression pin.
    monkeypatch.setitem(package.REGISTRY, "pac_bayes_mcallester", _fca9707_pac_bayes_mcallester)
    with pytest.raises(AssertionError):
        _assert_loop_halts_on_the_formula_value_error(package, args, fragment)


# Halt reasons the fca9707 formula produces under this composer for the cases
# the former check ("raised" and the formula name in the halt reason, no
# receipt) accepted. None carries the formula's own ValueError message.
FCA9707_HALTS_THAT_SATISFIED_THE_FORMER_CHECK = {
    "negative-risk": LOOP_HALT_PREFIX + "pac_bayes_mcallester bound must be a finite nonnegative real number",
    "huge-n": LOOP_HALT_PREFIX + "int too large to convert to float",
    "huge-kl": LOOP_HALT_PREFIX + "int too large to convert to float",
}


@PACKAGE_PARAMS
def test_former_raised_check_accepted_these_fca9707_halts(package, monkeypatch):
    # Why the shared assertion checks the message: the assertions this file
    # used before pass for the pre-fix formula in exactly these cases.
    monkeypatch.setitem(package.REGISTRY, "pac_bayes_mcallester", _fca9707_pac_bayes_mcallester)
    accepted = {}
    for case in LOOP_HALT_CASES:
        args, _fragment = case.values
        chain = _run_single_pac_step(package, args)
        if (
            chain["halted"] is True
            and "raised" in chain["halt_reason"]
            and "pac_bayes_mcallester" in chain["halt_reason"]
            and chain["receipts"] == []
            and chain["lambda_aggregate"] == 0.0
            and chain["root_hash"] == package.GENESIS
            and chain["replay_ok"] is True
        ):
            accepted[case.id] = chain["halt_reason"]
    assert accepted == FCA9707_HALTS_THAT_SATISFIED_THE_FORMER_CHECK


def test_only_labelled_rejection_rows_pass_on_the_fca9707_formula():
    # Discrimination guard: exactly the REJECTED_REGRESSION_PINS rows already
    # held at fca9707; every other row needs the domain fix.
    satisfied = set()
    for args, fragment, case_id in REJECTED:
        try:
            _fca9707_pac_bayes_mcallester(*args)
        except ValueError as exc:
            if re.search(re.escape(fragment), str(exc)):
                satisfied.add(case_id)
        except Exception:
            pass
    assert satisfied == REJECTED_REGRESSION_PINS


# REGRESSION PIN: fca9707 already scored this valid bound 1 - b; not a fix proof.
@PACKAGE_PARAMS
def test_governed_loop_valid_step_scalar_is_one_minus_the_bound(package):
    chain = package.run_governed_loop(
        [{"formula_name": "pac_bayes_mcallester", "args": list(BASELINE_ARGS)}]
    )
    assert chain["halted"] is False
    assert chain["halt_reason"] is None
    assert len(chain["receipts"]) == 1
    assert chain["receipts"][0]["scalar"] == 1.0 - BASELINE_VALUE
    assert chain["lambda_aggregate"] == pytest.approx(1.0 - BASELINE_VALUE)
    assert chain["replay_ok"] is True


# HARNESS check, not a behaviour claim: the composer and registry under test
# belong to the package copy, so the REGISTRY monkeypatches above reach it.
@PACKAGE_PARAMS
def test_composer_and_registry_belong_to_the_package_copy(package):
    composer = importlib.import_module(f"{package.__name__}._composer")
    formulas = importlib.import_module(f"{package.__name__}._formulas")
    assert composer.F is formulas
    assert package.REGISTRY is formulas.REGISTRY
    assert Path(composer.__file__).parent == Path(package.__file__).parent


@PACKAGE_PARAMS
def test_pac_scalar_is_nonincreasing_across_vacuous_boundary(package):
    composer = importlib.import_module(f"{package.__name__}._composer")
    bounds = [
        0, 0.0, BASELINE_VALUE, 0.5, math.nextafter(1.0, 0.0),
        1, 1.0, math.nextafter(1.0, INF), 2.0, 22.460759633307337,
        sys.float_info.max, 10**400,
    ]
    scores = [composer._to_scalar(bound, "pac_bayes_mcallester") for bound in bounds]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert all(left >= right for left, right in zip(scores, scores[1:]))
    assert scores[:5] == [1.0, 1.0, 1.0 - BASELINE_VALUE, 0.5, 2**-53]
    assert scores[5:] == [0.0] * (len(bounds) - 5)


@PACKAGE_PARAMS
def test_increasing_actual_kl_never_improves_the_advisory_score(package):
    bounds, scores = [], []
    for kl in [0.0, 1.0, 10.0, 100.0, 1000.0, 1e6, 2**1023]:
        args = [0.1, kl, 1000, 0.05]
        bound = package.pac_bayes_mcallester(*args)
        chain = package.run_governed_loop([{"formula_name": "pac_bayes_mcallester", "args": args}])
        receipt = chain["receipts"][0]
        assert receipt["output_repr"] == repr(bound)[:120]
        assert chain["replay_ok"] is True
        if bound >= 1:
            assert receipt["scalar"] == 0.0
            assert chain["halted"] is True
            assert chain["root_hash"] == package.GENESIS
            assert chain["lambda_aggregate"] == 0.0
        bounds.append(bound)
        scores.append(receipt["scalar"])
    assert all(left < right for left, right in zip(bounds, bounds[1:]))
    assert all(left >= right for left, right in zip(scores, scores[1:]))
    assert scores[-2:] == [0.0, 0.0]


def _halt_calls(package, monkeypatch, with_prefix):
    def forbidden_tail():
        pytest.fail("a step ran after the loop should have halted")

    monkeypatch.setitem(package.REGISTRY, "madhava_series", forbidden_tail)
    prefix_calls = [{"formula_name": "reed_solomon_singleton", "args": [255, 223]}] if with_prefix else []
    prefix = package.run_governed_loop(prefix_calls)
    calls = prefix_calls + [
        {"formula_name": "pac_bayes_mcallester", "args": []},
        {"formula_name": "madhava_series", "args": []},
    ]
    return prefix, calls


@PACKAGE_PARAMS
@pytest.mark.parametrize("with_prefix", [False, True], ids=["first-step", "accepted-prefix"])
# bound=1.0 is a REGRESSION PIN: fca9707 already scored it 0.0 and halted on
# axis_floor with this exact receipt. The larger bounds prove the fix.
@pytest.mark.parametrize("bound", [1.0, math.nextafter(1.0, INF), 2.0, sys.float_info.max, 10**400])
def test_vacuous_bound_retains_failed_receipt_and_last_good_state(package, monkeypatch, with_prefix, bound):
    prefix, calls = _halt_calls(package, monkeypatch, with_prefix)
    monkeypatch.setitem(package.REGISTRY, "pac_bayes_mcallester", lambda: bound)
    chain = package.run_governed_loop(calls)
    index = int(with_prefix)
    assert chain["halted"] is True
    assert chain["halt_reason"] == f"step {index} (pac_bayes_mcallester) HALT on validators ['axis_floor']"
    assert chain["receipts"][:-1] == prefix["receipts"]
    assert len(chain["receipts"]) == index + 1
    digest = sha256(b"pac_bayes_mcallester|[]|{}").hexdigest()
    receipt_hash = sha256(
        f'{prefix["root_hash"]}|{index}|pac_bayes_mcallester|{digest}|0.000000000'.encode()
    ).hexdigest()
    assert chain["receipts"][-1] == {
        "index": index, "formula_name": "pac_bayes_mcallester", "args_digest": digest,
        "output_repr": repr(bound)[:120], "scalar": 0.0,
        "prev_hash": prefix["root_hash"], "receipt_hash": receipt_hash,
        "validators": {"state_transition": True, "drift_bounds": True, "human_gate": True, "axis_floor": False},
    }
    assert chain["root_hash"] == prefix["root_hash"]
    assert chain["lambda_aggregate"] == prefix["lambda_aggregate"]
    assert chain["replay_ok"] is True
    assert package.verify_chain(chain, calls) is True
    tampered = deepcopy(chain)
    tampered["receipts"][-1]["scalar"] = 1.0
    assert package.verify_chain(tampered, calls) is False


@PACKAGE_PARAMS
@pytest.mark.parametrize("with_prefix", [False, True], ids=["first-step", "accepted-prefix"])
@pytest.mark.parametrize("output", [NAN, INF, -INF, -5e-324, -1, True, False, None, "0.1", [], {}, 1j])
def test_invalid_pac_output_halts_before_receipt_and_preserves_prefix(package, monkeypatch, with_prefix, output):
    prefix, calls = _halt_calls(package, monkeypatch, with_prefix)
    monkeypatch.setitem(package.REGISTRY, "pac_bayes_mcallester", lambda: output)
    chain = package.run_governed_loop(calls)
    assert chain["halted"] is True
    assert chain["halt_reason"] == (
        f"step {int(with_prefix)} (pac_bayes_mcallester) raised: "
        "pac_bayes_mcallester bound must be a finite nonnegative real number"
    )
    assert chain["receipts"] == prefix["receipts"]
    assert chain["root_hash"] == prefix["root_hash"]
    assert chain["lambda_aggregate"] == prefix["lambda_aggregate"]
    assert chain["replay_ok"] is True
    assert package.verify_chain(chain, calls) is True


# REGRESSION PIN: fca9707 already applied this floor tolerance; not a fix proof.
@PACKAGE_PARAMS
@pytest.mark.parametrize("bound,halted", [(0.0, False), (0.5, False), (0.5 + 5e-10, False), (0.5 + 2e-9, True)])
def test_pac_score_respects_existing_axis_floor_tolerance(package, monkeypatch, bound, halted):
    monkeypatch.setitem(package.REGISTRY, "pac_bayes_mcallester", lambda: bound)
    calls = [{"formula_name": "pac_bayes_mcallester", "args": []}]
    chain = package.run_governed_loop(calls)
    assert chain["halted"] is halted
    receipt = chain["receipts"][0]
    assert receipt["scalar"] == 1.0 - bound
    assert receipt["validators"]["axis_floor"] is not halted
    assert chain["root_hash"] == (package.GENESIS if halted else receipt["receipt_hash"])
    assert chain["lambda_aggregate"] == pytest.approx(0.0 if halted else 1.0 - bound)
    assert chain["replay_ok"] is True


@PACKAGE_PARAMS
def test_non_pac_conversion_overflow_is_a_governed_halt(package, monkeypatch):
    prefix, calls = _halt_calls(package, monkeypatch, True)
    calls[1]["formula_name"] = "lambda_bounded"
    monkeypatch.setitem(package.REGISTRY, "lambda_bounded", lambda: 10**400)
    chain = package.run_governed_loop(calls)
    assert chain["halted"] is True
    assert chain["halt_reason"].startswith("step 1 (lambda_bounded) raised:")
    assert chain["receipts"] == prefix["receipts"]
    assert chain["root_hash"] == prefix["root_hash"]
    assert chain["lambda_aggregate"] == prefix["lambda_aggregate"]
    assert chain["replay_ok"] is True


# REGRESSION PIN: fca9707 converted outside the try and already propagated these;
# this guards that the converted-inside-try fix does not start swallowing them.
@PACKAGE_PARAMS
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_scalar_conversion_does_not_swallow_process_control(package, monkeypatch, exception):
    composer = importlib.import_module(f"{package.__name__}._composer")

    def interrupted_conversion(*args):
        raise exception("stop")

    monkeypatch.setattr(composer, "_to_scalar", interrupted_conversion)
    with pytest.raises(exception, match="stop"):
        package.run_governed_loop([{"formula_name": "pac_bayes_mcallester", "args": list(BASELINE_ARGS)}])


# REGRESSION PIN apart from the RISK_LIKE 2.0 assertion: fca9707 already
# produced every other mapping below.
@PACKAGE_PARAMS
def test_unrelated_formula_scalar_mappings_are_unchanged(package):
    composer = importlib.import_module(f"{package.__name__}._composer")
    for name in composer.RISK_LIKE - {"pac_bayes_mcallester"}:
        assert composer._to_scalar(0.25, name) == 0.75
        # Changed after this file was written (audit P3): every RISK_LIKE value
        # >= 1 now saturates at 0.0 like PAC-Bayes. It used to score
        # 2 / (1 + 2) here, rewarding a larger risk; the full policy is pinned
        # in tests/test_composer_risk_scores.py.
        assert composer._to_scalar(2.0, name) == 0.0
    assert composer._to_scalar(0.25, "lambda_bounded") == 0.25
    assert composer._to_scalar(2.0, "lambda_bounded") == 1.0 / 3.0
    for name in composer.STRUCTURAL:
        assert composer._to_scalar({"verdict": "test"}, name) == 1.0


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


@PACKAGE_PARAMS
def test_atlas_executable_entry_and_locked_count_are_unchanged(package):
    atlas = package.load_formula_atlas()
    rows = [row for row in atlas["executable_formulas"] if row["name"] == "pac_bayes_mcallester"]
    assert rows == [{"name": "pac_bayes_mcallester", "proof_status": "SORRY(PACBayes)"}]
    assert package.proof_status("pac_bayes_mcallester") == "SORRY(PACBayes)"
    assert package.LOCKED_PROVEN_COUNT == 8
    assert package.registry_count() == 21
