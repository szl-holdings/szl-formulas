# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""Independent decimal enclosure oracle for ``szl.lambda/v1`` (R05 / P-MATH-03).

NUMERIC CERTIFICATION ONLY. This test helper brackets the exact real number

    Λ = ∏ xᵢ^{wᵢ} = exp(Σ wᵢ · ln xᵢ)

for the EXACT binary64 inputs the kernel receives (``Decimal(float)`` is
exact), and bounds how far the kernel's float result,
``szl_formulas.lambda_aggregate``, may legitimately sit from that real. It
is a test-side helper, not a runtime API and not part of the Hub payload.

What this is NOT
----------------
* NOT a gate verdict. ``classify``/``certify`` only say whether the float
  comparison ``Λ >= τ`` is numerically forced (``SEPARATED_ABOVE`` /
  ``SEPARATED_BELOW``) or numerically undecidable at binary64 accuracy
  (``UNKNOWN``). They never produce GO / NO_GO / ABSTAIN.
* It does NOT model szl-lambda-gate's ``NUMERIC_TIE`` policy band: the
  fixture vectors ``tie_inside_above`` / ``tie_inside_below`` are numerically
  SEPARATED here (4e-10 from τ, far outside any float error) but ABSTAIN in
  the gate by policy. Do not read an oracle relation as a gate outcome.
* Λ uniqueness remains **Conjecture 1 (open)**. Nothing here depends on it or
  strengthens it; the locked-proven canonical set stays exactly 8.

Contract
--------
* Validation is the kernel's own ``_lambda_v1_validate`` (run FIRST, with the
  kernel's own ``weights=None`` expansion), so error precedence and
  ``LambdaV1Error`` codes are identical to the kernel by construction. Only
  the numeric core is independent of the kernel.
* Weights are NEVER normalised. The v1 tolerance ``|fsum(w) - 1| <= 1e-12`` is
  respected as-is; a weight vector summing to ``1 + 4e-13`` is evaluated with
  that sum. ``weights=None`` reproduces the kernel's binary64 ``[1.0/k]*k``,
  not the rational ``1/k``.
* Any ``x == 0`` (including ``-0.0``) gives exactly ``(0, 0)``; ``x == 1``
  contributes exactly 0 to the exponent sum.
* ``ln`` and ``exp`` use ``decimal`` under ROUND_HALF_EVEN, which CPython
  documents as correctly rounded, then are widened by one ulp each way
  (``next_minus`` / ``next_plus``). Products and sums use ROUND_FLOOR for the
  lower bound and ROUND_CEILING for the upper bound; ``w > 0`` keeps every
  step monotone. The result is clamped to ``[0, 1]``, which is rigorous
  because ``x <= 1`` and ``w > 0`` give ``Σ wᵢ ln xᵢ <= 0``.
* Defence in depth: ``certify`` recomputes at ``check_prec`` (90) and raises
  ``OracleInconsistency`` unless the two enclosures nest.
* ``float_error_bound`` is exact ``Fraction`` arithmetic with ONE labelled
  ASSUMPTION (``LIBM_ULP_ASSUMPTION``): the platform libm ``math.log`` and
  ``math.exp`` are each within 4 ulp of the exact result. IEEE 754 does not
  mandate this; it is not verified here. A platform that breaks it makes the
  containment tests fail loudly instead of passing silently.

Stdlib only (decimal, fractions, math, struct, sys, pathlib, typing). No
mpmath, no numpy, no network, no I/O.
"""
from __future__ import annotations

import math
import struct
import sys
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Clamped,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    Subnormal,
    Underflow,
)
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREC = 60
CHECK_PREC = 90

# ASSUMPTION (labelled, owner decision): |math.log(x) - ln x| <= 4 ulp(ln x) and
# |math.exp(s) - e^s| <= 4 ulp(e^s), where ulp(y) = 2**(floor(log2|y|) - 52) for
# normal y and 2**-1074 for subnormal y. CPython defers both to the platform libm
# (glibc on ubuntu CI, UCRT on Windows); IEEE 754 does not mandate correct
# rounding for them. glibc documents <= 1 ulp for both on x86_64; 4 is a margin.
LIBM_ULP_ASSUMPTION = 4

U = Fraction(1, 2**53)  # binary64 unit roundoff under round-to-nearest
TINY = Fraction(1, 2**1074)  # smallest positive subnormal = ulp of every subnormal
ZERO = Decimal(0)
ONE = Decimal(1)

SEPARATED_ABOVE = "SEPARATED_ABOVE"
SEPARATED_BELOW = "SEPARATED_BELOW"
UNKNOWN = "UNKNOWN"
ENCLOSED = "ENCLOSED"
VIOLATION = "VIOLATION"

SCOPE = (
    "numeric certification only; NOT a gate verdict; does not model szl-lambda-gate "
    "NUMERIC_TIE policy; Lambda uniqueness remains Conjecture 1 (open)"
)

ASSUMPTIONS: Tuple[str, ...] = (
    "Decimal(float) and Decimal(int) are exact: the oracle sees the exact binary64 inputs "
    "the kernel receives.",
    "decimal.Decimal.ln and decimal.Decimal.exp are correctly rounded under ROUND_HALF_EVEN "
    "(CPython decimal documentation); each result is widened by one ulp each way.",
    f"ASSUMPTION (not IEEE-754 mandated, not verified here): platform libm math.log and "
    f"math.exp are each within {LIBM_ULP_ASSUMPTION} ulp of the exact result.",
    "math.fsum is correctly rounded (CPython documentation).",
    "math.log(1.0) == 0.0 and math.exp(0.0) == 1.0 exactly (C Annex F); see "
    "check_platform_identities().",
    "Products w*ln(x) and the exponent sum are modelled with round-to-nearest relative "
    "error u = 2**-53 plus a 2**-1074 absolute floor per operation for subnormal results.",
)

# Trap everything that would silently reduce rigor; a trap means a bug to investigate.
_TRAPS = [InvalidOperation, DivisionByZero, Overflow, Underflow, Subnormal, Clamped]


class OracleInconsistency(RuntimeError):
    """The oracle contradicted itself (e.g. prec-60/90 enclosures not nested). Never swallowed."""


def _ctx(prec: int, rounding: str) -> Context:
    return Context(prec=prec, rounding=rounding, Emin=-999999, Emax=999999, traps=_TRAPS)


# --------------------------------------------------------------------------- #
# Kernel plumbing: validation is the kernel's, by construction                 #
# --------------------------------------------------------------------------- #
def _default_impl():
    """The torch-ext kernel copy; tests pass the build/torch-universal copy explicitly."""
    torch_ext = str(ROOT / "torch-ext")
    if torch_ext not in sys.path:
        sys.path.insert(0, torch_ext)
    from szl_formulas import _formulas  # stdlib-only module

    return _formulas


def kernel_weights(axes, weights):
    """The weights the kernel actually uses: binary64 ``[1.0/k]*k`` when ``weights`` is None."""
    if weights is not None:
        return weights
    k = len(axes) if isinstance(axes, (list, tuple)) else 0
    return [1.0 / k] * k if k else []


def validate(axes, weights=None, *, impl=None):
    """Run the kernel's own ``_lambda_v1_validate`` FIRST; return ``(impl, concrete_weights)``.

    Raises ``impl.LambdaV1Error`` with exactly the kernel's code and precedence
    (every check runs over every element before the next). Nothing is clamped,
    renormalised or rounded.
    """
    impl = impl if impl is not None else _default_impl()
    ws = kernel_weights(axes, weights)
    impl._lambda_v1_validate(axes, ws)
    return impl, ws


def decode_f64(value):
    """``'f64:<16 hex>'`` (the fixture encoding) -> float; anything else passes through."""
    if isinstance(value, str) and value.startswith("f64:") and len(value) == 20:
        return struct.unpack(">d", bytes.fromhex(value[4:]))[0]
    return value


# --------------------------------------------------------------------------- #
# Numeric core: independent of the kernel                                      #
# --------------------------------------------------------------------------- #
def _ln_bounds(x: Decimal, he: Context) -> Tuple[Decimal, Decimal]:
    """Rigorous ``[lo, hi]`` around ``ln x`` for ``0 < x <= 1``.

    ``x.ln(he)`` is correctly rounded (|error| <= 0.5 ulp), so the true value
    lies strictly between the two neighbouring representables.
    """
    if x == 1:
        return ZERO, ZERO  # ln 1 = 0 exactly
    ln = x.ln(he)
    return ln.next_minus(he), ln.next_plus(he)


def _scale(ln_bound: Decimal, w: Union[Decimal, Fraction], ctx: Context) -> Decimal:
    """Directed-rounding ``w * ln_bound`` for ``w > 0`` (monotone, so the bound side is kept)."""
    if isinstance(w, Decimal):
        return ctx.multiply(w, ln_bound)
    # Exact rational weight p/q: two directed roundings, both order-preserving for q > 0.
    return ctx.divide(ctx.multiply(ln_bound, Decimal(w.numerator)), Decimal(w.denominator))


def enclose_exact(
    xs: Sequence[Decimal], ws: Sequence[Union[Decimal, Fraction]], prec: int = DEFAULT_PREC
) -> Tuple[Decimal, Decimal]:
    """Core enclosure of ``exp(Σ wᵢ ln xᵢ)`` for exact Decimal axes in [0,1] and weights > 0.

    No validation and no normalisation happen here; callers go through
    ``enclose`` (kernel-validated) or ``enclose_rational_weights`` (test support).
    """
    if len(xs) != len(ws):
        raise ValueError("axes and weights must have the same length")
    he = _ctx(prec, ROUND_HALF_EVEN)
    dn = _ctx(prec, ROUND_FLOOR)
    up = _ctx(prec, ROUND_CEILING)
    if any(x == 0 for x in xs):
        return ZERO, ZERO  # zero veto, exactly (Decimal('-0') == 0)
    s_lo = ZERO
    s_hi = ZERO
    for x, w in zip(xs, ws):
        if x == 1:
            continue  # contributes exactly 0
        ln_lo, ln_hi = _ln_bounds(x, he)
        s_lo = dn.add(s_lo, _scale(ln_lo, w, dn))
        s_hi = up.add(s_hi, _scale(ln_hi, w, up))
    lo = s_lo.exp(he).next_minus(he)
    hi = s_hi.exp(he).next_plus(he)
    # S <= 0 since every x <= 1 and w > 0, so the true value is in (0, 1]: clamping is rigorous.
    return max(lo, ZERO), min(hi, ONE)


def _exact_inputs(axes, ws) -> Tuple[List[Decimal], List[Decimal]]:
    return [Decimal(x) for x in axes], [Decimal(w) for w in ws]


def enclose(axes, weights=None, prec: int = DEFAULT_PREC, *, impl=None) -> Tuple[Decimal, Decimal]:
    """Rigorous ``(lo, hi)`` Decimals enclosing the exact real ``∏ xᵢ^{wᵢ}`` of the exact inputs.

    Kernel validation runs first (same ``LambdaV1Error`` code and precedence as
    ``lambda_aggregate``). Weights are used exactly as given; ``weights=None``
    is the kernel's binary64 ``[1.0/k]*k``. Any zero axis gives exactly ``(0, 0)``.
    Numeric certification only, not a gate verdict.
    """
    _, ws = validate(axes, weights, impl=impl)
    xs, wd = _exact_inputs(axes, ws)
    return enclose_exact(xs, wd, prec)


def enclose_rational_weights(
    axes, weights: Sequence[Fraction], prec: int = DEFAULT_PREC
) -> Tuple[Decimal, Decimal]:
    """Test support: enclose with EXACT rational weights (e.g. Fraction-normalised ones).

    This is how a test demonstrates that the kernel does NOT normalise. There
    is no kernel validation here; axes must be real numbers in [0, 1] and every
    weight must be a positive Fraction.
    """
    xs = [Decimal(x) for x in axes]
    if any(not (0 <= x <= 1) for x in xs):
        raise ValueError("axes must lie in [0, 1]")
    ws = [Fraction(w) for w in weights]
    if any(w <= 0 for w in ws):
        raise ValueError("weights must be > 0")
    return enclose_exact(xs, ws, prec)


def enclose_nested(
    axes, weights=None, prec: int = DEFAULT_PREC, check_prec: int = CHECK_PREC, *, impl=None
) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
    """``(lo, hi, lo_check, hi_check)``; raises ``OracleInconsistency`` unless nested."""
    _, ws = validate(axes, weights, impl=impl)
    xs, wd = _exact_inputs(axes, ws)
    lo, hi = enclose_exact(xs, wd, prec)
    lo2, hi2 = enclose_exact(xs, wd, check_prec)
    if not (lo <= lo2 <= hi2 <= hi):
        raise OracleInconsistency(
            f"prec-{prec} enclosure [{lo}, {hi}] does not contain prec-{check_prec} "
            f"enclosure [{lo2}, {hi2}] for axes={axes!r} weights={ws!r}"
        )
    return lo, hi, lo2, hi2


# --------------------------------------------------------------------------- #
# Float error model: exact rationals, one labelled libm assumption              #
# --------------------------------------------------------------------------- #
def abs_ln_upper(x, prec: int = 30) -> Fraction:
    """Rigorous rational upper bound on ``|ln x|`` for ``0 < x <= 1`` (never ``math.log``)."""
    d = Decimal(x)
    if d == 1:
        return Fraction(0)
    he = _ctx(prec, ROUND_HALF_EVEN)
    return Fraction(abs(d.ln(he)).next_plus(he))


def float_error_bound(
    axes, weights, hi, *, libm_ulps: int = LIBM_ULP_ASSUMPTION
) -> Fraction:
    """Exact-rational bound on ``|lambda_aggregate(axes, weights) - Λ|`` for VALIDATED inputs.

    Models the kernel's ``exp(fsum(w * log(x)))`` step by step:

    * ``log``: |error| <= ``libm_ulps`` ulp <= ``2*libm_ulps*u`` relative (ASSUMPTION);
    * ``w * log(x)``: round-to-nearest, relative ``u`` plus a ``2**-1074`` floor;
    * ``fsum``: correctly rounded, relative ``u`` on the perturbed sum plus a floor;
    * ``exp``: |error| <= ``libm_ulps`` ulp, i.e. ``2*libm_ulps*u`` relative for a
      normal result plus ``libm_ulps * 2**-1074`` for a subnormal one (ASSUMPTION);
    * ``|e^(S+d) - e^S| <= e^S (d + d²)`` for ``0 <= d <= 1``, with ``e^S <= hi``.

    ``hi`` is the oracle's upper bound (Decimal or Fraction); scaling by it keeps
    the bound tight for subnormal results. ``weights=None`` means the kernel's
    binary64 uniform weights. A zero axis gives exactly ``Fraction(0)`` because
    the kernel returns ``0.0`` without any float arithmetic.
    """
    ws = kernel_weights(axes, weights)
    if any(x == 0 for x in axes):
        return Fraction(0)
    k = len(axes)
    c = Fraction(libm_ulps)
    rel_log = 2 * c * U
    rel_exp = 2 * c * U
    # A bounds |S| = Σ wᵢ |ln xᵢ| (all terms share the sign of ln xᵢ <= 0).
    a_sum = sum((Fraction(w) * abs_ln_upper(x) for x, w in zip(axes, ws)), Fraction(0))
    g = (1 + rel_log) * (1 + U) - 1  # log error then product rounding, per term, relative
    d_s = a_sum * g + k * TINY  # per-term relative errors + subnormal/underflow floors
    d_s = d_s + U * a_sum * (1 + g) + TINY  # correctly rounded fsum of the perturbed terms
    if d_s > 1:
        raise OracleInconsistency("error model needs |S_float - S| <= 1; impossible for v1 inputs")
    e_ds = 1 + d_s + d_s * d_s  # e^t <= 1 + t + t² for 0 <= t <= 1
    hi_frac = Fraction(hi)
    return hi_frac * ((1 + rel_exp) * e_ds - 1) + c * TINY


def check_platform_identities() -> Dict[str, bool]:
    """Exact identities the error model relies on for ``x == 1`` terms (C Annex F)."""
    return {
        "log_1_is_zero": math.log(1.0) == 0.0,
        "exp_0_is_one": math.exp(0.0) == 1.0,
        "fsum_single_is_exact": math.fsum([0.1]) == 0.1,
    }


# --------------------------------------------------------------------------- #
# Threshold relation and certificate                                            #
# --------------------------------------------------------------------------- #
def _exact_tau(tau) -> Fraction:
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau):
        raise ValueError(
            "tau must be the exact finite binary64 (or int) the code compares against, "
            f"got {tau!r}"
        )
    return Fraction(tau)


def classify(lo, hi, eps, tau) -> str:
    """Relation of the kernel's float comparison ``lambda_aggregate(...) >= tau`` to the enclosure.

    With ``W = [lo - eps, hi + eps]`` (exact rationals, no rounding):
    ``SEPARATED_ABOVE`` iff ``lo - eps >= tau`` (the float result is >= tau for
    sure); ``SEPARATED_BELOW`` iff ``hi + eps < tau`` (the float result is < tau
    for sure); otherwise ``UNKNOWN``. ``tau`` must be the exact binary64 the code
    compares against, e.g. ``fl(AXIS_FLOOR - EPS)`` for the composer.

    This is numeric certification only, NOT a gate verdict; it does not model
    szl-lambda-gate's NUMERIC_TIE policy band, and Λ uniqueness remains
    Conjecture 1 (open).
    """
    t = _exact_tau(tau)
    e = Fraction(eps)
    if e < 0:
        raise ValueError("eps must be >= 0")
    if Fraction(lo) - e >= t:
        return SEPARATED_ABOVE
    if Fraction(hi) + e < t:
        return SEPARATED_BELOW
    return UNKNOWN


def ulp_distance(value: float, lo, hi) -> Fraction:
    """How far the kernel float sits outside ``[lo, hi]`` in units of ``ulp(value)``; 0 when inside."""
    v = Fraction(value)
    outside = max(Fraction(0), Fraction(lo) - v, v - Fraction(hi))
    if outside == 0:
        return Fraction(0)
    return outside / Fraction(math.ulp(value))


def _upper_str(fr: Fraction, digits: int = 20) -> str:
    """A ``digits``-significant-digit decimal string that is >= ``fr`` (never understated)."""
    if fr == 0:
        return "0"
    ctx = _ctx(digits, ROUND_CEILING)
    return str(ctx.divide(Decimal(fr.numerator), Decimal(fr.denominator)))


def certify(
    axes,
    weights=None,
    tau=None,
    *,
    prec: int = DEFAULT_PREC,
    check_prec: int = CHECK_PREC,
    impl=None,
    libm_ulps: int = LIBM_ULP_ASSUMPTION,
) -> dict:
    """Plain-dict numeric certificate for one kernel evaluation. NOT a gate verdict.

    Keys: ``lo``/``hi`` (exact Decimal strings), ``width`` and ``eps`` (upper-bound
    decimal strings), ``float_value`` (the kernel's output), ``contained`` (float
    within ``[lo - eps, hi + eps]``), ``ulp_outside`` (distance of the float from
    ``[lo, hi]`` in ulps), ``tau``, ``threshold_relation`` (``SEPARATED_ABOVE`` /
    ``SEPARATED_BELOW`` / ``UNKNOWN`` / None), ``assumptions`` (list of strings),
    ``status`` in {``ENCLOSED``, ``VIOLATION``, ``UNKNOWN``}: VIOLATION when the
    float is outside the error band (kernel defect or a libm violating the
    labelled assumption), UNKNOWN when ``tau`` is given and the relation is
    numerically undecidable, ENCLOSED otherwise.

    Input violations raise the kernel's ``LambdaV1Error`` (same code). A failed
    prec-60/90 nesting check raises ``OracleInconsistency``; nothing falls back.
    It does not model szl-lambda-gate's NUMERIC_TIE policy, and Λ uniqueness
    remains Conjecture 1 (open).
    """
    impl, ws = validate(axes, weights, impl=impl)
    lo, hi, lo2, hi2 = enclose_nested(axes, weights, prec, check_prec, impl=impl)
    value = impl.lambda_aggregate(axes, weights)
    eps = float_error_bound(axes, ws, hi, libm_ulps=libm_ulps)
    v = Fraction(value)
    contained = Fraction(lo) - eps <= v <= Fraction(hi) + eps
    relation: Optional[str] = classify(lo, hi, eps, tau) if tau is not None else None
    if not contained:
        status = VIOLATION
    elif relation == UNKNOWN:
        status = UNKNOWN
    else:
        status = ENCLOSED
    return {
        "scope": SCOPE,
        "kernel": getattr(impl, "__file__", repr(impl)),
        "prec": prec,
        "check_prec": check_prec,
        "nested": True,
        "lo": str(lo),
        "hi": str(hi),
        "width": _upper_str(Fraction(hi) - Fraction(lo)),
        "width_check_prec": _upper_str(Fraction(hi2) - Fraction(lo2)),
        "float_value": value,
        "eps": _upper_str(eps),
        "libm_ulp_assumption": libm_ulps,
        "contained": bool(contained),
        "ulp_outside": float(ulp_distance(value, lo, hi)),
        "tau": None if tau is None else repr(tau),
        "threshold_relation": relation,
        "assumptions": list(ASSUMPTIONS),
        "status": status,
    }
