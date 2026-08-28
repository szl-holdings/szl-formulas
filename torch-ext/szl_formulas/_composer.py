# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl_formulas._composer — governed-loop composer over the canonical formulas.

Composes registry formulas into a hash-chained governed loop. Each step's
receipt links to the previous receipt; four HARD-STOP validators gate every
step before its receipt is appended:

    1. state_transition — the formula name is on the allowed transition set
    2. drift_bounds     — the step's scalar output stays within [0,1]
    3. human_gate       — steps tagged requires_human carry an approval token
    4. axis_floor       — the running Λ-aggregate stays ≥ AXIS_FLOOR

On ANY validator failure the loop HALTS and the ReceiptChain seals at the last
good step. The Λ composition is **ADVISORY ONLY** — Λ uniqueness is Conjecture 1
(open); a high Λ is a non-compensatory advisory roll-up, NOT proven trust.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Optional, TypedDict

from . import _formulas as F

GENESIS = "0" * 64

ALLOWED_STEPS = set(F.REGISTRY)
AXIS_FLOOR = 0.5

RISK_LIKE = {
    "pac_bayes_mcallester",
    "hoeffding_tail",
    "pinsker_kl_bound",
    "fisher_rao_distance",
    "bekenstein_cascade",
}

STRUCTURAL = {
    "reed_solomon_singleton",
    "shor_codeword_distance",
}


class FormulaCall(TypedDict, total=False):
    formula_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    requires_human: bool
    approval_token: Optional[str]


class StepReceipt(TypedDict):
    index: int
    formula_name: str
    args_digest: str
    output_repr: str
    scalar: float
    prev_hash: str
    receipt_hash: str
    validators: Dict[str, bool]


class ReceiptChain(TypedDict):
    receipts: List[StepReceipt]
    lambda_aggregate: float
    lambda_label: str
    halted: bool
    halt_reason: Optional[str]
    replay_ok: bool
    root_hash: str


def _to_scalar(out: Any, formula_name: str = "") -> float:
    if formula_name in STRUCTURAL:
        return 1.0
    base = _raw_scalar(out)
    if formula_name in RISK_LIKE:
        return max(0.0, min(1.0, 1.0 - base))
    return base


def _raw_scalar(out: Any) -> float:
    if isinstance(out, bool):
        return 1.0 if out else 0.0
    if isinstance(out, (int, float)):
        v = float(out)
        if v != v:
            return 0.0
        if 0.0 <= v <= 1.0:
            return v
        return 1.0 / (1.0 + abs(v)) if v > 1.0 else max(0.0, v)
    if isinstance(out, (bytes, str)):
        b = out if isinstance(out, bytes) else out.encode()
        return (int.from_bytes(sha256(b).digest()[:4], "big") % 1_000_000) / 1_000_000
    if isinstance(out, (list, tuple)):
        return 1.0 if len(out) > 0 else 0.0
    if isinstance(out, dict):
        return 1.0
    return 0.5


def _args_digest(call: FormulaCall) -> str:
    body = f'{call["formula_name"]}|{call.get("args", [])}|{call.get("kwargs", {})}'
    return sha256(body.encode()).hexdigest()


def _receipt_hash(prev_hash: str, idx: int, name: str, args_digest: str, scalar: float) -> str:
    body = f"{prev_hash}|{idx}|{name}|{args_digest}|{scalar:.9f}"
    return sha256(body.encode()).hexdigest()


def _validate(call: FormulaCall, scalar: float, running_lambda: float) -> Dict[str, bool]:
    name = call.get("formula_name", "")
    return {
        "state_transition": name in ALLOWED_STEPS,
        "drift_bounds": 0.0 <= scalar <= 1.0,
        "human_gate": (not call.get("requires_human", False))
        or bool(call.get("approval_token")),
        "axis_floor": running_lambda >= AXIS_FLOOR - F.EPS,
    }


LAMBDA_LABEL = "ADVISORY — Λ = Conjecture 1 (open); non-compensatory roll-up, NOT proven trust"


def run_governed_loop(calls: List[FormulaCall]) -> ReceiptChain:
    """Execute formula calls as a hash-chained governed loop with hard-stops.
    Λ composition is ADVISORY only (Conjecture 1)."""
    receipts: List[StepReceipt] = []
    scalars: List[float] = []
    prev_hash = GENESIS
    halted = False
    halt_reason: Optional[str] = None

    for idx, call in enumerate(calls):
        name = call.get("formula_name", "")
        fn = F.REGISTRY.get(name)
        if fn is None:
            halted, halt_reason = True, f"unknown formula: {name}"
            break
        try:
            out = fn(*call.get("args", []), **call.get("kwargs", {}))
        except Exception as exc:
            halted, halt_reason = True, f"step {idx} ({name}) raised: {exc}"
            break

        scalar = _to_scalar(out, name)
        running_lambda = (
            F.lambda_aggregate(scalars + [scalar]) if (scalars + [scalar]) else scalar
        )
        validators = _validate(call, scalar, running_lambda)

        rh = _receipt_hash(prev_hash, idx, name, _args_digest(call), scalar)
        receipts.append(
            StepReceipt(
                index=idx,
                formula_name=name,
                args_digest=_args_digest(call),
                output_repr=repr(out)[:120],
                scalar=scalar,
                prev_hash=prev_hash,
                receipt_hash=rh,
                validators=validators,
            )
        )

        if not all(validators.values()):
            failed = [k for k, v in validators.items() if not v]
            halted, halt_reason = True, f"step {idx} ({name}) HALT on validators {failed}"
            break

        scalars.append(scalar)
        prev_hash = rh

    lam = F.lambda_aggregate(scalars) if scalars else 0.0
    chain = ReceiptChain(
        receipts=receipts,
        lambda_aggregate=lam,
        lambda_label=LAMBDA_LABEL,
        halted=halted,
        halt_reason=halt_reason,
        replay_ok=False,
        root_hash=prev_hash,
    )
    chain["replay_ok"] = verify_chain(chain, calls)
    return chain


def verify_chain(chain: ReceiptChain, calls: List[FormulaCall]) -> bool:
    """Pure replay verifier: recompute every receipt hash + the final Λ."""
    prev = GENESIS
    good_scalars: List[float] = []
    for r in chain["receipts"]:
        expected = _receipt_hash(
            prev, r["index"], r["formula_name"], r["args_digest"], r["scalar"]
        )
        if expected != r["receipt_hash"]:
            return False
        if r["prev_hash"] != prev:
            return False
        if all(r["validators"].values()):
            good_scalars.append(r["scalar"])
            prev = r["receipt_hash"]
        else:
            break
    lam = F.lambda_aggregate(good_scalars) if good_scalars else 0.0
    return abs(lam - chain["lambda_aggregate"]) <= 1e-9
