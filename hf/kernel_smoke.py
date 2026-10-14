# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish (szl-formulas).

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Hub (verify). Raise, or return False, to fail the run.
CPU-only, deterministic and fast. The result's repr is written to the receipt.

It checks the contract on the served bytes: the registry holds 21 formulas with a
verbatim PROOF_STATUS each and the locked-proven set is exactly 8; Λ is a weighted
geometric mean that a zero axis vetoes and that refuses an out-of-range axis; a clean
governed loop replays and a tampered one does not; an out-of-domain PAC-Bayes step halts
before any receipt; a vacuous risk bound scores zero and halts; the packaged atlas loads.
"""
from __future__ import annotations

LOCKED_8 = frozenset({"F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"})
CLEAN = [
    {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
    {"formula_name": "reed_solomon_singleton", "args": [255, 223]},
]


def smoke(kernel):
    if kernel.registry_count() != 21 or set(kernel.PROOF_STATUS) != set(kernel.REGISTRY):
        raise AssertionError("the registry must hold 21 formulas, each with a PROOF_STATUS")
    if kernel.proof_status("madhava_series") != kernel.PROOF_STATUS["madhava_series"]:
        raise AssertionError("proof_status must return the mirrored PROOF_STATUS verbatim")
    try:
        kernel.proof_status("not_a_formula")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown formula must raise KeyError, never a default status")
    if kernel.LOCKED_PROVEN_FORMULA_IDS != LOCKED_8 or kernel.LOCKED_PROVEN_COUNT != 8:
        raise AssertionError("the locked-proven set must be exactly the 8 locked ids")

    if kernel.lambda_aggregate([0.0, 0.9, 0.95]) != 0.0:
        raise AssertionError("a zero axis must veto Λ to exactly 0.0")
    if abs(kernel.lambda_aggregate([0.81, 0.81, 0.81]) - 0.81) > 1e-12:
        raise AssertionError("Λ of equal axes must be that axis")
    try:
        kernel.lambda_aggregate([1.5, 0.9])
    except ValueError as exc:
        if getattr(exc, "code", None) != "LAMBDA_AXIS_OUT_OF_RANGE":
            raise AssertionError(f"wrong Λ error code: {exc}") from None
    else:
        raise AssertionError("an axis above 1 must be refused, not aggregated")

    chain = kernel.run_governed_loop(CLEAN)
    if chain["halted"] or chain["replay_ok"] is not True or len(chain["receipts"]) != 2:
        raise AssertionError(f"a clean governed loop must run both steps and replay: {chain['halt_reason']}")
    if not kernel.verify_chain(chain, CLEAN):
        raise AssertionError("verify_chain must accept the clean chain")
    if not str(chain["lambda_label"]).startswith("ADVISORY"):
        raise AssertionError("the Λ roll-up must be labelled ADVISORY")
    chain["receipts"][0]["receipt_hash"] = "0" * 64
    if kernel.verify_chain(chain, CLEAN):
        raise AssertionError("verify_chain must reject a tampered receipt")

    if kernel.pac_bayes_mcallester(0.1, 1.0, 1000, 0.05) != 0.1638073549585195:
        raise AssertionError("the PAC-Bayes baseline must stay bit-exact")
    raised = kernel.run_governed_loop([
        {"formula_name": "pac_bayes_mcallester", "args": [0.1, 1.0, 1000.0, 0.05]},
    ])
    if not raised["halted"] or raised["receipts"] or raised["root_hash"] != kernel.GENESIS:
        raise AssertionError("an out-of-domain PAC-Bayes step must halt before any receipt")
    vacuous = kernel.run_governed_loop([{"formula_name": "bekenstein_cascade", "args": [1.0, 1.0]}])
    if (not vacuous["halted"] or "axis_floor" not in vacuous["halt_reason"]
            or [r["scalar"] for r in vacuous["receipts"]] != [0.0]):
        raise AssertionError("a vacuous risk bound must score 0.0 and halt on axis_floor")

    check = kernel.selfcheck()
    if check["falsifiable_demonstrated"] is not True or check["registry_count"] != 21:
        raise AssertionError(f"selfcheck failed: {check}")
    summary = kernel.atlas_summary()
    if summary["executable_formula_count"] != 21:
        raise AssertionError(f"the packaged atlas must list the 21 executable formulas: {summary}")
    return {"registry_count": 21, "locked_proven": sorted(LOCKED_8), "clean_replay_ok": True,
            "tamper_detected": True, "vacuous_risk_halts": True,
            "atlas_attributed": summary["attributed_formula_count"]}
