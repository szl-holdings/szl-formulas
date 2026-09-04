# SPDX-License-Identifier: Apache-2.0
# © 2026 SZL Holdings · Stephen P. Lutar · ORCID 0009-0001-0110-4173
"""szl_formulas — the 21 canonical SZL formulas as a governed-kernel model.

============================ HONEST SCOPE BOX ============================
This is NOT a trained model. There are NO weights (.safetensors/.bin/.pt/.gguf).
It is a pure-Python, stdlib-only governance kernel: an OFFLINE REPLAY of the
live Alloy formula surface — the 21 canonical, pure, typed formulas of
SZLHOLDINGS/canonical-formulas-v1 plus the Codex-Kernel governed-loop composer.
No network, no torch, no tensors. `get_kernel`-discoverable purely so the SZL
family loads the same way; it does not import torch.

PROOF-STATUS honesty (binding — Doctrine v11):
  - Each formula carries the EXACT per-obligation PROOF-STATUS the
    canonical-formulas-v1 dataset declares, mirrored VERBATIM: one of
    PROVEN / AXIOM / SORRY / CONJECTURE (see PROOF_STATUS).
  - A per-obligation "PROVEN(...)" tag is an OBLIGATION-LEVEL label (that narrow
    property is discharged or exact); it is NOT a claim that the formula is a
    member of the locked-proven canonical set.
  - The LOCKED-PROVEN canonical set is EXACTLY 8 — machine-enforced by the
    no-axiom theorem `locked_count_eight` in szl-holdings/lutar-lean
    (PROVEN_FORMULAS.md): {F1, F4, F7, F11, F12, F18, F19, F22}. That count is
    frozen at 8; see LOCKED_PROVEN_FORMULA_IDS / LOCKED_PROVEN_COUNT.
  - The F-numbering is lutar-lean's own corpus (>=22 formulas). Its mapping onto
    these 21 registry entries is NOT asserted here (UNKNOWN — never fabricated).
  - Λ uniqueness stays CONJECTURE 1 (open) on every surface; the composer's
    Λ roll-up is ADVISORY only — a high Λ is a non-compensatory advisory signal,
    NEVER proven trust. Trust never reaches 100%.
=========================================================================

Quickstart (offline):

    from kernels import get_kernel
    fx = get_kernel("SZLHOLDINGS/szl-formulas", revision="main", trust_remote_code=True)

    print(fx.registry_count())                    # 21
    print(fx.lambda_aggregate([0.9, 0.8, 0.95]))  # weighted geometric mean
    print(fx.proof_status("madhava_series"))       # exact dataset PROOF-STATUS
    print(fx.LOCKED_PROVEN_FORMULA_IDS)            # frozenset of exactly 8 F-ids

    chain = fx.run_governed_loop([
        {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
        {"formula_name": "reed_solomon_singleton", "args": [255, 223]},
    ])
    print(chain["replay_ok"], chain["lambda_label"])  # True, ADVISORY (Conjecture 1)
"""
from __future__ import annotations

from typing import Any, Dict, List

from ._formulas import (
    AXIS_BANDS,
    DEFAULT_AXIS_COUNT,
    EPS,
    LEGACY_AXIS_COUNT,
    PROOF_STATUS,
    REGISTRY,
    axis_floors,
    bekenstein_cascade,
    bohr_complementarity_floor,
    css_ingress_verify,
    dsse_envelope,
    fisher_rao_distance,
    gleason_quantum_lambda,
    hoeffding_tail,
    khipu_merkle_root,
    kitaev_surface_correct,
    kochen_specker_18vector_witness,
    lambda_aggregate,
    lambda_bounded,
    lambda_homogeneous,
    madhava_series,
    pac_bayes_mcallester,
    pinsker_kl_bound,
    reed_solomon_singleton,
    registry_count,
    reidemeister_invariant,
    schur_concave_lambda_two_axis,
    shor_codeword_distance,
    two_witness_ks18_soundness,
)
from ._composer import (
    ALLOWED_STEPS,
    AXIS_FLOOR,
    GENESIS,
    LAMBDA_LABEL,
    run_governed_loop,
    verify_chain,
)

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# Locked-proven canonical set — EXACTLY 8 (machine-enforced, no-axiom theorem  #
# `locked_count_eight` in szl-holdings/lutar-lean PROVEN_FORMULAS.md).         #
# This is DISTINCT from the per-obligation PROOF_STATUS "PROVEN(...)" tags.    #
# --------------------------------------------------------------------------- #
LOCKED_PROVEN_FORMULA_IDS = frozenset({"F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"})
LOCKED_PROVEN_COUNT = 8

LOCKED_PROVEN_NOTE = (
    "The locked-proven canonical set is EXACTLY 8 {F1,F4,F7,F11,F12,F18,F19,F22}, "
    "machine-enforced by the no-axiom Lean theorem `locked_count_eight` "
    "(szl-holdings/lutar-lean PROVEN_FORMULAS.md). A per-formula PROOF_STATUS of "
    "'PROVEN(...)' is an OBLIGATION-LEVEL label (that narrow property is discharged "
    "or numerically exact) and is NOT a claim of membership in this locked-8 set. "
    "The F-numbering belongs to the lutar-lean corpus (>=22 formulas); its mapping "
    "onto these 21 registry entries is NOT asserted here (UNKNOWN — never fabricated)."
)

DOCTRINE = (
    "21 canonical formulas, pure + typed + stdlib-only. Every PROOF_STATUS is "
    "mirrored VERBATIM from SZLHOLDINGS/canonical-formulas-v1. Locked-proven = "
    "exactly 8 (separate machine-enforced set). Λ = weighted geometric mean (D2); "
    "Λ uniqueness = Conjecture 1 (open); the composer's Λ roll-up is ADVISORY only."
)

LAMBDA_STATUS = "Conjecture 1 (open) — uniqueness unproven (Uniqueness.lean:120 sorry); advisory only"

PROVENANCE = {
    "mirrors": "SZLHOLDINGS/canonical-formulas-v1 (code/python/formulas.py + composer.py)",
    "lean_repo": "szl-holdings/lutar-lean",
    "doi_lutar_lean": "10.5281/zenodo.20434308",
    "doi_concept": "10.5281/zenodo.19944926",
    "lambda_status": LAMBDA_STATUS,
    "locked_proven_count": LOCKED_PROVEN_COUNT,
    "trained_weights_present": False,
}

DOCTRINE_FOOTER = (
    "SZL Holdings · 21 canonical formulas · PROOF_STATUS mirrored verbatim · "
    "locked-proven = exactly 8 (machine-enforced) · Λ = Conjecture 1, advisory only "
    "· honesty over checklist"
)

__all__ = [
    # canonical formulas (21)
    "lambda_aggregate",
    "lambda_homogeneous",
    "lambda_bounded",
    "pac_bayes_mcallester",
    "bekenstein_cascade",
    "reidemeister_invariant",
    "khipu_merkle_root",
    "dsse_envelope",
    "gleason_quantum_lambda",
    "hoeffding_tail",
    "pinsker_kl_bound",
    "fisher_rao_distance",
    "bohr_complementarity_floor",
    "kochen_specker_18vector_witness",
    "two_witness_ks18_soundness",
    "shor_codeword_distance",
    "css_ingress_verify",
    "kitaev_surface_correct",
    "reed_solomon_singleton",
    "madhava_series",
    "schur_concave_lambda_two_axis",
    # registry + proof status
    "REGISTRY",
    "PROOF_STATUS",
    "registry_count",
    "list_formulas",
    "proof_status",
    "lambda_status",
    # locked-proven canonical set (exactly 8)
    "LOCKED_PROVEN_FORMULA_IDS",
    "LOCKED_PROVEN_COUNT",
    "LOCKED_PROVEN_NOTE",
    # composer
    "run_governed_loop",
    "verify_chain",
    "GENESIS",
    "ALLOWED_STEPS",
    "AXIS_FLOOR",
    "LAMBDA_LABEL",
    # axis schema helpers
    "axis_floors",
    "AXIS_BANDS",
    "DEFAULT_AXIS_COUNT",
    "LEGACY_AXIS_COUNT",
    "EPS",
    # metadata
    "DOCTRINE",
    "LAMBDA_STATUS",
    "PROVENANCE",
    "DOCTRINE_FOOTER",
    "selfcheck",
    "__version__",
]


def proof_status(name: str) -> str:
    """Return the EXACT per-obligation PROOF-STATUS the canonical dataset declares
    for `name` (mirrored verbatim). Raises KeyError for an unknown formula —
    never coerced to a silent default."""
    return PROOF_STATUS[name]


def list_formulas() -> List[Dict[str, str]]:
    """List every canonical formula with its declared (verbatim) PROOF-STATUS.

    NOTE: `proof_status` here is the OBLIGATION-LEVEL label from the dataset; it is
    NOT the locked-proven-8 membership claim (see LOCKED_PROVEN_NOTE)."""
    return [
        {"name": name, "proof_status": PROOF_STATUS[name]}
        for name in REGISTRY
    ]


def lambda_status() -> str:
    """Λ status — always Conjecture 1 (open). Never 'proven'."""
    return LAMBDA_STATUS


def selfcheck() -> Dict[str, Any]:
    """One-shot stdlib-only health check. FALSIFIABLE — a wrong registry count,
    a coerced proof-status, a broken locked-8 count, or a tampered receipt chain
    would flip these. Does NOT touch Λ (Conjecture 1)."""
    # A clean governed loop replays; a tampered one must NOT.
    clean = run_governed_loop([
        {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
        {"formula_name": "reed_solomon_singleton", "args": [255, 223]},
    ])
    tampered = run_governed_loop([
        {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
    ])
    calls_t = [{"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]}]
    if tampered["receipts"]:
        tampered["receipts"][0]["receipt_hash"] = "0" * 64
    tamper_detected = not verify_chain(tampered, calls_t)

    return {
        "version": __version__,
        "registry_count": registry_count(),  # 21
        "proof_status_covers_all": set(PROOF_STATUS) == set(REGISTRY),
        "locked_proven_count": LOCKED_PROVEN_COUNT,  # 8
        "locked_proven_ids": sorted(LOCKED_PROVEN_FORMULA_IDS),
        "locked_count_is_eight": len(LOCKED_PROVEN_FORMULA_IDS) == LOCKED_PROVEN_COUNT == 8,
        "clean_replay_ok": clean["replay_ok"],
        "tamper_detected": tamper_detected,
        "falsifiable_demonstrated": clean["replay_ok"] and tamper_detected,
        "lambda_status": LAMBDA_STATUS,
        "lambda_label": LAMBDA_LABEL,
    }

# Attributed formula/quant atlas. This does not alter locked-proof membership.
from .atlas import (  # noqa: E402
    atlas_summary,
    list_attributed_formulas,
    load_formula_atlas,
    quant_domains,
)

__all__.extend([
    "load_formula_atlas",
    "atlas_summary",
    "quant_domains",
    "list_attributed_formulas",
])
