/-
Candidate restatement of the Memory Covenant wedge formulas.

These are candidates. They are not locked formulas. They are not entries in
contracts/locked-formula-reference.v1.json. This file does not certify an exit
code, a sorry count, or a kernel check.

MC1 is not F23. F23 in the formula atlas and in lutar-lean is the Λ uniqueness
claim (Conjecture 1).

Carriers, compiled as separate files under the toolchain in lean/lean-toolchain
(leanprover/lean4:v4.34.1):
  lean/MC1_deny_by_default.lean
  lean/F24_F25_covenant.lean
-/

-- CANDIDATE MC1, deny-by-default. Not atlas F23.
-- For any decision g over admitted facts, any memory m, and any fact f
-- with f ∉ m.admitted:
--   g m.admitted = g (insertUnadmitted m f).admitted
-- The carrier proves this by rfl: insertUnadmitted copies admitted, and the
-- non-membership hypothesis is unused.
-- Name: Puriq.Formula.Candidate.mc1_deny_by_default_safety

-- CANDIDATE F24, staleness propagation.
-- For any list of tiers containing Tier.stale, derive (the worst-take fold)
-- is stale or unavailable.
-- Name: Puriq.Formula.Candidate.f24_staleness_propagation
-- The carrier is an induction on the list, not a definitional equality.

-- CANDIDATE F25, learning monotonicity.
-- (a) Every previously admitted fact stays admitted.
--     Name: Puriq.Formula.Candidate.f25_admitted_monotone
--     True because reviewCycle appends onto admitted.
-- (b) A fact outside the old admitted list and inside the new one is in the
--     new reviewed list, because both fields append the same filter.
--     Name: Puriq.Formula.Candidate.f25_no_unreviewed_entry
--     This does not, by itself, say the fact was reviewed before the cycle.
-- (c) A fact added by reviewCycle was already in the old reviewed list.
--     Name: Puriq.Formula.Candidate.f25_new_admit_was_reviewed
--     True because the filter predicate is membership in that list.
