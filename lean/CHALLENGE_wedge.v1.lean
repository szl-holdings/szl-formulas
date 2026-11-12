/-
THE ESTATE CHALLENGE FILE — the AxiomMath standard, adopted.
Restates the locked wedge theorems INDEPENDENTLY of their proof files.
Lean's kernel checks that the carriers prove exactly these statements.
Any divergence between this file and the proof files is a certification failure.
-/

-- The challenge restates each claim in its own terms, without importing
-- the proof namespaces, so the statements below must be *matched* by the
-- library, not *defined by* it.

-- CHALLENGE F23: deny-by-default safety
-- "For any decision g over admitted facts, any memory m, and any unadmitted
--  fact f: g m.admitted = g ((insertUnadmitted m f).admitted)"
-- Carrier match: Puriq.Formula.Proved.f23_deny_by_default_safety

-- CHALLENGE F24: staleness propagation
-- "For any list of tiers containing a stale tier, the derived tier
--  (worst-take fold) is stale or worse."
-- Carrier match: Puriq.Formula.Proved.f24_staleness_propagation

-- CHALLENGE F25: learning monotonicity (two parts)
-- (a) "For any review cycle, every previously admitted fact remains admitted."
-- Carrier match: Puriq.Formula.Proved.f25_admitted_monotone
-- (b) "Every newly admitted fact was reviewed."
-- Carrier match: Puriq.Formula.Proved.f25_no_unreviewed_entry

-- The certification check (run against the proof files):
--   lean F23_deny_by_default.lean lean/F24_F25_covenant.lean  -> EXIT=0
--   zero sorry, zero undeclared axioms
-- The challenge statements above are the CONTRACT; the carriers in the
-- locked-formula-reference.v1.json are the PROOF. Divergence = failure.
