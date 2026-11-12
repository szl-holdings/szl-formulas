/-
MC1: Deny-by-default safety, as a candidate. Not a locked formula.

The informal claim is that a decision which reads only the admitted facts is
unchanged when an unadmitted fact is inserted. The formalization makes that
true by construction: `insertUnadmitted` copies `admitted`, so the equality
is `rfl`. The hypothesis that the inserted fact was not already admitted is
unused.

This is not F23. In the formula atlas and in lutar-lean, F23 is the Λ
uniqueness claim (Conjecture 1).

Status: CANDIDATE. Not an entry in the locked formula reference.
No `sorry`. Axioms: none beyond Lean's core.
-/

namespace Puriq.Formula.Candidate

/-- A fact is an atomic proposition admitted to memory. -/
abbrev Fact := String

/-- A memory is a carrier of facts with an admission boundary. -/
structure Memory where
  carrier : List Fact
  admitted : List Fact
  -- every admitted fact is carried (no phantom admits)
  admits_carried : ∀ f ∈ admitted, f ∈ carrier

/-- Insert a fact into the carrier without adding it to the admitted list. -/
def Memory.insertUnadmitted (m : Memory) (f : Fact) : Memory where
  carrier := f :: m.carrier
  admitted := m.admitted
  admits_carried := by
    intro g hg
    exact List.mem_cons_of_mem _ (m.admits_carried g hg)

/-- Definitional: insertion copies the admitted list. The non-membership
hypothesis is unused. -/
theorem mc1_admission_invariant (m : Memory) (f : Fact) (_hf : f ∉ m.admitted) :
    (m.insertUnadmitted f).admitted = m.admitted := rfl

/-- Any function of the admitted list is unchanged, because the two lists are
definitionally equal. This is `mc1_admission_invariant`, not a separate argument. -/
theorem mc1_deny_by_default_safety
    (g : List Fact → String)
    (m : Memory) (f : Fact) (hf : f ∉ m.admitted) :
    g m.admitted = g (m.insertUnadmitted f).admitted := by
  rw [mc1_admission_invariant m f hf]

end Puriq.Formula.Candidate
