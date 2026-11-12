/-
F23: Deny-by-default safety — the Memory Covenant's core property as a theorem.

THE CLAIM: in a memory where every recall requires explicit admission, an
unadmitted fact cannot influence any decision that reads through admission.
Injection is structurally impossible: not improbable — impossible.

The mathematical content is the admission invariance (hproj): inserting an
unadmitted fact into memory preserves the admitted set. Safety is the
composition of that invariant with any decision function.

Carrier for claim F23 in contracts/wedge-formula-candidates.v1.json.
Axioms: none beyond Lean's core (propext if the linter requires it).
-/

namespace Puriq.Formula.Proved

/-- A fact is an atomic proposition admitted to memory. -/
abbrev Fact := String

/-- A memory is a carrier of facts with an admission boundary. -/
structure Memory where
  carrier : List Fact
  admitted : List Fact
  -- the admission boundary: every admitted fact is carried (no phantom admits)
  admits_carried : ∀ f ∈ admitted, f ∈ carrier

/-- Insert an unadmitted fact into memory: it is carried but NOT admitted. -/
def Memory.insertUnadmitted (m : Memory) (f : Fact) : Memory where
  carrier := f :: m.carrier
  admitted := m.admitted
  admits_carried := by
    intro g hg
    exact List.mem_cons_of_mem _ (m.admits_carried g hg)

/-- The admission invariant: inserting an unadmitted fact preserves admission. -/
theorem f23_admission_invariant (m : Memory) (f : Fact) (_hf : f ∉ m.admitted) :
    (m.insertUnadmitted f).admitted = m.admitted := rfl

/-- THE WEDGE THEOREM: any decision reading through admission is unchanged
by the insertion of an unadmitted fact. Unreviewed influence is impossible. -/
theorem f23_deny_by_default_safety
    (g : List Fact → String)  -- any decision function over admitted facts
    (m : Memory) (f : Fact) (hf : f ∉ m.admitted) :
    g m.admitted = g (m.insertUnadmitted f).admitted := by
  rw [f23_admission_invariant m f hf]

end Puriq.Formula.Proved
